import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from ai_search.chunking import chunk_text
from ai_search.config import get_settings
from ai_search.document_loader import (
    MIN_TEXT_LEN,
    extract_pdf_pypdf,
    extract_text,
    ocr_pdf_progress,
)
from ai_search.models import Chunk, Document
from ai_search.openai_client import embed_texts


settings = get_settings()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_FILENAME_LEN = 120
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ALLOWED_SUFFIXES = {".pdf", ".txt"}


def _validate_tenant_id(tenant_id: str) -> str:
    if not TENANT_ID_RE.match(tenant_id or ""):
        raise ValueError("tenant_id must be 1-64 chars of letters, digits, '_' or '-'")
    return tenant_id


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.txt").name  # strip any directory components
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or "upload"
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only .txt and .pdf files are supported in the MVP")
    return name[:MAX_FILENAME_LEN]


def _check_size(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    if not data:
        raise ValueError("Uploaded file is empty")


async def ingest_upload(db: Session, tenant_id: str, file: UploadFile) -> tuple[Document, int]:
    tenant_id = _validate_tenant_id(tenant_id)
    data = await file.read()
    _check_size(data)
    filename = _safe_filename(file.filename or "upload.txt")
    content_type = file.content_type or "application/octet-stream"

    text = extract_text(filename, content_type, data)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("Document did not contain extractable text")

    storage_path = _store_file(tenant_id, filename, data)
    document = Document(
        tenant_id=tenant_id,
        filename=filename,
        content_type=content_type,
        storage_path=str(storage_path),
        status="processing",
    )
    db.add(document)
    db.flush()

    embeddings: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings.extend(await embed_texts(batch))

    for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        db.add(
            Chunk(
                tenant_id=tenant_id,
                document_id=document.id,
                chunk_index=index,
                content=content,
                embedding=embedding,
            )
        )

    document.status = "ready"
    db.commit()
    db.refresh(document)
    return document, len(chunks)


async def ingest_upload_stream(
    db: Session, tenant_id: str, file: UploadFile
) -> AsyncIterator[dict]:
    """Async generator yielding step-by-step progress events for the UI."""
    try:
        tenant_id = _validate_tenant_id(tenant_id)
    except ValueError as exc:
        yield {"type": "error", "detail": str(exc)}
        return
    data = await file.read()
    try:
        _check_size(data)
        filename = _safe_filename(file.filename or "upload.txt")
    except ValueError as exc:
        yield {"type": "error", "detail": str(exc)}
        return
    content_type = file.content_type or "application/octet-stream"
    yield {"type": "received", "filename": filename, "bytes": len(data)}

    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_txt = content_type.startswith("text/") or filename.lower().endswith(".txt")

    if is_pdf:
        yield {"type": "phase", "name": "extract_pdf", "label": "Extracting PDF text"}
        text = await asyncio.to_thread(extract_pdf_pypdf, data)
        yield {"type": "extracted_pypdf", "chars": len(text.strip())}
        if len(text.strip()) < MIN_TEXT_LEN:
            yield {"type": "phase", "name": "ocr", "label": "Falling back to OCR"}
            async for ev in ocr_pdf_progress(data):
                if ev["type"] == "ocr_done":
                    text = ev["text"]
                else:
                    yield ev
    elif is_txt:
        yield {"type": "phase", "name": "decode", "label": "Decoding text"}
        text = data.decode("utf-8", errors="ignore")
    else:
        yield {"type": "error", "detail": "Only .txt and .pdf files are supported in the MVP"}
        return

    yield {"type": "extracted", "chars": len(text)}

    yield {"type": "phase", "name": "chunk", "label": "Chunking text"}
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        yield {"type": "error", "detail": "Document did not contain extractable text"}
        return
    yield {"type": "chunked", "count": len(chunks)}

    storage_path = _store_file(tenant_id, filename, data)
    document = Document(
        tenant_id=tenant_id,
        filename=filename,
        content_type=content_type,
        storage_path=str(storage_path),
        status="processing",
    )
    db.add(document)
    db.flush()

    yield {"type": "phase", "name": "embed", "label": "Embedding chunks"}
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    embeddings: list[list[float]] = []
    for batch_index, start in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[start : start + batch_size]
        embeddings.extend(await embed_texts(batch))
        yield {
            "type": "embedded_batch",
            "batch": batch_index + 1,
            "total": total_batches,
            "chunks_done": min(start + batch_size, len(chunks)),
            "chunks_total": len(chunks),
        }

    yield {"type": "phase", "name": "save", "label": "Saving to database"}
    for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        db.add(
            Chunk(
                tenant_id=tenant_id,
                document_id=document.id,
                chunk_index=index,
                content=content,
                embedding=embedding,
            )
        )
    document.status = "ready"
    db.commit()
    db.refresh(document)

    yield {
        "type": "done",
        "document_id": document.id,
        "filename": document.filename,
        "chunk_count": len(chunks),
        "status": document.status,
    }


def _store_file(tenant_id: str, filename: str, data: bytes) -> Path:
    # tenant_id and filename are pre-validated; resolve and confirm containment.
    storage_root = settings.storage_dir.resolve()
    tenant_dir = (storage_root / tenant_id).resolve()
    if storage_root not in tenant_dir.parents and tenant_dir != storage_root:
        raise ValueError("Invalid tenant path")
    tenant_dir.mkdir(parents=True, exist_ok=True)
    path = (tenant_dir / filename).resolve()
    if tenant_dir not in path.parents:
        raise ValueError("Invalid file path")
    path.write_bytes(data)
    return path
