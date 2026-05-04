import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from ai_search.db import get_db, init_db
from ai_search.ingestion import ingest_upload, ingest_upload_stream
from ai_search.openai_client import embed_texts, generate_answer
from ai_search.retrieval import hybrid_search
from ai_search.schemas import DocumentResponse, SearchRequest, SearchResponse


app = FastAPI(title="AI Search RAG MVP")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(
    tenant_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    try:
        document, chunk_count = await ingest_upload(db, tenant_id, file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentResponse(
        document_id=document.id,
        filename=document.filename,
        chunk_count=chunk_count,
        status=document.status,
    )


@app.post("/documents/stream")
async def upload_document_stream(
    tenant_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in ingest_upload_stream(db, tenant_id, file):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    embeddings = await embed_texts([request.question])
    sources = hybrid_search(
        db=db,
        tenant_id=request.tenant_id,
        question=request.question,
        question_embedding=embeddings[0],
        top_k=request.top_k,
    )

    if not sources:
        return SearchResponse(
            answer="I do not have enough information in the provided documents to answer this.",
            sources=[],
        )

    answer = await generate_answer(request.question, sources)
    return SearchResponse(answer=answer, sources=sources)