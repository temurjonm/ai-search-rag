import asyncio
from collections.abc import AsyncIterator
from io import BytesIO

import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader

OCR_DPI = 300
MIN_TEXT_LEN = 20


def extract_text(filename: str, content_type: str, data: bytes) -> str:
    lower_name = filename.lower()

    if content_type == "application/pdf" or lower_name.endswith(".pdf"):
        return _extract_pdf(data)

    if content_type.startswith("text/") or lower_name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    raise ValueError("Only .txt and .pdf files are supported in the MVP")


def _extract_pdf(data: bytes) -> str:
    text = extract_pdf_pypdf(data)
    if len(text.strip()) >= MIN_TEXT_LEN:
        return text
    return _ocr_pdf_sync(data)


def extract_pdf_pypdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _ocr_pdf_sync(data: bytes) -> str:
    images = convert_from_bytes(data, dpi=OCR_DPI)
    pages = [pytesseract.image_to_string(image) for image in images]
    return "\n\n".join(pages)


async def ocr_pdf_progress(data: bytes) -> AsyncIterator[dict]:
    """Async generator yielding OCR progress events; final event has type=ocr_done with text."""
    images = await asyncio.to_thread(convert_from_bytes, data, dpi=OCR_DPI)
    total = len(images)
    yield {"type": "ocr_start", "pages": total}
    page_texts: list[str] = []
    for i, image in enumerate(images):
        text = await asyncio.to_thread(pytesseract.image_to_string, image)
        page_texts.append(text)
        yield {"type": "ocr_page", "page": i + 1, "total": total, "chars": len(text)}
    yield {"type": "ocr_done", "text": "\n\n".join(page_texts)}
