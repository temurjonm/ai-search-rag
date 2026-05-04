# ai-search-rag

![RAG system design](./rag-system-design.png)

A production-shaped Retrieval-Augmented Generation (RAG) search service. Upload documents, then ask questions and get grounded answers with citations.

Built with **FastAPI**, **PostgreSQL + pgvector**, and the **OpenAI API**. Combines vector and keyword search via Reciprocal Rank Fusion, falls back to OCR for scanned PDFs, and ships with a single-page UI that streams live pipeline progress over Server-Sent Events.

---

## Highlights

- **Hybrid retrieval** — pgvector cosine similarity fused with PostgreSQL full-text search (`ts_rank_cd` over `websearch_to_tsquery`) using Reciprocal Rank Fusion (`k=60`).
- **OCR fallback for scanned PDFs** — `pypdf` first; if a PDF returns no extractable text, the page images are rasterized via `pdf2image` and OCR'd with Tesseract, page-by-page.
- **Live progress streaming** — `POST /documents/stream` emits SSE events for each phase (extract → OCR → chunk → embed → save). The bundled UI renders them as an animated timeline so users can see exactly where time is being spent.
- **Multi-tenant data isolation** — every row is scoped to a `tenant_id`; all queries filter by it.
- **Defensive uploads** — validated tenant IDs, sanitized filenames with path-containment checks, 25 MB size cap, file-type allowlist.
- **Cited answers** — the LLM is constrained to cite `[Source N]` from the retrieved chunks and to refuse when sources don't support an answer.
- **Zero-build frontend** — vanilla HTML/CSS/JS in a single file, served from FastAPI.

---

## Architecture

The diagram at the top of this README shows the full request/response paths between the browser, FastAPI, the ingestion and retrieval pipelines, PostgreSQL + pgvector, and the OpenAI API.

### Ingestion pipeline

1. **Receive** — read upload bytes; validate size, filename, tenant.
2. **Extract** — `pypdf.PdfReader` for PDFs; UTF-8 decode for `.txt`.
3. **OCR fallback** *(only when extracted text < 20 chars)* — `pdf2image` rasterizes each page at 300 DPI, `pytesseract` reads each image. Runs page-by-page in a thread pool so the SSE stream can yield progress.
4. **Chunk** — sliding window over normalized whitespace (default 1200 chars / 200 overlap).
5. **Embed** — OpenAI `text-embedding-3-small` (1536 dims), batched 100 at a time.
6. **Persist** — atomic write of `Document` + N `Chunk` rows in a single transaction.

### Retrieval pipeline

1. Embed the query with the same model.
2. **Vector candidates**: cosine distance via pgvector (`<=>`), top 4×k.
3. **Keyword candidates**: `to_tsvector('english', ...)` matched against `websearch_to_tsquery`, ranked by `ts_rank_cd`, top 4×k.
4. **Fusion**: Reciprocal Rank Fusion combines both lists, keep top `k`.
5. **Answer**: `gpt-4o-mini` (Responses API) is given the retrieved chunks and constrained to cite them or refuse.

---

## Tech stack

| Layer        | Tool                                              |
| ------------ | ------------------------------------------------- |
| API          | FastAPI · Uvicorn · Pydantic v2                   |
| Database     | PostgreSQL 16 · pgvector                          |
| ORM          | SQLAlchemy 2.0 · psycopg 3                        |
| LLM provider | OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)  |
| PDF / OCR    | pypdf · pdf2image (poppler) · pytesseract         |
| Frontend     | Vanilla HTML/CSS/JS (no build step)               |
| Tests        | pytest                                            |
| Lint         | ruff                                              |

---

## Quick start

### Prerequisites

- Python ≥ 3.11
- Docker (for the Postgres+pgvector container)
- macOS or Linux. For OCR support on macOS:
  ```bash
  brew install poppler tesseract
  ```
  On Debian/Ubuntu: `sudo apt-get install -y poppler-utils tesseract-ocr`

### Setup

```bash
git clone <your-repo-url> ai-search-rag
cd ai-search-rag

# 1. Start Postgres + pgvector
docker compose up -d

# 2. Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# edit .env: paste your OPENAI_API_KEY

# 4. Run
uvicorn ai_search.main:app --reload
```

Open <http://127.0.0.1:8000/> for the UI.

> The provided `docker-compose.yml` exposes Postgres on **5433** to avoid colliding with a local Postgres install on 5432. The default `DATABASE_URL` matches.

---

## Using the UI

1. Set a **tenant ID** (any short string — `demo` is fine).
2. **Upload** a `.pdf` or `.txt`. A live timeline appears showing each phase: upload → extract → (OCR if needed) → chunk → embed → save. Each step has a status dot and, where applicable, a progress bar (per-page for OCR, per-batch for embeddings).
3. Type a question and press **Ask** (or `⌘/Ctrl + Enter`). The answer is shown with citations, followed by the source chunks used.

---

## API

### `POST /documents`

Synchronous upload. Returns once the document is fully indexed.

```bash
curl -F tenant_id=demo -F file=@./paper.pdf http://127.0.0.1:8000/documents
```

```json
{
  "document_id": 1,
  "filename": "paper.pdf",
  "chunk_count": 42,
  "status": "ready"
}
```

### `POST /documents/stream`

Same input, but returns `text/event-stream`. Each event is a JSON object with a `type` discriminator (`received`, `phase`, `extracted_pypdf`, `ocr_start`, `ocr_page`, `chunked`, `embedded_batch`, `done`, `error`). Used by the UI for the timeline.

### `POST /search`

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"demo","question":"What is RRF?","top_k":5}'
```

```json
{
  "answer": "Reciprocal Rank Fusion combines ranked lists ... [Source 1]",
  "sources": [
    {
      "document_id": 1,
      "filename": "paper.pdf",
      "chunk_id": 17,
      "chunk_index": 3,
      "score": 0.0312,
      "excerpt": "..."
    }
  ]
}
```

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

---

## Configuration

All settings are env-driven via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) and read from `.env`.

| Variable          | Default                                                 | Notes                                          |
| ----------------- | ------------------------------------------------------- | ---------------------------------------------- |
| `OPENAI_API_KEY`  | *(required)*                                            | Server fails fast at import if unset           |
| `DATABASE_URL`    | `postgresql+psycopg://rag:rag@localhost:5433/rag`       | Matches the bundled docker-compose mapping     |
| `STORAGE_DIR`     | `storage`                                               | Per-tenant subdirectories                      |
| `EMBEDDING_MODEL` | `text-embedding-3-small`                                | Must produce 1536-dim vectors                  |
| `ANSWER_MODEL`    | `gpt-4o-mini`                                           | Used via the Responses API                     |
| `CHUNK_SIZE`      | `1200`                                                  | Characters per chunk                           |
| `CHUNK_OVERLAP`   | `200`                                                   | Must be `< CHUNK_SIZE`                         |

---

## Project layout

```
ai-search-rag/
├── docker-compose.yml          Postgres + pgvector
├── pyproject.toml              Dependencies and tool config
├── SECURITY.md                 Threat model and deployment caveats
├── src/ai_search/
│   ├── main.py                 FastAPI app + routes
│   ├── config.py               Settings via pydantic-settings
│   ├── db.py                   Engine, session, init
│   ├── models.py               SQLAlchemy + pgvector models
│   ├── schemas.py              Pydantic request/response models
│   ├── ingestion.py            Upload, validate, chunk, embed, persist
│   ├── document_loader.py      pypdf + Tesseract OCR fallback
│   ├── chunking.py             Sliding-window chunker
│   ├── retrieval.py            Vector + keyword + RRF
│   ├── openai_client.py        Embeddings + answer generation
│   └── static/index.html       Single-page UI
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
└── storage/                    Per-tenant uploaded files (gitignored)
```

---

## Testing

```bash
.venv/bin/pytest -q
```

Tests cover the chunker boundaries and the retrieval fusion logic without requiring the database or OpenAI. They are intentionally lightweight; an end-to-end suite would require a Postgres test fixture and recorded fixture responses.

---

## Security

This MVP is intended for **local development**. Before deploying anywhere reachable from the public internet, read [`SECURITY.md`](./SECURITY.md). It enumerates what is in place (tenant scoping, path-traversal hardening, parameterized SQL, upload limits, output escaping) and — more importantly — what is missing (authentication, rate limiting, HTTPS, virus scanning, auth-bound tenancy).

---

## Roadmap ideas

- Authentication and per-user tenant binding
- Streaming answer generation (token-by-token in the UI)
- Re-ranking pass with a cross-encoder
- Background ingestion worker (Celery / arq) to free the API thread
- Per-document delete endpoint and tenant data export
- Configurable embedding dimensions (`text-embedding-3-large`)

---

## License

TBD — pick one before publishing (MIT and Apache-2.0 are common defaults).
