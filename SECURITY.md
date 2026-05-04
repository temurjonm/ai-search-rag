# Security Notes

This is an MVP intended for **local development only**. Do not expose it to the public internet without adding the controls listed below.

## What's already in place

- **Secrets are not in git.** `.env` is gitignored; only `.env.example` is committed.
- **Tenant isolation at the data layer.** All retrieval queries filter by `tenant_id`.
- **Path-traversal hardening.** `tenant_id` is validated against `^[A-Za-z0-9_-]{1,64}$`; uploaded filenames are sanitized and confined under `storage/<tenant>/`.
- **Upload size limit.** 25 MB per file (`MAX_UPLOAD_BYTES` in `src/ai_search/ingestion.py`).
- **File-type allowlist.** Only `.pdf` and `.txt`.
- **Parameterized SQL.** Keyword search uses bind parameters; vector search goes through SQLAlchemy ORM.
- **HTML output is escaped** in the frontend.

## What's NOT in place — add before deploying

- **No authentication or authorization.** Anyone who can reach the HTTP port can upload and search any tenant's data. Add auth (API keys, OAuth, mTLS) before exposing.
- **No rate limiting.** A single client can drain your OpenAI quota by uploading large PDFs in a loop. Add a reverse proxy (nginx, Caddy) or middleware (slowapi).
- **No CORS policy.** The default allows same-origin only, but if you add a separate frontend you'll need to configure `CORSMiddleware` explicitly.
- **No HTTPS.** Terminate TLS at a reverse proxy.
- **No virus scanning of uploads.** Files are written to disk verbatim.
- **OCR uses subprocess tools (`tesseract`, `pdftoppm`).** Keep them updated.
- **Tenant_id is client-supplied with no auth tie-in.** Any caller can claim any tenant_id; data is only "isolated" in the sense that two different tenant_id strings see different data — there's no enforcement that a given user owns a given tenant_id.

## If a key leaks

1. Rotate immediately at <https://platform.openai.com/api-keys>.
2. Revoke the old key.
3. Audit usage on the OpenAI dashboard.
4. Check `git log` and `git reflog` to confirm no commit captured the key. If one did: rewrite history (`git filter-repo`) and force-push, then still rotate — assume the key is public the moment it touched a repo.
