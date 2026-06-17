# RAGu frontend

A small Vite + React + TypeScript UI for the RAGu showcase. It calls the FastAPI
backend (`ragu.web`) to run grounded queries and renders each citation **on the
source page** — the page image is served raw and the word boxes are drawn as
scalable SVG overlays on top (so they stay crisp at any size and animate in).

## Run it (no Node needed)

The production bundle in `dist/` is already built and is served directly by the
Python backend. Just start the server from the **repo root**:

```bash
pip install -e '.[web,ocr,l2,local]'   # web = FastAPI; ocr = page rendering
python -m ragu.web                      # http://127.0.0.1:8000
```

`/api/page` rendering needs the `ocr` extra (pymupdf + pillow); the answer
pipeline needs whatever providers your `.env` is configured for.

## Rebuilding the frontend (needs Node 18+)

```bash
cd frontend
npm install
npm run build        # tsc typecheck + vite build -> dist/  (served by FastAPI)
npm run dev          # optional: hot-reload dev server on :5173, proxies /api -> :8000
```

> **WSL note:** a Windows `node.exe` on PATH cannot build a project that lives on
> the WSL filesystem (it misresolves the CWD). Use a Linux Node — e.g. install
> one in WSL, or download the official linux-x64 tarball — to run `npm`.

## Layout

- `src/api.ts` — typed client + response types (mirrors the backend JSON).
- `src/components/PageViewer.tsx` — the page-image modal with the SVG box overlay.
- `src/components/{QueryBar,AnswerPanel,CitationCard,TracePanel,WorkingSet}.tsx`
- `src/App.tsx` — wiring and state; `src/styles.css` — the whole look.
