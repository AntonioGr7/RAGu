"""FastAPI app powering the RAGu showcase frontend.

Three concerns, kept deliberately small:

* ``POST /api/query``  — run the full L1+L2 pipeline with grounding and return the
  answer, its span-level citations (page + word boxes), the retrieved working set,
  and the reasoning trace, all as plain JSON.
* ``GET  /api/page``   — render one source page to a PNG (no boxes baked in — the
  frontend overlays them as scalable SVG, using the boxes/dims from the citation).
* static files          — serve the built frontend (``frontend/dist``) if present.

The (expensive) :class:`Ragu` facade is built once and shared; queries are
serialised with a lock because the underlying models are not concurrency-safe and
this is a single-user demo, not a service.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path

from pydantic import BaseModel

from ragu.app import Ragu
from ragu.core import Answer, Citation, WorkingSet

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Body of ``POST /api/query``. Defined at module level so FastAPI can resolve
    the annotation as a request body under ``from __future__ import annotations``."""

    query: str
    grounding_source: str = "trajectory"  # trajectory | document | raw

# Repo root -> where the built frontend lands (frontend/dist).
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _citation_json(c: Citation) -> dict:
    """Serialise a citation, flattening boxes to plain lists for JSON/SVG use."""
    return {
        "doc_id": str(c.doc_id),
        "source": c.source,
        "quote": c.quote,
        "start_char": c.start_char,
        "end_char": c.end_char,
        "highlights": [
            {
                "page": h.page,
                "boxes": [list(b) for b in h.boxes],
                "width": h.width,
                "height": h.height,
            }
            for h in c.highlights
        ],
    }


def _answer_json(answer: Answer, working_set: WorkingSet, elapsed_ms: int) -> dict:
    return {
        "answer": answer.text,
        "used_reasoning": answer.used_reasoning,
        "trace": answer.trace,
        "citations": [_citation_json(c) for c in answer.citations],
        "working_set": [{"id": str(d.id), "source": d.source} for d in working_set.documents],
        "elapsed_ms": elapsed_ms,
    }


def create_app() -> "FastAPI":  # type: ignore[name-defined]  # noqa: F821
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response

    app = FastAPI(title="RAGu", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # Allow the Vite dev server (localhost:5173) to call the API during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ragu = Ragu()
    query_lock = asyncio.Lock()
    # Sources we have indexed — the only paths /api/page is allowed to render
    # (so a crafted ?source= can't read arbitrary files off disk).
    allowed_sources: set[str] = set()

    async def _refresh_allowed_sources() -> None:
        refs = await ragu.list_documents()
        allowed_sources.clear()
        allowed_sources.update(r.source for r in refs)

    @app.on_event("startup")
    async def _startup() -> None:
        await _refresh_allowed_sources()
        logger.info("RAGu web ready — %d document(s) indexed.", len(allowed_sources))

    @app.get("/api/documents")
    async def documents() -> dict:
        refs = await ragu.list_documents()
        return {"documents": [{"id": str(r.id), "source": r.source} for r in refs]}

    @app.post("/api/query")
    async def query(req: QueryRequest) -> dict:
        if not req.query.strip():
            raise HTTPException(400, "query is empty")
        async with query_lock:
            t0 = time.perf_counter()
            # Retrieve once for the working-set panel, then answer (which retrieves
            # again internally — cheap relative to L2, and keeps the facade simple).
            working_set = await ragu.retrieve(req.query)
            answer = await ragu.answer(
                req.query, ground=True, grounding_source=req.grounding_source
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return _answer_json(answer, working_set, elapsed_ms)

    @app.get("/api/page")
    def page(
        source: str = Query(...),
        page: int = Query(0, ge=0),
        dpi: int = Query(150, ge=72, le=400),
    ) -> Response:
        if source not in allowed_sources:
            raise HTTPException(404, "unknown source")
        from ragu.render import render_page

        image = render_page(source, page, dpi)
        if image is None:
            raise HTTPException(404, "page not found")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "documents": len(allowed_sources)}

    _mount_frontend(app)
    return app


def _mount_frontend(app: "FastAPI") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Serve the built SPA from ``frontend/dist`` if it exists, with an index
    fallback so client-side routing works. No-op (API only) before the build."""
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    if not _FRONTEND_DIST.is_dir():
        logger.warning("Frontend not built (%s missing) — API only. Run `npm run build`.",
                       _FRONTEND_DIST)
        return

    assets = _FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = _FRONTEND_DIST / "index.html"

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(index)
