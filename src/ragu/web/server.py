"""FastAPI app powering the RAGu chat frontend.

The frontend is a chat: a session lives until the user refreshes (or hits "New
chat"). Each user message starts one reasoning *turn* over the corpus. Because
L2 (vomero) can now **ask the user back** mid-reasoning, a turn may pause: the
server returns the question, the worker thread stays blocked on it, and a later
``/api/respond`` feeds the answer in and resumes the same run.

Concerns, kept deliberately small:

* ``POST /api/chat``     — start a turn for a session; returns either the final
  grounded answer (citations = page + word boxes, working set, trace) or a
  clarifying ``question`` if L2 paused.
* ``POST /api/respond``  — answer L2's pending question; resumes the turn and
  again returns an answer-or-question.
* ``POST /api/reset``    — cancel any in-flight turn for a session and drop it.
* ``GET  /api/page``     — render one source page to a PNG (boxes overlaid by
  the frontend as scalable SVG).
* static files            — serve the built frontend (``frontend/dist``).

The (expensive) :class:`Ragu` facade is built once and shared; turns are
serialised with a lock because the underlying models are not concurrency-safe
and this is a single-user demo, not a service.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import time
from pathlib import Path

from pydantic import BaseModel

from ragu.adapters.reasoning.vomero import (
    ask_handler_var,
    format_step,
    trace_handler_var,
    transcript_var,
)
from ragu.app import Ragu
from ragu.core import Answer, Citation, WorkingSet

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat`` — start a new turn for a session."""

    session_id: str
    message: str
    grounding_source: str = "trajectory"  # trajectory | document | raw
    # Skip L1; L2 reasons over every indexed document. None => use the server's
    # configured default (RAGU_VOMERO__FULL_CORPUS, on).
    full_corpus: bool | None = None


class RespondRequest(BaseModel):
    """Body of ``POST /api/respond`` — answer L2's pending clarifying question."""

    session_id: str
    answer: str


class ResetRequest(BaseModel):
    """Body of ``POST /api/reset`` — cancel any in-flight turn for a session."""

    session_id: str


# Repo root -> where the built frontend lands (frontend/dist).
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

# Sentinel fed to a paused worker to make it stop waiting on a human (on reset).
_CANCELLED = "(the user ended this turn — answer with what you have so far)"


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
        "type": "answer",
        "answer": answer.text,
        "used_reasoning": answer.used_reasoning,
        "trace": answer.trace,
        "citations": [_citation_json(c) for c in answer.citations],
        "working_set": [
            {"id": str(d.id), "source": d.source} for d in working_set.documents
        ],
        "elapsed_ms": elapsed_ms,
    }


class _Turn:
    """One reasoning turn for a session. Runs as a background task; bridges
    vomero's *synchronous* ask-handler (called in a worker thread) to the async
    HTTP layer with two thread-safe queues."""

    def __init__(
        self, session_id: str, message: str, grounding_source: str,
        full_corpus: bool | None = None,
    ) -> None:
        self.session_id = session_id
        self.message = message
        self.grounding_source = grounding_source
        self.full_corpus = full_corpus
        # Worker thread -> server: ("ask", question) / ("done", None) / ("error", exc).
        self.events: queue.Queue = queue.Queue()
        # Server -> worker thread: the user's answer to a pending question.
        self.replies: queue.Queue = queue.Queue()
        self.result: Answer | None = None
        self.working_set: WorkingSet | None = None
        self.t0 = 0.0
        self.elapsed_ms = 0
        self.task: asyncio.Task | None = None
        # L2's reasoning log, appended live by the worker thread; the /api/trace
        # SSE stream polls this list and ``finished`` to drive the log box.
        self.trace_log: list[str] = []
        self.finished = False

    def ask(self, question: str) -> str:
        """vomero's ask-handler — runs in the worker thread. Surfaces the
        question to the server and blocks the run until a reply arrives."""
        self.events.put(("ask", question))
        return self.replies.get()

    def on_event(self, step) -> None:
        """vomero's trace handler — runs in the worker thread, once per step."""
        line = format_step(step)
        if line is not None:
            self.trace_log.append(line)


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
    # Models aren't concurrency-safe; serialise actual reasoning. The lock is held
    # for a whole turn (across any ask-back pause) so two runs never interleave.
    turn_lock = asyncio.Lock()
    # In-flight turn per session (absent when the session is idle).
    turns: dict[str, _Turn] = {}
    # Conversation transcript per session — vomero's resumable history, mutated
    # in place each turn so follow-ups build on what was already asked/answered.
    # Created on first turn, chained across turns, dropped on reset.
    transcripts: dict[str, list] = {}
    # Per-session retrieval continuity: the working set assembled last turn (so a
    # follow-up accumulates evidence instead of retrieving cold) and the previous
    # (question, answer) used to make follow-up retrieval conversation-aware — a
    # multi-hop follow-up retrieves over the resolved intermediate answers, not
    # just the latest message. Both dropped on reset.
    working_sets: dict[str, WorkingSet] = {}
    last_qa: dict[str, tuple[str, str]] = {}
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
        # Build/open L2's search index (and materialize the full corpus) now, so
        # the first question isn't slow. Best-effort: a warmup failure must not
        # stop the server from booting.
        try:
            status = await ragu.warmup()
            if status:
                logger.info("L2 search warmed — %s.", status)
        except Exception:
            logger.exception("L2 warmup failed (continuing; first query pays the cost)")
        logger.info("RAGu web ready — %d document(s) indexed.", len(allowed_sources))

    async def _run(turn: _Turn) -> None:
        """Drive one turn to completion. Holds the model lock for its lifetime
        (including any ask-back pause) so runs never overlap."""
        async with turn_lock:
            ask_handler_var.set(turn.ask)
            trace_handler_var.set(turn.on_event)
            # Chain this session's prior turns in (and capture this turn into the
            # same list) so L2 answers follow-ups in the conversation's context.
            transcript_var.set(transcripts.setdefault(turn.session_id, []))
            try:
                turn.t0 = time.perf_counter()
                # None => server default (L1 off / full-corpus, per config).
                full_corpus = (
                    ragu.full_corpus_default if turn.full_corpus is None else turn.full_corpus
                )
                if full_corpus:
                    turn.working_set = await ragu.full_working_set()
                else:
                    # Conversation-aware retrieval: on a follow-up, retrieve over
                    # the prior (question, answer) plus the new message so a
                    # resolved intermediate hop ("English Channel") pulls in the
                    # next hop's evidence. Accumulate with last turn's working set
                    # so earlier evidence is never dropped.
                    prev = last_qa.get(turn.session_id)
                    retrieval_text = (
                        f"{prev[0]}\n{prev[1]}\n{turn.message}" if prev else turn.message
                    )
                    turn.working_set = await ragu.retrieve(
                        retrieval_text, prior=working_sets.get(turn.session_id)
                    )
                # Pass the assembled set straight to L2 (don't re-retrieve from
                # the bare message, which would discard the conversation context).
                turn.result = await ragu.answer(
                    turn.message,
                    ground=True,
                    grounding_source=turn.grounding_source,
                    full_corpus=full_corpus,
                    working_set=turn.working_set,
                )
                turn.elapsed_ms = int((time.perf_counter() - turn.t0) * 1000)
                # Remember this turn for the next follow-up's retrieval + accumulation.
                working_sets[turn.session_id] = turn.working_set
                last_qa[turn.session_id] = (turn.message, turn.result.text)
                turn.events.put(("done", None))
            except Exception as exc:  # surface to the awaiting HTTP handler
                logger.exception("turn failed for session %s", turn.session_id)
                turn.events.put(("error", exc))
            finally:
                turn.finished = True  # closes any open /api/trace stream

    async def _pump(turn: _Turn) -> dict:
        """Wait for the turn's next event and shape it into a JSON response.
        A pending question leaves the turn alive (worker blocked on ``replies``);
        done/error completes it and drops it from the session map."""
        kind, payload = await asyncio.to_thread(turn.events.get)
        if kind == "ask":
            return {
                "type": "question",
                "session_id": turn.session_id,
                "question": payload,
            }
        turns.pop(turn.session_id, None)
        if kind == "error":
            raise HTTPException(500, f"reasoning failed: {payload}")
        assert turn.result is not None and turn.working_set is not None
        return {
            "session_id": turn.session_id,
            "reasoning_log": list(turn.trace_log),
            **_answer_json(turn.result, turn.working_set, turn.elapsed_ms),
        }

    @app.get("/api/documents")
    async def documents() -> dict:
        refs = await ragu.list_documents()
        return {"documents": [{"id": str(r.id), "source": r.source} for r in refs]}

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> dict:
        if not req.message.strip():
            raise HTTPException(400, "message is empty")
        if req.session_id in turns:
            raise HTTPException(409, "a turn is already in flight for this session")
        turn = _Turn(
            req.session_id, req.message.strip(), req.grounding_source, req.full_corpus
        )
        turns[req.session_id] = turn
        turn.task = asyncio.create_task(_run(turn))
        return await _pump(turn)

    @app.get("/api/trace")
    async def trace(session_id: str = Query(...), after: int = Query(0, ge=0)) -> dict:
        """Poll L2's reasoning log for a session's in-flight turn. Returns the log
        lines after index ``after`` (so the client only fetches what's new) plus
        ``finished``. A short-poll endpoint, not a persistent connection — robust
        behind dev proxies and immune to per-origin connection limits.

        Once a turn completes it's removed from ``turns``; the authoritative full
        log then rides back with the answer (``reasoning_log``), so a missed final
        poll loses nothing."""
        turn = turns.get(session_id)
        if turn is None:
            return {"lines": [], "next": after, "finished": True}
        return {
            "lines": turn.trace_log[after:],
            "next": len(turn.trace_log),
            "finished": turn.finished,
        }

    @app.post("/api/respond")
    async def respond(req: RespondRequest) -> dict:
        turn = turns.get(req.session_id)
        if turn is None:
            raise HTTPException(409, "no pending question for this session")
        turn.replies.put(req.answer)
        return await _pump(turn)

    @app.post("/api/reset")
    async def reset(req: ResetRequest) -> dict:
        """End a session. If a turn is mid-question, unblock the worker so it can
        finish and release the lock instead of stranding it forever."""
        turn = turns.pop(req.session_id, None)
        transcripts.pop(req.session_id, None)  # forget the conversation history
        working_sets.pop(req.session_id, None)  # and the accumulated evidence
        last_qa.pop(req.session_id, None)  # and the conversation-aware retrieval context
        if turn is not None:
            turn.replies.put(_CANCELLED)
            if turn.task is not None:
                turn.task.add_done_callback(lambda _: None)  # fire-and-forget
        return {"ok": True}

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
