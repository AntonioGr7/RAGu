"""Vomero (RLM) behind the ``ReasoningEngine`` port — the L2 layer.

RAGu hands vomero the L1 working set as a **corpus** (a temp folder of text
files, one per document) and lets the RLM navigate it agentically — grep/read/
recurse/multihop — instead of stuffing chunks into a prompt. The engine's
answer + trajectory come back, and we map them to a RAGu ``Answer`` with
document-level citations derived from which corpus files the model actually
touched.

vomero is imported lazily (only here), runs synchronously, and is driven in a
worker thread so ``reason`` stays async. Nothing outside this module references
vomero — swapping in another RLM means another adapter, nothing else.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import logging
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from ragu.config import VomeroSettings
from ragu.core import Answer, Citation, Document, DocumentId, EvidenceSpan, Query, WorkingSet

logger = logging.getLogger(__name__)


# When set, this overrides the terminal ask-handler so a non-TTY caller (the web
# server) can answer L2's clarifying questions itself. ``asyncio.to_thread``
# copies the current context into the worker thread, so a handler set on the
# task driving ``reason`` reaches ``_reason_sync`` inside that thread.
ask_handler_var: contextvars.ContextVar[Callable[[str], str] | None] = (
    contextvars.ContextVar("ragu_ask_handler", default=None)
)

# When set, receives each vomero trajectory step as it happens (the same events
# the ``--verbose`` printer formats). The web server uses this to stream L2's
# reasoning log live. Like ``ask_handler_var`` it rides ``asyncio.to_thread``'s
# context copy into the worker thread.
trace_handler_var: contextvars.ContextVar[Callable[[Any], None] | None] = (
    contextvars.ContextVar("ragu_trace_handler", default=None)
)

# Conversation continuity across turns of one session. When set to a (possibly
# empty) list, it is passed to vomero as BOTH ``history`` (prior turns, read in)
# and ``transcript_sink`` (the updated transcript, written back) — vomero reads
# history before it clears/refills the sink, so the same list safely does both.
# The list is mutated in place, so the caller (the web server, holding one list
# per session) sees the new transcript and chains it into the next turn. The
# engine itself stays stateless; the caller owns the store. Rides the same
# ``asyncio.to_thread`` context copy as the handlers above. ``None`` = a
# one-shot run with no memory (the CLI/eval default).
transcript_var: contextvars.ContextVar[list[Any] | None] = (
    contextvars.ContextVar("ragu_transcript", default=None)
)


class _Engine(Protocol):
    """The slice of vomero's RLMEngine we use (kept narrow for testability)."""

    def run(
        self,
        question: str,
        source: Any,
        *,
        return_trajectory: bool = False,
        on_event: Any = None,
        ask_handler: Any = None,
        history: Any = None,
        transcript_sink: Any = None,
    ) -> Any: ...


class VomeroReasoningEngine:
    # Only back a working set with the persistent on-disk search index when it
    # is large enough to be worth it — i.e. the full corpus. A small per-query
    # working set (L1+L2 mode) gets a cheap lazy in-memory index instead, so we
    # never thrash the shared on-disk index with per-query rebuilds.
    _SEARCH_INDEX_MIN_DOCS = 200

    def __init__(
        self,
        settings: VomeroSettings,
        *,
        engine: _Engine | None = None,
        search_index_dir: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine  # injectable for tests; built lazily otherwise
        # Where L2's persistent lexical search index lives (None => no search
        # index; corpus.search() degrades to a lazy in-memory build). Built once
        # from the corpus and opened read-only — see `_ensure_search_index`.
        self._search_index_dir = (
            Path(search_index_dir).expanduser().resolve() if search_index_dir else None
        )
        # Doc-id set the persistent index currently covers (this process), so a
        # repeat full-corpus turn skips the staleness check entirely.
        self._indexed_signature: frozenset[str] | None = None
        # Materialized-corpus cache. The corpus written to disk is identical for
        # every turn (and, in full-corpus mode, every session), so we write it
        # once and reuse the temp dir instead of re-dumping every document on
        # each turn. Keyed by the working set's identity *and* its document-id
        # set: a different working set evicts the old dir and rebuilds. Guarded
        # by a lock because ``reason`` runs in worker threads. The held dir is
        # removed at process exit (see ``_cleanup_corpus``).
        self._corpus_lock = threading.Lock()
        self._corpus_cache: tuple[int, frozenset[str], Path, dict[str, str]] | None = None
        atexit.register(self._cleanup_corpus)

    # -- ReasoningEngine port -------------------------------------------
    async def reason(self, query: Query, working_set: WorkingSet) -> Answer:
        return await asyncio.to_thread(self._reason_sync, query, working_set)

    def _corpus_dir(
        self, working_set: WorkingSet, *, progress: bool = False
    ) -> tuple[Path, dict[str, str]]:
        """Materialize ``working_set`` to a temp dir, reusing the one written for
        a previous identical working set (the common full-corpus case) instead of
        rewriting every document. A changed working set evicts the stale dir.
        ``progress`` shows a write bar (used at warmup, not per-turn)."""
        signature = frozenset(str(d.id) for d in working_set.documents)
        key = id(working_set)
        with self._corpus_lock:
            if self._corpus_cache is not None:
                c_key, c_sig, root, rel_to_doc = self._corpus_cache
                if c_key == key and c_sig == signature and root.exists():
                    return root, rel_to_doc
                shutil.rmtree(root, ignore_errors=True)
                self._corpus_cache = None
            root = Path(tempfile.mkdtemp(prefix="ragu-corpus-"))
            rel_to_doc = materialize_corpus(working_set, root, progress=progress)
            self._corpus_cache = (key, signature, root, rel_to_doc)
            return root, rel_to_doc

    def _cleanup_corpus(self) -> None:
        with self._corpus_lock:
            if self._corpus_cache is not None:
                shutil.rmtree(self._corpus_cache[2], ignore_errors=True)
                self._corpus_cache = None

    def _ensure_search_index(self, working_set: WorkingSet) -> Path | None:
        """Build (once) / locate the persistent lexical index for ``working_set``
        and return its dir to hand to ``Corpus``, or ``None`` when search should
        fall back to a lazy in-memory index.

        Built lexical-only (no embedder) from the same corpus filenames the temp
        corpus uses, so a ``Hit.doc`` resolves through ``rel_to_doc`` just like a
        read/grep. A repeat full-corpus turn short-circuits on the cached
        signature; otherwise an unchanged on-disk index is reused (``is_stale``)
        and only a genuinely changed corpus triggers a rebuild."""
        if self._search_index_dir is None:
            return None
        if len(working_set.documents) < self._SEARCH_INDEX_MIN_DOCS:
            return None  # small per-query set — cheap in-memory index instead
        from vomero.context.index import PersistentIndex

        signature = frozenset(str(d.id) for d in working_set.documents)
        with self._corpus_lock:
            d = self._search_index_dir
            if self._indexed_signature == signature and PersistentIndex.exists(d):
                return d
            docs = [(corpus_filename(doc), doc.content) for doc in working_set.documents]
            if PersistentIndex.exists(d) and not PersistentIndex(d).is_stale(docs):
                logger.info("L2 search index up to date (%d docs) — %s", len(docs), d)
            else:
                logger.info("L2 search index: building lexical index over %d docs → %s",
                            len(docs), d)
                PersistentIndex.build(docs, d)  # lexical-only (no embedder)
                logger.info("L2 search index built.")
            self._indexed_signature = signature
            return d

    def warmup(self, working_set: WorkingSet) -> str:
        """Pay the corpus materialization + search-index build now (server
        startup) instead of on the first user's question, and prime the read-only
        index. Returns a human-readable status for the boot log."""
        root, _ = self._corpus_dir(working_set, progress=True)  # materialize once (cached)
        index_dir = self._ensure_search_index(working_set)
        if index_dir is None:
            # Search disabled (or set too small for a persistent index): the
            # corpus is materialized, but don't read every file to build a
            # throwaway in-memory index we wouldn't reuse.
            return f"corpus materialized ({len(working_set.documents)} docs); search index off"
        from vomero import Corpus

        return Corpus(root, index_dir=index_dir).warmup()

    def _reason_sync(self, query: Query, working_set: WorkingSet) -> Answer:
        engine = self._engine or self._build_engine()
        root, rel_to_doc = self._corpus_dir(working_set)
        source = self._make_source(root, working_set)
        # A caller-supplied trace handler (the web server, streaming the log
        # live) wins; otherwise print to stderr when verbose.
        on_event = trace_handler_var.get() or (
            _trace_printer() if self._settings.verbose else None
        )
        # A caller-supplied handler (e.g. the web server, which drives the
        # question over HTTP) wins; otherwise prompt on the terminal when L2
        # is interactive and a TTY is present.
        ask_handler = ask_handler_var.get() or (
            _terminal_ask_handler() if self._settings.interactive else None
        )
        # A per-session transcript (from the web server) chains turns: vomero
        # reads it as prior history and writes the updated transcript back
        # into the same list. Absent (CLI/eval), the run carries no memory.
        transcript = transcript_var.get()
        result = engine.run(
            query.text, source, return_trajectory=True,
            on_event=on_event, ask_handler=ask_handler,
            history=transcript or None, transcript_sink=transcript,
        )
        answer_text, steps, tokens, calls, provenance = _unpack(result)
        sources = {d.id: d.source for d in working_set.documents}
        # A Context source logs provenance by document *index*; resolve those
        # back to ids via the working-set order. (Corpus logs by file path.)
        doc_by_index = [d.id for d in working_set.documents]
        # Prefer vomero's structured access log; fall back to scanning the
        # trajectory code when it's empty (e.g. the gVisor backend, whose
        # in-pod corpus never reaches the host access log).
        citations = citations_from_provenance(
            provenance, rel_to_doc, sources, doc_by_index
        ) or citations_from_trajectory(steps, rel_to_doc, sources)
        return Answer(
            text=answer_text,
            citations=tuple(citations),
            used_reasoning=True,
            # The text the model actually read (read/grep output) — used by
            # the LLM grounding passes to anchor citations in the real evidence.
            evidence=evidence_from_trajectory(steps),
            # Structured grep hits with their home doc — used by "raw" grounding.
            evidence_spans=evidence_spans_from_provenance(
                provenance, rel_to_doc, sources, doc_by_index
            ),
            trace={
                "engine": "vomero",
                "tokens": str(tokens),
                "calls": str(calls),
                "steps": str(len(steps)),
                "working_set_docs": str(len(working_set.documents)),
            },
        )

    def _make_source(self, root: Path, working_set: WorkingSet) -> Any:
        from vomero import Context, Corpus

        if self._settings.handoff == "context":
            return Context([d.content for d in working_set.documents])
        # Pass the persistent lexical index (when one covers this corpus) so
        # corpus.search() opens it read-only instead of building in-memory.
        return Corpus(root, index_dir=self._ensure_search_index(working_set))

    def _build_engine(self) -> _Engine:
        from vomero import Settings, build_engine

        s = self._settings
        settings = Settings(
            provider=s.provider,
            model=s.model,
            base_url=s.base_url,
            api_key=s.api_key or _api_key_from_env(),
            max_steps=s.max_steps,
            max_depth=s.max_depth,
            max_parallel_calls=s.max_parallel_calls,
            max_total_tokens=s.max_total_tokens,
            max_total_calls=s.max_total_calls,
            max_output_chars=s.max_output_chars,
            context_window=s.context_window,
            compact_ratio=s.compact_ratio,
            exec_backend="gvisor" if s.sandbox else "inprocess",
            enable_planning=s.plan,
            enable_interaction=s.interactive,
        )
        return build_engine(settings)


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + " …"


def format_step(step: Any) -> str | None:
    """Render one vomero trajectory step as a single human-readable log line, or
    ``None`` for steps with nothing to show. Shared by the stderr ``--verbose``
    printer and the web server's live reasoning-log stream so both stay in sync.
    """
    tag = f"[d{getattr(step, 'depth', 0)}.{getattr(step, 'index', 0)}]"
    if getattr(step, "code", None) is not None:
        return f"{tag} python:\n{step.code}"
    if getattr(step, "output", None) is not None:
        return f"{tag} → {_clip(step.output, 1500)}"
    if getattr(step, "message", None) is not None:
        return f"{tag} 💬 {_clip(step.message, 500)}"
    if getattr(step, "llm_call", None) is not None:
        c = step.llm_call
        return f"{tag} llm() (+{c.tokens:,} tok) → {_clip(c.response, 300)}"
    if getattr(step, "usage", None) is not None:
        u = step.usage
        return f"{tag} ctx {u.context_tokens:,} tok | total {u.cumulative_tokens:,} tok"
    if getattr(step, "note", None) is not None:
        return f"{tag} ⚠ {step.note}"
    if getattr(step, "final", None) is not None:
        return f"{tag} FINAL: {_clip(step.final, 1000)}"
    return None


def _trace_printer() -> Callable[[Any], None]:
    """An ``on_event`` callback that streams vomero's per-step trace to stderr
    as it reasons — the code L2 runs, the read/grep output, its messages, token
    usage, and the final answer. Mirrors vomero's own CLI ``--verbose`` view but
    kept here so the adapter never imports vomero's private CLI helpers.

    Bind stderr now: vomero fires these events from inside ``env.execute()``,
    which redirects ``sys.stderr`` to capture model output, so a late lookup
    would land trace lines in that buffer instead of the terminal."""
    import sys

    stream = sys.stderr

    def clip(text: str, n: int) -> str:
        text = text.replace("\n", " ")
        return text if len(text) <= n else text[:n] + " …"

    def emit(step: Any) -> None:
        pad = "  " * step.depth
        tag = f"{pad}[d{step.depth}.{step.index}]"
        if step.code is not None:
            print(f"\n{tag} python:\n{pad}  " + step.code.replace("\n", "\n" + pad + "  "),
                  file=stream)
        elif step.output is not None:
            print(f"{tag} -> " + clip(step.output, 1500), file=stream)
        elif step.message is not None:
            print(f"{tag} 💬 " + clip(step.message, 500), file=stream)
        elif step.llm_call is not None:
            c = step.llm_call
            print(f"{tag} llm() (+{c.tokens:,} tok) -> " + clip(c.response, 300), file=stream)
        elif step.usage is not None:
            u = step.usage
            print(f"{tag} ctx {u.context_tokens:,} tok | total {u.cumulative_tokens:,} tok",
                  file=stream)
        elif step.note is not None:
            print(f"{tag} ⚠ {step.note}", file=stream)
        elif step.final is not None:
            print(f"{tag} FINAL: " + clip(step.final, 1000), file=stream)
        stream.flush()

    return emit


def _api_key_from_env() -> str | None:
    """Resolve the L2 API key from the environment when ``VomeroSettings.api_key``
    is unset. We build vomero's ``Settings`` explicitly (not via its
    ``from_env``), so vomero's own key fallback never runs — we mirror its
    precedence here so a plain ``GEMINI_API_KEY`` / ``OPENAI_API_KEY`` in ``.env``
    just works."""
    for var in ("VOMERO_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(var)
        if value:
            return value
    return None


def corpus_filename(doc: Document) -> str:
    """A corpus-safe, text-readable filename for a document.

    Preserves the document's relative path (so vomero's grep/read names line up
    with the source tree) and ensures a text extension vomero will read."""
    rel = doc.metadata.get("rel_path") or str(doc.id)
    text_exts = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}
    return rel if Path(rel).suffix.lower() in text_exts else f"{rel}.txt"


def materialize_corpus(
    working_set: WorkingSet, root: Path, *, progress: bool = False
) -> dict[str, str]:
    """Write each working-set document into ``root`` as a text file.

    Returns a mapping of corpus-relative path -> document id, used to turn the
    files vomero touched back into citations. With ``progress=True`` a tqdm bar
    tracks the write — worth it for the full corpus (tens of thousands of files)
    at server warmup, where the wait is otherwise silent."""
    rel_to_doc: dict[str, str] = {}
    bar = _progress_bar(len(working_set.documents), enabled=progress, desc="Materializing corpus")
    for doc in working_set.documents:
        name = corpus_filename(doc)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.content, encoding="utf-8")
        rel_to_doc[name] = str(doc.id)
        bar.update(1)
    bar.close()
    return rel_to_doc


class _NullBar:
    """No-op stand-in for tqdm when progress is disabled or tqdm is missing."""

    def update(self, _n: int = 1) -> None: ...
    def close(self) -> None: ...


def _progress_bar(total: int, *, enabled: bool, desc: str):
    """A tqdm bar of ``total`` items, or a no-op bar when disabled/unavailable."""
    if not enabled:
        return _NullBar()
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm ships with the 'ocr' extra
        return _NullBar()
    return tqdm(total=total, unit="doc", desc=desc)


def citations_from_trajectory(
    steps: list[Any],
    rel_to_doc: dict[str, str],
    sources: dict[Any, str],
) -> list[Citation]:
    """Derive document-level citations from the REPL code the model ran.

    A document is cited when its corpus filename appears in any step's code
    (a read/peek/grep against it). Order of first reference is preserved."""
    code = "\n".join(getattr(s, "code", None) or "" for s in steps)
    cited: list[Citation] = []
    seen: set[str] = set()
    # Longer paths first so a nested path isn't masked by a shorter prefix.
    for rel in sorted(rel_to_doc, key=len, reverse=True):
        doc_id = rel_to_doc[rel]
        if doc_id in seen:
            continue
        if re.search(re.escape(rel), code):
            seen.add(doc_id)
            cited.append(Citation(doc_id=doc_id, source=sources.get(doc_id, rel)))
    return cited


def evidence_from_trajectory(steps: list[Any]) -> tuple[str, ...]:
    """The verbatim text the model read during reasoning — each step's ``output``
    (the result of its read/grep code). This is exactly the evidence the answer
    rests on, so grounding it yields more specific citations than re-reading the
    whole document. Empty/blank outputs are skipped."""
    return tuple(
        out for s in steps if (out := (getattr(s, "output", None) or "").strip())
    )


def _resolve_doc_id(
    doc: Any, rel_to_doc: dict[str, str], doc_by_index: list[Any]
) -> str | None:
    """Map a vomero ``AccessEvent.doc`` back to a RAGu document id. A ``Corpus``
    reports a relative file path (look up in ``rel_to_doc``); a ``Context``
    reports a document index into the working set (look up by position)."""
    if isinstance(doc, int):
        return str(doc_by_index[doc]) if 0 <= doc < len(doc_by_index) else None
    return rel_to_doc.get(doc)


def citations_from_provenance(
    provenance: list[Any],
    rel_to_doc: dict[str, str],
    sources: dict[Any, str],
    doc_by_index: list[Any],
) -> list[Citation]:
    """Document-level citations from vomero's structured access log: every
    document the model read, peeked, or grep'd is cited, in order of first
    touch. More reliable than scanning REPL code — it's what vomero recorded it
    actually retrieved. Empty when no access log is available."""
    cited: list[Citation] = []
    seen: set[str] = set()
    for event in provenance:
        if getattr(event, "op", None) not in ("read", "peek", "grep", "search"):
            continue
        doc_id = _resolve_doc_id(event.doc, rel_to_doc, doc_by_index)
        if doc_id is None or doc_id in seen:
            continue
        seen.add(doc_id)
        cited.append(
            Citation(doc_id=DocumentId(doc_id), source=sources.get(DocumentId(doc_id), doc_id))
        )
    return cited


def evidence_spans_from_provenance(
    provenance: list[Any],
    rel_to_doc: dict[str, str],
    sources: dict[Any, str],
    doc_by_index: list[Any],
) -> tuple[EvidenceSpan, ...]:
    """The grep/search hits from vomero's access log, each tagged with its home
    doc. These are clean source fragments (a matched line or a ranked-result
    snippet — no REPL stdout to strip) that "raw" grounding places straight into
    their own document."""
    spans: list[EvidenceSpan] = []
    for event in provenance:
        if getattr(event, "op", None) not in ("grep", "search"):
            continue
        text = (getattr(event, "text", None) or "").strip()
        if not text:
            continue
        doc_id = _resolve_doc_id(event.doc, rel_to_doc, doc_by_index)
        if doc_id is None:
            continue
        spans.append(EvidenceSpan(doc_id=DocumentId(doc_id), text=text))
    return tuple(spans)


def _terminal_ask_handler() -> Callable[[str], str] | None:
    """An ``ask_handler`` that prompts the user on the terminal when L2 needs a
    clarification — but only when stdin/stdout are a real TTY. Headless (server,
    eval, piped) there's no human, so we return ``None`` and let vomero degrade
    gracefully (it proceeds with its best judgment instead of hanging)."""
    import sys

    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return None

    def ask(question: str) -> str:
        print(f"\n❓ L2 needs your input:\n   {question}\n   > ", end="",
              file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        return line.rstrip("\n") if line else "(no answer provided)"

    return ask


def _unpack(result: Any) -> tuple[str, list[Any], int, int, list[Any]]:
    """Normalize vomero's run output (RunResult or bare string)."""
    if isinstance(result, str):
        return result, [], 0, 0, []
    return (
        getattr(result, "answer", str(result)),
        list(getattr(result, "trajectory", [])),
        int(getattr(result, "tokens", 0)),
        int(getattr(result, "calls", 0)),
        list(getattr(result, "provenance", []) or []),
    )
