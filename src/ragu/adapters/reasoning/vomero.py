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
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol

from ragu.config import VomeroSettings
from ragu.core import Answer, Citation, Document, Query, WorkingSet


class _Engine(Protocol):
    """The slice of vomero's RLMEngine we use (kept narrow for testability)."""

    def run(self, question: str, source: Any, *, return_trajectory: bool = False) -> Any: ...


class VomeroReasoningEngine:
    def __init__(
        self,
        settings: VomeroSettings,
        *,
        engine: _Engine | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine  # injectable for tests; built lazily otherwise

    # -- ReasoningEngine port -------------------------------------------
    async def reason(self, query: Query, working_set: WorkingSet) -> Answer:
        return await asyncio.to_thread(self._reason_sync, query, working_set)

    def _reason_sync(self, query: Query, working_set: WorkingSet) -> Answer:
        engine = self._engine or self._build_engine()
        root = Path(tempfile.mkdtemp(prefix="ragu-corpus-"))
        try:
            rel_to_doc = materialize_corpus(working_set, root)
            source = self._make_source(root, working_set)
            result = engine.run(query.text, source, return_trajectory=True)
            answer_text, steps, tokens, calls = _unpack(result)
            sources = {d.id: d.source for d in working_set.documents}
            citations = citations_from_trajectory(steps, rel_to_doc, sources)
            return Answer(
                text=answer_text,
                citations=tuple(citations),
                used_reasoning=True,
                trace={
                    "engine": "vomero",
                    "tokens": str(tokens),
                    "calls": str(calls),
                    "steps": str(len(steps)),
                    "working_set_docs": str(len(working_set.documents)),
                },
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _make_source(self, root: Path, working_set: WorkingSet) -> Any:
        from vomero import Context, Corpus

        if self._settings.handoff == "context":
            return Context([d.content for d in working_set.documents])
        return Corpus(root)

    def _build_engine(self) -> _Engine:
        from vomero import Settings, build_engine

        s = self._settings
        settings = Settings(
            provider=s.provider,
            model=s.model,
            base_url=s.base_url,
            api_key=s.api_key,
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
            enable_interaction=False,  # no human in the RAG loop
        )
        return build_engine(settings)


def corpus_filename(doc: Document) -> str:
    """A corpus-safe, text-readable filename for a document.

    Preserves the document's relative path (so vomero's grep/read names line up
    with the source tree) and ensures a text extension vomero will read."""
    rel = doc.metadata.get("rel_path") or str(doc.id)
    text_exts = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}
    return rel if Path(rel).suffix.lower() in text_exts else f"{rel}.txt"


def materialize_corpus(working_set: WorkingSet, root: Path) -> dict[str, str]:
    """Write each working-set document into ``root`` as a text file.

    Returns a mapping of corpus-relative path -> document id, used to turn the
    files vomero touched back into citations."""
    rel_to_doc: dict[str, str] = {}
    for doc in working_set.documents:
        name = corpus_filename(doc)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.content, encoding="utf-8")
        rel_to_doc[name] = str(doc.id)
    return rel_to_doc


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


def _unpack(result: Any) -> tuple[str, list[Any], int, int]:
    """Normalize vomero's run output (RunResult or bare string)."""
    if isinstance(result, str):
        return result, [], 0, 0
    return (
        getattr(result, "answer", str(result)),
        list(getattr(result, "trajectory", [])),
        int(getattr(result, "tokens", 0)),
        int(getattr(result, "calls", 0)),
    )
