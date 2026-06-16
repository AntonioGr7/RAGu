"""The ``Ragu`` facade — the high-level entry point users interact with.

Wires the L1 components from settings and exposes a small surface: index files
or documents, and retrieve a working set for a query. L2 reasoning will hang off
this same object once the engine adapter lands.
"""

from __future__ import annotations

from pathlib import Path

from ragu.adapters.ingestion import (
    dump_documents,
    file_fingerprint,
    iter_files,
    load_files,
    load_paths,
)
from ragu.adapters.ingestion.ocr import OcrEngine
from ragu.config import RaguSettings
from ragu.core import Answer, Document, DocumentId, IndexReport, Query, WorkingSet
from ragu.factory import (
    build_chunker,
    build_embedder,
    build_ocr_engine,
    build_reasoning_engine,
    build_retriever,
    build_stores,
    build_token_counter,
)
from ragu.pipeline import Indexer, build_working_set
from ragu.ports import ReasoningEngine


def _is_under(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` itself or nested within it."""
    return path == root or path.is_relative_to(root)


class Ragu:
    def __init__(self, settings: RaguSettings | None = None) -> None:
        self._settings = settings or RaguSettings()
        self._counter = build_token_counter(self._settings)
        self._embedder = build_embedder(self._settings)
        self._vector_store, self._document_store = build_stores(
            self._settings, self._embedder.dim
        )
        self._chunker = build_chunker(self._settings, self._counter)
        self._indexer = Indexer(
            self._chunker,
            self._embedder,
            self._vector_store,
            self._document_store,
            embed_batch_size=self._settings.embedding.batch_size,
        )
        self._retriever = build_retriever(self._settings, self._embedder, self._vector_store)
        self._ocr: OcrEngine | None = None
        self._reasoning: ReasoningEngine | None = None  # built lazily (needs vomero)

    def _ensure_ocr(self) -> OcrEngine | None:
        if self._ocr is None:
            self._ocr = build_ocr_engine(self._settings)
        return self._ocr

    async def index_paths(
        self, paths: list[str | Path], *, prune: bool = True, progress: bool = False
    ) -> IndexReport:
        """Incrementally index files/folders (text + OCR'd images/PDFs).

        Each file is fingerprinted (a hash of its bytes) and compared against
        what is already stored: unchanged files are skipped *before* OCR, new and
        changed files are (re)loaded and indexed in place. When ``prune`` is set,
        documents whose source file has disappeared from the indexed directories
        are removed from both stores. Returns an :class:`IndexReport`."""
        stored = {ref.id: ref for ref in await self._document_store.fingerprints()}

        to_load: list[tuple[Path, Path, str | None]] = []
        skipped = 0
        for file, base in iter_files(paths):
            doc_id = DocumentId(file.relative_to(base).as_posix())
            digest = file_fingerprint(file)
            ref = stored.get(doc_id)
            if ref is not None and ref.content_hash == digest:
                skipped += 1
                continue
            to_load.append((file, base, digest))

        cfg = self._settings.ocr
        # Build the (expensive) OCR engine only when there is something to load —
        # a re-run where everything is unchanged must not pay to spin up PaddleOCR.
        documents = (
            load_files(
                to_load,
                ocr=self._ensure_ocr(),
                pdf_mode=cfg.pdf_mode,
                pdf_dpi=cfg.pdf_dpi,
                pdf_min_text_chars=cfg.pdf_min_text_chars,
                progress=progress,
            )
            if to_load
            else []
        )
        updated = sum(1 for d in documents if d.id in stored)
        chunks = await self.index_documents(documents)

        pruned = await self._prune(paths, stored.values()) if prune else 0
        return IndexReport(
            chunks=chunks,
            new=len(documents) - updated,
            updated=updated,
            skipped=skipped,
            pruned=pruned,
        )

    async def _prune(self, paths: list[str | Path], stored) -> int:
        """Remove stored documents whose source file no longer exists, limited to
        documents that live under one of the indexed *directory* roots (so
        indexing one folder never prunes another)."""
        roots = [Path(p).resolve() for p in paths if Path(p).is_dir()]
        if not roots:
            return 0
        dead = [
            ref.id
            for ref in stored
            if any(_is_under(Path(ref.source), root) for root in roots)
            and not Path(ref.source).exists()
        ]
        if dead:
            await self._vector_store.delete(dead)
            await self._document_store.delete(dead)
        return len(dead)

    async def index_documents(self, documents: list[Document]) -> int:
        return await self._indexer.index(documents)

    def extract_paths(self, paths: list[str | Path], out_dir: str | Path) -> int:
        """Load files/folders (text + OCR'd images/PDFs) and dump the extracted
        text to ``out_dir``, mirroring the source structure. Inspection tool —
        does not index. Returns the number of files written."""
        return dump_documents(self._load(paths), out_dir)

    def _load(self, paths: list[str | Path]) -> list[Document]:
        cfg = self._settings.ocr
        return load_paths(
            paths,
            ocr=self._ensure_ocr(),
            pdf_mode=cfg.pdf_mode,
            pdf_dpi=cfg.pdf_dpi,
            pdf_min_text_chars=cfg.pdf_min_text_chars,
        )

    async def retrieve(self, text: str, **query_kwargs: str) -> WorkingSet:
        """Run L1 and assemble the token-bounded working set for the query."""
        return await self._working_set(Query(text=text, **query_kwargs))

    async def answer(self, text: str, **query_kwargs: str) -> Answer:
        """Full pipeline: L1 selects a working set, L2 (vomero) reasons over it."""
        query = Query(text=text, **query_kwargs)
        working_set = await self._working_set(query)
        return await self._ensure_reasoning().reason(query, working_set)

    async def _working_set(self, query: Query) -> WorkingSet:
        result = await self._retriever.retrieve(query, k=self._settings.retrieval.candidate_k)
        ranked = result.to_documents()
        return await build_working_set(
            ranked,
            self._document_store,
            self._counter,
            max_tokens=self._settings.working_set.max_tokens,
            max_documents=self._settings.retrieval.document_k,
        )

    def _ensure_reasoning(self) -> ReasoningEngine:
        if self._reasoning is None:
            self._reasoning = build_reasoning_engine(self._settings)
        return self._reasoning
