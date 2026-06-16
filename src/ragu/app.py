"""The ``Ragu`` facade — the high-level entry point users interact with.

Wires the L1 components from settings and exposes a small surface: index files
or documents, and retrieve a working set for a query. L2 reasoning will hang off
this same object once the engine adapter lands.
"""

from __future__ import annotations

from pathlib import Path

from ragu.adapters.ingestion import dump_documents, load_paths
from ragu.adapters.ingestion.ocr import OcrEngine
from ragu.config import RaguSettings
from ragu.core import Document, Query, WorkingSet
from ragu.factory import (
    build_chunker,
    build_embedder,
    build_ocr_engine,
    build_retriever,
    build_stores,
    build_token_counter,
)
from ragu.pipeline import Indexer, build_working_set


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

    def _ensure_ocr(self) -> OcrEngine | None:
        if self._ocr is None:
            self._ocr = build_ocr_engine(self._settings)
        return self._ocr

    async def index_paths(self, paths: list[str | Path]) -> int:
        """Load files/folders (text + OCR'd images/PDFs) and index them.

        Returns the number of chunks written."""
        documents = load_paths(paths, ocr=self._ensure_ocr())
        return await self.index_documents(documents)

    async def index_documents(self, documents: list[Document]) -> int:
        return await self._indexer.index(documents)

    def extract_paths(self, paths: list[str | Path], out_dir: str | Path) -> int:
        """Load files/folders (text + OCR'd images/PDFs) and dump the extracted
        text to ``out_dir``, mirroring the source structure. Inspection tool —
        does not index. Returns the number of files written."""
        documents = load_paths(paths, ocr=self._ensure_ocr())
        return dump_documents(documents, out_dir)

    async def retrieve(self, text: str, **query_kwargs: str) -> WorkingSet:
        """Run L1 and assemble the token-bounded working set for the query."""
        query = Query(text=text, **query_kwargs)
        result = await self._retriever.retrieve(query, k=self._settings.retrieval.candidate_k)
        ranked = result.to_documents()
        return await build_working_set(
            ranked,
            self._document_store,
            self._counter,
            max_tokens=self._settings.working_set.max_tokens,
            max_documents=self._settings.retrieval.document_k,
        )
