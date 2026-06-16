"""Storage contracts: the vector index and the parent-document store.

These are deliberately separate. The ``VectorStore`` holds chunk vectors + text
for retrieval; the ``DocumentStore`` holds whole documents for working-set
assembly. They can be backed by the same engine (LanceDB does both) but the
roles are distinct — L1 searches the former, L2's working set is built from the
latter.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragu.core import Chunk, Document, DocumentId, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Persists embedded chunks and serves dense + lexical search.

    Hybrid fusion is the *retriever's* job; the store only exposes the two raw
    channels so the fusion strategy stays swappable.
    """

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    async def search_dense(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """Vector similarity search; scores are similarities (higher = better)."""
        ...

    async def search_lexical(
        self,
        query_text: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """Full-text/BM25 search; scores are BM25 relevance (higher = better)."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Stores and retrieves whole documents by id."""

    async def put(self, documents: list[Document]) -> None:
        ...

    async def get(self, doc_ids: list[DocumentId]) -> list[Document]:
        """Return documents for the given ids, preserving order; missing ids are
        skipped (callers rank before fetching, so absence is non-fatal)."""
        ...
