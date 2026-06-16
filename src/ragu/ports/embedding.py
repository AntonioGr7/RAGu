"""Embedding contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors.

    Query and document embeddings are separate methods because SOTA embedders
    (Voyage, e5, BGE) are asymmetric — they prepend different task instructions
    to queries vs. passages, and conflating them quietly degrades recall.
    """

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts (batched)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...
