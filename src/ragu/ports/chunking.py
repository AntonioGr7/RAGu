"""Chunking contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragu.core import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrieval units.

    Async because SOTA chunkers (contextual retrieval) call an LLM to generate
    per-chunk context. A purely lexical chunker simply doesn't await anything.
    """

    async def chunk(self, document: Document) -> list[Chunk]:
        ...
