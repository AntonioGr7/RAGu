"""Retrieval contracts: reranking and the L1 retriever orchestrator."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragu.core import Query, RetrievalResult, ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate chunks with a cross-encoder (or LLM) for precision.

    Returns at most ``top_k`` chunks, best-first, with ``rerank_score`` set.
    """

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        ...


@runtime_checkable
class Retriever(Protocol):
    """L1: given a query, return scored chunks (which roll up to documents).

    This is the working-set *selector*. Implementations fuse dense + lexical
    channels and optionally rerank; the pipeline turns the result into a
    ``WorkingSet`` for L2.
    """

    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        ...
