"""L1 hybrid retriever.

Fuses the dense and lexical channels with Reciprocal Rank Fusion (RRF) — rank-
based, so it needs no score normalization between the two very different score
scales (cosine vs BM25), which is exactly why it's the robust default. An
optional reranker refines the top candidates before returning.

Remember L1's job: select the working set. We pull a wide candidate pool per
channel and return fused chunks; the pipeline rolls these up to parent
documents.
"""

from __future__ import annotations

from ragu.core import Chunk, Query, RetrievalResult, ScoredChunk
from ragu.ports import Embedder, Reranker, VectorStore


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        reranker: Reranker | None = None,
        candidate_k: int = 200,
        rrf_k: int = 60,
        rerank_top_k: int = 50,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._rerank_top_k = rerank_top_k

    async def retrieve(self, query: Query, k: int) -> RetrievalResult:
        filters = query.filters or None
        query_vec = await self._embedder.embed_query(query.text)

        dense = await self._store.search_dense(query_vec, self._candidate_k, filters)
        lexical = await self._store.search_lexical(query.text, self._candidate_k, filters)

        fused = _reciprocal_rank_fusion([dense, lexical], self._rrf_k)

        if self._reranker is not None and fused:
            head = fused[: self._rerank_top_k]
            reranked = await self._reranker.rerank(query.text, head, self._rerank_top_k)
            fused = reranked + fused[self._rerank_top_k :]

        return RetrievalResult(query_text=query.text, chunks=tuple(fused[:k]))


def _reciprocal_rank_fusion(
    channels: list[list[ScoredChunk]], rrf_k: int
) -> list[ScoredChunk]:
    """Fuse ranked channels: score(d) = Σ 1 / (rrf_k + rank_in_channel(d)).

    Preserves each channel's component score on the surviving ScoredChunk for
    transparency.
    """
    fused_score: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for channel in channels:
        for rank, sc in enumerate(channel):
            cid = sc.chunk.id
            fused_score[cid] = fused_score.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_by_id[cid] = sc.chunk
            # Merge component scores across channels for the same chunk.
            prev = best.get(cid)
            best[cid] = ScoredChunk(
                chunk=sc.chunk,
                score=0.0,  # filled below
                dense_score=sc.dense_score or (prev.dense_score if prev else None),
                sparse_score=sc.sparse_score or (prev.sparse_score if prev else None),
                rerank_score=sc.rerank_score or (prev.rerank_score if prev else None),
            )

    out = [
        ScoredChunk(
            chunk=chunk_by_id[cid],
            score=score,
            dense_score=best[cid].dense_score,
            sparse_score=best[cid].sparse_score,
            rerank_score=best[cid].rerank_score,
        )
        for cid, score in fused_score.items()
    ]
    out.sort(key=lambda s: s.score, reverse=True)
    return out
