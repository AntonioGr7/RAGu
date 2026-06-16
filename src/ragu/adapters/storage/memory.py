"""In-memory storage: brute-force dense search + a compact BM25 lexical index.

Dependency-free and exact, so it doubles as the reference implementation the
LanceDB adapter is checked against. Fine for tests and small corpora; not for
production scale.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ragu.core import Chunk, Document, DocumentId, ScoredChunk

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _matches(chunk: Chunk, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(chunk.metadata.get(key) == val for key, val in filters.items())


class InMemoryVectorStore:
    """Holds chunks + their embeddings; serves dense and BM25 search."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must align 1:1")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    async def search_dense(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        scored = []
        for chunk, emb in zip(self._chunks, self._embeddings, strict=True):
            if not _matches(chunk, filters):
                continue
            sim = _dot(query_embedding, emb)
            scored.append(ScoredChunk(chunk=chunk, score=sim, dense_score=sim))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def search_lexical(
        self,
        query_text: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        candidates = [c for c in self._chunks if _matches(c, filters)]
        scores = _bm25(query_text, candidates)
        scored = [
            ScoredChunk(chunk=chunk, score=score, sparse_score=score)
            for chunk, score in zip(candidates, scores, strict=True)
            if score > 0
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _bm25(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Classic BM25 over the candidate set (recomputed per query; the candidate
    set is small in the in-memory backend)."""
    if not chunks:
        return []
    docs = [_tokenize(c.embedding_text) for c in chunks]
    lengths = [len(d) for d in docs]
    avgdl = sum(lengths) / len(docs) if docs else 0.0
    n = len(docs)

    df: Counter[str] = Counter()
    for terms in docs:
        for term in set(terms):
            df[term] += 1

    q_terms = set(_tokenize(query))
    scores = [0.0] * n
    for i, terms in enumerate(docs):
        tf = Counter(terms)
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * (lengths[i] / avgdl if avgdl else 0))
            scores[i] += idf * (tf[term] * (k1 + 1)) / denom
    return scores


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._docs: dict[DocumentId, Document] = {}

    async def put(self, documents: list[Document]) -> None:
        for doc in documents:
            self._docs[doc.id] = doc

    async def get(self, doc_ids: list[DocumentId]) -> list[Document]:
        return [self._docs[i] for i in doc_ids if i in self._docs]
