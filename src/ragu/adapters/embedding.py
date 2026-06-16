"""Embedder adapters.

``FakeEmbedder`` is deterministic and dependency-free — it lets the whole
pipeline (and its tests) run offline while still exercising real cosine
geometry. ``VoyageEmbedder`` is the production path (Voyage is Anthropic's
recommended embedding partner) and imports ``voyageai`` lazily so the dependency
stays optional.
"""

from __future__ import annotations

import hashlib
import math


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class FakeEmbedder:
    """Hashing-based deterministic embedder.

    Each token is hashed into the vector via the hashing trick, so semantically
    identical text yields identical vectors and lexical overlap yields cosine
    similarity. Not for production — for tests and offline development.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in text.lower().split():
            h = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self._dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class VoyageEmbedder:
    """Voyage embeddings with asymmetric query/document input types."""

    def __init__(self, model: str = "voyage-3", dim: int = 1024) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "VoyageEmbedder requires the 'voyage' extra: pip install 'ragu[voyage]'"
            ) from exc
        self._client = voyageai.AsyncClient()
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embed(texts, model=self._model, input_type="document")
        return resp.embeddings

    async def embed_query(self, text: str) -> list[float]:
        resp = await self._client.embed([text], model=self._model, input_type="query")
        return resp.embeddings[0]
