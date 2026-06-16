"""Embedder adapters.

* ``LocalEmbedder`` — a sentence-transformers model running locally (CPU or
  GPU). Default path: no API, no per-call cost, multilingual, and small enough
  for a modest GPU.
* ``VoyageEmbedder`` — hosted Voyage embeddings (optional ``voyage`` extra).
* ``FakeEmbedder`` — deterministic and dependency-free, so the whole pipeline
  (and its tests) runs offline while exercising real cosine geometry.

Heavy deps are imported lazily so only the provider you use needs installing.
"""

from __future__ import annotations

import asyncio
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


class LocalEmbedder:
    """A local sentence-transformers embedder.

    Query and document texts get the model's asymmetric prefixes (E5 wants
    "query:"/"passage:"); pass empty prefixes for models that don't use them.
    Encoding is synchronous in the library, so it's run in a worker thread to
    keep the async port contract.

    Two ways to apply a model's asymmetric query/document instructions:

    * **prompt names** (``query_prompt_name``/``document_prompt_name``) — use the
      prompts the model itself registers with sentence-transformers. Correct for
      models like Snowflake Arctic that ship their own prompts. When a prompt
      name is set, the corresponding string prefix is ignored.
    * **string prefixes** (``query_prefix``/``document_prefix``) — prepend a
      literal string. The simple path for E5 ("query:"/"passage:").
    """

    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-small",
        *,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 64,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LocalEmbedder requires the 'local' extra: pip install 'ragu[local]'"
            ) from exc
        # Some models (e.g. Arctic v2.0 on GTE-multilingual) ship custom modeling
        # code and require trust_remote_code=True to load.
        self._model = SentenceTransformer(model, device=device, trust_remote_code=trust_remote_code)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._normalize = normalize
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._query_prompt_name = query_prompt_name
        self._document_prompt_name = document_prompt_name

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name=prompt_name,
        )
        return vectors.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._document_prompt_name is not None:
            return await asyncio.to_thread(self._encode, texts, self._document_prompt_name)
        prefixed = [f"{self._document_prefix}{t}" for t in texts]
        return await asyncio.to_thread(self._encode, prefixed, None)

    async def embed_query(self, text: str) -> list[float]:
        if self._query_prompt_name is not None:
            out = await asyncio.to_thread(self._encode, [text], self._query_prompt_name)
        else:
            out = await asyncio.to_thread(self._encode, [f"{self._query_prefix}{text}"], None)
        return out[0]


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
