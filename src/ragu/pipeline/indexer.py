"""Indexing flow: document → chunks → embeddings → stores.

Keeps the whole document in the ``DocumentStore`` (L2 reads it later) while the
chunks + vectors go to the ``VectorStore`` (L1 searches them). Embeddings are
computed over ``embedding_text`` (context + body) so contextual retrieval is in
effect whenever the chunker produced context.
"""

from __future__ import annotations

from ragu.core import Document
from ragu.ports import Chunker, DocumentStore, Embedder, VectorStore


class Indexer:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        document_store: DocumentStore,
        *,
        embed_batch_size: int = 128,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._document_store = document_store
        self._batch = embed_batch_size

    async def index(self, documents: list[Document]) -> int:
        """Index documents; returns the number of chunks written.

        Idempotent per document: a document's previously-indexed chunks are
        deleted before its new chunks are added, so re-indexing a changed (or
        unchanged) document never leaves stale or duplicate chunks behind."""
        if not documents:
            return 0
        await self._document_store.put(documents)
        await self._vector_store.delete([doc.id for doc in documents])

        all_chunks = []
        for doc in documents:
            all_chunks.extend(await self._chunker.chunk(doc))

        for start in range(0, len(all_chunks), self._batch):
            batch = all_chunks[start : start + self._batch]
            embeddings = await self._embedder.embed_documents(
                [c.embedding_text for c in batch]
            )
            await self._vector_store.add(batch, embeddings)

        return len(all_chunks)
