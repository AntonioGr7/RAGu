"""LanceDB-backed storage: the production vector + document store.

LanceDB is embedded (on-disk, no server) and serves both channels L1 needs:
vector search (cosine) and native BM25 full-text search (Tantivy). Connection
and tables are created lazily on first use, so construction is cheap and
import-safe.

Filtering note: filters are applied as SQL predicates over flat metadata columns
the caller indexed; arbitrary nested metadata filtering is out of scope for v1.
"""

from __future__ import annotations

import json

from ragu.core import Chunk, Document, DocumentId, DocumentRef, ScoredChunk


def _id_list(doc_ids: list[DocumentId]) -> str:
    """SQL ``IN`` list, single-quote-escaped, for a doc_id predicate."""
    return ", ".join("'" + str(i).replace("'", "''") + "'" for i in doc_ids)


class LanceVectorStore:
    def __init__(
        self,
        uri: str,
        dim: int,
        *,
        table: str = "chunks",
    ) -> None:
        self._uri = uri
        self._dim = dim
        self._table_name = table
        self._conn = None  # type: ignore[var-annotated]
        self._table = None  # type: ignore[var-annotated]
        self._fts_ready = False

    async def _connect(self):  # type: ignore[no-untyped-def]
        if self._conn is None:
            import lancedb

            self._conn = await lancedb.connect_async(self._uri)
        return self._conn

    def _schema(self):  # type: ignore[no-untyped-def]
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("context", pa.string()),
                pa.field("ordinal", pa.int32()),
                pa.field("start_char", pa.int32()),
                pa.field("end_char", pa.int32()),
                pa.field("metadata_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._dim)),
            ]
        )

    async def _ensure_table(self):  # type: ignore[no-untyped-def]
        if self._table is not None:
            return self._table
        conn = await self._connect()
        if self._table_name in await conn.table_names():
            self._table = await conn.open_table(self._table_name)
            self._fts_ready = True
        else:
            self._table = await conn.create_table(self._table_name, schema=self._schema())
        return self._table

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must align 1:1")
        table = await self._ensure_table()
        records = [
            {
                "id": c.id,
                "doc_id": str(c.doc_id),
                "text": c.text,
                "context": c.context,
                "ordinal": c.ordinal,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "metadata_json": json.dumps(c.metadata),
                "vector": emb,
            }
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        await table.add(records)
        await self._ensure_fts(table)

    async def delete(self, doc_ids: list[DocumentId]) -> None:
        if not doc_ids:
            return
        table = await self._ensure_table()
        await table.delete(f"doc_id IN ({_id_list(doc_ids)})")

    async def _ensure_fts(self, table):  # type: ignore[no-untyped-def]
        """Create the full-text index once there is data to index."""
        if self._fts_ready:
            return
        from lancedb.index import FTS

        # Index over context+body would be ideal, but FTS indexes a single
        # column; we index the body and rely on dense for context recall.
        await table.create_index("text", config=FTS(), replace=True)
        self._fts_ready = True

    async def search_dense(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        table = await self._ensure_table()
        q = table.query().nearest_to(query_embedding).distance_type("cosine").limit(k)
        q = _apply_filters(q, filters)
        rows = await q.to_list()
        # cosine distance in [0,2]; similarity = 1 - distance.
        return [_row_to_scored(r, score=1.0 - r["_distance"], channel="dense") for r in rows]

    async def search_lexical(
        self,
        query_text: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        table = await self._ensure_table()
        if not self._fts_ready:
            return []
        q = table.query().nearest_to_text(query_text).limit(k)
        q = _apply_filters(q, filters)
        rows = await q.to_list()
        return [_row_to_scored(r, score=r["_score"], channel="lexical") for r in rows]


def _apply_filters(query, filters: dict[str, str] | None):  # type: ignore[no-untyped-def]
    if not filters:
        return query
    clause = " AND ".join(f"{key} = '{val}'" for key, val in filters.items())
    return query.where(clause)


def _row_to_scored(row: dict, score: float, channel: str) -> ScoredChunk:
    chunk = Chunk(
        id=row["id"],
        doc_id=DocumentId(row["doc_id"]),
        text=row["text"],
        context=row.get("context", ""),
        ordinal=row.get("ordinal", 0),
        start_char=row.get("start_char", 0),
        end_char=row.get("end_char", 0),
        metadata=json.loads(row.get("metadata_json") or "{}"),
    )
    return ScoredChunk(
        chunk=chunk,
        score=score,
        dense_score=score if channel == "dense" else None,
        sparse_score=score if channel == "lexical" else None,
    )


class LanceDocumentStore:
    def __init__(self, uri: str, *, table: str = "documents") -> None:
        self._uri = uri
        self._table_name = table
        self._conn = None  # type: ignore[var-annotated]
        self._table = None  # type: ignore[var-annotated]

    async def _ensure_table(self):  # type: ignore[no-untyped-def]
        if self._table is not None:
            return self._table
        import lancedb
        import pyarrow as pa

        if self._conn is None:
            self._conn = await lancedb.connect_async(self._uri)
        if self._table_name in await self._conn.table_names():
            self._table = await self._conn.open_table(self._table_name)
        else:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("content", pa.string()),
                    pa.field("metadata_json", pa.string()),
                    # Structured ingestion byproducts (e.g. OCR geometry) as JSON,
                    # so they survive indexing rather than being dropped.
                    pa.field("artifacts_json", pa.string()),
                ]
            )
            self._table = await self._conn.create_table(self._table_name, schema=schema)
        return self._table

    async def put(self, documents: list[Document]) -> None:
        table = await self._ensure_table()
        merge = table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all()
        await merge.execute(
            [
                {
                    "id": str(d.id),
                    "source": d.source,
                    "content": d.content,
                    "metadata_json": json.dumps(d.metadata),
                    "artifacts_json": json.dumps(d.artifacts),
                }
                for d in documents
            ]
        )

    async def delete(self, doc_ids: list[DocumentId]) -> None:
        if not doc_ids:
            return
        table = await self._ensure_table()
        await table.delete(f"id IN ({_id_list(doc_ids)})")

    async def fingerprints(self) -> list[DocumentRef]:
        table = await self._ensure_table()
        # Pull only the small columns — never the document body.
        rows = await table.query().select(["id", "source", "metadata_json"]).to_list()
        refs: list[DocumentRef] = []
        for r in rows:
            meta = json.loads(r.get("metadata_json") or "{}")
            refs.append(
                DocumentRef(
                    id=DocumentId(r["id"]),
                    source=r["source"],
                    content_hash=meta.get("content_hash"),
                )
            )
        return refs

    async def get(self, doc_ids: list[DocumentId]) -> list[Document]:
        if not doc_ids:
            return []
        table = await self._ensure_table()
        rows = await table.query().where(f"id IN ({_id_list(doc_ids)})").to_list()
        by_id = {
            r["id"]: Document(
                id=DocumentId(r["id"]),
                source=r["source"],
                content=r["content"],
                metadata=json.loads(r.get("metadata_json") or "{}"),
                artifacts=json.loads(r.get("artifacts_json") or "{}"),
            )
            for r in rows
        }
        # Preserve the caller's requested order.
        return [by_id[str(i)] for i in doc_ids if str(i) in by_id]
