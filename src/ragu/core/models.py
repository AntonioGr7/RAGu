"""Core domain models.

These types are the contract between layers. They are deliberately free of any
retrieval/embedding/LLM machinery so they can be reasoned about and tested in
isolation. Behaviour that is *intrinsic* to the data (e.g. "what text do we
embed for a chunk", "how do chunk scores roll up to documents") lives here;
behaviour that depends on infrastructure lives in adapters.
"""

from __future__ import annotations

from typing import Any, NewType

from pydantic import BaseModel, ConfigDict, Field

# A stable identifier for a source document. Kept as a distinct type so the type
# checker catches accidental mixing with chunk ids or arbitrary strings.
DocumentId = NewType("DocumentId", str)


class Document(BaseModel):
    """A whole source document — the unit L1 selects and L2 reasons over.

    The two-level design hinges on this: L1 retrieves *chunks* only to find
    their parent ``Document``; L2 then navigates the full ``content``. Chunk
    boundaries therefore affect findability, not answer assembly.
    """

    model_config = ConfigDict(frozen=True)

    id: DocumentId
    source: str  # path or URI the document was ingested from
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    # Structured, JSON-serializable byproducts of ingestion that don't fit the
    # flat string ``metadata`` — e.g. OCR geometry (word boxes, polys, scores)
    # under key "ocr". Carried alongside the document so it is never lost; how
    # it is persisted/indexed is the storage layer's choice.
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_bytes(self) -> int:
        return len(self.content.encode("utf-8"))


class DocumentRef(BaseModel):
    """A lightweight handle to a stored document — its id, source path, and
    content fingerprint — without the (potentially large) content.

    Used for incremental indexing: comparing the on-disk fingerprint against the
    stored ``content_hash`` decides skip vs re-index, and ``source`` lets a prune
    pass detect documents whose source file has been deleted."""

    model_config = ConfigDict(frozen=True)

    id: DocumentId
    source: str
    content_hash: str | None = None


class IndexReport(BaseModel):
    """Summary of one incremental index run, surfaced to the CLI."""

    model_config = ConfigDict(frozen=True)

    chunks: int = 0  # chunks written this run
    new: int = 0  # documents indexed for the first time
    updated: int = 0  # changed documents re-indexed in place
    skipped: int = 0  # unchanged documents left untouched (no OCR/embed)
    pruned: int = 0  # documents removed because their source file is gone


class Chunk(BaseModel):
    """A retrieval unit carved from a ``Document``.

    ``context`` is the contextual-retrieval blurb (an LLM-generated sentence or
    two situating the chunk within its document). We embed ``embedding_text``
    (context + body) but always carry ``doc_id`` so retrieval can climb back to
    the parent.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: DocumentId
    text: str
    context: str = ""
    ordinal: int = 0  # position of this chunk within its document
    start_char: int = 0
    end_char: int = 0
    section_path: tuple[str, ...] = ()  # e.g. ("Chapter 2", "Risks")
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """The text actually fed to the embedder (contextual retrieval)."""
        return f"{self.context}\n\n{self.text}".strip() if self.context else self.text


class ScoredChunk(BaseModel):
    """A chunk with its retrieval score and, where available, the component
    scores that produced it — kept for transparency and eval/debugging."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class ScoredDocument(BaseModel):
    """A parent document with the score it earned via its best chunk, plus the
    chunks that hit. This is L1's real output unit."""

    model_config = ConfigDict(frozen=True)

    doc_id: DocumentId
    score: float
    hits: tuple[ScoredChunk, ...]


class RetrievalResult(BaseModel):
    """The raw chunk-level output of an L1 query, with a pure roll-up to the
    document level. Keeping the roll-up here (not in the retriever) means every
    backend dedups identically."""

    model_config = ConfigDict(frozen=True)

    query_text: str
    chunks: tuple[ScoredChunk, ...]

    def to_documents(self) -> list[ScoredDocument]:
        """Collapse scored chunks to scored documents.

        A document's score is its best chunk's score (max-pooling — robust and
        order-independent). Documents come back sorted best-first.
        """
        by_doc: dict[DocumentId, list[ScoredChunk]] = {}
        for sc in self.chunks:
            by_doc.setdefault(sc.chunk.doc_id, []).append(sc)

        docs = [
            ScoredDocument(
                doc_id=doc_id,
                score=max(sc.score for sc in hits),
                hits=tuple(sorted(hits, key=lambda s: s.score, reverse=True)),
            )
            for doc_id, hits in by_doc.items()
        ]
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs


class WorkingSet(BaseModel):
    """The bounded set of documents handed to L2.

    Built by loading parent documents (best-ranked first) until a token budget
    is hit — the budget is on *bytes/tokens*, not document count, because what
    bounds L2 cost is how much text it must navigate, not how many docs."""

    model_config = ConfigDict(frozen=True)

    documents: tuple[Document, ...]
    token_count: int
    truncated: bool = False  # True if docs were dropped to stay within budget

    @property
    def doc_ids(self) -> tuple[DocumentId, ...]:
        return tuple(d.id for d in self.documents)


class Query(BaseModel):
    """A user query plus the session/user context that shapes retrieval."""

    text: str
    session_id: str | None = None
    user_id: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    """A pointer from an answer back to the source it rests on.

    Provenance must survive the whole pipeline: L2 navigates documents and must
    be able to say *which* document (and ideally which span) grounds each claim.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: DocumentId
    source: str
    quote: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class Answer(BaseModel):
    """The system's response to a query, with citations and a light trace."""

    model_config = ConfigDict(frozen=True)

    text: str
    citations: tuple[Citation, ...] = ()
    used_reasoning: bool = False  # True if L2 (RLM) was invoked
    trace: dict[str, str] = Field(default_factory=dict)
