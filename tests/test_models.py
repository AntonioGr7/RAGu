from ragu.core import Chunk, DocumentId, RetrievalResult, ScoredChunk


def _sc(doc: str, ordinal: int, score: float) -> ScoredChunk:
    doc_id = DocumentId(doc)
    return ScoredChunk(
        chunk=Chunk(id=f"{doc}::{ordinal}", doc_id=doc_id, text="x"),
        score=score,
    )


def test_to_documents_max_pools_and_sorts() -> None:
    result = RetrievalResult(
        query_text="q",
        chunks=(
            _sc("A", 0, 0.2),
            _sc("A", 1, 0.9),  # A's best chunk
            _sc("B", 0, 0.5),
        ),
    )
    docs = result.to_documents()

    assert [d.doc_id for d in docs] == [DocumentId("A"), DocumentId("B")]
    assert docs[0].score == 0.9  # max-pooled
    assert len(docs[0].hits) == 2
    # Hits within a document are sorted best-first.
    assert docs[0].hits[0].score == 0.9


def test_embedding_text_prepends_context() -> None:
    c = Chunk(id="d::0", doc_id=DocumentId("d"), text="body", context="ctx")
    assert c.embedding_text == "ctx\n\nbody"
    bare = Chunk(id="d::1", doc_id=DocumentId("d"), text="body")
    assert bare.embedding_text == "body"
