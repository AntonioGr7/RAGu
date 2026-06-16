import pytest

from ragu.adapters.chunking import ContextualChunker, RecursiveTokenSplitter
from ragu.adapters.tokens import TiktokenCounter
from ragu.core import Document, DocumentId

COUNTER = TiktokenCounter()


def test_splitter_offsets_round_trip() -> None:
    text = "\n\n".join(f"Paragraph number {i}. " * 20 for i in range(8))
    splitter = RecursiveTokenSplitter(COUNTER, target_tokens=60, overlap_tokens=0)
    atoms = splitter.split(text)

    assert len(atoms) > 1
    for atom in atoms:
        # Offsets must slice back to exactly the atom's text.
        assert text[atom.start : atom.end] == atom.text


def test_splitter_respects_target_size() -> None:
    text = " ".join(f"word{i}" for i in range(2000))
    splitter = RecursiveTokenSplitter(COUNTER, target_tokens=100, overlap_tokens=0)
    atoms = splitter.split(text)
    # Allow modest overshoot from atom granularity, but nothing wild.
    assert all(COUNTER.count(a.text) <= 200 for a in atoms)


@pytest.mark.asyncio
async def test_chunker_assigns_ids_and_doc_id() -> None:
    doc = Document(id=DocumentId("doc1"), source="mem", content="Hello world. " * 200)
    chunker = ContextualChunker(RecursiveTokenSplitter(COUNTER, target_tokens=50))
    chunks = await chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(c.doc_id == DocumentId("doc1") for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].context == ""  # no contextualizer wired
