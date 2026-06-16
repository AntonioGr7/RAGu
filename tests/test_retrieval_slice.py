"""End-to-end L1 vertical slice on offline backends:
index → hybrid retrieve → doc-level dedup → working-set assembly.
"""

import pytest

from ragu.adapters.chunking import ContextualChunker, RecursiveTokenSplitter
from ragu.adapters.embedding import FakeEmbedder
from ragu.adapters.retrieval import HybridRetriever
from ragu.adapters.storage import InMemoryDocumentStore, InMemoryVectorStore
from ragu.adapters.tokens import TiktokenCounter
from ragu.core import Document, DocumentId, Query
from ragu.pipeline import Indexer, build_working_set

COUNTER = TiktokenCounter()

CORPUS = {
    "physics": "Quantum entanglement links particles across distance. "
    "Photons exhibit wave particle duality in the double slit experiment.",
    "cooking": "To make risotto, toast the arborio rice, then add warm broth "
    "slowly while stirring until creamy.",
    "finance": "Compound interest grows principal exponentially over time as "
    "interest accrues on prior interest.",
}


async def _build_index() -> tuple[HybridRetriever, InMemoryDocumentStore]:
    embedder = FakeEmbedder(dim=128)
    vstore = InMemoryVectorStore()
    dstore = InMemoryDocumentStore()
    chunker = ContextualChunker(RecursiveTokenSplitter(COUNTER, target_tokens=40))
    indexer = Indexer(chunker, embedder, vstore, dstore)

    docs = [Document(id=DocumentId(k), source="mem", content=v) for k, v in CORPUS.items()]
    n = await indexer.index(docs)
    assert n >= len(docs)

    retriever = HybridRetriever(embedder, vstore, candidate_k=50, rrf_k=60)
    return retriever, dstore


@pytest.mark.asyncio
async def test_relevant_document_ranks_first() -> None:
    retriever, _ = await _build_index()
    result = await retriever.retrieve(Query(text="quantum entanglement photons"), k=20)
    docs = result.to_documents()

    assert docs, "expected at least one document"
    assert docs[0].doc_id == DocumentId("physics")


@pytest.mark.asyncio
async def test_working_set_respects_token_budget() -> None:
    retriever, dstore = await _build_index()
    result = await retriever.retrieve(Query(text="risotto broth rice"), k=20)
    ranked = result.to_documents()

    tiny = await build_working_set(ranked, dstore, COUNTER, max_tokens=5)
    # Budget so small only the first doc fits; rest dropped + flagged.
    assert len(tiny.documents) == 1
    assert tiny.truncated is True

    big = await build_working_set(ranked, dstore, COUNTER, max_tokens=100_000)
    assert big.truncated is False
    assert big.token_count == sum(COUNTER.count(d.content) for d in big.documents)


@pytest.mark.asyncio
async def test_working_set_doc_count_bound_ignores_tokens() -> None:
    # Corpus handoff: max_tokens=None -> bound purely by document count, so a
    # large first document never starves out the rest.
    retriever, dstore = await _build_index()
    result = await retriever.retrieve(Query(text="risotto broth rice"), k=20)
    ranked = result.to_documents()

    ws = await build_working_set(ranked, dstore, COUNTER, max_tokens=None)
    assert len(ws.documents) == len(ranked)
    assert ws.truncated is False

    capped = await build_working_set(
        ranked, dstore, COUNTER, max_tokens=None, max_documents=1
    )
    assert len(capped.documents) == 1
    # Dropped by the count cap (when more were ranked), so coverage is flagged.
    assert capped.truncated is (len(ranked) > 1)
