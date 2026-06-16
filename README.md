# RAGu

A two-level RAG system.

- **L1 — Retrieval (working-set selector).** Hybrid dense + BM25 retrieval with
  contextual-retrieval chunking. Its job is *not* to find the answer but to
  narrow a large corpus to the documents worth reasoning over. Chunks roll up to
  their parent documents (document-level recall is the target).
- **L2 — Reasoning (agentic RLM).** An agentic engine ([vomero](https://github.com/AntonioGr7/vomero),
  isolated behind a port) navigates the selected working set — grep/read/recurse/
  multihop — and answers with citations. Invoked only when a query needs it.

## Architecture

Hexagonal / ports-and-adapters — the pipeline depends only on `ragu.ports`, so
every backend is swappable.

```
src/ragu/
  core/        # pure domain types (Document, Chunk, RetrievalResult, WorkingSet, Answer)
  ports/       # Protocols: Embedder, Chunker, VectorStore, DocumentStore,
               #            Reranker, Retriever, ReasoningEngine, ChatModel,
               #            PreferenceStore, SessionMemory, TokenCounter
  adapters/    # concrete backends (embedding, chunking, storage, retrieval, llm, ...)
  pipeline/    # orchestration: Indexer, working-set assembly
  config/      # pydantic-settings (incl. LLM provider + vomero knobs)
```

Three memories, kept distinct: **preferences** (mem0, long-term per-user),
**working set** (per-session bytes-aware LRU of documents), **reasoning
transcript** (owned by L2). All LLM access goes through a generic `ChatModel`
port — OpenAI-compatible, Gemini, or self-hosted; never a hardcoded vendor.

## Indexing data

Via the CLI (config comes from env / `.env`):

```bash
ragu index ./docs ./more_docs        # ingest text files + OCR'd images/PDFs
ragu retrieve "what blocks P-BEACON?" # L1: show the selected working set
```

Or programmatically through the facade:

```python
from ragu.app import Ragu

ragu = Ragu()                          # backends wired from settings
await ragu.index_paths(["./docs"])     # text (.md/.txt/...) + images/PDFs (if OCR enabled)
ws = await ragu.retrieve("what blocks P-BEACON?")
```

Ingestion reads text directly and routes images/PDFs through PP-OCRv6 OCR when
`RAGU_OCR__ENABLED=true`. Document ids are the path relative to the ingestion
root, so re-indexing updates in place.

### OCR (PP-OCRv6) install

CPU:

```bash
uv pip install 'ragu[ocr,ocr-cpu]'
```

GPU — the paddle runtime is not on PyPI; install the build matching your CUDA
(e.g. 12.x → cu126), then select the device:

```bash
uv pip install 'ragu[ocr]'
uv pip uninstall paddlepaddle    # if the CPU build is present
uv pip install paddlepaddle-gpu --index https://www.paddlepaddle.org.cn/packages/stable/cu126/
export RAGU_OCR__ENABLED=true RAGU_OCR__DEVICE=gpu:0
```

On a small GPU (≈4 GB) use the tiny/small tier and cap detection resolution:
`RAGU_OCR__DET_MODEL_NAME=PP-OCRv6_small_det`,
`RAGU_OCR__REC_MODEL_NAME=PP-OCRv6_small_rec`,
`RAGU_OCR__DET_LIMIT_SIDE_LEN=736`. PaddleOCR 3.7 defaults to PP-OCRv6.

## Status

L1 vertical slice is implemented and tested: ingest → contextual chunking →
hybrid retrieval (RRF) → doc-level dedup → token-bounded working set, on both an
in-memory backend and **LanceDB**.

## Develop

```bash
uv sync --all-extras        # install
uv run pytest               # tests (run fully offline on the fake embedder)
uvx ruff check src tests    # lint
```
