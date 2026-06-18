# RAGu

**Agentic RAG over your whole corpus.** Instead of retrieving a handful of chunks
and stuffing them into a prompt, RAGu hands the *entire* indexed corpus to an
agentic reasoning engine ([vomero](https://github.com/AntonioGr7/vomero)) that
navigates it the way a person would — ranked search, grep, read, recurse,
multi-hop — and answers with citations. The model decides what evidence it needs,
so retrieval recall never caps the answer. This is the default and the point:
**maximum expressivity**, which matters most for multi-hop questions whose
final-hop evidence isn't anywhere near the raw query.

- **L2 — Reasoning (the main path).** The agentic engine (vomero, isolated behind
  a `ReasoningEngine` port) reasons over the full corpus: it `search()`es
  (hybrid BM25 ranked retrieval, built-in), `grep`s, reads, and recurses, then
  answers with document-level citations — span-level with grounding on.
- **L1 — Retrieval (optional pre-filter).** Hybrid dense + BM25 retrieval with
  contextual-retrieval chunking that narrows the corpus to a working set *before*
  L2 reasons. Off by default — turn it on when the corpus is large but the
  questions are simple and don't need full agentic navigation. Its job is never
  to find the answer, only to shrink what L2 looks at.

All LLM access goes through a generic `ChatModel` port, so every model-backed
feature works with **OpenAI, any OpenAI-compatible server, or Gemini** — the
provider is pure configuration, never hardcoded.

## Install

```bash
uv sync                       # core library
uv sync --all-extras          # everything (OCR, L2, all providers)
```

Optional extras, install what you need:

| Extra | Enables |
|---|---|
| `local` | Local embeddings via sentence-transformers (default embedder) |
| `ocr`, `ocr-cpu` | Read images / scanned PDFs (PP-OCRv6); see [OCR install](#ocr-pp-ocrv6) |
| `gemini` | Gemini chat + embeddings (`google-genai` SDK) |
| `voyage` | Hosted Voyage embeddings |
| `l2` | L2 reasoning (vomero) |

```bash
uv pip install 'ragu[local]'          # e.g. just local embeddings
uv pip install 'ragu[ocr,ocr-cpu,l2]' # OCR + reasoning
```

## Configure

All settings come from environment variables or a `.env` file. Copy the template
and edit what you need — every option there shows its default:

```bash
cp .env.example .env
```

Naming convention:

- `RAGU_<GROUP>__<KEY>` — RAGu settings, note the **double** underscore
  (e.g. `RAGU_RETRIEVAL__CANDIDATE_K=500`).
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, … — provider keys have **no** prefix; the
  provider SDKs read them directly.

A minimal setup is usually just a provider key plus, if you ingest scans, OCR:

```bash
OPENAI_API_KEY=sk-...
RAGU_OCR__ENABLED=true
```

### Choosing an LLM provider

The chat model (used for contextual chunking and answer grounding) is configured
under `RAGU_LLM__*`, independent of the L2 engine (`RAGU_VOMERO__*`) — you can run
them on different models.

```bash
# OpenAI or any OpenAI-compatible server (vLLM, Ollama, LM Studio, Together, …)
RAGU_LLM__PROVIDER=openai_compat
RAGU_LLM__MODEL=gpt-4o-mini
RAGU_LLM__BASE_URL=http://localhost:11434/v1   # omit for OpenAI itself
RAGU_LLM__API_KEY=...                           # omit for keyless local servers

# Gemini (needs the `gemini` extra)
RAGU_LLM__PROVIDER=gemini
RAGU_LLM__MODEL=gemini-flash-latest
```

## Indexing

```bash
ragu index ./docs ./more_docs     # ingest text files + (if enabled) OCR'd images/PDFs
```

Ingestion reads text files directly and routes images/PDFs through PP-OCRv6 OCR
when `RAGU_OCR__ENABLED=true`. Each document's id is its path relative to the
*common ancestor* of the paths you pass, so the same filename in different
folders never collides, however you invoke the command.

**Incremental & idempotent.** Every file is fingerprinted (a hash of its bytes);
on re-index, unchanged files are skipped *before* OCR, changed files are
re-indexed in place (old chunks replaced, never duplicated), and — by default —
documents whose source file has been deleted are pruned from the store. The
summary line reports what happened:

```text
Indexed 12 chunks (new=2, updated=1, skipped=40, pruned=1).
```

Pass `--no-prune` to keep documents whose source files are gone.

### OCR (PP-OCRv6)

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

Both born-digital PDFs (text layer) and scanned ones (OCR) come back with the
same structured per-word geometry, which powers inline citations (below).

## Browsing the index

```bash
ragu list                 # every indexed document (id + source)
ragu show <doc_id>        # one document: id, metadata, then full content
ragu show <doc_id> --json # the whole record as JSON, incl. OCR geometry
```

`list`/`show` only touch the document store, so they're fast — they don't load
the embedder.

## Retrieving (L1, optional)

L1 is the optional pre-filter. Select a working set without generating an answer:

```bash
ragu retrieve "what blocks P-BEACON?"
```

```text
Working set: 3 docs, 48210 tokens (truncated=False)
  - guide.md      [/abs/guide.md]
  - design.pdf    [/abs/design.pdf]
  ...
```

## Answering (agentic, full-corpus by default)

```bash
ragu answer "what blocks P-BEACON?"          # needs the `l2` extra + an LLM key
ragu answer "what blocks P-BEACON?" --cite   # also produce inline citations (below)
```

By default L2 reasons over the **entire** indexed corpus — L1 retrieval is
skipped, so dense recall never becomes the ceiling. RAGu materializes the corpus
as a folder of files, and [vomero](https://github.com/AntonioGr7/vomero) (behind
the `ReasoningEngine` port) navigates it agentically — `search()` (built-in
hybrid BM25 ranked retrieval), `grep`, read, recurse, multi-hop — answering with
citations for the documents it used. Configure via `RAGU_VOMERO__*`
(provider/model/base_url/limits).

**Pre-filter with L1 for simple sets.** When the corpus is large but the
questions don't need full agentic navigation, narrow it with L1 first:

```bash
ragu answer "what blocks P-BEACON?" --no-full-corpus   # L1 selects a working set, then L2 reasons over it
```

or globally with `RAGU_VOMERO__FULL_CORPUS=false`. Either way it's a per-query
choice — the same corpus serves both paths.

> **Search index.** L2's `search()` is backed by a persistent lexical index
> (SQLite FTS5) built once from the corpus and opened read-only, so the first
> question isn't slow. It lives next to the LanceDB store by default
> (`RAGU_VOMERO__SEARCH_INDEX_DIR`; set `off` to disable). The web server warms
> it at startup.

### Inline citations (`--cite`)

By default citations are document-level. With `--cite` (or
`RAGU_VOMERO__GROUND_CITATIONS=true`) RAGu runs an extra grounding pass that
returns **span-level** citations anchored to the source:

- the **verbatim quote** that supports each claim,
- the **page** it appears on, and
- **word-level bounding boxes** in that page's pixel space — ready to highlight
  on the page rendered at any resolution.

```text
Sources:
  - mutuo.pdf  [/abs/mutuo.pdf]
      “…al tasso annuo nominale variabile del 2,453%…”
      page 4: ((443, 1659, 1287, 1689), (258, 1739, 1288, 1769))
```

This costs one extra LLM call (via the `RAGU_LLM__*` chat model — any provider),
so it's **off by default** to keep the standard answer path fast. It is
best-effort: if a quote can't be located in the source it's still listed (without
boxes), and any failure falls back to the plain answer — enabling it never breaks
the answer. Boxes come from the document's OCR/text-layer geometry, so they're
available for indexed images and PDFs.

The grounding evidence can come from three places (`--cite-source`, or
`RAGU_VOMERO__GROUNDING_SOURCE`):

- `trajectory` (default) — an LLM extracts quotes from the text L2 *actually
  read* while reasoning. Most specific, since it's exactly the evidence the
  answer rests on.
- `document` — an LLM extracts quotes from the full cited document(s).
- `raw` — **no extra LLM call** (and no API key needed): the lines L2 read that
  share a number or name with the answer are highlighted directly. Cheapest, but
  coarser — whole read lines rather than the precise supporting clause.

```bash
ragu answer "…" --cite-source raw        # free highlights, no extra LLM call
ragu answer "…" --cite-source document   # implies --cite
```

`trajectory` and `document` cost one extra LLM call; `raw` costs none.

## Extracting text (no indexing)

Dump extracted text (incl. OCR) to a folder, mirroring the source structure —
useful for inspecting what ingestion sees:

```bash
ragu extract ./scans --out ./scans_text
```

## Programmatic API

The `Ragu` facade wires every backend from settings and exposes the same
operations as the CLI:

```python
from ragu.app import Ragu

ragu = Ragu()

# index (incremental; returns a report)
report = await ragu.index_paths(["./docs"])
print(report.new, report.updated, report.skipped, report.pruned)

# browse
for ref in await ragu.list_documents():     # lightweight: id, source, hash
    print(ref.id)
doc = await ragu.get_document("design.pdf")  # full content + metadata + artifacts

# answer over the full corpus (default); pass full_corpus=False to pre-filter with L1
answer = await ragu.answer("what blocks P-BEACON?", ground=True)
ws = await ragu.retrieve("what blocks P-BEACON?")   # L1 only, when you want the working set
for c in answer.citations:
    print(c.doc_id, c.quote)
    for h in c.highlights:                   # page + boxes, when grounded
        print(h.page, h.boxes)
```

## Architecture

Hexagonal / ports-and-adapters — the pipeline depends only on `ragu.ports`, so
every backend is swappable.

```
src/ragu/
  core/        # pure domain types (Document, Chunk, RetrievalResult, WorkingSet, Answer)
  ports/       # Protocols: Embedder, Chunker, VectorStore, DocumentStore,
               #            Reranker, Retriever, ReasoningEngine, ChatModel,
               #            PreferenceStore, SessionMemory, TokenCounter
  adapters/    # concrete backends (embedding, chunking, storage, retrieval, llm, ingestion, ...)
  pipeline/    # orchestration: Indexer, working-set assembly, citation grounding
  config/      # pydantic-settings (incl. LLM provider + vomero knobs)
```

Three memories, kept distinct: **preferences** (mem0, long-term per-user),
**working set** (per-session bytes-aware LRU of documents), **reasoning
transcript** (owned by L2).

## Develop

```bash
uv sync --all-extras        # install
uv run pytest               # tests (run fully offline on the fake embedder)
uvx ruff check src tests    # lint
```
