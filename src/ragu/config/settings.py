"""Typed, environment-driven configuration.

One nested settings tree, loaded from environment variables (prefix ``RAGU_``)
or a ``.env`` file. Nested groups use ``__`` as the delimiter, e.g.
``RAGU_RETRIEVAL__CANDIDATE_K=500``.

The vomero knobs (depth, token/call budgets, ...) are surfaced *here* rather
than left inside the RLM, so L2 cost is governed from one place — a stated goal
of the design.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """Generic chat-model selection. Used for contextual retrieval, routing,
    and any other LLM glue — never tied to one vendor."""

    provider: str = "openai_compat"  # "openai_compat" | "gemini" | "scripted"
    model: str = "gpt-4o-mini"
    base_url: str | None = None  # e.g. self-hosted or Anthropic compat endpoint
    api_key: str | None = None  # falls back to the provider SDK's env var if None
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    # "local" (sentence-transformers, runs offline/on-GPU), "voyage" (hosted),
    # "fake" (deterministic, for dev/tests).
    provider: str = "local"
    # Default: a small, strong, multilingual model that fits a modest GPU.
    model: str = "intfloat/multilingual-e5-small"
    # dim is informational for hosted providers; the local provider reports the
    # model's true dimension regardless of this value.
    dim: int = 384
    batch_size: int = 64
    device: str = "cpu"  # "cpu" or "cuda" / "cuda:0" for the local provider
    normalize: bool = True
    # Asymmetric retrieval prefixes. E5 models expect "query:"/"passage:"; set
    # both empty for models that don't use instructions (e.g. BGE-M3).
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    # Alternatively use a model's own registered sentence-transformers prompts
    # (correct for Snowflake Arctic). When set, the matching prefix is ignored.
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    # Needed by models shipping custom code (e.g. Arctic v2.0 / GTE-multilingual).
    trust_remote_code: bool = False


class ChunkingSettings(BaseModel):
    target_tokens: int = 400  # ~SOTA passage size for dense retrieval
    overlap_tokens: int = 64
    # Contextual retrieval: generate a situating blurb per chunk via the LLM
    # port. Disable to fall back to plain (offline) chunking.
    contextual: bool = True
    # Optional cheaper model override for blurb generation; falls back to
    # LLMSettings.model when None.
    context_model: str | None = None


class RetrievalSettings(BaseModel):
    # L1 casts a wide net: it selects the working set, not the answer.
    candidate_k: int = 200  # chunks pulled per channel before fusion
    rrf_k: int = 60  # reciprocal-rank-fusion constant
    rerank: bool = False  # off until a reranker adapter is wired
    rerank_top_k: int = 50
    # Final number of parent documents handed toward working-set assembly.
    document_k: int = 100


class OcrSettings(BaseModel):
    """PP-OCRv6 OCR ingestion (PaddleOCR 3.x). Disabled by default; enable to
    read images/PDFs.

    Select a tier by model name, e.g. ``PP-OCRv6_small_det`` /
    ``PP-OCRv6_small_rec`` (tiers: tiny/small/medium); ``*_model_dir`` overrides
    with a local directory. None leaves PaddleOCR's defaults (already PP-OCRv6).
    """

    enabled: bool = False
    det_model_name: str | None = None
    rec_model_name: str | None = None
    det_model_dir: str | None = None
    rec_model_dir: str | None = None
    lang: str = "en"
    device: str = "cpu"  # e.g. "cpu" or "gpu:0"
    # paddlepaddle 3.x oneDNN CPU path crashes on PP-OCRv6 detection; keep off.
    # (Ignored on GPU.)
    enable_mkldnn: bool = False
    # Detection input resolution lever. With limit_type "max", side_len caps the
    # LONGEST side (downscales big scans, never upscales) — intuitive and stable.
    # With "min" (PaddleOCR default), it upscales the SHORTEST side, which can
    # blow tiny images up enormously. None = PaddleOCR default.
    det_limit_side_len: int | None = None
    det_limit_type: str | None = None  # "max" | "min" (None = PaddleOCR default)
    # Recognition batch size; lower to cut peak VRAM. None = PaddleOCR default.
    rec_batch_size: int | None = None
    # PDF strategy:
    #   "auto" — use the text layer, OCR only text-poor pages (fast, default)
    #   "ocr"  — OCR every page, ignoring the text layer (best for bad scans)
    #   "text" — text layer only, never OCR
    pdf_mode: str = "auto"
    pdf_dpi: int = 200
    pdf_min_text_chars: int = 16


class StorageSettings(BaseModel):
    backend: str = "lance"  # "lance" (persistent) | "memory" (ephemeral, dev/tests)
    lancedb_uri: str = "./.ragu/lancedb"
    chunks_table: str = "chunks"
    documents_table: str = "documents"


class MemorySettings(BaseModel):
    # Per-session working-set LRU, bounded by bytes (not entries).
    session_max_bytes: int = 256 * 1024 * 1024
    warm_boost: float = 0.1  # additive boost to warm-doc scores in L1
    preferences_enabled: bool = False  # mem0 (advanced feature, later)


class VomeroSettings(BaseModel):
    """L2 reasoning-engine knobs, owned by RAGu and mapped onto vomero.Settings.

    Provider-agnostic like the rest of RAGu: ``provider`` + ``base_url`` reach
    OpenAI-compatible servers (incl. Anthropic's compat endpoint) or Gemini.
    """

    provider: str = "openai"  # vomero providers: "openai" | "gemini"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None  # falls back to vomero's env resolution if None

    # RLM loop limits (map to vomero Settings of the same meaning).
    max_steps: int = 24
    max_depth: int = 3
    max_parallel_calls: int = 8
    max_total_tokens: int = 0  # 0 = unlimited (vomero convention)
    max_total_calls: int = 0
    context_window: int = 128_000
    compact_ratio: float = 0.8
    max_output_chars: int = 10_000
    plan: bool = False  # -> vomero enable_planning
    sandbox: bool = False  # True -> exec_backend="gvisor"; trusted corpus default

    # How L1's working set is materialised for vomero: "corpus" (temp folder) or
    # "context" (in-memory). Corpus pairs naturally with grep/files navigation
    # and is what enables document-level citations.
    handoff: str = "corpus"

    # Optional post-answer grounding pass: a cheap LLM call (via the chat-model
    # port) extracts verbatim supporting quotes and resolves them to page + word
    # boxes (inline citations). Off by default — it costs one extra LLM call, so
    # leave it off for the fastest answers.
    ground_citations: bool = False
    # Where grounding extracts quotes from: "trajectory" (what L2 actually read —
    # more specific) or "document" (the whole cited document).
    grounding_source: str = "trajectory"


class WorkingSetSettings(BaseModel):
    # The real bound on L2 cost is text volume, not document count.
    max_tokens: int = 150_000


class RaguSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGU_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    vomero: VomeroSettings = Field(default_factory=VomeroSettings)
    working_set: WorkingSetSettings = Field(default_factory=WorkingSetSettings)
