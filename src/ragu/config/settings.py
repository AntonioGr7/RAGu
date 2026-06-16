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
    provider: str = "fake"  # "voyage" | "fake" (deterministic, for dev/tests)
    model: str = "voyage-3"
    dim: int = 1024
    batch_size: int = 128


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
    # Max image side length fed to detection. The main VRAM lever on small GPUs:
    # lower it (e.g. 960 → 736) if detection OOMs. None = PaddleOCR default.
    det_limit_side_len: int | None = None
    # Recognition batch size; lower to cut peak VRAM. None = PaddleOCR default.
    rec_batch_size: int | None = None


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
    """L2 reasoning-engine knobs, owned by RAGu and passed through to vomero."""

    model: str = "claude-opus-4-8"
    max_depth: int = 3
    max_total_tokens: int = 200_000
    max_total_calls: int = 40
    context_window: int = 128_000
    compact_ratio: float = 0.8
    max_output_chars: int = 10_000
    plan: bool = False
    sandbox: bool = False  # trusted corpus for v1
    # How L1's working set is materialised for vomero: "corpus" (temp folder) or
    # "context" (in-memory). Corpus pairs naturally with grep/files navigation.
    handoff: str = "corpus"


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
