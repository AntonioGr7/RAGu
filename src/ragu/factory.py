"""Composition root: build wired components from ``RaguSettings``.

This is the *one* place that knows which concrete adapter backs each port.
Everything else depends on ports only. Selection is driven entirely by config,
so swapping a backend is a settings change, not a code change.
"""

from __future__ import annotations

from ragu.adapters.chunking import ContextualChunker, RecursiveTokenSplitter
from ragu.adapters.contextualizer import LLMContextualizer
from ragu.adapters.embedding import FakeEmbedder, LocalEmbedder, VoyageEmbedder
from ragu.adapters.ingestion.ocr import OcrEngine
from ragu.adapters.llm import GeminiChat, OpenAICompatChat, ScriptedChat
from ragu.adapters.retrieval import HybridRetriever
from ragu.adapters.storage import (
    InMemoryDocumentStore,
    InMemoryVectorStore,
    LanceDocumentStore,
    LanceVectorStore,
)
from ragu.adapters.tokens import TiktokenCounter
from ragu.config import RaguSettings
from ragu.ports import (
    ChatModel,
    Chunker,
    DocumentStore,
    Embedder,
    ReasoningEngine,
    Retriever,
    TokenCounter,
    VectorStore,
)


def build_token_counter(settings: RaguSettings) -> TokenCounter:
    return TiktokenCounter()


def build_ocr_engine(settings: RaguSettings) -> OcrEngine | None:
    """Build the OCR engine from settings, or None when OCR is disabled."""
    cfg = settings.ocr
    if not cfg.enabled:
        return None
    from ragu.adapters.ingestion import PaddleOcrEngine

    return PaddleOcrEngine(
        det_model_name=cfg.det_model_name,
        rec_model_name=cfg.rec_model_name,
        det_model_dir=cfg.det_model_dir,
        rec_model_dir=cfg.rec_model_dir,
        lang=cfg.lang,
        device=cfg.device,
        enable_mkldnn=cfg.enable_mkldnn,
        det_limit_side_len=cfg.det_limit_side_len,
        det_limit_type=cfg.det_limit_type,
        rec_batch_size=cfg.rec_batch_size,
    )


def build_chat_model(settings: RaguSettings) -> ChatModel:
    cfg = settings.llm
    if cfg.provider == "scripted":
        return ScriptedChat()
    if cfg.provider == "gemini":
        return GeminiChat(cfg.model, api_key=cfg.api_key)
    if cfg.provider == "openai_compat":
        return OpenAICompatChat(cfg.model, base_url=cfg.base_url, api_key=cfg.api_key)
    raise ValueError(f"unknown LLM provider: {cfg.provider!r}")


def build_embedder(settings: RaguSettings) -> Embedder:
    cfg = settings.embedding
    if cfg.provider == "fake":
        return FakeEmbedder(dim=cfg.dim)
    if cfg.provider == "local":
        return LocalEmbedder(
            cfg.model,
            device=cfg.device,
            normalize=cfg.normalize,
            batch_size=cfg.batch_size,
            query_prefix=cfg.query_prefix,
            document_prefix=cfg.document_prefix,
            query_prompt_name=cfg.query_prompt_name,
            document_prompt_name=cfg.document_prompt_name,
            trust_remote_code=cfg.trust_remote_code,
        )
    if cfg.provider == "voyage":
        return VoyageEmbedder(model=cfg.model, dim=cfg.dim)
    raise ValueError(f"unknown embedding provider: {cfg.provider!r}")


def build_chunker(settings: RaguSettings, counter: TokenCounter) -> Chunker:
    splitter = RecursiveTokenSplitter(
        counter,
        target_tokens=settings.chunking.target_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
    )
    contextualizer = None
    if settings.chunking.contextual:
        model = build_chat_model(settings)
        contextualizer = LLMContextualizer(model)
    return ContextualChunker(splitter, contextualizer)


def build_stores(settings: RaguSettings, dim: int) -> tuple[VectorStore, DocumentStore]:
    if settings.storage.backend == "memory":
        return InMemoryVectorStore(), InMemoryDocumentStore()
    if settings.storage.backend == "lance":
        uri = settings.storage.lancedb_uri
        vector = LanceVectorStore(uri, dim=dim, table=settings.storage.chunks_table)
        documents = LanceDocumentStore(uri, table=settings.storage.documents_table)
        return vector, documents
    raise ValueError(f"unknown storage backend: {settings.storage.backend!r}")


def build_reasoning_engine(settings: RaguSettings) -> ReasoningEngine:
    """Build the L2 engine (vomero). Importing the adapter does not import
    vomero — that happens lazily on first use, so this is import-safe."""
    from ragu.adapters.reasoning import VomeroReasoningEngine

    return VomeroReasoningEngine(settings.vomero)


def build_retriever(settings: RaguSettings, embedder: Embedder, store: VectorStore) -> Retriever:
    return HybridRetriever(
        embedder,
        store,
        candidate_k=settings.retrieval.candidate_k,
        rrf_k=settings.retrieval.rrf_k,
        rerank_top_k=settings.retrieval.rerank_top_k,
    )
