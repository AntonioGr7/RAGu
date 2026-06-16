"""Ports — the abstract contracts every adapter implements.

This is the seam that makes RAGu swappable end-to-end: the pipeline depends only
on these Protocols, never on a concrete backend. Want Qdrant instead of
LanceDB, or a different RLM than vomero? Implement the Protocol; nothing else
changes.

All I/O-bound ports are ``async`` because real implementations hit the network
(embeddings, LLMs, vector stores). CPU-only contracts (token counting) are sync.
"""

from ragu.ports.chunking import Chunker
from ragu.ports.embedding import Embedder
from ragu.ports.llm import ChatMessage, ChatModel
from ragu.ports.memory import PreferenceStore, SessionMemory
from ragu.ports.reasoning import ReasoningEngine
from ragu.ports.retrieval import Reranker, Retriever
from ragu.ports.storage import DocumentStore, VectorStore
from ragu.ports.tokens import TokenCounter

__all__ = [
    "ChatMessage",
    "ChatModel",
    "Chunker",
    "DocumentStore",
    "Embedder",
    "PreferenceStore",
    "ReasoningEngine",
    "Reranker",
    "Retriever",
    "SessionMemory",
    "TokenCounter",
    "VectorStore",
]
