"""RAGu — a two-level RAG system.

L1 hybrid retrieval selects a *working set* of documents; an agentic RLM (L2)
reasons over it. See ``ragu.core`` for the domain model and ``ragu.ports`` for
the swappable component contracts.
"""

__version__ = "0.0.1"
