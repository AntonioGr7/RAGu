"""Memory contracts — the two long-/short-lived stores.

RAGu has three memories (see architecture notes); two of them are ports:

* ``PreferenceStore`` — long-term, cross-session, per-user preferences (mem0).
* ``SessionMemory`` — short-term, per-session **working set** of parsed
  documents. This is a *document-level* reuse cache, NOT a query→result cache:
  follow-up questions are reworded, so query keying yields a ~0 hit rate, while
  the document set stays stable across a conversation.

The third memory (the RLM's internal reasoning transcript) is owned by the L2
engine and is intentionally not modelled here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragu.core import Document, DocumentId


@runtime_checkable
class PreferenceStore(Protocol):
    """Long-term per-user preferences, surfaced into prompt construction."""

    async def get(self, user_id: str) -> list[str]:
        """Return preference statements relevant to this user."""
        ...

    async def remember(self, user_id: str, messages: list[str]) -> None:
        """Extract and persist any durable preferences from these messages."""
        ...


@runtime_checkable
class SessionMemory(Protocol):
    """A bytes-aware, per-session LRU of parsed documents.

    Eviction is by total bytes, not entry count, because one large document can
    dwarf hundreds of small ones. ``touch`` records access for both warming and
    recency; ``warm_ids`` lets L1 boost documents already in play this session.
    """

    async def touch(self, session_id: str, documents: list[Document]) -> None:
        """Insert/refresh documents as most-recently-used for the session."""
        ...

    async def get(self, session_id: str, doc_ids: list[DocumentId]) -> list[Document]:
        """Return cached documents (subset that is present), MRU-refreshed."""
        ...

    async def warm_ids(self, session_id: str) -> list[DocumentId]:
        """Ids currently warm for the session, for L1 score boosting."""
        ...
