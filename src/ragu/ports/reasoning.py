"""Reasoning contract — the L2 boundary.

This is where vomero (or any other RLM) plugs in. RAGu never imports vomero
directly; it depends only on this Protocol. The adapter is responsible for
translating a ``WorkingSet`` into whatever input form the engine wants (corpus
folder, in-memory context, ...) and for translating the engine's output back
into an ``Answer`` with citations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragu.core import Answer, Query, WorkingSet


@runtime_checkable
class ReasoningEngine(Protocol):
    """Answers a query by agentically exploring a bounded working set."""

    async def reason(self, query: Query, working_set: WorkingSet) -> Answer:
        ...
