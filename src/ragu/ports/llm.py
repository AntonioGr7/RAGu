"""Generic chat-model contract.

Every LLM call in RAGu (contextual-retrieval blurbs, query routing, reasoning
glue, ...) goes through this port — never a vendor SDK directly — so the
provider is pure configuration. Adapters cover OpenAI-compatible servers
(OpenAI, self-hosted, Anthropic's compat endpoint) and Gemini.

Provider-specific capabilities (prompt caching, structured output modes) live
inside adapters; the port stays minimal so any provider can satisfy it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class ChatModel(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant's text reply for the given conversation."""
        ...
