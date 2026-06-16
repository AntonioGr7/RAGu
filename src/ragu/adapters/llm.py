"""Chat-model adapters.

* ``OpenAICompatChat`` — talks to any OpenAI-compatible endpoint. Set
  ``base_url`` to reach a self-hosted server or Anthropic's compat endpoint;
  leave it unset for OpenAI proper.
* ``GeminiChat`` — Google Gemini, via the optional ``gemini`` extra.
* ``ScriptedChat`` — deterministic, offline; returns a fixed reply or echoes.
  For tests and dry runs.

SDKs are imported lazily so only the provider you actually use needs to be
installed.
"""

from __future__ import annotations

from ragu.ports.llm import ChatMessage


class OpenAICompatChat:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        # api_key=None lets the SDK fall back to OPENAI_API_KEY; some local
        # servers ignore it, so pass a placeholder when none is configured.
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model

    async def complete(
        self, messages: list[ChatMessage], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return (resp.choices[0].message.content or "").strip()


class GeminiChat:
    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GeminiChat requires the 'gemini' extra: pip install 'ragu[gemini]'"
            ) from exc
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(
        self, messages: list[ChatMessage], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        from google.genai import types

        system = "\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(role="user" if m.role != "assistant" else "model", parts=[
                types.Part(text=m.content)
            ])
            for m in messages
            if m.role != "system"
        ]
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return (resp.text or "").strip()


class ScriptedChat:
    """Offline chat model: returns ``reply`` for every call (or echoes the last
    user message when ``reply`` is None). Used by tests and offline pipelines."""

    def __init__(self, reply: str | None = None) -> None:
        self._reply = reply

    async def complete(
        self, messages: list[ChatMessage], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        if self._reply is not None:
            return self._reply
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return last_user
