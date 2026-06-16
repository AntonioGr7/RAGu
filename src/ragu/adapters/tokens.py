"""Token counting via tiktoken.

For Claude this is an *estimate* (tiktoken encodes for OpenAI models), which is
fine for budgeting the working set — we want a stable, fast proxy for text
volume, not exact billing. ``o200k_base`` is a reasonable modern default.
"""

from __future__ import annotations

import tiktoken


class TiktokenCounter:
    def __init__(self, encoding: str = "o200k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))
