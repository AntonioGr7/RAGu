"""Token-counting contract.

Used to bound the working set by tokens rather than document count. Sync because
it is pure CPU work with no I/O.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        ...
