"""Storage backends. In-memory for tests/small corpora; LanceDB for real use.

The LanceDB classes import ``lancedb``/``pyarrow`` lazily (inside methods), so
importing this package never requires those packages to be installed.
"""

from ragu.adapters.storage.lance import LanceDocumentStore, LanceVectorStore
from ragu.adapters.storage.memory import InMemoryDocumentStore, InMemoryVectorStore

__all__ = [
    "InMemoryDocumentStore",
    "InMemoryVectorStore",
    "LanceDocumentStore",
    "LanceVectorStore",
]
