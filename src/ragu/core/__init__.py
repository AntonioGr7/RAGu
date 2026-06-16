"""Pure domain layer: data types with no infrastructure dependencies.

Nothing in this package may import an adapter, a network client, or a
configuration object. It is the vocabulary the rest of the system speaks.
"""

from ragu.core.models import (
    Answer,
    Chunk,
    Citation,
    Document,
    DocumentId,
    Query,
    RetrievalResult,
    ScoredChunk,
    ScoredDocument,
    WorkingSet,
)

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "DocumentId",
    "Query",
    "RetrievalResult",
    "ScoredChunk",
    "ScoredDocument",
    "WorkingSet",
]
