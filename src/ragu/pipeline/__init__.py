"""Orchestration: composes ports into the indexing and query flows.

The pipeline is the only layer that knows the *order* of operations; every step
it calls is a port, so the wiring stays backend-agnostic.
"""

from ragu.pipeline.indexer import Indexer
from ragu.pipeline.working_set import build_working_set

__all__ = ["Indexer", "build_working_set"]
