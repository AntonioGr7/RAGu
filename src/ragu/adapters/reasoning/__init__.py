"""L2 reasoning adapters. Currently vomero (the RLM); isolated here so the rest
of RAGu depends only on the ``ReasoningEngine`` port, never on vomero."""

from ragu.adapters.reasoning.vomero import VomeroReasoningEngine

__all__ = ["VomeroReasoningEngine"]
