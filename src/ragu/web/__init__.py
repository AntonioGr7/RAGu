"""Web demo layer: a small FastAPI app over the :class:`ragu.app.Ragu` facade.

Exposes just enough to drive the TypeScript frontend — run a grounded query and
render a source page — without leaking the pipeline internals. Optional: install
with ``pip install 'ragu[web]'`` (and ``[ocr]`` for page rendering)."""

from ragu.web.server import create_app

__all__ = ["create_app"]
