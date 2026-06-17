"""Run the RAGu web demo: ``python -m ragu.web`` (or with --host/--port).

Loads .env (provider keys), builds the FastAPI app, and serves it with uvicorn.
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the RAGu showcase frontend + API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload (dev)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers", "transformers", "paddle", "paddlex"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()

    import uvicorn

    # `reload`/string-import path lets uvicorn re-import the app on file changes.
    uvicorn.run(
        "ragu.web.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
