"""Command-line entry point.

    ragu index ./docs ./more_docs     # ingest + index files/folders
    ragu retrieve "what blocks X?"     # L1: show the selected working set

Configuration comes from the environment / ``.env`` (see ``RaguSettings``), so
the CLI stays thin — it only parses arguments and drives the ``Ragu`` facade.
"""

from __future__ import annotations

import argparse
import asyncio

from ragu.app import Ragu


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragu", description="Two-level RAG system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="ingest and index files or folders")
    p_index.add_argument("paths", nargs="+", help="files or directories to index")

    p_retrieve = sub.add_parser("retrieve", help="L1 retrieve a working set for a query")
    p_retrieve.add_argument("query", help="the query text")

    p_extract = sub.add_parser(
        "extract", help="extract text (incl. OCR) from a folder into another folder"
    )
    p_extract.add_argument("paths", nargs="+", help="input files or directories")
    p_extract.add_argument("-o", "--out", required=True, help="output directory")

    args = parser.parse_args(argv)

    if args.command == "index":
        n = asyncio.run(Ragu().index_paths(args.paths))
        print(f"Indexed {n} chunks from {len(args.paths)} path(s).")
        return 0

    if args.command == "retrieve":
        ws = asyncio.run(Ragu().retrieve(args.query))
        print(f"Working set: {len(ws.documents)} docs, {ws.token_count} tokens "
              f"(truncated={ws.truncated})")
        for doc in ws.documents:
            print(f"  - {doc.id}  [{doc.source}]")
        return 0

    if args.command == "extract":
        # Extraction needs only ingestion + OCR — skip the full facade (no
        # embedder / vector store / LLM are constructed).
        from ragu.adapters.ingestion import dump_documents, load_paths
        from ragu.config import RaguSettings
        from ragu.factory import build_ocr_engine

        ocr = build_ocr_engine(RaguSettings())
        documents = load_paths(args.paths, ocr=ocr)
        n = dump_documents(documents, args.out)
        print(f"Extracted {n} file(s) into {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
