"""Command-line entry point.

    ragu index ./docs ./more_docs     # ingest + index files/folders
    ragu retrieve "what blocks X?"     # L1: show the selected working set

Configuration comes from the environment / ``.env`` (see ``RaguSettings``), so
the CLI stays thin — it only parses arguments and drives the ``Ragu`` facade.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ragu.app import Ragu


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Chatty third-party libraries log every model file fetch / HTTP request at
    # INFO, which drowns out RAGu's own output. Lift them to WARNING.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers",
                  "huggingface_hub", "paddlex", "paddle", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Load .env into the process environment so provider SDKs (Gemini/OpenAI)
    # see their API keys. pydantic-settings reads .env for RAGU_* but does not
    # export non-prefixed keys to os.environ.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is a normal dependency
        pass
    parser = argparse.ArgumentParser(prog="ragu", description="Two-level RAG system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="ingest and index files or folders")
    p_index.add_argument("paths", nargs="+", help="files or directories to index")
    p_index.add_argument(
        "--no-prune",
        action="store_true",
        help="keep documents whose source file has been deleted (default: prune them)",
    )

    p_retrieve = sub.add_parser("retrieve", help="L1 retrieve a working set for a query")
    p_retrieve.add_argument("query", help="the query text")

    p_answer = sub.add_parser("answer", help="L1+L2: reason over the working set (needs vomero)")
    p_answer.add_argument("query", help="the query text")
    p_answer.add_argument(
        "--cite",
        action="store_true",
        help="ground the answer: inline citations with quotes + page/word boxes (extra LLM call)",
    )
    p_answer.add_argument(
        "--cite-source",
        choices=["trajectory", "document", "raw"],
        default=None,
        help="grounding evidence: 'trajectory' (LLM over what L2 read, default), "
        "'document' (LLM over the whole document), or 'raw' (no extra LLM call). "
        "Implies --cite.",
    )

    p_extract = sub.add_parser(
        "extract", help="extract text (incl. OCR) from a folder into another folder"
    )
    p_extract.add_argument("paths", nargs="+", help="input files or directories")
    p_extract.add_argument("-o", "--out", required=True, help="output directory")

    sub.add_parser("list", help="list every indexed document")

    p_show = sub.add_parser("show", help="show one indexed document's content + metadata")
    p_show.add_argument("doc_id", help="the document id (as shown by `ragu list`)")
    p_show.add_argument(
        "--json", action="store_true", help="emit the full document (incl. artifacts) as JSON"
    )

    args = parser.parse_args(argv)

    if args.command == "index":
        report = asyncio.run(
            Ragu().index_paths(args.paths, prune=not args.no_prune, progress=True)
        )
        print(
            f"Indexed {report.chunks} chunks "
            f"(new={report.new}, updated={report.updated}, "
            f"skipped={report.skipped}, pruned={report.pruned})."
        )
        return 0

    if args.command == "retrieve":
        ws = asyncio.run(Ragu().retrieve(args.query))
        print(f"Working set: {len(ws.documents)} docs, {ws.token_count} tokens "
              f"(truncated={ws.truncated})")
        for doc in ws.documents:
            print(f"  - {doc.id}  [{doc.source}]")
        return 0

    if args.command == "answer":
        ground = args.cite or args.cite_source is not None or None
        answer = asyncio.run(
            Ragu().answer(args.query, ground=ground, grounding_source=args.cite_source)
        )
        print(answer.text)
        if answer.citations:
            print("\nSources:")
            for c in answer.citations:
                print(f"  - {c.doc_id}  [{c.source}]")
                if c.quote:
                    print(f"      “{c.quote.strip()[:160]}”")
                for h in c.highlights:
                    print(f"      page {h.page}: {len(h.boxes)} box(es) {h.boxes}")
        print(f"\n(trace: {answer.trace})")
        return 0

    if args.command == "list":
        refs = asyncio.run(Ragu().list_documents())
        for ref in sorted(refs, key=lambda r: str(r.id)):
            print(f"  - {ref.id}  [{ref.source}]")
        print(f"{len(refs)} document(s).")
        return 0

    if args.command == "show":
        doc = asyncio.run(Ragu().get_document(args.doc_id))
        if doc is None:
            print(f"No document with id '{args.doc_id}'.")
            return 1
        if args.json:
            import json

            print(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2))
        else:
            print(f"id:     {doc.id}")
            print(f"source: {doc.source}")
            if doc.metadata:
                print("metadata:")
                for key, val in doc.metadata.items():
                    print(f"  {key}: {val}")
            if doc.artifacts:
                print(f"artifacts: {', '.join(doc.artifacts)} (use --json for full detail)")
            print("\n--- content ---")
            print(doc.content)
        return 0

    if args.command == "extract":
        # Extraction needs only ingestion + OCR — skip the full facade (no
        # embedder / vector store / LLM are constructed).
        from ragu.adapters.ingestion import dump_documents, load_paths
        from ragu.config import RaguSettings
        from ragu.factory import build_ocr_engine

        settings = RaguSettings()
        documents = load_paths(
            args.paths,
            ocr=build_ocr_engine(settings),
            pdf_mode=settings.ocr.pdf_mode,
            pdf_dpi=settings.ocr.pdf_dpi,
            pdf_min_text_chars=settings.ocr.pdf_min_text_chars,
            progress=True,
        )
        n = dump_documents(documents, args.out)
        print(f"Extracted {n} file(s) into {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
