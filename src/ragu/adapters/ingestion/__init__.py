"""Ingestion: turn raw sources into ``Document`` objects.

``load_paths`` walks files/folders and dispatches by extension: text files are
read directly; images and PDFs are routed through an injected ``OcrEngine``
(PP-OCRv6 by default). Files needing OCR are skipped when no engine is supplied.
Richer parsing (tables, layout) plugs in here as further loaders.
"""

from __future__ import annotations

from pathlib import Path

from ragu.adapters.ingestion.files import (
    TEXT_SUFFIXES,
    iter_files,
    load_text_file,
)
from ragu.adapters.ingestion.ocr import (
    OCR_SUFFIXES,
    OcrEngine,
    PaddleOcrEngine,
    load_ocr_file,
)
from ragu.core import Document

__all__ = [
    "OcrEngine",
    "PaddleOcrEngine",
    "dump_documents",
    "load_paths",
]


def load_paths(
    paths: list[str | Path],
    *,
    ocr: OcrEngine | None = None,
) -> list[Document]:
    """Load every supported file under the given files/directories."""
    documents: list[Document] = []
    for file, base in iter_files(paths):
        suffix = file.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            documents.append(load_text_file(file, base))
        elif suffix in OCR_SUFFIXES and ocr is not None:
            documents.append(load_ocr_file(file, base, ocr))
        # other suffixes (or OCR files without an engine) are skipped
    return documents


def dump_documents(documents: list[Document], out_dir: str | Path) -> int:
    """Write each document's extracted text under ``out_dir``, mirroring the
    original folder structure (``rel_path`` + ``.txt``). For inspecting what
    ingestion/OCR produced. Returns the number of files written.
    """
    out = Path(out_dir)
    for doc in documents:
        rel = doc.metadata.get("rel_path", str(doc.id))
        target = out / f"{rel}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.content, encoding="utf-8")
    return len(documents)
