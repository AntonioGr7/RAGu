"""Ingestion: turn raw sources into ``Document`` objects.

``load_paths`` walks files/folders and dispatches by extension:

* text files (.md/.txt/...) → read directly;
* PDFs → text layer extracted directly, OCR-ing only text-poor pages (so PDFs
  load even without OCR enabled);
* images → require an ``OcrEngine``; skipped (with a warning) when none is given.

Richer parsing (tables, layout) plugs in here as further loaders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ragu.adapters.ingestion.files import (
    TEXT_SUFFIXES,
    file_fingerprint,
    iter_files,
    load_text_file,
)
from ragu.adapters.ingestion.ocr import (
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    OcrArtifact,
    OcrEngine,
    OcrLine,
    OcrPage,
    OcrWord,
    PaddleOcrEngine,
    load_ocr_file,
)
from ragu.adapters.ingestion.geometry import (
    WordLayout,
    WordSpan,
    build_word_layout,
    highlights_for_span,
)
from ragu.adapters.ingestion.pdf import load_pdf_file
from ragu.core import Document

logger = logging.getLogger(__name__)

__all__ = [
    "OcrArtifact",
    "OcrEngine",
    "OcrLine",
    "OcrPage",
    "OcrWord",
    "PaddleOcrEngine",
    "WordLayout",
    "WordSpan",
    "build_word_layout",
    "dump_documents",
    "file_fingerprint",
    "highlights_for_span",
    "iter_files",
    "load_files",
    "load_one",
    "load_paths",
]


def load_one(
    file: Path,
    base: Path,
    *,
    ocr: OcrEngine | None = None,
    pdf_mode: str = "auto",
    pdf_dpi: int = 200,
    pdf_min_text_chars: int = 16,
    content_hash: str | None = None,
    on_page: Callable[[int, int], None] | None = None,
) -> Document | None:
    """Load a single file into a ``Document``, dispatching by extension.

    Returns ``None`` (with a WARNING) for unsupported types and for images when
    no OCR engine is given, so a "0 files" result is never silent. The document's
    ``metadata['content_hash']`` is stamped with ``content_hash`` (computed from
    the file bytes when not supplied) so re-indexing can detect changes."""
    suffix = file.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        doc = load_text_file(file, base)
    elif suffix in PDF_SUFFIXES:
        doc = load_pdf_file(
            file, base, ocr=ocr, mode=pdf_mode, dpi=pdf_dpi,
            min_text_chars=pdf_min_text_chars, on_page=on_page,
        )
    elif suffix in IMAGE_SUFFIXES:
        if ocr is None:
            logger.warning("Skipping image (OCR disabled): %s", file)
            return None
        doc = load_ocr_file(file, base, ocr)
    else:
        logger.warning("Skipping unsupported file type %s: %s", suffix, file)
        return None

    if content_hash is None:
        content_hash = file_fingerprint(file)
    return doc.model_copy(
        update={"metadata": {**doc.metadata, "content_hash": content_hash}}
    )


def load_files(
    items: list[tuple[Path, Path, str | None]],
    *,
    ocr: OcrEngine | None = None,
    pdf_mode: str = "auto",
    pdf_dpi: int = 200,
    pdf_min_text_chars: int = 16,
    progress: bool = False,
) -> list[Document]:
    """Load explicit ``(file, base, content_hash)`` triples into Documents.

    The ``base`` is carried per item so a file's id stays stable whether it was
    reached via a directory walk or named directly. ``content_hash`` may be a
    precomputed fingerprint (to avoid re-reading) or ``None`` to compute lazily.
    With ``progress=True`` a per-file ``tqdm`` bar is shown (per-page for PDFs),
    since OCR-ing a large scan can otherwise sit for minutes with no output."""
    documents: list[Document] = []
    bar = _progress_bar(items, enabled=progress)
    for file, base, content_hash in items:
        bar.set_description(file.name[:40])
        doc = load_one(
            file, base, ocr=ocr, pdf_mode=pdf_mode, pdf_dpi=pdf_dpi,
            pdf_min_text_chars=pdf_min_text_chars, content_hash=content_hash,
            on_page=_page_reporter(bar),
        )
        if doc is not None:
            documents.append(doc)
        bar.update(1)
    bar.close()
    return documents


def load_paths(
    paths: list[str | Path],
    *,
    ocr: OcrEngine | None = None,
    pdf_mode: str = "auto",
    pdf_dpi: int = 200,
    pdf_min_text_chars: int = 16,
    progress: bool = False,
) -> list[Document]:
    """Load every supported file under the given files/directories.

    A convenience wrapper over :func:`load_files` that walks ``paths`` and lets
    each document's fingerprint be computed lazily. Incremental indexing uses
    :func:`load_files` directly so it can skip unchanged files before loading."""
    items: list[tuple[Path, Path, str | None]] = [
        (file, base, None) for file, base in iter_files(paths)
    ]
    return load_files(
        items, ocr=ocr, pdf_mode=pdf_mode, pdf_dpi=pdf_dpi,
        pdf_min_text_chars=pdf_min_text_chars, progress=progress,
    )


class _NullBar:
    """No-op stand-in for tqdm when progress is disabled or tqdm is missing."""

    def set_description(self, _desc: str) -> None: ...
    def set_postfix_str(self, _s: str) -> None: ...
    def update(self, _n: int = 1) -> None: ...
    def close(self) -> None: ...


def _progress_bar(items: list, *, enabled: bool):
    """A tqdm bar sized to ``items``, or a no-op bar when disabled/unavailable."""
    if not enabled:
        return _NullBar()
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm ships with the 'ocr' extra
        return _NullBar()
    return tqdm(total=len(items), unit="file")


def _page_reporter(bar) -> Callable[[int, int], None]:
    """Show per-page progress for a PDF as a postfix on the file bar, so a large
    scan reports ``page 3/40`` rather than appearing frozen on one file."""
    def report(done: int, total: int) -> None:
        bar.set_postfix_str(f"page {done}/{total}")
    return report


def dump_documents(documents: list[Document], out_dir: str | Path) -> int:
    """Write each document's extracted text under ``out_dir``, mirroring the
    original folder structure (``rel_path`` + ``.txt``). When a document carries
    structured OCR (``artifacts['ocr']``), also write a ``.ocr.json`` sidecar
    with the full geometry (lines, word boxes, scores). Returns files written.
    """
    out = Path(out_dir)
    for doc in documents:
        rel = doc.metadata.get("rel_path", str(doc.id))
        target = out / f"{rel}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.content, encoding="utf-8")
        ocr = doc.artifacts.get("ocr")
        if ocr is not None:
            sidecar = out / f"{rel}.ocr.json"
            sidecar.write_text(json.dumps(ocr, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(documents)
