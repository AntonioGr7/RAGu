"""PDF loading — text layer, OCR, or a smart mix.

Three modes, chosen by the corpus:

* ``auto``  — use the embedded text layer, OCR only text-poor pages. Fast and
  accurate for born-digital PDFs.
* ``ocr``   — OCR every page, ignoring the text layer. Best when the embedded
  text is a bad scan-OCR (garbled headers, wrong characters) and our SOTA OCR
  does better.
* ``text``  — text layer only, never OCR.

A page is OCR-ed only when an ``OcrEngine`` is available; ``ocr`` mode without an
engine degrades to ``auto`` with a warning rather than failing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import groupby
from pathlib import Path

from ragu.adapters.ingestion.files import make_document
from ragu.adapters.ingestion.ocr import OcrArtifact, OcrEngine, OcrLine, OcrPage, OcrWord
from ragu.core import Document

logger = logging.getLogger(__name__)


def load_pdf_file(
    file: Path,
    base: Path,
    *,
    ocr: OcrEngine | None = None,
    mode: str = "auto",
    dpi: int = 200,
    min_text_chars: int = 16,
    on_page: Callable[[int, int], None] | None = None,
) -> Document:
    """Load a PDF per ``mode``, returning one Document for the whole file."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PDF ingestion requires PyMuPDF: pip install 'ragu[ocr]' (or pymupdf)"
        ) from exc

    if mode == "ocr" and ocr is None:
        logger.warning(
            "pdf_mode='ocr' but OCR is disabled; using text layer for %s", file
        )
        mode = "text"

    ocr_results: list[OcrPage] = []
    ocr_pages = 0
    with fitz.open(file) as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        total_pages = doc.page_count
        for i, page in enumerate(doc):
            if on_page is not None:
                on_page(i + 1, total_pages)
            text = page.get_text().strip()
            ocr_this = ocr is not None and (
                mode == "ocr" or (mode == "auto" and len(text) < min_text_chars)
            )
            if ocr_this:
                pixmap = page.get_pixmap(matrix=matrix)
                ocr_page = ocr.read_page(pixmap.tobytes("png"), index=i)  # type: ignore[union-attr]
                ocr_results.append(ocr_page)
                ocr_pages += 1
            else:
                # Text-layer page: same structured shape as an OCR page (lines +
                # word boxes from PyMuPDF), so downstream consumers treat both
                # uniformly and only read ``page.source`` to know the provenance.
                ocr_results.append(_text_layer_page(page, index=i, zoom=zoom, text=text))

    content = "\n\n".join(f"[page {p.index + 1}]\n{p.text}" for p in ocr_results)
    if ocr_pages == 0:
        loader = "pdf-text"
    elif ocr_pages == len(ocr_results):
        loader = "pdf-ocr"
    else:
        loader = f"pdf-mixed:{ocr_pages}-ocr"

    engine_name = ocr.name if ocr is not None else "text-layer"
    artifact = OcrArtifact(engine=engine_name, pages=ocr_results)
    return make_document(
        file,
        base,
        content=content,
        extra={"loader": loader, "pages": str(len(ocr_results))},
        artifacts={"ocr": artifact.model_dump()},
    )


def _text_layer_page(page, *, index: int, zoom: float, text: str) -> OcrPage:
    """Build a structured ``OcrPage`` from a PDF's embedded text layer.

    Mirrors the OCR page shape — words (from ``page.get_text("words")``) grouped
    into lines, each with an axis-aligned box. Coordinates are scaled by ``zoom``
    so they share the same DPI pixel space as OCR pages of the same document (a
    mixed PDF then has one coordinate convention throughout). ``score`` is 1.0:
    text-layer characters are exact, not recognized."""

    def scale(v: float) -> int:
        return int(round(v * zoom))

    # Each word: (x0, y0, x1, y1, "word", block_no, line_no, word_no).
    words = [w for w in page.get_text("words") if w[4].strip()]
    words.sort(key=lambda w: (w[5], w[6], w[7]))

    lines: list[OcrLine] = []
    for _key, group in groupby(words, key=lambda w: (w[5], w[6])):
        g = list(group)
        ocr_words = [
            OcrWord(text=w[4], box=(scale(w[0]), scale(w[1]), scale(w[2]), scale(w[3])))
            for w in g
        ]
        box = (
            scale(min(w[0] for w in g)),
            scale(min(w[1] for w in g)),
            scale(max(w[2] for w in g)),
            scale(max(w[3] for w in g)),
        )
        poly = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
        lines.append(
            OcrLine(
                text=" ".join(w[4] for w in g),
                score=1.0,
                poly=poly,
                box=box,
                words=ocr_words,
            )
        )

    rect = page.rect
    return OcrPage(
        index=index,
        source="text-layer",
        text=text,  # plain extracted text preserves reading order for `content`
        width=scale(rect.width),
        height=scale(rect.height),
        lines=lines,
    )
