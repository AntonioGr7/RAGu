"""Debug helper: render cited source pages with their highlight boxes drawn.

Given citations that carry page + word boxes (produced by the grounding pass),
this renders each referenced page of the source PDF/image and draws the boxes on
it, so you can eyeball whether a citation actually points where it claims. Boxes
are stored in each page's own pixel space (``Highlight.width``/``height``); the
rendered page is scaled to match, so the overlay lines up at any render DPI.

Not part of the normal pipeline — call it explicitly (or via ``--cite-images``)
when debugging. Requires pymupdf + pillow (the ``ocr`` extra).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from ragu.core import Citation, Highlight

logger = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}


def save_citation_overlays(
    citations: list[Citation],
    out_dir: str | Path,
    *,
    dpi: int = 200,
    color: tuple[int, int, int] = (220, 30, 30),
    line_width: int = 3,
) -> list[Path]:
    """Render every cited page with its highlight boxes drawn, one PNG per page.

    Boxes from all citations that land on the same source page are drawn together.
    Returns the saved image paths (empty if no citation carries highlights)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_page: dict[tuple[str, int], list[Highlight]] = defaultdict(list)
    for citation in citations:
        for highlight in citation.highlights:
            by_page[(citation.source, highlight.page)].append(highlight)

    if not by_page:
        logger.warning("No highlights to render — citations carry no boxes.")
        return []

    saved: list[Path] = []
    for (source, page), highlights in sorted(by_page.items()):
        image = _render_page(source, page, dpi)
        if image is None:
            continue
        _draw_boxes(image, highlights, color, line_width)
        path = out / f"{Path(source).stem}_p{page}.png"
        image.save(path)
        boxes = sum(len(h.boxes) for h in highlights)
        logger.info("Saved %s (%d box(es))", path, boxes)
        saved.append(path)
    return saved


def render_page(source: str, page_index: int, dpi: int = 200):  # type: ignore[no-untyped-def]
    """Render one page of a PDF (or an image file) to a PIL RGB image (or ``None``
    if the source is missing / the page is out of range). Public entry point used
    by the debug overlay above and by the web viewer (which draws its own boxes)."""
    return _render_page(source, page_index, dpi)


def _render_page(source: str, page_index: int, dpi: int):  # type: ignore[no-untyped-def]
    """Render one page of a PDF (or an image file) to a PIL RGB image."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("citation overlays need pillow: pip install 'ragu[ocr]'") from exc

    path = Path(source)
    if not path.exists():
        logger.warning("Source not found, skipping: %s", source)
        return None

    if path.suffix.lower() in PDF_SUFFIXES:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise ImportError("rendering PDF pages needs pymupdf: pip install 'ragu[ocr]'") from exc
        with fitz.open(source) as doc:
            if not 0 <= page_index < doc.page_count:
                logger.warning("Page %d out of range for %s", page_index, source)
                return None
            pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    if page_index != 0:
        logger.warning("Image source %s has only page 0 (asked for %d)", source, page_index)
        return None
    return Image.open(source).convert("RGB")


def _draw_boxes(image, highlights: list[Highlight], color, line_width: int) -> None:  # type: ignore[no-untyped-def]
    """Draw each highlight's boxes, scaling from the stored page dims to the
    rendered image size so the overlay aligns regardless of render DPI."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for highlight in highlights:
        scale_x = image.width / highlight.width if highlight.width else 1.0
        scale_y = image.height / highlight.height if highlight.height else 1.0
        for x1, y1, x2, y2 in highlight.boxes:
            draw.rectangle(
                [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                outline=color,
                width=line_width,
            )
