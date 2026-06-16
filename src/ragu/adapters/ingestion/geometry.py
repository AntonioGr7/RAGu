"""Map character spans in a document back to word-level boxes on its pages.

A document's OCR/text-layer artifact carries, per page, the recognized lines and
their per-word boxes (see :mod:`ragu.adapters.ingestion.ocr`). This module builds
a *canonical* text from those words — with a char-offset for every word — so a
span found in that text can be resolved to the boxes that cover it. The canonical
text is rebuilt on demand from the artifact; it is not stored and does not affect
``Document.content`` (what retrieval and L2 see).

The flow a caller uses (the citation grounding pass):

    layout = build_word_layout(doc.artifacts["ocr"])
    start = layout.text.find(quote)              # or fuzzy-locate the quote
    highlights = highlights_for_span(layout, start, start + len(quote))

``highlights_for_span`` returns one line-level box per line the span touches,
grouped by page — the rectangles you'd draw over the page rendered at its DPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragu.adapters.ingestion.ocr import OcrArtifact
from ragu.core import Box, Highlight


@dataclass(frozen=True)
class WordSpan:
    """One word's place in the canonical text and on its page."""

    page_index: int
    line_index: int
    char_start: int
    char_end: int
    box: Box


@dataclass
class WordLayout:
    """The canonical text of a document plus the word→box index into it."""

    text: str
    words: list[WordSpan] = field(default_factory=list)
    # page index -> (width, height) in the page's own pixel space
    page_dims: dict[int, tuple[int, int]] = field(default_factory=dict)


def build_word_layout(artifact: OcrArtifact | dict) -> WordLayout:
    """Build the canonical text + word index for a document's OCR artifact.

    Pages are joined by a blank line and prefixed with ``[page N]`` (mirroring
    ``Document.content``); within a page, words are space-joined and lines
    newline-joined, with every word's char span recorded against its box. A line
    that carries text but no per-word boxes falls back to a single span mapped to
    the whole-line box, so its text is still locatable (just coarser)."""
    art = artifact if isinstance(artifact, OcrArtifact) else OcrArtifact.model_validate(artifact)

    parts: list[str] = []
    words: list[WordSpan] = []
    page_dims: dict[int, tuple[int, int]] = {}
    cursor = 0

    def emit(text: str) -> int:
        nonlocal cursor
        parts.append(text)
        start = cursor
        cursor += len(text)
        return start

    for page_pos, page in enumerate(art.pages):
        if page_pos > 0:
            emit("\n\n")
        page_dims[page.index] = (page.width or 0, page.height or 0)
        emit(f"[page {page.index + 1}]\n")
        for line_index, line in enumerate(page.lines):
            if line_index > 0:
                emit("\n")
            if line.words:
                for word_pos, word in enumerate(line.words):
                    if word_pos > 0:
                        emit(" ")
                    start = emit(word.text)
                    words.append(
                        WordSpan(page.index, line_index, start, cursor, tuple(word.box))
                    )
            elif line.text:
                start = emit(line.text)
                words.append(
                    WordSpan(page.index, line_index, start, cursor, tuple(line.box))
                )

    return WordLayout(text="".join(parts), words=words, page_dims=page_dims)


def highlights_for_span(layout: WordLayout, start: int, end: int) -> list[Highlight]:
    """Resolve a ``[start, end)`` char span in ``layout.text`` to per-page
    highlights — one line-level box per line the span overlaps, in reading order.
    Returns an empty list if the span covers no words (e.g. it falls entirely on
    page markers or whitespace)."""
    if end <= start:
        return []

    # Group overlapping words by (page, line), preserving first-seen order.
    groups: dict[tuple[int, int], list[WordSpan]] = {}
    order: list[tuple[int, int]] = []
    for w in layout.words:
        if w.char_start < end and w.char_end > start:  # half-open overlap
            key = (w.page_index, w.line_index)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(w)

    # One merged box per line, collected per page (page order preserved).
    by_page: dict[int, list[Box]] = {}
    page_order: list[int] = []
    for key in order:
        ws = groups[key]
        line_box: Box = (
            min(w.box[0] for w in ws),
            min(w.box[1] for w in ws),
            max(w.box[2] for w in ws),
            max(w.box[3] for w in ws),
        )
        page_index = key[0]
        if page_index not in by_page:
            by_page[page_index] = []
            page_order.append(page_index)
        by_page[page_index].append(line_box)

    highlights: list[Highlight] = []
    for page_index in page_order:
        width, height = layout.page_dims.get(page_index, (0, 0))
        highlights.append(
            Highlight(
                page=page_index,
                boxes=tuple(by_page[page_index]),
                width=width or None,
                height=height or None,
            )
        )
    return highlights
