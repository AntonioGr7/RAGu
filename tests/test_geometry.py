"""Char-span -> word-box resolution off an OCR artifact."""

from ragu.adapters.ingestion import build_word_layout, highlights_for_span
from ragu.adapters.ingestion.ocr import OcrArtifact, OcrLine, OcrPage, OcrWord


def _word(text: str, box: tuple[int, int, int, int]) -> OcrWord:
    return OcrWord(text=text, box=box)


def _line(words: list[OcrWord]) -> OcrLine:
    x0 = min(w.box[0] for w in words)
    y0 = min(w.box[1] for w in words)
    x1 = max(w.box[2] for w in words)
    y1 = max(w.box[3] for w in words)
    return OcrLine(
        text=" ".join(w.text for w in words),
        score=1.0,
        poly=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        box=(x0, y0, x1, y1),
        words=words,
    )


def _artifact() -> OcrArtifact:
    page0 = OcrPage(
        index=0,
        source="text-layer",
        text="Tasso variabile\nEuribor 6 mesi",
        width=1000,
        height=1400,
        lines=[
            _line([_word("Tasso", (10, 10, 90, 30)), _word("variabile", (100, 10, 250, 30))]),
            _line([_word("Euribor", (10, 50, 110, 70)), _word("6", (120, 50, 140, 70)),
                   _word("mesi", (150, 50, 210, 70))]),
        ],
    )
    page1 = OcrPage(
        index=1,
        source="text-layer",
        text="spread 1,20",
        width=1000,
        height=1400,
        lines=[_line([_word("spread", (10, 10, 100, 30)), _word("1,20", (110, 10, 190, 30))])],
    )
    return OcrArtifact(engine="text-layer", pages=[page0, page1])


def test_layout_text_is_word_aligned() -> None:
    layout = build_word_layout(_artifact())
    assert layout.text.startswith("[page 1]\n")
    assert "[page 2]\n" in layout.text
    # Every recorded word span slices back to exactly that word's text.
    for w in layout.words:
        assert layout.text[w.char_start : w.char_end] in {
            "Tasso", "variabile", "Euribor", "6", "mesi", "spread", "1,20"
        }


def test_single_word_span_maps_to_its_box() -> None:
    layout = build_word_layout(_artifact())
    start = layout.text.index("variabile")
    hits = highlights_for_span(layout, start, start + len("variabile"))
    assert len(hits) == 1
    h = hits[0]
    assert h.page == 0 and h.width == 1000 and h.height == 1400
    assert h.boxes == ((100, 10, 250, 30),)


def test_multiword_span_merges_to_one_line_box() -> None:
    layout = build_word_layout(_artifact())
    start = layout.text.index("Tasso")
    end = layout.text.index("variabile") + len("variabile")
    hits = highlights_for_span(layout, start, end)
    assert len(hits) == 1
    # Two words on one line collapse to a single covering rectangle.
    assert hits[0].boxes == ((10, 10, 250, 30),)


def test_span_across_lines_yields_one_box_per_line() -> None:
    layout = build_word_layout(_artifact())
    start = layout.text.index("variabile")
    end = layout.text.index("Euribor") + len("Euribor")
    hits = highlights_for_span(layout, start, end)
    assert len(hits) == 1  # same page
    assert hits[0].boxes == ((100, 10, 250, 30), (10, 50, 110, 70))


def test_span_across_pages_yields_one_highlight_per_page() -> None:
    layout = build_word_layout(_artifact())
    start = layout.text.index("mesi")
    end = layout.text.index("spread") + len("spread")
    hits = highlights_for_span(layout, start, end)
    assert [h.page for h in hits] == [0, 1]
    assert hits[0].boxes == ((150, 50, 210, 70),)
    assert hits[1].boxes == ((10, 10, 100, 30),)


def test_span_on_page_marker_only_has_no_boxes() -> None:
    layout = build_word_layout(_artifact())
    # The "[page 1]\n" prefix carries no words.
    hits = highlights_for_span(layout, 0, len("[page 1]"))
    assert hits == []
