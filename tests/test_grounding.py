"""Locate quotes and ground an answer into span-level citations."""

import pytest

from ragu.adapters.ingestion.ocr import OcrArtifact, OcrLine, OcrPage, OcrWord
from ragu.core import Answer, Citation, Document, DocumentId, EvidenceSpan, WorkingSet
from ragu.pipeline.grounding import (
    _parse_quotes,
    ground_answer,
    locate,
)

# ── locate() ────────────────────────────────────────────────────────────────
SRC = "Il tasso\nvariabile è\nindicizzato all'Euribor a 6 mesi."


def test_locate_exact() -> None:
    span = locate(SRC, "Euribor a 6 mesi")
    assert span is not None
    assert SRC[span[0] : span[1]] == "Euribor a 6 mesi"


def test_locate_whitespace_normalized() -> None:
    # Quote uses single spaces where the source has newlines.
    span = locate(SRC, "Il tasso variabile è indicizzato")
    assert span is not None
    assert SRC[span[0] : span[1]] == "Il tasso\nvariabile è\nindicizzato"


def test_locate_fuzzy_tolerates_small_diff() -> None:
    # A stray character inside the quote; fuzzy matching still anchors it.
    span = locate(SRC, "indicizzato allxEuribor a 6 mesi")
    assert span is not None
    assert "Euribor a 6 mesi" in SRC[span[0] : span[1]]


def test_locate_returns_none_when_absent() -> None:
    assert locate(SRC, "completely unrelated sentence about sailboats") is None


def test_locate_rejects_scattered_match() -> None:
    # "alpha" and "beta" both occur, but far apart with unrelated text between —
    # a real quote spans ~its own length, so this must not resolve to the whole
    # span (which would yield a giant, wrong highlight).
    scattered = "alpha " + "filler " * 40 + "beta end"
    assert locate(scattered, "alpha beta") is None


def test_parse_quotes_tolerates_fences_and_prose() -> None:
    raw = 'Here you go:\n```json\n["primo", "secondo"]\n```'
    assert _parse_quotes(raw) == ["primo", "secondo"]
    assert _parse_quotes('[{"quote": "terzo"}]') == ["terzo"]
    assert _parse_quotes("no json here") == []


# ── ground_answer() ──────────────────────────────────────────────────────────
class FakeChat:
    """Returns a fixed reply, recording the prompt it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    async def complete(self, messages, *, max_tokens=512, temperature=0.0) -> str:
        self.last_prompt = messages[-1].content
        return self.reply


def _doc_with_geometry() -> Document:
    page = OcrPage(
        index=0,
        source="text-layer",
        text="tasso variabile Euribor",
        width=1000,
        height=1400,
        lines=[
            OcrLine(
                text="tasso variabile Euribor",
                score=1.0,
                poly=[(10, 10), (300, 10), (300, 30), (10, 30)],
                box=(10, 10, 300, 30),
                words=[
                    OcrWord(text="tasso", box=(10, 10, 90, 30)),
                    OcrWord(text="variabile", box=(100, 10, 240, 30)),
                    OcrWord(text="Euribor", box=(250, 10, 300, 30)),
                ],
            )
        ],
    )
    return Document(
        id=DocumentId("mutuo.pdf"),
        source="/abs/mutuo.pdf",
        content="ignored-by-grounding",
        artifacts={"ocr": OcrArtifact(engine="text-layer", pages=[page]).model_dump()},
    )


def _working_set(doc: Document) -> WorkingSet:
    return WorkingSet(documents=(doc,), token_count=10, truncated=False)


@pytest.mark.asyncio
async def test_ground_answer_fills_highlights() -> None:
    doc = _doc_with_geometry()
    answer = Answer(text="Il tasso è variabile.", used_reasoning=True)
    chat = FakeChat('["tasso variabile"]')

    grounded = await ground_answer(answer, _working_set(doc), chat)

    assert len(grounded.citations) == 1
    cit = grounded.citations[0]
    assert cit.doc_id == "mutuo.pdf"
    assert cit.quote == "tasso variabile"
    assert len(cit.highlights) == 1
    h = cit.highlights[0]
    assert h.page == 0 and h.width == 1000 and h.height == 1400
    # "tasso variabile" merges to one line box spanning both words.
    assert h.boxes == ((10, 10, 240, 30),)
    # The model was grounded against the canonical (word-derived) text.
    assert "[page 1]" in (chat.last_prompt or "")


@pytest.mark.asyncio
async def test_ground_answer_keeps_quote_even_if_unlocatable() -> None:
    doc = _doc_with_geometry()
    answer = Answer(text="x", used_reasoning=True)
    chat = FakeChat('["nowhere in the source"]')

    grounded = await ground_answer(answer, _working_set(doc), chat)

    assert len(grounded.citations) == 1
    assert grounded.citations[0].quote == "nowhere in the source"
    assert grounded.citations[0].highlights == ()  # no box, still cited


@pytest.mark.asyncio
async def test_ground_answer_drops_off_topic_quote() -> None:
    doc = _doc_with_geometry()
    # Answer asserts a specific number; the returned quote shares no number and
    # only the generic word "tasso" — it must be rejected as boilerplate.
    answer = Answer(text="Il tasso applicato è il 2,453%.", used_reasoning=True)
    chat = FakeChat('["tasso variabile Euribor"]')

    grounded = await ground_answer(answer, _working_set(doc), chat)

    assert grounded is answer  # nothing survived the relevance guard


@pytest.mark.asyncio
async def test_ground_answer_trajectory_source_uses_evidence() -> None:
    doc = _doc_with_geometry()
    # The model is shown the trajectory evidence, not the whole document.
    answer = Answer(
        text="Il tasso è variabile.",
        used_reasoning=True,
        evidence=("EVIDENCE: il tasso variabile applicato al mutuo.",),
    )
    chat = FakeChat('["tasso variabile"]')

    grounded = await ground_answer(answer, _working_set(doc), chat, source="trajectory")

    assert "EVIDENCE:" in (chat.last_prompt or "")  # extracted from evidence
    assert len(grounded.citations) == 1
    cit = grounded.citations[0]
    assert cit.quote == "tasso variabile"
    # Quote still located in the document's canonical text -> boxes.
    assert cit.highlights and cit.highlights[0].boxes == ((10, 10, 240, 30),)


@pytest.mark.asyncio
async def test_ground_answer_trajectory_falls_back_to_document() -> None:
    doc = _doc_with_geometry()
    answer = Answer(text="Il tasso è variabile.", used_reasoning=True)  # no evidence
    chat = FakeChat('["tasso variabile"]')

    grounded = await ground_answer(answer, _working_set(doc), chat, source="trajectory")

    # With no evidence it falls back to the document text (the canonical text).
    assert "[page 1]" in (chat.last_prompt or "")
    assert grounded.citations[0].highlights[0].boxes == ((10, 10, 240, 30),)


def _rate_doc() -> Document:
    line = OcrLine(
        text="tasso variabile 2,453% Euribor",
        score=1.0,
        poly=[(10, 10), (500, 10), (500, 30), (10, 30)],
        box=(10, 10, 500, 30),
        words=[
            OcrWord(text="tasso", box=(10, 10, 90, 30)),
            OcrWord(text="variabile", box=(100, 10, 240, 30)),
            OcrWord(text="2,453%", box=(250, 10, 360, 30)),
            OcrWord(text="Euribor", box=(370, 10, 500, 30)),
        ],
    )
    page = OcrPage(
        index=0, source="text-layer", text="tasso variabile 2,453% Euribor",
        width=1000, height=1400, lines=[line],
    )
    return Document(
        id=DocumentId("rate.pdf"),
        source="/abs/rate.pdf",
        content="ignored",
        artifacts={"ocr": OcrArtifact(engine="text-layer", pages=[page]).model_dump()},
    )


@pytest.mark.asyncio
async def test_ground_answer_raw_source_needs_no_llm() -> None:
    doc = _rate_doc()
    answer = Answer(
        text="Il tasso applicato è il 2,453%.",
        used_reasoning=True,
        # Structured grep provenance: each hit already carries its home doc.
        evidence_spans=(
            EvidenceSpan(doc_id=DocumentId("rate.pdf"),
                         text="tasso variabile 2,453% Euribor"),  # shares 2,453
            EvidenceSpan(doc_id=DocumentId("rate.pdf"),
                         text="clausola generica di rimborso senza dettagli"),  # dropped
        ),
    )

    # No chat_model passed at all — raw mode must not call an LLM.
    grounded = await ground_answer(answer, _working_set(doc), source="raw")

    assert len(grounded.citations) == 1
    cit = grounded.citations[0]
    assert "2,453%" in cit.quote
    assert cit.highlights  # the grep line was located + boxed
    assert cit.highlights[0].boxes == ((10, 10, 500, 30),)


@pytest.mark.asyncio
async def test_ground_answer_raw_places_span_in_its_own_doc() -> None:
    # Two near-duplicate docs differing only by the rate. The grep hit is tagged
    # with rate.pdf, so it must box onto rate.pdf — never the sibling.
    rate = _rate_doc()
    sibling = rate.model_copy(update={"id": DocumentId("other.pdf"), "source": "/abs/other.pdf"})
    answer = Answer(
        text="Il tasso applicato è il 2,453%.",
        used_reasoning=True,
        evidence_spans=(
            EvidenceSpan(doc_id=DocumentId("rate.pdf"), text="tasso variabile 2,453% Euribor"),
        ),
    )
    ws = WorkingSet(documents=(sibling, rate), token_count=20, truncated=False)

    grounded = await ground_answer(answer, ws, source="raw")

    assert len(grounded.citations) == 1
    assert grounded.citations[0].doc_id == "rate.pdf"


@pytest.mark.asyncio
async def test_ground_answer_drops_code_junk_quote() -> None:
    doc = _doc_with_geometry()
    answer = Answer(text="Il tasso è il 2,453%.", used_reasoning=True)
    chat = FakeChat('["000004763n000"]')  # stamped reference number, no real word

    grounded = await ground_answer(answer, _working_set(doc), chat)

    assert grounded is answer


@pytest.mark.asyncio
async def test_ground_answer_no_quotes_returns_original() -> None:
    doc = _doc_with_geometry()
    original = Answer(
        text="x",
        citations=(Citation(doc_id=DocumentId("mutuo.pdf"), source="/abs/mutuo.pdf"),),
        used_reasoning=True,
    )
    grounded = await ground_answer(original, _working_set(doc), FakeChat("[]"))
    assert grounded is original  # unchanged when nothing is extracted
