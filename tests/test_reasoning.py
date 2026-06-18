"""L2 adapter tests — fully offline, with a fake vomero engine.

We test the parts RAGu owns: materializing the working set into a corpus,
deriving citations from the trajectory, and mapping the result to an Answer.
The real vomero engine (which makes LLM calls) is substituted by a fake.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ragu.adapters.reasoning.vomero import (
    VomeroReasoningEngine,
    citations_from_provenance,
    citations_from_trajectory,
    corpus_filename,
    evidence_spans_from_provenance,
    materialize_corpus,
)
from ragu.config import VomeroSettings
from ragu.core import Document, DocumentId, Query, WorkingSet


@dataclass
class AccessEvent:
    """Mirror of vomero's provenance record (op + doc + located line)."""

    op: str
    doc: str | int
    lineno: int | None = None
    text: str | None = None


def _ws() -> WorkingSet:
    docs = (
        Document(id=DocumentId("invoices/inv1.pdf"), source="/abs/inv1.pdf", content="total 1250",
                 metadata={"rel_path": "invoices/inv1.pdf"}),
        Document(id=DocumentId("notes/memo.md"), source="/abs/memo.md", content="memo body",
                 metadata={"rel_path": "notes/memo.md"}),
    )
    return WorkingSet(documents=docs, token_count=4)


def test_corpus_filename_ensures_text_ext() -> None:
    pdf = Document(
        id=DocumentId("a/b.pdf"), source="s", content="x", metadata={"rel_path": "a/b.pdf"}
    )
    md = Document(id=DocumentId("c.md"), source="s", content="x", metadata={"rel_path": "c.md"})
    assert corpus_filename(pdf) == "a/b.pdf.txt"  # non-text ext gets .txt
    assert corpus_filename(md) == "c.md"  # already text


def test_materialize_corpus_writes_tree(tmp_path: Path) -> None:
    rel_to_doc = materialize_corpus(_ws(), tmp_path)
    assert (tmp_path / "invoices/inv1.pdf.txt").read_text() == "total 1250"
    assert (tmp_path / "notes/memo.md").read_text() == "memo body"
    assert rel_to_doc == {
        "invoices/inv1.pdf.txt": "invoices/inv1.pdf",
        "notes/memo.md": "notes/memo.md",
    }


def test_citations_only_for_touched_files() -> None:
    rel_to_doc = {"invoices/inv1.pdf.txt": "invoices/inv1.pdf", "notes/memo.md": "notes/memo.md"}
    sources = {
        DocumentId("invoices/inv1.pdf"): "/abs/inv1.pdf",
        DocumentId("notes/memo.md"): "/abs/memo.md",
    }

    @dataclass
    class Step:
        code: str | None

    steps = [Step(code="corpus.read('invoices/inv1.pdf.txt')"), Step(code="print(1)")]
    cites = citations_from_trajectory(steps, rel_to_doc, sources)
    # Only the file the model actually read is cited.
    assert [c.doc_id for c in cites] == ["invoices/inv1.pdf"]
    assert cites[0].source == "/abs/inv1.pdf"


@dataclass
class FakeRunResult:
    answer: str
    trajectory: list[Any] = field(default_factory=list)
    tokens: int = 0
    calls: int = 0
    provenance: list[Any] = field(default_factory=list)


@dataclass
class FakeStep:
    code: str | None


class FakeEngine:
    def __init__(self, captured: dict, provenance: list[Any] | None = None) -> None:
        self._captured = captured
        self._provenance = provenance or []

    def run(
        self,
        question: str,
        source: Any,
        *,
        return_trajectory: bool = False,
        on_event: Any = None,
        ask_handler: Any = None,
        history: Any = None,
        transcript_sink: Any = None,
    ) -> FakeRunResult:
        self._captured["question"] = question
        self._captured["source"] = source
        self._captured["ask_handler"] = ask_handler
        self._captured["history"] = history
        self._captured["transcript_sink"] = transcript_sink
        return FakeRunResult(
            answer="The total is 1250.",
            trajectory=[FakeStep(code="corpus.read('invoices/inv1.pdf.txt')")],
            tokens=4321,
            calls=3,
            provenance=self._provenance,
        )


@pytest.mark.asyncio
async def test_reason_maps_answer_and_citations() -> None:
    captured: dict = {}
    engine = VomeroReasoningEngine(
        VomeroSettings(handoff="corpus"), engine=FakeEngine(captured)
    )
    answer = await engine.reason(Query(text="what is the total?"), _ws())

    assert answer.text == "The total is 1250."
    assert answer.used_reasoning is True
    assert answer.trace["tokens"] == "4321"
    assert answer.trace["engine"] == "vomero"
    # Citation derived from the file the fake model read.
    assert [c.doc_id for c in answer.citations] == ["invoices/inv1.pdf"]
    assert captured["question"] == "what is the total?"


# ── provenance ────────────────────────────────────────────────────────────────
def test_citations_from_provenance_orders_by_first_touch() -> None:
    rel_to_doc = {"invoices/inv1.pdf.txt": "invoices/inv1.pdf", "notes/memo.md": "notes/memo.md"}
    sources = {
        DocumentId("invoices/inv1.pdf"): "/abs/inv1.pdf",
        DocumentId("notes/memo.md"): "/abs/memo.md",
    }
    prov = [
        AccessEvent("grep", "notes/memo.md", 3, "total due 1250"),
        AccessEvent("read", "invoices/inv1.pdf.txt"),
        AccessEvent("read", "notes/memo.md"),  # already seen — not duplicated
    ]
    cites = citations_from_provenance(prov, rel_to_doc, sources, [])
    assert [c.doc_id for c in cites] == ["notes/memo.md", "invoices/inv1.pdf"]
    assert cites[0].source == "/abs/memo.md"


def test_evidence_spans_from_provenance_keeps_grep_hits_with_doc() -> None:
    rel_to_doc = {"invoices/inv1.pdf.txt": "invoices/inv1.pdf"}
    prov = [
        AccessEvent("grep", "invoices/inv1.pdf.txt", 42, "total 1250"),
        AccessEvent("read", "invoices/inv1.pdf.txt"),  # not a grep — dropped
        AccessEvent("grep", "invoices/inv1.pdf.txt", 7, "   "),  # blank — dropped
    ]
    spans = evidence_spans_from_provenance(prov, rel_to_doc, {}, [])
    assert [(str(s.doc_id), s.text) for s in spans] == [("invoices/inv1.pdf", "total 1250")]


@pytest.mark.asyncio
async def test_reason_prefers_provenance_for_citations() -> None:
    # Provenance points at memo.md even though the trajectory code reads inv1 —
    # the structured access log wins when present.
    prov = [AccessEvent("read", "notes/memo.md")]
    engine = VomeroReasoningEngine(
        VomeroSettings(handoff="corpus"), engine=FakeEngine({}, provenance=prov)
    )
    answer = await engine.reason(Query(text="q"), _ws())
    assert [c.doc_id for c in answer.citations] == ["notes/memo.md"]
