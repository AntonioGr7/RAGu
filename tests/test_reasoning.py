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
    citations_from_trajectory,
    corpus_filename,
    materialize_corpus,
)
from ragu.config import VomeroSettings
from ragu.core import Document, DocumentId, Query, WorkingSet


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


@dataclass
class FakeStep:
    code: str | None


class FakeEngine:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def run(self, question: str, source: Any, *, return_trajectory: bool = False) -> FakeRunResult:
        self._captured["question"] = question
        self._captured["source"] = source
        return FakeRunResult(
            answer="The total is 1250.",
            trajectory=[FakeStep(code="corpus.read('invoices/inv1.pdf.txt')")],
            tokens=4321,
            calls=3,
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
