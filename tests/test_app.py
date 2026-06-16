"""Facade-level test: the path a user actually takes (offline, in-memory)."""

from pathlib import Path

import pytest

from ragu.app import Ragu
from ragu.config import RaguSettings
from ragu.core import DocumentId


def _offline_settings() -> RaguSettings:
    settings = RaguSettings()
    settings.storage.backend = "memory"
    settings.embedding.provider = "fake"
    settings.chunking.contextual = False  # no LLM calls
    settings.chunking.target_tokens = 40
    settings.working_set.max_tokens = 100_000
    return settings


@pytest.mark.asyncio
async def test_index_and_retrieve(tmp_path: Path) -> None:
    (tmp_path / "physics.md").write_text(
        "Quantum entanglement links particles across distance. "
        "Photons show wave particle duality."
    )
    (tmp_path / "cooking.md").write_text(
        "Toast arborio rice, then add warm broth slowly while stirring for risotto."
    )

    ragu = Ragu(_offline_settings())
    report = await ragu.index_paths([tmp_path])
    assert report.chunks >= 2
    assert report.new == 2
    assert report.updated == 0
    assert report.skipped == 0

    ws = await ragu.retrieve("quantum entanglement photons")
    assert ws.documents
    assert str(ws.documents[0].id) == "physics.md"
    assert ws.token_count > 0


@pytest.mark.asyncio
async def test_incremental_reindex_skips_unchanged(tmp_path: Path) -> None:
    physics = tmp_path / "physics.md"
    physics.write_text("Quantum entanglement links particles across distance.")
    (tmp_path / "cooking.md").write_text("Toast arborio rice for risotto.")

    ragu = Ragu(_offline_settings())
    first = await ragu.index_paths([tmp_path])
    assert first.new == 2

    # Re-index unchanged: both files skipped, nothing reloaded or rewritten.
    second = await ragu.index_paths([tmp_path])
    assert (second.new, second.updated, second.skipped, second.chunks) == (0, 0, 2, 0)

    # Change one file: only it is re-indexed; the other stays skipped.
    physics.write_text("Photons show wave particle duality across the lab.")
    third = await ragu.index_paths([tmp_path])
    assert (third.new, third.updated, third.skipped) == (0, 1, 1)


@pytest.mark.asyncio
async def test_reindex_replaces_chunks_no_duplicates(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("alpha beta gamma delta")

    ragu = Ragu(_offline_settings())
    await ragu.index_paths([tmp_path])
    note.write_text("epsilon zeta eta theta")
    await ragu.index_paths([tmp_path])

    # The vector store must hold only the new chunks — no stale "alpha" text
    # lingering beside the fresh "epsilon" chunks.
    chunks = ragu._vector_store._chunks  # white-box: in-memory backend
    texts = " ".join(c.text for c in chunks)
    assert "alpha" not in texts
    assert "epsilon" in texts
    # And the parent document content is the updated text.
    [doc] = await ragu._document_store.get([DocumentId("note.md")])
    assert doc.content == "epsilon zeta eta theta"


@pytest.mark.asyncio
async def test_prune_removes_deleted_source_files(tmp_path: Path) -> None:
    keep = tmp_path / "keep.md"
    drop = tmp_path / "drop.md"
    keep.write_text("kept document about turbines")
    drop.write_text("doomed document about sailboats")

    ragu = Ragu(_offline_settings())
    await ragu.index_paths([tmp_path])

    drop.unlink()
    report = await ragu.index_paths([tmp_path])
    assert report.pruned == 1
    assert report.skipped == 1

    ws = await ragu.retrieve("doomed sailboats")
    assert all(str(d.id) != "drop.md" for d in ws.documents)
