"""Facade-level test: the path a user actually takes (offline, in-memory)."""

from pathlib import Path

import pytest

from ragu.app import Ragu
from ragu.config import RaguSettings


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
    n = await ragu.index_paths([tmp_path])
    assert n >= 2

    ws = await ragu.retrieve("quantum entanglement photons")
    assert ws.documents
    assert str(ws.documents[0].id) == "physics.md"
    assert ws.token_count > 0
