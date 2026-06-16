"""Plain text/markdown file loading + filesystem walking shared by all loaders.

Document ids are derived from each file's path relative to the *common ancestor*
of everything being indexed, so they are stable and human-readable (re-indexing
the same file updates it in place rather than duplicating). Using a single shared
base — rather than per-argument bases — also means two files with the same name
in different folders never collide on the same id, regardless of whether you
index the parent folder or list the subfolders/files separately.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ragu.core import Document, DocumentId

# Extensions read directly as UTF-8 text.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}


def file_fingerprint(file: Path) -> str:
    """SHA-256 of the raw file bytes — a content fingerprint for change detection.

    Hashed from the bytes on disk (not the extracted text) so the skip decision
    can be made *before* the expensive OCR/parse step. Streamed in blocks so
    large scans don't load wholly into memory."""
    h = hashlib.sha256()
    with file.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(paths: list[str | Path]) -> list[tuple[Path, Path]]:
    """Return ``(file, base)`` pairs for every file under the given paths.

    All pairs share one ``base`` — the common-ancestor directory of the path
    arguments — so a file's id (``file.relative_to(base)``) depends on *what* is
    indexed, not on *how* the arguments are spelled. The base is computed from
    the arguments (not from the files found), so ids stay stable as the corpus
    grows. Files are resolved and de-duplicated (overlapping arguments, or a file
    that also lives inside an indexed directory, are ingested once).

    Because every file hangs off a single base, two distinct files can never map
    to the same id; the explicit check below is defensive insurance against any
    future change to id derivation."""
    roots = [Path(raw).resolve() for raw in paths]
    if not roots:
        return []

    base = Path(os.path.commonpath([str(r) for r in roots]))
    if not base.is_dir():  # single-file arg: commonpath is the file itself
        base = base.parent

    files: set[Path] = set()
    for root in roots:
        if root.is_dir():
            files.update(p for p in root.rglob("*") if p.is_file())
        else:
            files.add(root)  # a file, or a missing path we let the loader report

    out: list[tuple[Path, Path]] = []
    seen: dict[str, Path] = {}
    for file in sorted(files):
        rel = file.relative_to(base).as_posix()
        if rel in seen:
            raise ValueError(
                f"Document id collision: {file} and {seen[rel]} both map to "
                f"id '{rel}'. Index them from a shared parent folder instead."
            )
        seen[rel] = file
        out.append((file, base))
    return out


def make_document(
    file: Path,
    base: Path,
    content: str,
    extra: dict[str, str],
    artifacts: dict[str, Any] | None = None,
) -> Document:
    rel = file.relative_to(base).as_posix()
    return Document(
        id=DocumentId(rel),
        source=str(file.resolve()),
        content=content,
        metadata={
            "filename": file.name,
            "suffix": file.suffix.lower(),
            # Path relative to the ingestion base folder — preserves the original
            # folder structure for citations, filtering, and re-export.
            "rel_path": rel,
            **extra,
        },
        artifacts=artifacts or {},
    )


def load_text_file(file: Path, base: Path) -> Document:
    content = file.read_text(encoding="utf-8", errors="replace")
    return make_document(file, base, content, {"loader": "text"})
