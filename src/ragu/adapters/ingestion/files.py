"""Plain text/markdown file loading + filesystem walking shared by all loaders.

Document ids are derived from the path relative to the ingestion root, so they
are stable and human-readable (re-indexing the same file updates it in place
rather than duplicating).
"""

from __future__ import annotations

from pathlib import Path

from ragu.core import Document, DocumentId

# Extensions read directly as UTF-8 text.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}


def iter_files(paths: list[str | Path]) -> list[tuple[Path, Path]]:
    """Yield (file, base) pairs for every file under the given paths.

    ``base`` is the root the file's id is made relative to (the directory for a
    directory arg, the file's parent for a file arg)."""
    out: list[tuple[Path, Path]] = []
    for raw in paths:
        root = Path(raw)
        if root.is_dir():
            out.extend((p, root) for p in sorted(root.rglob("*")) if p.is_file())
        else:
            out.append((root, root.parent))
    return out


def make_document(file: Path, base: Path, content: str, extra: dict[str, str]) -> Document:
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
    )


def load_text_file(file: Path, base: Path) -> Document:
    content = file.read_text(encoding="utf-8", errors="replace")
    return make_document(file, base, content, {"loader": "text"})
