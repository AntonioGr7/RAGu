"""Contextual recursive chunker.

Two responsibilities, kept separate so each is swappable:

1. **Splitting** (``RecursiveTokenSplitter``) — carve the document into
   token-sized passages along natural boundaries (paragraphs → sentences →
   hard cut), preserving character offsets so citations can point back precisely.
2. **Contextualizing** (``Contextualizer``) — optionally prepend an
   LLM-generated blurb situating each passage in its document. This is
   Anthropic's *contextual retrieval*, which materially lifts recall. It is a
   plug-in so chunking works fully offline when disabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ragu.core import Chunk, Document, DocumentId
from ragu.ports import TokenCounter

# Boundary patterns in descending strength. We split on the strongest boundary
# that still yields pieces under the size limit.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Atom:
    """An indivisible span of source text with its offsets."""

    text: str
    start: int
    end: int


@runtime_checkable
class Contextualizer(Protocol):
    """Produces a situating context blurb for a chunk within its document."""

    async def contextualize(self, document: Document, chunk_text: str) -> str:
        ...


class RecursiveTokenSplitter:
    """Splits text into offset-tracked passages near ``target_tokens``."""

    def __init__(
        self, counter: TokenCounter, target_tokens: int = 400, overlap_tokens: int = 64
    ) -> None:
        self._counter = counter
        self._target = target_tokens
        self._overlap = overlap_tokens

    def split(self, text: str) -> list[_Atom]:
        atoms = self._atomize(text, 0)
        return self._pack(atoms, text)

    def _atomize(self, text: str, base: int) -> list[_Atom]:
        """Break text into atoms each under the target size, recursively."""
        if not text.strip():
            return []
        if self._counter.count(text) <= self._target:
            return [_Atom(text, base, base + len(text))]

        for pattern in (_PARAGRAPH, _SENTENCE):
            pieces = self._split_keep_offsets(text, base, pattern)
            if len(pieces) > 1:
                out: list[_Atom] = []
                for piece in pieces:
                    out.extend(self._atomize(piece.text, piece.start))
                return out

        # No boundary helped (one giant token-dense span): hard-cut by chars.
        return self._hard_split(text, base)

    @staticmethod
    def _split_keep_offsets(text: str, base: int, pattern: re.Pattern[str]) -> list[_Atom]:
        atoms: list[_Atom] = []
        cursor = 0
        for m in pattern.finditer(text):
            segment = text[cursor : m.start()]
            if segment.strip():
                atoms.append(_Atom(segment, base + cursor, base + m.start()))
            cursor = m.end()
        tail = text[cursor:]
        if tail.strip():
            atoms.append(_Atom(tail, base + cursor, base + len(text)))
        return atoms

    def _hard_split(self, text: str, base: int) -> list[_Atom]:
        # Approximate chars-per-token from this text to size a char window.
        tokens = max(1, self._counter.count(text))
        chars_per_tok = max(1, len(text) // tokens)
        window = self._target * chars_per_tok
        return [
            _Atom(text[i : i + window], base + i, base + min(i + window, len(text)))
            for i in range(0, len(text), window)
        ]

    def _pack(self, atoms: list[_Atom], source: str) -> list[_Atom]:
        """Greedily merge atoms up to the target, with token overlap carryover.

        Chunk text is sliced from ``source`` between the first and last atom
        offsets so separators (spaces, newlines) are preserved and offsets
        round-trip exactly.
        """
        chunks: list[_Atom] = []
        bucket: list[_Atom] = []

        def flush() -> None:
            if not bucket:
                return
            start, end = bucket[0].start, bucket[-1].end
            chunks.append(_Atom(source[start:end], start, end))

        for atom in atoms:
            tentative = source[bucket[0].start : atom.end] if bucket else atom.text
            if bucket and self._counter.count(tentative) > self._target:
                flush()
                bucket = self._overlap_tail(bucket)
            bucket.append(atom)
        flush()
        return chunks

    def _overlap_tail(self, bucket: list[_Atom]) -> list[_Atom]:
        if self._overlap <= 0:
            return []
        tail: list[_Atom] = []
        total = 0
        for atom in reversed(bucket):
            total += self._counter.count(atom.text)
            tail.insert(0, atom)
            if total >= self._overlap:
                break
        return tail


class ContextualChunker:
    """``Chunker`` port implementation: split, then optionally contextualize."""

    def __init__(
        self,
        splitter: RecursiveTokenSplitter,
        contextualizer: Contextualizer | None = None,
    ) -> None:
        self._splitter = splitter
        self._contextualizer = contextualizer

    async def chunk(self, document: Document) -> list[Chunk]:
        atoms = self._splitter.split(document.content)
        chunks: list[Chunk] = []
        for ordinal, atom in enumerate(atoms):
            context = ""
            if self._contextualizer is not None:
                context = await self._contextualizer.contextualize(document, atom.text)
            chunks.append(
                Chunk(
                    id=_chunk_id(document.id, ordinal),
                    doc_id=document.id,
                    text=atom.text,
                    context=context,
                    ordinal=ordinal,
                    start_char=atom.start,
                    end_char=atom.end,
                )
            )
        return chunks


def _chunk_id(doc_id: DocumentId, ordinal: int) -> str:
    return f"{doc_id}::{ordinal:04d}"
