"""Ground an L2 answer in the source with inline, located citations.

L2 (vomero) returns an answer with *document-level* citations. This optional
pass turns those into *span-level* ones: a cheap LLM call extracts the verbatim
quotes that support the answer, each quote is located in the document's canonical
word-aligned text (see :mod:`ragu.adapters.ingestion.geometry`), and the
resulting char span is resolved to page + word boxes. The model is shown the
**canonical** text — not ``Document.content`` — so the offsets it copies from
map straight onto boxes.

Locating is robust to the messy text OCR/text-layer extraction produces:
exact match → whitespace-normalised match → fuzzy (longest matching blocks).
A quote that can't be located still becomes a citation, just without highlights.

This module depends only on the ``ChatModel`` port and the geometry helper, so it
is provider-agnostic and never imports vomero.
"""

from __future__ import annotations

import difflib
import json
import logging
import re

from ragu.adapters.ingestion import build_word_layout, highlights_for_span
from ragu.core import Answer, Citation, Document, WorkingSet
from ragu.ports import ChatModel
from ragu.ports.llm import ChatMessage

logger = logging.getLogger(__name__)

_SYSTEM = ChatMessage(
    role="system",
    content=(
        "You extract the exact sentences from the SOURCE that state the facts "
        "asserted in the ANSWER. Focus on the concrete details the answer gives — "
        "numbers, rates, amounts, percentages, dates, names — and for each, copy "
        "the single shortest verbatim span from the SOURCE that actually states "
        "that detail, character-for-character (including any odd spacing or line "
        "breaks). Do NOT quote generic boilerplate (article headings, standard "
        "clauses) that only mentions the topic without the specific fact. Every "
        "span MUST appear verbatim in the SOURCE. "
        "Return ONLY a JSON array of strings, e.g. [\"...\", \"...\"]. "
        "If the SOURCE does not state a given fact, omit it; if none, return []."
    ),
)


# Grounding evidence sources — where the supporting quotes come from.
TRAJECTORY = "trajectory"  # LLM-extracted from what L2 actually read — specific
DOCUMENT = "document"  # LLM-extracted from the whole cited document(s) — broad
RAW = "raw"  # no LLM: the lines L2 read that overlap the answer — cheap, coarser


async def ground_answer(
    answer: Answer,
    working_set: WorkingSet,
    chat_model: ChatModel | None = None,
    *,
    source: str = TRAJECTORY,
    max_source_chars: int = 150_000,
) -> Answer:
    """Return ``answer`` with span-level, box-resolved citations.

    ``source`` chooses where the supporting quotes come from:

    * ``"trajectory"`` — an LLM extracts quotes from the text L2 actually read
      (``Answer.evidence``). Most specific; falls back to ``"document"`` if no
      evidence is available.
    * ``"document"`` — an LLM extracts quotes from the full cited document(s).
    * ``"raw"`` — **no extra LLM call**: the lines L2 read that share a number or
      name with the answer are highlighted directly. Cheapest and needs no API
      key, but coarser (whole read lines, not the precise supporting clause).

    Quotes are located in each document's canonical word-aligned text and
    resolved to page + word boxes. On any failure the original answer is returned
    unchanged — grounding never breaks the answer."""
    targets = _target_docs(answer, working_set)
    if not targets:
        return answer
    layouts = {
        doc.id: (build_word_layout(doc.artifacts["ocr"]) if "ocr" in doc.artifacts else None)
        for doc in targets
    }

    # Each candidate is a quote plus the documents it should be located in.
    candidates: list[tuple[str, list[Document]]] = []
    if source == RAW:
        if not answer.evidence:
            logger.info("Grounding: raw source has no L2 evidence to ground")
            return answer
        for line in _relevant_evidence_lines(answer.evidence, answer.text):
            candidates.append((line, targets))
    elif source == TRAJECTORY and answer.evidence:
        evidence = _truncate("\n\n".join(answer.evidence), max_source_chars, "trajectory")
        for quote in await _extract_quotes(chat_model, answer.text, "evidence", evidence):
            candidates.append((quote, targets))  # search every cited doc
    else:
        if source == TRAJECTORY:
            logger.info("Grounding: no L2 evidence available; using document source")
        for doc in targets:
            layout = layouts[doc.id]
            doc_text = _truncate(
                layout.text if layout is not None else doc.content,
                max_source_chars, str(doc.id),
            )
            for quote in await _extract_quotes(chat_model, answer.text, str(doc.id), doc_text):
                candidates.append((quote, [doc]))  # search only its home doc

    # Raw lines are pre-filtered for relevance and only kept if they actually
    # locate on a page (an un-boxable read line is noise, not a citation).
    require_location = source == RAW
    grounded: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for quote, docs in candidates:
        if source != RAW and not _is_relevant(answer.text, quote):
            logger.info("Grounding: dropped off-topic quote: %r", quote[:80])
            continue
        citation = _place_quote(quote, docs, layouts)
        if require_location and citation.start_char is None:
            continue
        key = (str(citation.doc_id), quote)
        if key in seen:
            continue
        seen.add(key)
        grounded.append(citation)

    if not grounded:
        return answer
    located = sum(1 for c in grounded if c.highlights)
    logger.info(
        "Grounding (%s): %d quote(s), %d located on a page", source, len(grounded), located
    )
    return answer.model_copy(update={"citations": tuple(grounded)})


# vomero's read output is REPL stdout, not clean text: grep hits carry a
# "<file>:<lineno>:" prefix, and list prints wrap fragments in quotes. We strip
# those so the underlying source text can be located in the canonical text.
_GREP_PREFIX = re.compile(r"[^\s,\[\]:]+:\d+:\s*")
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_LEADING_NOISE = re.compile(r"^[-•*–—\s]+")


def _evidence_fragments(block: str) -> list[str]:
    """Recover candidate source spans from one block of L2 read output.

    Handles the two shapes vomero emits: Python list prints (``['a', 'b']`` →
    the quoted fragments) and grep dumps (``file.txt:43: text`` → the text after
    the prefix). Anything else is treated as plain lines."""
    quoted = _QUOTED.findall(block)
    if quoted:
        raw = [a or b for a, b in quoted]
    else:
        raw = [block]
    fragments: list[str] = []
    for piece in raw:
        # One fragment per line: a multi-line read dump would otherwise resolve to
        # a giant span (dozens of boxes) instead of a tidy line-level highlight.
        for line in _GREP_PREFIX.sub("\n", piece).splitlines():
            line = _LEADING_NOISE.sub("", line.strip().strip("[]")).strip(" ,=\t")
            if line:
                fragments.append(line)
    return fragments


def _relevant_evidence_lines(
    evidence: tuple[str, ...], answer_text: str, *, min_len: int = 14, cap: int = 40
) -> list[str]:
    """Source fragments from L2's read output that overlap the answer's facts
    (share a number or name), de-duplicated and capped. The selection that the
    LLM does in the other modes, here done by the relevance heuristic — no model
    call. vomero's REPL formatting (grep prefixes, list quotes) is stripped first
    so the fragments can actually be located in the document."""
    seen: set[str] = set()
    lines: list[str] = []
    for block in evidence:
        for fragment in _evidence_fragments(block):
            if len(fragment) < min_len or fragment in seen:
                continue
            if not _is_relevant(answer_text, fragment):
                continue
            seen.add(fragment)
            lines.append(fragment)
            if len(lines) >= cap:
                return lines
    return lines


def _place_quote(quote: str, docs: list[Document], layouts: dict) -> Citation:
    """Locate ``quote`` across ``docs`` and build a citation with page/word boxes.

    Tries an exact/normalised match in every doc first, only falling back to
    fuzzy matching if none has one — so a quote that belongs verbatim to one
    document isn't fuzzily mis-attributed to a near-duplicate sibling (these
    template contracts differ by a digit or two). If it can't be placed at all,
    the quote is still cited (attributed to the first doc) without highlights."""
    for fuzzy in (False, True):  # exact/normalised pass first, then fuzzy
        for doc in docs:
            layout = layouts.get(doc.id)
            source_text = layout.text if layout is not None else doc.content
            span = locate(source_text, quote, fuzzy=fuzzy)
            if span is None:
                continue
            highlights = (
                tuple(highlights_for_span(layout, *span)) if layout is not None else ()
            )
            return Citation(
                doc_id=doc.id,
                source=doc.source,
                quote=quote,
                start_char=span[0],
                end_char=span[1],
                highlights=highlights,
            )
    first = docs[0]
    return Citation(doc_id=first.id, source=first.source, quote=quote)


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) > limit:
        logger.warning("Grounding source (%s) truncated to %d chars (was %d)",
                       label, limit, len(text))
        return text[:limit]
    return text


# Numbers (incl. 2,453 / 1.20) and distinctive capitalised words (names, terms).
_NUM_RE = re.compile(r"\d[\d.,]*")
_CAP_RE = re.compile(r"\b[A-ZÀ-Ý][\wÀ-ÿ]{3,}\b")


def _salient(text: str) -> tuple[set[str], set[str]]:
    """Salient tokens of a text: numeric tokens and capitalised words (lowered).

    Spaces *inside* a number are stripped first, so OCR's ``2, 453`` matches a
    model's ``2,453`` (a common, otherwise-silent cause of false rejections)."""
    joined = re.sub(r"(?<=[\d.,])[ \t]+(?=[\d.,])", "", text)
    nums = set(_NUM_RE.findall(joined))
    caps = {w.lower() for w in _CAP_RE.findall(text)}
    return nums, caps


def _is_relevant(answer_text: str, quote: str) -> bool:
    """Whether ``quote`` plausibly grounds the answer, used to reject boilerplate
    the extractor sometimes returns. The quote must share a *number* with the
    answer, or at least two distinctive capitalised words — a single shared name
    (e.g. "Banca") is too common to count. If the answer has no salient tokens at
    all, nothing can be checked, so the quote is kept."""
    if not re.search(r"[^\W\d_]{3,}", quote):
        return False  # no real word — OCR code/number junk like "000004763n000"
    a_nums, a_caps = _salient(answer_text)
    if not a_nums and not a_caps:
        return True
    q_nums, q_caps = _salient(quote)
    if a_nums & q_nums:
        return True
    return len(a_caps & q_caps) >= 2


def _target_docs(answer: Answer, working_set: WorkingSet) -> list[Document]:
    """Documents to ground against: the ones L2 cited, else the whole set."""
    by_id = {d.id: d for d in working_set.documents}
    cited = [by_id[c.doc_id] for c in answer.citations if c.doc_id in by_id]
    return cited or list(working_set.documents)


async def _extract_quotes(
    chat_model: ChatModel, answer_text: str, doc_id: str, source_text: str
) -> list[str]:
    user = ChatMessage(
        role="user",
        content=f"ANSWER:\n{answer_text}\n\nSOURCE ({doc_id}):\n{source_text}",
    )
    try:
        reply = await chat_model.complete([_SYSTEM, user], max_tokens=1024, temperature=0.0)
    except Exception as exc:  # grounding is best-effort; never break the answer
        logger.warning("Quote extraction failed for %s: %s", doc_id, exc)
        return []
    return _parse_quotes(reply)


def _parse_quotes(raw: str) -> list[str]:
    """Parse a JSON array of strings out of the model reply (tolerating code
    fences / surrounding prose by grabbing the first ``[...]`` block)."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    quotes: list[str] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str) and item.strip():
            quotes.append(item)
        elif isinstance(item, dict):
            value = item.get("quote") or item.get("text")
            if value:
                quotes.append(str(value))
    return quotes


def _normalize(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, returning the normalised text
    and a map from each normalised index back to the original index."""
    chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            index_map.append(i)
            prev_space = True
        else:
            chars.append(ch)
            index_map.append(i)
            prev_space = False
    return "".join(chars), index_map


def locate(
    source: str, quote: str, *, min_ratio: float = 0.7, fuzzy: bool = True
) -> tuple[int, int] | None:
    """Find ``quote`` in ``source``, returning a ``(start, end)`` char span or
    ``None``. Tries exact, then whitespace-normalised, then (unless ``fuzzy`` is
    False) fuzzy matching — OCR text rarely matches a model's quote byte-for-byte.
    Pass ``fuzzy=False`` for a high-confidence-only match."""
    quote = quote.strip()
    if not quote:
        return None

    exact = source.find(quote)
    if exact >= 0:
        return (exact, exact + len(quote))

    norm_src, index_map = _normalize(source)
    norm_q = " ".join(quote.split())
    if not norm_q:
        return None

    hit = norm_src.find(norm_q)
    if hit >= 0:
        return (index_map[hit], index_map[hit + len(norm_q) - 1] + 1)

    if not fuzzy:
        return None

    # Fuzzy: the quote's matching blocks within the source. Tolerates a few
    # inserted/dropped characters (a stray newline glued into a word, etc.).
    matcher = difflib.SequenceMatcher(None, norm_src, norm_q, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    if matched / len(norm_q) < min_ratio:
        return None
    start_norm = blocks[0].a
    end_norm = min(blocks[-1].a + blocks[-1].size, len(index_map))
    # Reject scattered matches: a genuine hit spans about the quote's length, not
    # half the document. Without this, a quote that isn't really present matches
    # its common words all over and resolves to a giant, wrong span.
    if end_norm - start_norm > len(norm_q) * 2 + 16:
        return None
    return (index_map[start_norm], index_map[end_norm - 1] + 1)
