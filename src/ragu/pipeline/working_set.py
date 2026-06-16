"""Working-set assembly: turn ranked documents into the bounded set for L2.

How the set is bounded depends on how L2 receives it. When L2 navigates the
documents as files (``handoff="corpus"``), the bound is purely ``max_documents``
— L2 greps/reads on demand, so the inlined-text volume is irrelevant. When the
documents are inlined into the prompt (``handoff="context"``), the real cost is
text volume, so ``max_tokens`` caps it: documents are added best-first until the
next would overflow the budget, the rest are dropped, and ``truncated`` is set
so callers (and eval) can see coverage was capped rather than silently complete.
"""

from __future__ import annotations

from ragu.core import ScoredDocument, WorkingSet
from ragu.ports import DocumentStore, TokenCounter


async def build_working_set(
    ranked: list[ScoredDocument],
    document_store: DocumentStore,
    token_counter: TokenCounter,
    *,
    max_tokens: int | None,
    max_documents: int | None = None,
) -> WorkingSet:
    candidates = ranked[:max_documents] if max_documents else ranked
    docs = await document_store.get([d.doc_id for d in candidates])

    selected = []
    total = 0
    truncated = len(ranked) > len(candidates)  # docs dropped by the count cap
    for doc in docs:
        cost = token_counter.count(doc.content)
        # max_tokens=None -> bound by document count only (corpus handoff).
        if max_tokens is not None and selected and total + cost > max_tokens:
            truncated = True
            continue  # try smaller later docs rather than stopping outright
        selected.append(doc)
        total += cost

    return WorkingSet(documents=tuple(selected), token_count=total, truncated=truncated)
