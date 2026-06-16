"""Working-set assembly: turn ranked documents into the bounded set for L2.

The budget is on tokens, not document count — what bounds L2 cost is how much
text it must navigate. Documents are added best-first until the next one would
overflow the budget; the rest are dropped and ``truncated`` is set so callers
(and eval) can see coverage was capped rather than silently complete.
"""

from __future__ import annotations

from ragu.core import ScoredDocument, WorkingSet
from ragu.ports import DocumentStore, TokenCounter


async def build_working_set(
    ranked: list[ScoredDocument],
    document_store: DocumentStore,
    token_counter: TokenCounter,
    *,
    max_tokens: int,
    max_documents: int | None = None,
) -> WorkingSet:
    candidates = ranked[:max_documents] if max_documents else ranked
    docs = await document_store.get([d.doc_id for d in candidates])

    selected = []
    total = 0
    truncated = False
    for doc in docs:
        cost = token_counter.count(doc.content)
        if selected and total + cost > max_tokens:
            truncated = True
            continue  # try smaller later docs rather than stopping outright
        selected.append(doc)
        total += cost

    return WorkingSet(documents=tuple(selected), token_count=total, truncated=truncated)
