"""MuSiQue evaluation harness for RAGu — global-corpus, multi-hop regime.

Two steps:

    # 1. build ONE index from every unique paragraph in the dev set (21k docs),
    #    each paragraph = a Document (title + text). One-time, ~minutes.
    uv run python eval/musique.py index

    # 2. sample N dev questions (stratified by hop count), answer each against
    #    the global index, score EM / F1 / contains and support-recall.
    uv run python eval/musique.py run --n 100 --seed 0 --concurrency 4

The hard part this exercises: L1 must pull the right supporting paragraphs out
of ~21k before L2 can chain the hops. MuSiQue answers are short spans, so we
append a "give only the answer" instruction (L2 otherwise returns prose) and also
report a lenient ``contains`` metric. ``support_recall`` is measured directly on
the working set L1 hands to L2 — the retrieval ceiling for the answer.

Data: ``bdsaglam/musique`` (config ``answerable``, the official MuSiQue-Ans dev).
Index is isolated under ``./.ragu/musique`` and contextual chunking is disabled
(both via env defaults below) so this never touches the demo index or fires an
LLM call per paragraph at build time.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

DATASET, CONFIG = "bdsaglam/musique", "answerable"
RESULTS_DIR = Path("eval/results")

# Appended to each question: MuSiQue is span-scored, but L2 answers in prose.
CONCISE = (
    "\n\nAnswer with ONLY the exact answer — a name, date, number, or short "
    "phrase — and nothing else. Do not explain or add a sentence."
)


# ── data ──────────────────────────────────────────────────────────────────────
def load_split(spec: str):  # type: ignore[no-untyped-def]
    """Load one or both splits as a flat list of examples.

    ``spec`` is ``"validation"``, ``"train"``, or ``"train+val"``. The index and
    the questions must come from the SAME spec or evidence is missing (the HF
    viewer defaults to train, which is why hand-picked train questions fail
    against a validation-only index)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG)
    names = {"validation": ["validation"], "train": ["train"], "train+val": ["train", "validation"]}[spec]
    rows: list = []
    for name in names:
        rows.extend(ds[name])
    return rows


def para_id(title: str, text: str) -> str:
    """Stable id for a paragraph so gold supporting paragraphs map to the same id
    we indexed under (dedup is by exact title+text)."""
    h = hashlib.sha1(f"{title}␟{text}".encode()).hexdigest()[:16]
    return f"para_{h}"


def is_supporting(p: dict) -> bool:
    return str(p["is_supporting"]).lower() == "true"


def hops_of(ex: dict) -> int:
    return len(ex["question_decomposition"])


# ── index ─────────────────────────────────────────────────────────────────────
def build_corpus_documents(dev, max_paragraphs: int | None):  # type: ignore[no-untyped-def]
    """Every unique paragraph across the dev set as a Document (title + text)."""
    from ragu.core import Document

    seen: dict[str, Document] = {}
    for ex in dev:
        for p in ex["paragraphs"]:
            title, text = p["title"], p["paragraph_text"]
            pid = para_id(title, text)
            if pid not in seen:
                seen[pid] = Document(
                    id=pid,
                    source=f"musique://{title}",
                    content=f"{title}\n\n{text}",
                    metadata={"title": title},
                )
        if max_paragraphs and len(seen) >= max_paragraphs:
            break
    return list(seen.values())


def chunked_doc_ids(uri: str) -> set[str]:
    """doc_ids that already have >=1 chunk in the vector store. These are the
    fully-indexed documents — anything else (never-seen, or a crash that wrote
    the document row but no chunks) still needs indexing. Empty if no index yet."""
    import lancedb

    path = Path(uri) / "chunks.lance"
    if not path.exists():
        return set()
    table = lancedb.connect(uri).open_table("chunks")
    return set(table.to_arrow().column("doc_id").to_pylist())


def orphan_doc_ids(uri: str, done: set[str]) -> list[str]:
    """Document rows with no chunks — written by index() before it crashed
    mid-batch. Harmless to read but unretrievable; we re-index them, which
    overwrites the stale row and gives it chunks."""
    import lancedb

    path = Path(uri) / "documents.lance"
    if not path.exists():
        return []
    table = lancedb.connect(uri).open_table("documents")
    return [i for i in table.to_arrow().column("id").to_pylist() if i not in done]


async def cmd_index(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from ragu.app import Ragu

    uri = os.environ["RAGU_STORAGE__LANCEDB_URI"]
    dev = load_split(args.split)
    docs = build_corpus_documents(dev, args.max_paragraphs)
    print(f"corpus: {len(docs)} unique paragraph-documents → {uri}")

    if not args.reindex:
        done = chunked_doc_ids(uri)
        if done:
            orphans = orphan_doc_ids(uri, done)
            docs = [d for d in docs if d.id not in done]
            print(
                f"resume: {len(done)} docs already chunked, "
                f"{len(orphans)} orphan doc rows to repair, "
                f"{len(docs)} docs left to index"
            )
        if not docs:
            print("nothing to do — corpus already fully indexed.")
            return

    ragu = Ragu()
    total, t0, batch = 0, time.perf_counter(), args.batch
    bar = tqdm(total=len(docs), unit="doc", desc="indexing", smoothing=0.05)
    for i in range(0, len(docs), batch):
        total += await ragu.index_documents(docs[i : i + batch])
        bar.update(min(batch, len(docs) - i))
        bar.set_postfix(chunks=total)
    bar.close()
    print(f"done: {total} chunks in {time.perf_counter() - t0:.0f}s")


# ── sampling ────────────────────────────────────────────────────────────────--
def stratified_sample(dev, n: int, seed: int) -> list[int]:  # type: ignore[no-untyped-def]
    """~equal counts per hop-stratum (2/3/4-hop), deterministic for a given seed."""
    by_hop: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dev)):
        by_hop[hops_of(dev[idx])].append(idx)

    rng = random.Random(seed)
    strata = sorted(by_hop)
    per = n // len(strata)
    chosen: list[int] = []
    leftover: list[int] = []
    for h in strata:
        pool = by_hop[h][:]
        rng.shuffle(pool)
        chosen += pool[:per]
        leftover += pool[per:]
    rng.shuffle(leftover)
    chosen += leftover[: n - len(chosen)]  # top up to exactly n
    rng.shuffle(chosen)
    return chosen


# ── scoring (SQuAD/MuSiQue normalisation) ──────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalize(s: str) -> str:
    return " ".join(_ARTICLES.sub(" ", s.lower().translate(_PUNCT)).split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    shared = sum((Counter(p) & Counter(g)).values())
    if shared == 0:
        return 0.0
    prec, rec = shared / len(p), shared / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, golds: list[str]) -> tuple[int, float, int]:
    npred = normalize(pred)
    em = max(int(npred == normalize(g)) for g in golds)
    f1 = max(token_f1(pred, g) for g in golds)
    contains = max(int(bool(normalize(g)) and normalize(g) in npred) for g in golds)
    return em, f1, contains


# ── run ─────────────────────────────────────────────────────────────────────--
def cmd_sample(args: argparse.Namespace) -> None:
    """Print random in-index (validation) questions with answers and supporting
    titles — so manual `main.py` tests use questions whose evidence is indexed.
    The HF viewer defaults to the TRAIN split, whose paragraphs are NOT indexed."""
    import random

    dev = load_split(args.split)
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(dev)), min(args.n, len(dev)))
    for i in picks:
        ex = dev[i]
        sup = [p["title"] for p in ex["paragraphs"] if is_supporting(p)]
        print(f"\n[{hops_of(ex)}hop] {ex['id']}  answer={ex['answer']!r}")
        print(f"  Q: {ex['question']}")
        print(f"  supporting: {sup}")


async def cmd_run(args: argparse.Namespace) -> None:
    from ragu.app import Ragu

    dev = load_split(args.split)
    idxs = stratified_sample(dev, args.n, args.seed)
    print(f"evaluating {len(idxs)} questions (seed={args.seed}, concurrency={args.concurrency})")
    print(f"hop mix: {dict(Counter(hops_of(dev[i]) for i in idxs))}")

    ragu = Ragu()
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def one(i: int) -> dict:
        ex = dev[i]
        q = ex["question"]
        golds = [ex["answer"], *(ex.get("answer_aliases") or [])]
        gold_support = {
            para_id(p["title"], p["paragraph_text"]) for p in ex["paragraphs"] if is_supporting(p)
        }
        async with sem:
            t0 = time.perf_counter()
            ws = await ragu.retrieve(q)
            ws_ids = {str(d) for d in ws.doc_ids}
            recall = len(gold_support & ws_ids) / len(gold_support) if gold_support else None
            ans = await ragu.answer(q + CONCISE, ground=False)
            dt = time.perf_counter() - t0
        em, f1, contains = score(ans.text, golds)
        return {
            "id": ex["id"],
            "hops": hops_of(ex),
            "question": q,
            "gold": ex["answer"],
            "pred": ans.text.strip(),
            "em": em,
            "f1": round(f1, 3),
            "contains": contains,
            "support_recall": recall,
            "ws_size": len(ws_ids),
            "elapsed_s": round(dt, 1),
        }

    tasks = [asyncio.create_task(one(i)) for i in idxs]
    for k, fut in enumerate(asyncio.as_completed(tasks), 1):
        r = await fut
        results.append(r)
        rec = "n/a" if r["support_recall"] is None else f"{r['support_recall']:.2f}"
        print(
            f"[{k:>3}/{len(idxs)}] {r['hops']}hop em={r['em']} f1={r['f1']:.2f} "
            f"rec={rec} {r['elapsed_s']:>4.0f}s | {r['gold'][:30]!r} ⇐ {r['pred'][:46]!r}"
        )

    _summarize_and_save(results, args)


def _summarize_and_save(results: list[dict], args: argparse.Namespace) -> None:
    def agg(rows: list[dict]) -> dict:
        recs = [r["support_recall"] for r in rows if r["support_recall"] is not None]
        return {
            "n": len(rows),
            "em": round(100 * sum(r["em"] for r in rows) / len(rows), 1),
            "f1": round(100 * sum(r["f1"] for r in rows) / len(rows), 1),
            "contains": round(100 * sum(r["contains"] for r in rows) / len(rows), 1),
            "support_recall": round(100 * sum(recs) / len(recs), 1) if recs else None,
            "avg_s": round(sum(r["elapsed_s"] for r in rows) / len(rows), 1),
        }

    by_hop = defaultdict(list)
    for r in results:
        by_hop[r["hops"]].append(r)

    print("\n" + "=" * 64)
    print(f"{'split':>8} | {'n':>3} | {'EM':>5} | {'F1':>5} | {'cont':>5} | {'rec@ws':>6} | {'s/q':>5}")
    print("-" * 64)
    for hop in sorted(by_hop):
        a = agg(by_hop[hop])
        rec = "n/a" if a["support_recall"] is None else f"{a['support_recall']:>5.1f}"
        print(f"{hop:>6}hop | {a['n']:>3} | {a['em']:>5} | {a['f1']:>5} | {a['contains']:>5} | {rec:>6} | {a['avg_s']:>5}")
    a = agg(results)
    rec = "n/a" if a["support_recall"] is None else f"{a['support_recall']:>5.1f}"
    print("-" * 64)
    print(f"{'ALL':>8} | {a['n']:>3} | {a['em']:>5} | {a['f1']:>5} | {a['contains']:>5} | {rec:>6} | {a['avg_s']:>5}")
    print("=" * 64)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"run_n{len(results)}_seed{args.seed}.json"
    out.write_text(
        json.dumps({"overall": agg(results), "results": results}, ensure_ascii=False, indent=2)
    )
    print(f"\nwrote {out}")


# ── cli ─────────────────────────────────────────────────────────────────────--
def main() -> None:
    parser = argparse.ArgumentParser(description="MuSiQue eval for RAGu (global corpus).")
    parser.add_argument(
        "--lancedb-uri",
        default="./.ragu/musique",
        help="isolated index for MuSiQue (overrides RAGU_STORAGE__LANCEDB_URI / .env)",
    )
    parser.add_argument(
        "--document-k",
        type=int,
        default=30,
        help="docs L1 puts in the working set (overrides RAGU_RETRIEVAL__DOCUMENT_K). "
        ".env's 1000 is tuned for the 11-doc demo and drowns L2 on the 21k corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    splits = ["validation", "train", "train+val"]

    p_index = sub.add_parser("index", help="build the global paragraph index")
    p_index.add_argument("--split", choices=splits, default="train+val",
                         help="which split(s) of paragraphs to index")
    p_index.add_argument("--batch", type=int, default=1000, help="docs per index call")
    p_index.add_argument("--max-paragraphs", type=int, default=None, help="cap corpus (smoke test)")
    p_index.add_argument("--reindex", action="store_true",
                         help="re-index the whole corpus from scratch instead of resuming "
                         "(default: skip docs that already have chunks)")

    p_run = sub.add_parser("run", help="evaluate a stratified sample of dev questions")
    p_run.add_argument("--split", choices=splits, default="validation",
                       help="which split to draw eval questions from (must be ⊆ indexed split)")
    p_run.add_argument("--n", type=int, default=100)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--concurrency", type=int, default=4)

    p_sample = sub.add_parser(
        "sample", help="print random in-index questions to test by hand"
    )
    p_sample.add_argument("--split", choices=splits, default="train+val",
                          help="must match the indexed split so evidence is present")
    p_sample.add_argument("--n", type=int, default=5)
    p_sample.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    load_dotenv()
    # FORCE an isolated index and skip per-chunk contextual LLM calls — override
    # .env (which points at the demo's test index) so MuSiQue never touches it.
    os.environ["RAGU_STORAGE__LANCEDB_URI"] = args.lancedb_uri
    os.environ["RAGU_CHUNKING__CONTEXTUAL"] = "false"
    os.environ["RAGU_RETRIEVAL__DOCUMENT_K"] = str(args.document_k)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers", "transformers", "datasets", "openai"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.command == "sample":
        cmd_sample(args)
    else:
        asyncio.run(cmd_index(args) if args.command == "index" else cmd_run(args))


if __name__ == "__main__":
    main()
