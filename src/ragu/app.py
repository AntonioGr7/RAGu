"""The ``Ragu`` facade — the high-level entry point users interact with.

Wires the L1 components from settings and exposes a small surface: index files
or documents, and retrieve a working set for a query. L2 reasoning will hang off
this same object once the engine adapter lands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ragu.adapters.ingestion import (
    dump_documents,
    file_fingerprint,
    iter_files,
    load_files,
    load_paths,
)
from ragu.adapters.ingestion.ocr import OcrEngine
from ragu.config import RaguSettings
from ragu.core import (
    Answer,
    Document,
    DocumentId,
    DocumentRef,
    IndexReport,
    Query,
    ScoredDocument,
    WorkingSet,
)
from ragu.factory import (
    build_chat_model,
    build_chunker,
    build_document_store,
    build_embedder,
    build_ocr_engine,
    build_reasoning_engine,
    build_retriever,
    build_vector_store,
    build_token_counter,
)
from ragu.pipeline import Indexer, build_working_set
from ragu.pipeline.grounding import ground_answer
from ragu.ports import ChatModel, Embedder, ReasoningEngine, Retriever, VectorStore


def _is_under(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` itself or nested within it."""
    return path == root or path.is_relative_to(root)


class Ragu:
    def __init__(self, settings: RaguSettings | None = None) -> None:
        self._settings = settings or RaguSettings()
        self._counter = build_token_counter(self._settings)
        # The document store needs no embedding dimension, so it is built eagerly
        # and cheaply — read-only commands (list/show/get) never pay to load the
        # embedder. Everything that depends on the embedder is built lazily below.
        self._document_store = build_document_store(self._settings)
        self._embedder: Embedder | None = None
        self._vector_store: VectorStore | None = None
        self._indexer: Indexer | None = None
        self._retriever: Retriever | None = None
        self._chat_model: ChatModel | None = None
        self._ocr: OcrEngine | None = None
        self._reasoning: ReasoningEngine | None = None  # built lazily (needs vomero)
        # The full-corpus working set (the L1-skip path) is the same every turn
        # and every session, so it is loaded from the store and assembled once,
        # then reused. Invalidated whenever indexing changes the corpus.
        self._full_corpus_ws: WorkingSet | None = None

    def _ensure_chat_model(self) -> ChatModel:
        if self._chat_model is None:
            self._chat_model = build_chat_model(self._settings)
        return self._chat_model

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self._settings)
        return self._embedder

    def _ensure_vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = build_vector_store(
                self._settings, self._ensure_embedder().dim
            )
        return self._vector_store

    def _ensure_indexer(self) -> Indexer:
        if self._indexer is None:
            self._indexer = Indexer(
                build_chunker(self._settings, self._counter),
                self._ensure_embedder(),
                self._ensure_vector_store(),
                self._document_store,
                embed_batch_size=self._settings.embedding.batch_size,
            )
        return self._indexer

    def _ensure_retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = build_retriever(
                self._settings, self._ensure_embedder(), self._ensure_vector_store()
            )
        return self._retriever

    def _ensure_ocr(self) -> OcrEngine | None:
        if self._ocr is None:
            self._ocr = build_ocr_engine(self._settings)
        return self._ocr

    async def index_paths(
        self, paths: list[str | Path], *, prune: bool = True, progress: bool = False
    ) -> IndexReport:
        """Incrementally index files/folders (text + OCR'd images/PDFs).

        Each file is fingerprinted (a hash of its bytes) and compared against
        what is already stored: unchanged files are skipped *before* OCR, new and
        changed files are (re)loaded and indexed in place. When ``prune`` is set,
        documents whose source file has disappeared from the indexed directories
        are removed from both stores. Returns an :class:`IndexReport`."""
        stored = {ref.id: ref for ref in await self._document_store.fingerprints()}

        to_load: list[tuple[Path, Path, str | None]] = []
        skipped = 0
        for file, base in iter_files(paths):
            doc_id = DocumentId(file.relative_to(base).as_posix())
            digest = file_fingerprint(file)
            ref = stored.get(doc_id)
            if ref is not None and ref.content_hash == digest:
                skipped += 1
                continue
            to_load.append((file, base, digest))

        cfg = self._settings.ocr
        # Build the (expensive) OCR engine only when there is something to load —
        # a re-run where everything is unchanged must not pay to spin up PaddleOCR.
        documents = (
            load_files(
                to_load,
                ocr=self._ensure_ocr(),
                pdf_mode=cfg.pdf_mode,
                pdf_dpi=cfg.pdf_dpi,
                pdf_min_text_chars=cfg.pdf_min_text_chars,
                progress=progress,
            )
            if to_load
            else []
        )
        updated = sum(1 for d in documents if d.id in stored)
        chunks = await self.index_documents(documents)

        pruned = await self._prune(paths, stored.values()) if prune else 0
        return IndexReport(
            chunks=chunks,
            new=len(documents) - updated,
            updated=updated,
            skipped=skipped,
            pruned=pruned,
        )

    async def _prune(self, paths: list[str | Path], stored) -> int:
        """Remove stored documents whose source file no longer exists, limited to
        documents that live under one of the indexed *directory* roots (so
        indexing one folder never prunes another)."""
        roots = [Path(p).resolve() for p in paths if Path(p).is_dir()]
        if not roots:
            return 0
        dead = [
            ref.id
            for ref in stored
            if any(_is_under(Path(ref.source), root) for root in roots)
            and not Path(ref.source).exists()
        ]
        if dead:
            await self._ensure_vector_store().delete(dead)
            await self._document_store.delete(dead)
            self._full_corpus_ws = None  # corpus shrank — drop the cached set
        return len(dead)

    async def index_documents(self, documents: list[Document]) -> int:
        if documents:
            self._full_corpus_ws = None  # corpus changed — drop the cached set
        return await self._ensure_indexer().index(documents)

    def extract_paths(self, paths: list[str | Path], out_dir: str | Path) -> int:
        """Load files/folders (text + OCR'd images/PDFs) and dump the extracted
        text to ``out_dir``, mirroring the source structure. Inspection tool —
        does not index. Returns the number of files written."""
        return dump_documents(self._load(paths), out_dir)

    def _load(self, paths: list[str | Path]) -> list[Document]:
        cfg = self._settings.ocr
        return load_paths(
            paths,
            ocr=self._ensure_ocr(),
            pdf_mode=cfg.pdf_mode,
            pdf_dpi=cfg.pdf_dpi,
            pdf_min_text_chars=cfg.pdf_min_text_chars,
        )

    async def list_documents(self) -> list[DocumentRef]:
        """List every indexed document as a lightweight ref (id, source,
        content_hash) — no content loaded, so it stays cheap on large corpora.
        Use :meth:`get_document` to pull full bodies for the ids you want."""
        return await self._document_store.fingerprints()

    async def get_document(self, doc_id: str | DocumentId) -> Document | None:
        """Fetch one indexed document in full (content + metadata + artifacts),
        or ``None`` if no document has that id."""
        docs = await self._document_store.get([DocumentId(str(doc_id))])
        return docs[0] if docs else None

    async def get_documents(self, doc_ids: list[str | DocumentId]) -> list[Document]:
        """Fetch several indexed documents in full, preserving the requested
        order; ids with no stored document are skipped."""
        return await self._document_store.get([DocumentId(str(i)) for i in doc_ids])

    @property
    def full_corpus_default(self) -> bool:
        """Whether L1 is skipped by default — L2 reasons over the whole corpus
        (``RAGU_VOMERO__FULL_CORPUS``). Lets the web server pick the retrieval
        path before it builds the working set."""
        return self._settings.vomero.full_corpus

    async def retrieve(
        self, text: str, *, prior: WorkingSet | None = None, **query_kwargs: str
    ) -> WorkingSet:
        """Run L1 and assemble the token-bounded working set for the query.

        When ``prior`` is given (a previous turn's working set in the same
        session), its documents are carried forward and merged with this query's
        hits — so a follow-up keeps the evidence that grounded earlier answers
        instead of retrieving cold from the latest message alone."""
        return await self._working_set(Query(text=text, **query_kwargs), prior=prior)

    async def answer(
        self,
        text: str,
        *,
        ground: bool | None = None,
        grounding_source: str | None = None,
        full_corpus: bool | None = None,
        working_set: WorkingSet | None = None,
        **query_kwargs: str,
    ) -> Answer:
        """Full pipeline: L1 selects a working set, L2 (vomero) reasons over it.

        When ``full_corpus`` is set (defaults to ``RAGU_VOMERO__FULL_CORPUS``,
        on), L1 retrieval is skipped and L2 reasons over *every* indexed document
        — the embedder/vector store are never touched. Only sensible with
        ``handoff="corpus"`` (L2 navigates the docs as files); inlining the whole
        corpus into the prompt (``handoff="context"``) would overflow the context
        window, so that combination is rejected. Pass ``full_corpus=False`` to
        force the L1+L2 pipeline regardless of config.

        When ``ground`` is true (defaults to ``RAGU_VOMERO__GROUND_CITATIONS``),
        an extra LLM pass replaces the document-level citations with span-level
        ones anchored to page + word boxes. ``grounding_source`` (defaults to
        ``RAGU_VOMERO__GROUNDING_SOURCE``) chooses whether quotes are drawn from
        what L2 actually read (``"trajectory"``) or the full document
        (``"document"``). Leave grounding off for the fastest path."""
        query = Query(text=text, **query_kwargs)
        if full_corpus is None:
            full_corpus = self._settings.vomero.full_corpus
        # A caller (the web server) may pass a pre-built working set — e.g. one
        # assembled with conversation-aware retrieval and carried across turns.
        # Otherwise build it here from this query alone.
        if working_set is None:
            working_set = (
                await self.full_working_set() if full_corpus else await self._working_set(query)
            )
        answer = await self._ensure_reasoning().reason(query, working_set)
        if ground is None:
            ground = self._settings.vomero.ground_citations
        if ground:
            source = grounding_source or self._settings.vomero.grounding_source
            # "raw" needs no LLM (and no API key) — don't build the chat model.
            chat = None if source == "raw" else self._ensure_chat_model()
            answer = await ground_answer(answer, working_set, chat, source=source)
        # Never hand the (potentially large) transient evidence back to callers.
        if answer.evidence or answer.evidence_spans:
            return answer.model_copy(update={"evidence": (), "evidence_spans": ()})
        return answer

    async def _working_set(
        self, query: Query, *, prior: WorkingSet | None = None
    ) -> WorkingSet:
        result = await self._ensure_retriever().retrieve(
            query, k=self._settings.retrieval.candidate_k
        )
        ranked = result.to_documents()
        if prior is not None and prior.documents:
            # Accumulate across turns: this query's fresh hits lead (so the
            # document-count cap can't evict the new hop's evidence), then carry
            # forward prior-turn documents that weren't re-retrieved. A follow-up
            # thus never loses the evidence that grounded the earlier answer.
            have = {d.doc_id for d in ranked}
            ranked = ranked + [
                ScoredDocument(doc_id=did, score=0.0, hits=())
                for did in prior.doc_ids
                if did not in have
            ]
        # In corpus handoff L2 navigates the docs as files, so the token budget
        # (a context-window concern) doesn't apply — bound by document count only.
        token_budget = (
            None
            if self._settings.vomero.handoff == "corpus"
            else self._settings.working_set.max_tokens
        )
        return await build_working_set(
            ranked,
            self._document_store,
            self._counter,
            max_tokens=token_budget,
            max_documents=self._settings.retrieval.document_k,
        )

    async def full_working_set(self) -> WorkingSet:
        """Every indexed document as the working set — the L1-skip path.

        Reasoning over the full corpus only works when L2 navigates the docs as
        files; inlining them all into the prompt would blow the context window."""
        if self._settings.vomero.handoff != "corpus":
            raise ValueError(
                "full_corpus requires vomero.handoff='corpus' (the whole corpus "
                f"cannot be inlined into the prompt); got {self._settings.vomero.handoff!r}"
            )
        if self._full_corpus_ws is None:
            refs = await self._document_store.fingerprints()
            docs = await self._document_store.get([ref.id for ref in refs])
            # Corpus handoff lets L2 navigate the docs as files, so the token
            # count never bounds anything here — skip the full-corpus tiktoken
            # pass (it would re-tokenize every document on every turn for a
            # number nothing reads in this path).
            self._full_corpus_ws = WorkingSet(
                documents=tuple(docs), token_count=0, truncated=False
            )
        return self._full_corpus_ws

    async def warmup(self) -> str | None:
        """Prepare L2 to answer the first question fast: in full-corpus mode,
        assemble the corpus working set and build/open L2's persistent search
        index now (at server startup) rather than on the first user's request.

        Returns a human-readable status for the boot log, or ``None`` when there
        is nothing to warm (L1+L2 mode reasons over small per-query sets, so the
        cost isn't worth paying up front)."""
        if not self.full_corpus_default:
            return None
        ws = await self.full_working_set()
        warm = getattr(self._ensure_reasoning(), "warmup", None)
        if warm is None:
            return None
        return await asyncio.to_thread(warm, ws)

    def _ensure_reasoning(self) -> ReasoningEngine:
        if self._reasoning is None:
            self._reasoning = build_reasoning_engine(self._settings)
        return self._reasoning
