import { useEffect, useRef, useState } from "react";
import {
  Citation,
  fetchTrace,
  GroundingSource,
  listDocuments,
  resetSession,
  respond,
  sendMessage,
} from "./api";
import { ChatThread, ChatMessage } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { PageViewer } from "./components/PageViewer";
import { ReferencesPanel } from "./components/ReferencesPanel";

function newSessionId(): string {
  return crypto.randomUUID?.() ?? `s-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

let msgSeq = 0;
const nextId = () => `m${++msgSeq}`;

export function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [grounding, setGrounding] = useState<GroundingSource>("trajectory");
  // L1 off by default: L2 reasons over the whole corpus (matches the server's
  // RAGU_VOMERO__FULL_CORPUS default). Uncheck to use the L1+L2 retrieval pipeline.
  const [fullCorpus, setFullCorpus] = useState(true);
  const [pending, setPending] = useState(false);
  // True while L2 is waiting on the user — the composer answers the question.
  const [awaitingQuestion, setAwaitingQuestion] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [docCount, setDocCount] = useState<number | null>(null);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  // The answer message whose references the right panel shows (defaults to the
  // latest). Tracked by message id so the focused bubble can be highlighted.
  const [activeId, setActiveId] = useState<string | null>(null);
  // L2's reasoning log for the in-flight turn, refreshed by polling. Mirrored in
  // a ref so it can be snapshotted if the answer arrives mid-poll.
  const [liveTrace, setLiveTrace] = useState<string[]>([]);
  const liveTraceRef = useRef<string[]>([]);
  const pollingRef = useRef(false);
  const cursorRef = useRef(0);

  useEffect(() => {
    listDocuments()
      .then((d) => setDocCount(d.length))
      .catch(() => setDocCount(null));
    return () => {
      pollingRef.current = false;
    };
  }, []);

  const stopPolling = () => {
    pollingRef.current = false;
  };

  // Poll the reasoning log while a turn is in flight. Lightweight short requests
  // (no persistent connection) — survives the ask-back pause until `finished`.
  const startPolling = () => {
    liveTraceRef.current = [];
    setLiveTrace([]);
    cursorRef.current = 0;
    pollingRef.current = true;
    const tick = async () => {
      if (!pollingRef.current) return;
      try {
        const d = await fetchTrace(sessionId, cursorRef.current);
        if (d.lines.length) {
          cursorRef.current = d.next;
          liveTraceRef.current = [...liveTraceRef.current, ...d.lines];
          setLiveTrace(liveTraceRef.current);
        }
        if (d.finished) {
          pollingRef.current = false;
          return;
        }
      } catch {
        /* transient — keep polling */
      }
      if (pollingRef.current) setTimeout(tick, 400);
    };
    void tick();
  };

  const push = (m: ChatMessage) => setMessages((prev) => [...prev, m]);

  // Fold a server turn (answer or another question) into the thread.
  const consume = (turn: Awaited<ReturnType<typeof sendMessage>>) => {
    if (turn.type === "question") {
      push({ id: nextId(), role: "question", text: turn.question });
      setAwaitingQuestion(true);
    } else {
      const id = nextId();
      // The answer carries the authoritative full log; fall back to what we
      // polled if it's somehow absent.
      const log = turn.reasoning_log ?? liveTraceRef.current;
      push({ id, role: "assistant", turn, trace: [...log] });
      setActiveId(id);
      setAwaitingQuestion(false);
      stopPolling();
    }
  };

  const submit = async (text: string) => {
    setError(null);
    push({ id: nextId(), role: "user", text });
    setPending(true);
    const answering = awaitingQuestion;
    setAwaitingQuestion(false);
    if (!answering) startPolling();
    try {
      const turn = answering
        ? await respond(sessionId, text)
        : await sendMessage(sessionId, text, grounding, fullCorpus);
      consume(turn);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      stopPolling();
    } finally {
      setPending(false);
    }
  };

  const newChat = () => {
    stopPolling();
    resetSession(sessionId);
    setSessionId(newSessionId());
    setMessages([]);
    setActiveId(null);
    setAwaitingQuestion(false);
    setPending(false);
    setError(null);
    setLiveTrace([]);
    liveTraceRef.current = [];
  };

  const activeMsg = messages.find((m) => m.id === activeId);
  const activeTurn = activeMsg?.role === "assistant" ? activeMsg.turn : null;
  const started = messages.length > 0;

  return (
    <div className="shell">
      <main className="chat-col">
        <header className="masthead">
          <div className="brand">
            <span className="logo">RAGu</span>
            <span className="tagline">two-level RAG · answers grounded on the page</span>
          </div>
          <div className="masthead-right">
            {docCount !== null && (
              <span className="corpus-badge">{docCount} documents indexed</span>
            )}
            <button className="new-chat" onClick={newChat} disabled={pending}>
              + New chat
            </button>
          </div>
        </header>

        {!started && !pending && (
          <div className="empty-hero">
            <div className="hero-glow" />
            <h1>Ask. Get a grounded answer.</h1>
            <p>
              L2 can ask you back when it needs a detail — the conversation stays
              open until you start a new chat. Every claim is traced to the exact
              words on the source page.
            </p>
          </div>
        )}

        <ChatThread
          messages={messages}
          pending={pending}
          liveTrace={liveTrace}
          activeId={activeId}
          onFocus={setActiveId}
        />

        {error && <div className="error">{error}</div>}

        <Composer
          onSubmit={submit}
          pending={pending}
          awaitingQuestion={awaitingQuestion}
          grounding={grounding}
          onGroundingChange={setGrounding}
          fullCorpus={fullCorpus}
          onFullCorpusChange={setFullCorpus}
          showExamples={!started}
        />
      </main>

      <ReferencesPanel turn={activeTurn} onOpenCitation={setOpenCitation} />

      {openCitation && (
        <PageViewer citation={openCitation} onClose={() => setOpenCitation(null)} />
      )}
    </div>
  );
}
