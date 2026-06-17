import { useEffect, useRef, useState } from "react";
import {
  Citation,
  GroundingSource,
  listDocuments,
  resetSession,
  respond,
  sendMessage,
  traceStreamUrl,
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
  const [fullCorpus, setFullCorpus] = useState(false);
  const [pending, setPending] = useState(false);
  // True while L2 is waiting on the user — the composer answers the question.
  const [awaitingQuestion, setAwaitingQuestion] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [docCount, setDocCount] = useState<number | null>(null);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  // The answer message whose references the right panel shows (defaults to the
  // latest). Tracked by message id so the focused bubble can be highlighted.
  const [activeId, setActiveId] = useState<string | null>(null);
  // L2's reasoning log for the in-flight turn, streamed over SSE. Mirrored in a
  // ref so it can be snapshotted onto the answer message when the turn ends.
  const [liveTrace, setLiveTrace] = useState<string[]>([]);
  const liveTraceRef = useRef<string[]>([]);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    listDocuments()
      .then((d) => setDocCount(d.length))
      .catch(() => setDocCount(null));
    return () => esRef.current?.close();
  }, []);

  const closeStream = () => {
    esRef.current?.close();
    esRef.current = null;
  };

  // Open a fresh trace stream for a new turn (a continuation/respond keeps the
  // existing one — the same turn's log keeps growing on the server).
  const openStream = () => {
    closeStream();
    liveTraceRef.current = [];
    setLiveTrace([]);
    const es = new EventSource(traceStreamUrl(sessionId));
    es.onmessage = (e) => {
      try {
        const { line } = JSON.parse(e.data);
        liveTraceRef.current = [...liveTraceRef.current, line];
        setLiveTrace(liveTraceRef.current);
      } catch {
        /* ignore malformed frame */
      }
    };
    es.addEventListener("end", closeStream);
    es.onerror = closeStream;
    esRef.current = es;
  };

  const push = (m: ChatMessage) => setMessages((prev) => [...prev, m]);

  // Fold a server turn (answer or another question) into the thread.
  const consume = (turn: Awaited<ReturnType<typeof sendMessage>>) => {
    if (turn.type === "question") {
      push({ id: nextId(), role: "question", text: turn.question });
      setAwaitingQuestion(true);
    } else {
      const id = nextId();
      // Snapshot the reasoning log onto the answer so it stays inspectable.
      push({ id, role: "assistant", turn, trace: [...liveTraceRef.current] });
      setActiveId(id);
      setAwaitingQuestion(false);
      closeStream();
    }
  };

  const submit = async (text: string) => {
    setError(null);
    push({ id: nextId(), role: "user", text });
    setPending(true);
    const answering = awaitingQuestion;
    setAwaitingQuestion(false);
    if (!answering) openStream();
    try {
      const turn = answering
        ? await respond(sessionId, text)
        : await sendMessage(sessionId, text, grounding, fullCorpus);
      consume(turn);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      closeStream();
    } finally {
      setPending(false);
    }
  };

  const newChat = () => {
    closeStream();
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
