import { useEffect, useRef } from "react";
import { AnswerTurn } from "../api";
import { TraceBox } from "./TraceBox";

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "question"; text: string }
  | { id: string; role: "assistant"; turn: AnswerTurn; trace?: string[] };

interface Props {
  messages: ChatMessage[];
  pending: boolean;
  liveTrace: string[];
  activeId: string | null;
  onFocus: (id: string) => void;
}

// Compact reasoning stats shown under an assistant answer.
function TurnStats({ turn }: { turn: AnswerTurn }) {
  const t = turn.trace;
  const chips: string[] = [
    t.engine ?? (turn.used_reasoning ? "L2" : "L1"),
    `${(turn.elapsed_ms / 1000).toFixed(1)}s`,
  ];
  if (t.steps) chips.push(`${t.steps} steps`);
  // Budget L2 spent answering — the tokens/calls vomero reports back.
  const budget =
    t.tokens &&
    `${Number(t.tokens).toLocaleString()} tok${
      t.calls ? ` · ${t.calls} calls` : ""
    }`;
  return (
    <div className="msg-stats">
      {chips.map((c, i) => (
        <span className="msg-chip" key={i}>
          {c}
        </span>
      ))}
      {budget && (
        <span className="msg-chip budget" title="Budget L2 spent on this answer">
          ⛽ {budget}
        </span>
      )}
      {turn.citations.length > 0 && (
        <span className="msg-chip accent">
          {turn.citations.length} reference{turn.citations.length === 1 ? "" : "s"} →
        </span>
      )}
    </div>
  );
}

export function ChatThread({ messages, pending, liveTrace, activeId, onFocus }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  return (
    <div className="thread">
      {messages.map((m) => {
        if (m.role === "user") {
          return (
            <div className="msg msg-user" key={m.id}>
              <div className="bubble">{m.text}</div>
            </div>
          );
        }
        if (m.role === "question") {
          return (
            <div className="msg msg-question" key={m.id}>
              <div className="msg-avatar">?</div>
              <div className="bubble">
                <div className="q-label">L2 needs your input</div>
                {m.text}
              </div>
            </div>
          );
        }
        const active = m.id === activeId;
        return (
          <div className="msg msg-assistant" key={m.id}>
            <div className="msg-avatar">A</div>
            <div className="assistant-col">
              <div
                className={`bubble assistant-bubble ${active ? "bubble-active" : ""}`}
                onClick={() => onFocus(m.id)}
                title="Show this answer's references"
              >
                <div className="answer-text">{m.turn.answer}</div>
                <TurnStats turn={m.turn} />
              </div>
              {m.trace && m.trace.length > 0 && <TraceBox lines={m.trace} />}
            </div>
          </div>
        );
      })}

      {pending && (
        <div className="msg msg-assistant">
          <div className="msg-avatar">A</div>
          <div className="assistant-col">
            <div className="bubble assistant-bubble">
              <div className="typing">
                <span />
                <span />
                <span />
              </div>
            </div>
            <TraceBox lines={liveTrace} live />
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  );
}
