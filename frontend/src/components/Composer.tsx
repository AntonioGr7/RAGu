import { useState } from "react";
import { GroundingSource } from "../api";

interface Props {
  onSubmit: (text: string) => void;
  pending: boolean;
  awaitingQuestion: boolean;
  grounding: GroundingSource;
  onGroundingChange: (g: GroundingSource) => void;
  fullCorpus: boolean;
  onFullCorpusChange: (v: boolean) => void;
  showExamples: boolean;
}

const EXAMPLES = [
  "Qual è l'importo del mutuo di Banelli Alice e il tasso di interesse?",
  "Ci sono mutui dove le parti sono nate in Jugoslavia?",
  "Quali sono tutte le condizioni economiche del mutuo di Banelli Alice?",
];

const SOURCES: { id: GroundingSource; label: string; hint: string }[] = [
  { id: "trajectory", label: "trajectory", hint: "quotes from what L2 actually read" },
  { id: "document", label: "document", hint: "quotes from the full cited documents" },
  { id: "raw", label: "raw", hint: "no extra LLM — the lines L2 read" },
];

export function Composer({
  onSubmit,
  pending,
  awaitingQuestion,
  grounding,
  onGroundingChange,
  fullCorpus,
  onFullCorpusChange,
  showExamples,
}: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    const t = text.trim();
    if (t && !pending) {
      onSubmit(t);
      setText("");
    }
  };

  return (
    <div className={`composer ${awaitingQuestion ? "composer-answering" : ""}`}>
      {showExamples && (
        <div className="examples">
          <span className="ctrl-label">try</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="example"
              onClick={() => setText(ex)}
              disabled={pending}
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      <div className="composer-box">
        <textarea
          className="query-input"
          placeholder={
            awaitingQuestion
              ? "Answer L2's question…"
              : "Ask anything about the indexed documents…"
          }
          value={text}
          rows={2}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button className="ask-btn" onClick={submit} disabled={pending || !text.trim()}>
          {pending ? <span className="dots" /> : awaitingQuestion ? "Reply" : "Ask"}
        </button>
      </div>

      <div className="controls">
        <div className="grounding">
          <span className="ctrl-label">grounding</span>
          {SOURCES.map((s) => (
            <button
              key={s.id}
              title={s.hint}
              className={`pill ${grounding === s.id ? "pill-on" : ""}`}
              onClick={() => onGroundingChange(s.id)}
              disabled={awaitingQuestion}
            >
              {s.label}
            </button>
          ))}
        </div>
        <button
          className={`pill ${!fullCorpus ? "pill-on" : ""}`}
          title="Pre-filter the corpus with L1 retrieval (hybrid dense + BM25) before reasoning — for simple sets that don't need full agentic navigation. Off (default): the engine reasons over the whole corpus."
          onClick={() => onFullCorpusChange(!fullCorpus)}
          disabled={awaitingQuestion}
        >
          L1 filter
        </button>
      </div>
    </div>
  );
}
