import { useState } from "react";
import { GroundingSource } from "../api";

interface Props {
  onSubmit: (query: string) => void;
  loading: boolean;
  grounding: GroundingSource;
  onGroundingChange: (g: GroundingSource) => void;
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

export function QueryBar({ onSubmit, loading, grounding, onGroundingChange }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    if (text.trim() && !loading) onSubmit(text.trim());
  };

  return (
    <div className="querybar">
      <div className="input-row">
        <textarea
          className="query-input"
          placeholder="Ask anything about the indexed documents…"
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
        <button className="ask-btn" onClick={submit} disabled={loading || !text.trim()}>
          {loading ? <span className="dots" /> : "Ask"}
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
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="examples">
        <span className="ctrl-label">try</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} className="example" onClick={() => setText(ex)} disabled={loading}>
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
