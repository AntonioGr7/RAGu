import { Citation, QueryResult } from "../api";
import { CitationCard } from "./CitationCard";

interface Props {
  result: QueryResult;
  onOpenCitation: (c: Citation) => void;
}

export function AnswerPanel({ result, onOpenCitation }: Props) {
  return (
    <div className="answer-wrap">
      <div className="panel answer-panel">
        <div className="panel-title">Answer</div>
        <div className="answer-text">{result.answer}</div>
      </div>

      <div className="panel">
        <div className="panel-title">
          Citations <span className="panel-count">{result.citations.length}</span>
          <span className="panel-sub">click a card to see it on the page</span>
        </div>
        <div className="cite-grid">
          {result.citations.length === 0 && (
            <div className="page-empty">No citations returned for this answer.</div>
          )}
          {result.citations.map((c, i) => (
            <CitationCard
              key={`${c.doc_id}-${i}`}
              citation={c}
              index={i}
              onOpen={() => onOpenCitation(c)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
