import { AnswerTurn, Citation, docName } from "../api";
import { CitationCard } from "./CitationCard";

interface Props {
  turn: AnswerTurn | null;
  onOpenCitation: (c: Citation) => void;
}

// The right rail: grounded references for the focused answer, then the L1
// working set beneath. Empty state before the first answer lands.
export function ReferencesPanel({ turn, onOpenCitation }: Props) {
  const cited = new Set(turn?.citations.map((c) => c.doc_id) ?? []);

  return (
    <aside className="refs-col">
      <div className="refs-head">References</div>

      {!turn ? (
        <div className="refs-empty">
          Grounded citations for the answer will appear here — click one to see the
          highlighted box on the source page.
        </div>
      ) : (
        <>
          <div className="refs-section">
            <div className="panel-title">
              Citations <span className="panel-count">{turn.citations.length}</span>
            </div>
            <div className="cite-grid">
              {turn.citations.length === 0 && (
                <div className="refs-empty">No citations returned for this answer.</div>
              )}
              {turn.citations.map((c, i) => (
                <CitationCard
                  key={`${c.doc_id}-${i}`}
                  citation={c}
                  index={i}
                  onOpen={() => onOpenCitation(c)}
                />
              ))}
            </div>
          </div>

          <div className="refs-section">
            <div className="panel-title">
              Working set <span className="panel-count">{turn.working_set_count}</span>
            </div>
            <div className="ws-list">
              {turn.working_set.map((d) => (
                <div
                  key={d.id}
                  className={`ws-doc ${cited.has(d.id) ? "ws-cited" : ""}`}
                >
                  <span className="ws-dot" />
                  {docName(d.source)}
                  {cited.has(d.id) && <span className="ws-tag">cited</span>}
                </div>
              ))}
              {turn.working_set_count > turn.working_set.length && (
                <div className="ws-doc ws-more">
                  … and {turn.working_set_count - turn.working_set.length} more
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
