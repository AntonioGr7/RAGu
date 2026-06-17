import { docName, QueryResult } from "../api";

// The L1 output: the documents retrieval selected and handed to L2. Documents
// that ended up grounding the answer are marked.
export function WorkingSet({ result }: { result: QueryResult }) {
  const cited = new Set(result.citations.map((c) => c.doc_id));
  return (
    <div className="panel">
      <div className="panel-title">
        Working set <span className="panel-count">{result.working_set.length}</span>
      </div>
      <div className="ws-list">
        {result.working_set.map((d) => (
          <div key={d.id} className={`ws-doc ${cited.has(d.id) ? "ws-cited" : ""}`}>
            <span className="ws-dot" />
            {docName(d.source)}
            {cited.has(d.id) && <span className="ws-tag">cited</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
