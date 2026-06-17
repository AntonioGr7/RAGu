import { useEffect, useState } from "react";
import { Citation, GroundingSource, listDocuments, QueryResult, runQuery } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { PageViewer } from "./components/PageViewer";
import { QueryBar } from "./components/QueryBar";
import { TracePanel } from "./components/TracePanel";
import { WorkingSet } from "./components/WorkingSet";

export function App() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [grounding, setGrounding] = useState<GroundingSource>("trajectory");
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const [docCount, setDocCount] = useState<number | null>(null);
  const [lastQuery, setLastQuery] = useState("");

  useEffect(() => {
    listDocuments()
      .then((d) => setDocCount(d.length))
      .catch(() => setDocCount(null));
  }, []);

  const ask = async (query: string) => {
    setLoading(true);
    setError(null);
    setLastQuery(query);
    try {
      setResult(await runQuery(query, grounding));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <span className="logo">RAGu</span>
          <span className="tagline">two-level RAG · answers grounded on the page</span>
        </div>
        {docCount !== null && (
          <div className="corpus-badge">{docCount} documents indexed</div>
        )}
      </header>

      <QueryBar
        onSubmit={ask}
        loading={loading}
        grounding={grounding}
        onGroundingChange={setGrounding}
      />

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading-card">
          <div className="pulse-ring" />
          <div>
            <div className="loading-title">Reasoning over the corpus…</div>
            <div className="loading-sub">“{lastQuery}”</div>
            <div className="loading-steps">
              L1 retrieves a working set → L2 navigates the documents → citations are
              grounded to word boxes
            </div>
          </div>
        </div>
      )}

      {result && !loading && (
        <>
          <TracePanel result={result} />
          <div className="layout">
            <AnswerPanel result={result} onOpenCitation={setOpenCitation} />
            <WorkingSet result={result} />
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div className="empty-hero">
          <div className="hero-glow" />
          <h1>Ask. Get a grounded answer.</h1>
          <p>
            Every claim is traced back to the exact words on the source page — click a
            citation to see the highlighted box.
          </p>
        </div>
      )}

      {openCitation && (
        <PageViewer citation={openCitation} onClose={() => setOpenCitation(null)} />
      )}
    </div>
  );
}
