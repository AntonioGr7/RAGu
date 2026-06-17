import { Citation, docName } from "../api";

interface Props {
  citation: Citation;
  index: number;
  onOpen: () => void;
}

// One grounded citation. Clicking opens the page viewer. Shows whether it carries
// page geometry (boxes) so the user knows there's something to see.
export function CitationCard({ citation, index, onOpen }: Props) {
  const totalBoxes = citation.highlights.reduce((n, h) => n + h.boxes.length, 0);
  const pages = citation.highlights.map((h) => h.page + 1);
  const located = totalBoxes > 0;

  return (
    <button className={`cite-card ${located ? "" : "cite-flat"}`} onClick={onOpen}>
      <span className="cite-num">{index + 1}</span>
      <span className="cite-body">
        {citation.quote && <span className="cite-quote">“{citation.quote.trim()}”</span>}
        <span className="cite-meta">
          <span className="cite-doc">{docName(citation.source)}</span>
          {located ? (
            <span className="cite-badge">
              page{pages.length > 1 ? "s" : ""} {pages.join(", ")} · {totalBoxes} box
              {totalBoxes === 1 ? "" : "es"}
            </span>
          ) : (
            <span className="cite-badge muted">no page boxes</span>
          )}
        </span>
      </span>
      {located && <span className="cite-go">view ↗</span>}
    </button>
  );
}
