import { useEffect, useState } from "react";
import { Citation, docName, Highlight, pageImageUrl } from "../api";

interface Props {
  citation: Citation;
  onClose: () => void;
}

// Modal: the rendered source page with the citation's word boxes overlaid as
// scalable SVG rectangles. The SVG viewBox is the page's own pixel space, so the
// boxes line up at any display size without us doing any scaling math.
export function PageViewer({ citation, onClose }: Props) {
  const pages = citation.highlights;
  const [active, setActive] = useState(0);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setActive(0);
  }, [citation]);

  useEffect(() => {
    setLoaded(false);
  }, [active, citation]);

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const hl: Highlight | undefined = pages[active];
  const hasBoxes = hl && hl.boxes.length > 0;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <div>
            <div className="modal-doc">{docName(citation.source)}</div>
            {citation.quote && <div className="modal-quote">“{citation.quote.trim()}”</div>}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        {pages.length > 1 && (
          <div className="page-tabs">
            {pages.map((h, i) => (
              <button
                key={i}
                className={`page-tab ${i === active ? "active" : ""}`}
                onClick={() => setActive(i)}
              >
                page {h.page + 1} · {h.boxes.length} box{h.boxes.length === 1 ? "" : "es"}
              </button>
            ))}
          </div>
        )}

        <div className="page-stage">
          {!hl ? (
            <div className="page-empty">This citation has no page geometry to show.</div>
          ) : (
            <div className="page-frame">
              {!loaded && <div className="page-spinner">rendering page…</div>}
              <img
                className="page-img"
                src={pageImageUrl(citation.source, hl.page)}
                alt={`page ${hl.page + 1}`}
                onLoad={() => setLoaded(true)}
                style={{ opacity: loaded ? 1 : 0 }}
              />
              {loaded && hasBoxes && hl.width && hl.height && (
                <svg
                  className="overlay"
                  viewBox={`0 0 ${hl.width} ${hl.height}`}
                  preserveAspectRatio="none"
                >
                  {hl.boxes.map((b, i) => (
                    <rect
                      key={i}
                      x={b[0]}
                      y={b[1]}
                      width={b[2] - b[0]}
                      height={b[3] - b[1]}
                      className="hl-box"
                      style={{ animationDelay: `${i * 90}ms` }}
                      rx={4}
                    />
                  ))}
                </svg>
              )}
            </div>
          )}
        </div>

        <footer className="modal-foot">
          {hasBoxes ? (
            <span>
              page {hl!.page + 1} · {hl!.boxes.length} highlighted span
              {hl!.boxes.length === 1 ? "" : "s"}
            </span>
          ) : (
            <span>located in document, no word boxes</span>
          )}
        </footer>
      </div>
    </div>
  );
}
