import { useEffect, useRef, useState } from "react";

interface Props {
  lines: string[];
  live?: boolean; // currently streaming — default open, show a pulse
}

// Collapsible panel showing L2's reasoning log (one line per trajectory step).
// Live during a turn (auto-expanded, auto-scrolling); collapsed by default once
// attached to a finished answer.
export function TraceBox({ lines, live = false }: Props) {
  const [open, setOpen] = useState(live);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Follow the tail while streaming.
  useEffect(() => {
    if (open && live) bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
  }, [lines.length, open, live]);

  return (
    <div className={`trace-box ${live ? "trace-live" : ""}`}>
      <button className="trace-toggle" onClick={() => setOpen((v) => !v)}>
        <span className={`caret ${open ? "caret-open" : ""}`}>▸</span>
        {live && <span className="trace-pulse" />}
        <span>{live ? "Reasoning…" : "Reasoning log"}</span>
        <span className="trace-count">{lines.length} step{lines.length === 1 ? "" : "s"}</span>
      </button>
      {open && (
        <div className="trace-body" ref={bodyRef}>
          {lines.length === 0 ? (
            <div className="trace-waiting">waiting for L2…</div>
          ) : (
            lines.map((l, i) => (
              <pre className="trace-line" key={i}>
                {l}
              </pre>
            ))
          )}
        </div>
      )}
    </div>
  );
}
