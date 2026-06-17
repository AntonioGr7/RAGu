// Typed client for the RAGu FastAPI backend. Uses relative /api paths so it
// works both behind the Vite dev proxy and when served by FastAPI in production.

export type Box = [number, number, number, number];

export interface Highlight {
  page: number; // zero-based page index
  boxes: Box[]; // in the page's own pixel space
  width: number | null; // page pixel width  (SVG viewBox)
  height: number | null; // page pixel height
}

export interface Citation {
  doc_id: string;
  source: string;
  quote: string | null;
  start_char: number | null;
  end_char: number | null;
  highlights: Highlight[];
}

export interface WorkingDoc {
  id: string;
  source: string;
}

export interface QueryResult {
  answer: string;
  used_reasoning: boolean;
  trace: Record<string, string>;
  citations: Citation[];
  working_set: WorkingDoc[];
  elapsed_ms: number;
}

export type GroundingSource = "trajectory" | "document" | "raw";

export async function runQuery(
  query: string,
  groundingSource: GroundingSource,
): Promise<QueryResult> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, grounding_source: groundingSource }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Query failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function listDocuments(): Promise<WorkingDoc[]> {
  const res = await fetch("/api/documents");
  if (!res.ok) throw new Error(`documents failed: ${res.status}`);
  return (await res.json()).documents;
}

// URL of a rendered source page (no boxes baked in — overlaid as SVG client-side).
export function pageImageUrl(source: string, page: number, dpi = 170): string {
  const q = new URLSearchParams({ source, page: String(page), dpi: String(dpi) });
  return `/api/page?${q.toString()}`;
}

// Short, display-friendly document name from a full source path.
export function docName(source: string): string {
  return source.split("/").pop() || source;
}
