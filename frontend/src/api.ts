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

// A completed answer turn.
export interface AnswerTurn {
  type: "answer";
  session_id: string;
  answer: string;
  used_reasoning: boolean;
  trace: Record<string, string>;
  citations: Citation[];
  working_set: WorkingDoc[];
  elapsed_ms: number;
  reasoning_log: string[]; // L2's full step-by-step log for this answer
}

// L2 paused mid-reasoning to ask the user something.
export interface QuestionTurn {
  type: "question";
  session_id: string;
  question: string;
}

export type TurnResult = AnswerTurn | QuestionTurn;

export type GroundingSource = "trajectory" | "document" | "raw";

async function postTurn(path: string, body: unknown): Promise<TurnResult> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }
  return res.json();
}

// Start a new turn for a session (a fresh L1+L2 run over the corpus). When
// fullCorpus is set, L1 retrieval is skipped and L2 reasons over every document.
export function sendMessage(
  sessionId: string,
  message: string,
  groundingSource: GroundingSource,
  fullCorpus = false,
): Promise<TurnResult> {
  return postTurn("/api/chat", {
    session_id: sessionId,
    message,
    grounding_source: groundingSource,
    full_corpus: fullCorpus,
  });
}

// Answer L2's pending clarifying question; resumes the same turn.
export function respond(sessionId: string, answer: string): Promise<TurnResult> {
  return postTurn("/api/respond", { session_id: sessionId, answer });
}

// End a session — cancels any in-flight turn so the server lock is released.
export async function resetSession(sessionId: string): Promise<void> {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  }).catch(() => {});
}

export interface TracePoll {
  lines: string[]; // log lines after the requested index
  next: number; // cursor to pass as `after` on the next poll
  finished: boolean; // the turn has completed
}

// Poll L2's reasoning log for a session's in-flight turn (lines after `after`).
export async function fetchTrace(sessionId: string, after: number): Promise<TracePoll> {
  const q = new URLSearchParams({ session_id: sessionId, after: String(after) });
  const res = await fetch(`/api/trace?${q.toString()}`);
  if (!res.ok) throw new Error(`trace failed: ${res.status}`);
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
