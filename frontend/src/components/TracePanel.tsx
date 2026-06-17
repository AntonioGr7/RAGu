import { QueryResult } from "../api";

// The reasoning trace + timing as a row of stat chips — shows the two-level
// pipeline at work (L2 engine, tokens, steps, tool calls, docs navigated).
export function TracePanel({ result }: { result: QueryResult }) {
  const t = result.trace;
  const stats: { label: string; value: string }[] = [
    { label: "engine", value: t.engine ?? (result.used_reasoning ? "L2" : "L1") },
    { label: "elapsed", value: `${(result.elapsed_ms / 1000).toFixed(1)}s` },
  ];
  if (t.tokens) stats.push({ label: "tokens", value: Number(t.tokens).toLocaleString() });
  if (t.steps) stats.push({ label: "steps", value: t.steps });
  if (t.calls) stats.push({ label: "tool calls", value: t.calls });
  if (t.working_set_docs)
    stats.push({ label: "docs read", value: t.working_set_docs });

  return (
    <div className="trace">
      {stats.map((s) => (
        <div className="stat" key={s.label}>
          <div className="stat-value">{s.value}</div>
          <div className="stat-label">{s.label}</div>
        </div>
      ))}
    </div>
  );
}
