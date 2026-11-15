import { consensus } from "@/lib/format";

/** "8/10 mô hình đồng thuận" + a 10-segment meter (the text carries the meaning, the meter is decoration). */
export function Consensus({ votes, n = 10 }: { votes: number; n?: number }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-x-2 gap-y-1" title={consensus(votes, n)}>
      <span className="flex shrink-0 gap-[2px]" aria-hidden>
        {Array.from({ length: n }, (_, i) => (
          <span key={i} className={`h-2.5 w-1.5 rounded-[2px] ${i < votes ? "bg-series-1" : "bg-muted"}`} />
        ))}
      </span>
      <span className="whitespace-nowrap text-xs text-muted-foreground">{consensus(votes, n)}</span>
    </span>
  );
}

/** Lightweight variant for table rows (2 elements instead of 12 per row). */
export function ConsensusCompact({ votes, n = 10 }: { votes: number; n?: number }) {
  return (
    <span className="inline-flex items-center gap-2" title={consensus(votes, n)}>
      <span
        aria-hidden
        className="h-2 w-12 shrink-0 rounded-full bg-muted"
        style={{ backgroundImage: `linear-gradient(to right, var(--series-1) ${(votes / n) * 100}%, transparent 0)` }}
      />
      <span className="text-xs text-muted-foreground">
        {votes}/{n}
      </span>
    </span>
  );
}
