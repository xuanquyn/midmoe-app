import { ArrowDown, ArrowUp, Minus } from "lucide-react";

/** previous rank - current rank: positive = moved up. Icon + sign + color (never color alone). */
export function RankChange({ value }: { value: number | null }) {
  if (value === null) return <span className="text-subtle">—</span>;
  if (value === 0)
    return (
      <span className="inline-flex items-center gap-0.5 text-muted-foreground">
        <Minus className="h-3.5 w-3.5" aria-hidden />0<span className="sr-only"> (không đổi)</span>
      </span>
    );
  const up = value > 0;
  return (
    <span className={`inline-flex items-center gap-0.5 ${up ? "text-up" : "text-down"}`}>
      {up ? <ArrowUp className="h-3.5 w-3.5" aria-hidden /> : <ArrowDown className="h-3.5 w-3.5" aria-hidden />}
      {Math.abs(value)}
      <span className="sr-only">{up ? " bậc tăng" : " bậc giảm"}</span>
    </span>
  );
}
