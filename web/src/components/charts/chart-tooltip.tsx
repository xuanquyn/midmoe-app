// Shared tooltip shell for Recharts (text in ink tokens; the colored swatch carries series identity).
export function TooltipBox({ title, rows }: { title: string; rows: { label: string; value: string; color?: string }[] }) {
  return (
    <div className="rounded-md border bg-card px-3 py-2 text-xs shadow-md">
      <p className="mb-1 font-medium text-foreground">{title}</p>
      <ul className="space-y-0.5">
        {rows.map((r) => (
          <li key={r.label} className="flex items-center gap-2">
            {r.color && <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: r.color }} aria-hidden />}
            <span className="text-muted-foreground">{r.label}</span>
            <span className="ml-auto pl-3 font-medium text-foreground">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
