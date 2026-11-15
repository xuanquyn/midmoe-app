"use client";

import { ArrowDownUp, Search } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { ConsensusCompact } from "@/components/consensus";
import { RankChange } from "@/components/rank-change";
import { Skeleton } from "@/components/ui/primitives";
import type { RankingItem } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import { useDateHref } from "@/lib/use-selected-date";
import { cn } from "@/lib/utils";

type SortKey = "rank" | "ticker" | "icb_sector" | "score" | "top5_votes" | "rank_change";

const COLS: { key: SortKey; label: string; className?: string }[] = [
  { key: "rank", label: "Hạng", className: "w-14 text-right" },
  { key: "ticker", label: "Mã", className: "text-left" },
  { key: "icb_sector", label: "Ngành", className: "hidden text-left md:table-cell" },
  { key: "score", label: "Điểm", className: "hidden text-right sm:table-cell" },
  { key: "top5_votes", label: "Đồng thuận", className: "text-left" },
  { key: "rank_change", label: "Thay đổi", className: "text-right" },
];

export function RankingsTable({
  items,
  sectors,
  loading,
}: {
  items: RankingItem[] | null;
  sectors: string[];
  loading: boolean;
}) {
  const href = useDateHref();
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "rank", asc: true });
  const [sector, setSector] = useState("");
  const [q, setQ] = useState("");

  const rows = useMemo(() => {
    let r = items ?? [];
    if (sector) r = r.filter((x) => x.icb_sector === sector);
    if (q.trim()) {
      const s = q.trim().toUpperCase();
      r = r.filter((x) => x.ticker.includes(s) || (x.name ?? "").toUpperCase().includes(s));
    }
    const dir = sort.asc ? 1 : -1;
    return [...r].sort((a, b) => {
      const va = a[sort.key] ?? (sort.key === "rank_change" ? -Infinity : "");
      const vb = b[sort.key] ?? (sort.key === "rank_change" ? -Infinity : "");
      if (va < vb) return -dir;
      if (va > vb) return dir;
      return a.rank - b.rank;
    });
  }, [items, sector, q, sort]);

  const toggle = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, asc: !s.asc } : { key, asc: key === "rank" || key === "ticker" || key === "icb_sector" }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <label className="relative">
          <span className="sr-only">Tìm mã</span>
          <Search className="pointer-events-none absolute left-2 top-2.5 h-4 w-4 text-subtle" aria-hidden />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Tìm mã hoặc tên…"
            className="h-9 w-48 rounded-md border bg-card pl-8 pr-2 text-sm"
          />
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-muted-foreground">Ngành</span>
          <select value={sector} onChange={(e) => setSector(e.target.value)} className="h-9 max-w-[14rem] rounded-md border bg-card px-2 text-sm">
            <option value="">Tất cả</option>
            {sectors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span className="text-xs text-muted-foreground">{items ? `${rows.length}/${items.length} mã` : ""}</span>
      </div>

      <div className="overflow-x-auto rounded-lg border bg-card">
        <table className="w-full text-sm">
          <thead className="border-b text-xs text-muted-foreground">
            <tr>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  aria-sort={sort.key === c.key ? (sort.asc ? "ascending" : "descending") : "none"}
                  className={cn("px-3 py-2 font-medium", c.className)}
                >
                  <button onClick={() => toggle(c.key)} className="inline-flex items-center gap-1 hover:text-foreground">
                    {c.label}
                    <ArrowDownUp className={cn("h-3 w-3", sort.key === c.key ? "text-foreground" : "opacity-40")} aria-hidden />
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && !items
              ? Array.from({ length: 8 }, (_, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td colSpan={COLS.length} className="px-3 py-2">
                      <Skeleton className="h-5" />
                    </td>
                  </tr>
                ))
              : rows.map((r) => (
                  <tr key={r.ticker} className="border-b last:border-0 hover:bg-muted/60">
                    <td className="px-3 py-2 text-right font-medium">{r.rank}</td>
                    <td className="px-3 py-2">
                      <Link href={href(`/stocks/${r.ticker}`)} className="font-semibold text-primary hover:underline">
                        {r.ticker}
                      </Link>
                      <div className="max-w-[16rem] truncate text-xs text-muted-foreground">{r.name}</div>
                    </td>
                    <td className="hidden px-3 py-2 text-muted-foreground md:table-cell">{r.icb_sector ?? "—"}</td>
                    <td className="hidden px-3 py-2 text-right sm:table-cell">{fmtNum(r.score, 4)}</td>
                    <td className="px-3 py-2">
                      <ConsensusCompact votes={r.top5_votes} />
                    </td>
                    <td className="px-3 py-2 text-right">
                      <RankChange value={r.rank_change} />
                    </td>
                  </tr>
                ))}
            {items && rows.length === 0 && (
              <tr>
                <td colSpan={COLS.length} className="px-3 py-6 text-center text-muted-foreground">
                  Không có mã phù hợp.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
