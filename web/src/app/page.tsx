"use client";

import { MessageCircleQuestion } from "lucide-react";
import Link from "next/link";

import { usePageContext, useChat } from "@/components/chat/chat-context";
import { Consensus } from "@/components/consensus";
import { PipelineStatus } from "@/components/pipeline-status";
import { RankChange } from "@/components/rank-change";
import { RankingsTable } from "@/components/rankings-table";
import { Badge, Button, Card, EmptyBox, ErrorBox, Skeleton } from "@/components/ui/primitives";
import { api, type RecommendationItem } from "@/lib/api";
import { fmtDate } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { useDateHref, useSelectedDate } from "@/lib/use-selected-date";

function TopCard({ item, total, date }: { item: RecommendationItem; total: number; date: string }) {
  const { ask } = useChat();
  const href = useDateHref();
  return (
    <Card className="flex flex-col gap-2 p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <Link href={href(`/stocks/${item.ticker}`)} className="text-lg font-semibold text-primary hover:underline">
            {item.ticker}
          </Link>
          <p className="line-clamp-1 text-xs text-muted-foreground">{item.name ?? "—"}</p>
        </div>
        <Badge className="shrink-0 text-foreground">
          #{item.rank}
          <span className="text-muted-foreground">/{total}</span>
        </Badge>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span className="truncate">{item.icb_sector ?? "—"}</span>
        <span aria-hidden>·</span>
        <span className="inline-flex items-center gap-1">
          so với phiên trước <RankChange value={item.rank_change} />
        </span>
      </div>
      <Consensus votes={item.top5_votes} />
      <p className="line-clamp-3 flex-1 text-sm">{item.reason ?? "Chưa có giải thích cho ngày này."}</p>
      <Button
        variant="outline"
        size="sm"
        className="self-start"
        onClick={() => ask(`Vì sao ${item.ticker} được xếp hạng ${item.rank} ngày ${fmtDate(date)}?`, { ticker: item.ticker, date })}
      >
        <MessageCircleQuestion className="h-4 w-4" /> Hỏi trợ lý
      </Button>
    </Card>
  );
}

export default function DashboardPage() {
  const [date] = useSelectedDate();
  usePageContext(null, date);
  const rec = useApi(() => api.recommendations(date, 5), [date]);
  const rank = useApi(() => api.rankings(date), [date]);
  const sectors = useApi(() => api.sectors(), []);
  const shownDate = rec.data?.date ?? rank.data?.date;
  const total = rank.data?.items.length ?? 78;

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <section className="space-y-1">
        <h1 className="text-xl font-semibold">
          Top 5 khuyến nghị {shownDate ? `phiên ${fmtDate(shownDate)}` : ""}
        </h1>
        <p className="text-sm text-muted-foreground">
          Xếp hạng 78 mã theo kỳ vọng lợi suất phiên kế tiếp, trung bình của 10 mô hình. Đồng thuận = số mô hình xếp mã
          vào top 5.
        </p>
        {date && shownDate && date !== shownDate && (
          <p className="text-xs text-muted-foreground">
            Ngày {fmtDate(date)} không có phiên giao dịch, đang hiển thị phiên gần nhất trước đó.
          </p>
        )}
        <PipelineStatus />
      </section>

      <section aria-label="Top 5">
        {rec.error ? (
          rec.error.includes("Chưa có dữ liệu") ? (
            <EmptyBox>Chưa có dữ liệu dự đoán. Hãy chạy pipeline hoặc backfill.</EmptyBox>
          ) : (
            <ErrorBox message={rec.error} onRetry={rec.reload} />
          )
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {rec.loading || !rec.data
              ? Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-56" />)
              : rec.data.items.map((it) => <TopCard key={it.ticker} item={it} total={total} date={rec.data!.date} />)}
          </div>
        )}
      </section>

      <section className="space-y-2" aria-label="Bảng xếp hạng">
        <h2 className="text-base font-semibold">Bảng xếp hạng {total} mã</h2>
        {rank.error && !rank.error.includes("Chưa có dữ liệu") ? (
          <ErrorBox message={rank.error} onRetry={rank.reload} />
        ) : (
          <RankingsTable items={rank.data?.items ?? null} sectors={sectors.data ?? []} loading={rank.loading} />
        )}
      </section>
    </div>
  );
}
