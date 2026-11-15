"use client";

import { ArrowLeft, ExternalLink, MessageCircleQuestion } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";

import { CandleChart } from "@/components/charts/candle-chart";
import { RankHistoryChart } from "@/components/charts/rank-history-chart";
import { SourceChart } from "@/components/charts/source-chart";
import { useChat, usePageContext } from "@/components/chat/chat-context";
import { Consensus } from "@/components/consensus";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  EmptyBox,
  ErrorBox,
  Skeleton,
} from "@/components/ui/primitives";
import { api, type ExplanationPayload, type PriceBar } from "@/lib/api";
import { FEATURE_LABELS, fmtDate, fmtFeature, fmtNum } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { useDateHref, useSelectedDate } from "@/lib/use-selected-date";

const FEATURES = [
  "ret_1d",
  "ret_5d",
  "ret_10d",
  "rsi",
  "macd",
  "close_open_ratio",
  "high_low_ratio",
] as const;
const CANDLE_SESSIONS = 120;
const RANK_SESSIONS = 30;

function shiftDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function IndicatorTable({ payload, bar }: { payload: ExplanationPayload | null; bar: PriceBar | null }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-xs text-muted-foreground">
          <tr className="border-b">
            <th scope="col" className="py-1.5 text-left font-medium">
              Chỉ báo
            </th>
            <th scope="col" className="py-1.5 text-right font-medium">
              Giá trị
            </th>
            <th
              scope="col"
              className="py-1.5 text-right font-medium"
              title="Vị trí so với 78 mã trong cùng phiên"
            >
              Percentile
            </th>
            <th scope="col" className="py-1.5 pl-3 text-left font-medium">
              Diễn giải
            </th>
          </tr>
        </thead>
        <tbody>
          {FEATURES.map((f) => {
            const fromPayload = payload?.features[f];
            const value = fromPayload?.value ?? (bar ? (bar[f] as number | null) : null);
            return (
              <tr key={f} className="border-b last:border-0">
                <th scope="row" className="py-1.5 text-left font-normal">
                  {FEATURE_LABELS[f]}
                </th>
                <td className="py-1.5 text-right">{fmtFeature(f, value)}</td>
                <td className="py-1.5 text-right">
                  {fromPayload?.pct !== null && fromPayload?.pct !== undefined
                    ? `${Math.round(fromPayload.pct * 100)}%`
                    : "—"}
                </td>
                <td className="py-1.5 pl-3 text-muted-foreground">{fromPayload?.label ?? ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function StockPage({ params }: { params: { ticker: string } }) {
  const ticker = decodeURIComponent(params.ticker).toUpperCase();
  const [date] = useSelectedDate();
  usePageContext(ticker, date);
  const { ask } = useChat();
  const href = useDateHref();

  // All requests start in parallel (no waterfall). `date` may be a non-trading day: the backend resolves it to the
  // previous session for explanation/rankings, and the price window simply ends at the last session <= date.
  const exp = useApi(() => api.explanation(ticker, date), [ticker, date]);
  const stock = useApi(() => api.stock(ticker, date ? shiftDays(date, -200) : null, date), [ticker, date]);
  const ranking = useApi(() => api.rankings(date), [date]);
  const news = useApi(() => api.news(ticker, 14), [ticker]);
  const effDate = exp.data?.date ?? ranking.data?.date ?? date;

  const total = ranking.data?.items.length ?? 78;
  const row = ranking.data?.items.find((r) => r.ticker === ticker) ?? null;
  const bars = useMemo(() => (stock.data?.prices ?? []).slice(-CANDLE_SESSIONS), [stock.data]);
  const rankPts = useMemo(() => (stock.data?.rank_history ?? []).slice(-RANK_SESSIONS), [stock.data]);
  const lastBar = bars[bars.length - 1] ?? null;
  const payload = exp.data?.payload ?? null;

  const allNews = useMemo(() => {
    const seen = new Set<string>();
    const out: { date: string; title: string; url: string }[] = [];
    for (const n of [...(payload?.news ?? []), ...(news.data ?? [])]) {
      if (!seen.has(n.url)) {
        seen.add(n.url);
        out.push({ date: n.date, title: n.title, url: n.url });
      }
    }
    return out.sort((a, b) => b.date.localeCompare(a.date));
  }, [payload, news.data]);

  if (stock.error?.includes("không nằm trong danh sách")) {
    return (
      <div className="mx-auto max-w-3xl space-y-3">
        <Link
          href={href("/")}
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" /> Tổng quan
        </Link>
        <EmptyBox>{stock.error}</EmptyBox>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <Link
        href={href("/")}
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Tổng quan
      </Link>

      {/* fixed-height rows so late-arriving name/badges never move the content below (CLS) */}
      <section className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 space-y-1">
          <h1 className="text-2xl font-semibold">{ticker}</h1>
          <p className="h-5 truncate text-sm text-muted-foreground">{stock.data?.name ?? ""}</p>
          <div className="flex min-h-[1.375rem] flex-wrap gap-1.5">
            {stock.data?.icb_sector && <Badge>{stock.data.icb_sector}</Badge>}
            {stock.data?.exchange && <Badge>{stock.data.exchange}</Badge>}
            {effDate && <Badge>Phiên {fmtDate(effDate)}</Badge>}
          </div>
        </div>
        <Button
          variant="outline"
          className="self-start"
          onClick={() =>
            ask(`Vì sao ${ticker} được xếp hạng như vậy${effDate ? ` ngày ${fmtDate(effDate)}` : ""}?`, {
              ticker,
              date: effDate,
            })
          }
        >
          <MessageCircleQuestion className="h-4 w-4" /> Hỏi trợ lý về {ticker}
        </Button>
      </section>

      <Card>
        <CardContent className="grid gap-4 p-4 sm:grid-cols-[auto_1fr]">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Hạng</p>
            {row || payload ? (
              <p className="text-3xl font-semibold leading-none">
                {payload?.rank ?? row?.rank}
                <span className="text-base font-normal text-muted-foreground">/{total}</span>
              </p>
            ) : (
              <Skeleton className="h-8 w-20" />
            )}
            {payload || row ? (
              <Consensus votes={payload?.top5_votes ?? row!.top5_votes} />
            ) : (
              <Skeleton className="h-4 w-40" />
            )}
          </div>
          <div className="space-y-1 text-sm">
            {payload ? (
              <>
                <p>{payload.summary}</p>
                <p className="text-xs text-muted-foreground">
                  Hạng {payload.sector.rank_in_sector}/{payload.sector.n_in_sector} trong ngành{" "}
                  {payload.sector.name}
                  {payload.sector.top.length ? ` · Dẫn đầu ngành: ${payload.sector.top.join(", ")}` : ""}
                  {" · "}Điểm {fmtNum(payload.score, 4)} (độ lệch giữa 10 mô hình{" "}
                  {fmtNum(payload.score_std, 4)})
                </p>
              </>
            ) : exp.loading ? (
              <Skeleton className="h-[4.75rem] sm:h-10" />
            ) : (
              <p className="text-muted-foreground">{exp.error ?? "Chưa có giải thích cho phiên này."}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Giá và khối lượng</CardTitle>
          <CardDescription>{CANDLE_SESSIONS} phiên gần nhất tính đến phiên đang xem</CardDescription>
        </CardHeader>
        <CardContent>
          {stock.error ? (
            <ErrorBox message={stock.error} onRetry={stock.reload} />
          ) : stock.loading || !stock.data ? (
            <Skeleton className="h-[388px] sm:h-[368px]" />
          ) : (
            <CandleChart bars={bars} />
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Đóng góp của 4 nguồn thông tin</CardTitle>
            <CardDescription>
              Điểm giảm bao nhiêu khi tắt từng nguồn (trung bình 10 mô hình). Dương = đẩy mã lên.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {payload ? (
              <SourceChart payload={payload} total={total} />
            ) : (
              <EmptyBox>Chưa có dữ liệu giải thích.</EmptyBox>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Lịch sử hạng</CardTitle>
            <CardDescription>{RANK_SESSIONS} phiên gần nhất · hạng 1 ở trên cùng</CardDescription>
          </CardHeader>
          <CardContent>
            {stock.data ? (
              <RankHistoryChart points={rankPts} total={total} />
            ) : (
              <Skeleton className="h-[220px]" />
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>7 chỉ báo kỹ thuật</CardTitle>
            <CardDescription>
              Percentile = vị trí so với 78 mã trong cùng phiên (100% = cao nhất)
            </CardDescription>
          </CardHeader>
          <CardContent>
            {exp.loading && stock.loading ? (
              <Skeleton className="h-56" />
            ) : (
              <IndicatorTable payload={payload} bar={lastBar} />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Tin tức liên quan</CardTitle>
            <CardDescription>Nguồn CafeF</CardDescription>
          </CardHeader>
          <CardContent>
            {news.loading && !payload ? (
              <Skeleton className="h-40" />
            ) : allNews.length === 0 ? (
              <EmptyBox>Không có tin trong thời gian gần đây.</EmptyBox>
            ) : (
              <ul className="space-y-2">
                {allNews.slice(0, 10).map((n) => (
                  <li key={n.url} className="text-sm">
                    <a
                      href={n.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group inline-flex gap-1 hover:text-primary"
                    >
                      <span className="group-hover:underline">{n.title}</span>
                      <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 opacity-60" aria-hidden />
                    </a>
                    <p className="text-xs text-muted-foreground">{fmtDate(n.date)}</p>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
