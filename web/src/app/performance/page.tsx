"use client";

import { useState } from "react";

import { PerformanceChart } from "@/components/charts/performance-chart";
import { usePageContext } from "@/components/chat/chat-context";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  EmptyBox,
  ErrorBox,
  Skeleton,
} from "@/components/ui/primitives";
import { api, type BacktestMetrics } from "@/lib/api";
import { fmtDate, fmtNum, fmtPct } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { cn } from "@/lib/utils";

const LIVE_START = process.env.NEXT_PUBLIC_LIVE_START || "2026-01-01";
const K = 5;

const PERIODS = [
  { id: "all", label: "Toàn bộ", from: null, to: null },
  { id: "test", label: "Test 2025", from: "2025-01-01", to: "2025-12-31" },
  { id: "live", label: `Live từ ${fmtDate(LIVE_START)}`, from: LIVE_START, to: null },
] as const;

const ROWS: { key: "topk" | "random" | "vnindex"; label: string; swatch: string }[] = [
  { key: "topk", label: `Top ${K} mô hình`, swatch: "bg-series-1" },
  { key: "random", label: "Chọn ngẫu nhiên (TB 5 lần)", swatch: "bg-series-2" },
  { key: "vnindex", label: `VN-Index ×${K}`, swatch: "bg-series-3" },
];

function MetricsTable({
  metrics,
}: {
  metrics: Partial<Record<"topk" | "random" | "vnindex", BacktestMetrics>>;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-xs text-muted-foreground">
          <tr className="border-b">
            <th scope="col" className="py-2 text-left font-medium">
              Chiến lược
            </th>
            <th
              scope="col"
              className="py-2 text-right font-medium"
              title="Trung bình / độ lệch chuẩn lợi nhuận ngày"
            >
              Sharpe (ngày)
            </th>
            <th scope="col" className="py-2 text-right font-medium">
              Tổng lợi nhuận
            </th>
            <th scope="col" className="py-2 text-right font-medium">
              Sụt giảm tối đa
            </th>
            <th scope="col" className="py-2 text-right font-medium" title="Chất lượng xếp hạng top 5 (0–1)">
              nDCG@5
            </th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((r) => {
            const m = metrics[r.key];
            return (
              <tr key={r.key} className="border-b last:border-0">
                <th scope="row" className="py-2 text-left font-normal">
                  <span className="inline-flex items-center gap-2">
                    <span className={cn("h-2.5 w-2.5 rounded-full", r.swatch)} aria-hidden />
                    {r.label}
                  </span>
                </th>
                <td className="py-2 text-right">{m ? fmtNum(m.SR, 3) : "—"}</td>
                <td className="py-2 text-right">{m ? fmtPct(m.IRR, 1, true) : "—"}</td>
                <td className="py-2 text-right">{m ? fmtPct(m.MaxDD, 1) : "—"}</td>
                <td className="py-2 text-right">
                  {m?.nDCG5 !== null && m?.nDCG5 !== undefined ? fmtNum(m.nDCG5, 3) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function PerformancePage() {
  usePageContext(null, null);
  const [showTable, setShowTable] = useState(false);
  const [period, setPeriod] = useState<(typeof PERIODS)[number]["id"]>("all");
  const p = PERIODS.find((x) => x.id === period)!;
  const bt = useApi(() => api.backtest(p.from, p.to, K), [p.from, p.to]);

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <section className="space-y-1">
        <h1 className="text-xl font-semibold">Hiệu quả mô hình</h1>
        <p className="text-sm text-muted-foreground">
          Mỗi phiên mua đều {K} mã đứng đầu và bán ở phiên kế tiếp. Năm 2025 là năm kiểm thử trong paper; từ{" "}
          {fmtDate(LIVE_START)} là giai đoạn chạy thật.
        </p>
      </section>

      <div role="tablist" aria-label="Giai đoạn" className="inline-flex rounded-md border bg-card p-0.5">
        {PERIODS.map((x) => (
          <button
            key={x.id}
            role="tab"
            aria-selected={period === x.id}
            onClick={() => setPeriod(x.id)}
            className={cn(
              "rounded px-3 py-1.5 text-sm",
              period === x.id
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {x.label}
          </button>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Lợi nhuận tích luỹ</CardTitle>
          <CardDescription>
            {bt.data
              ? `${fmtDate(bt.data.from_date)} – ${fmtDate(bt.data.to_date)} · ${bt.data.n_days} phiên`
              : "Đang tải…"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {bt.error ? (
            bt.error.includes("Không có dự đoán") ? (
              <EmptyBox>Chưa có dữ liệu cho giai đoạn này.</EmptyBox>
            ) : (
              <ErrorBox message={bt.error} onRetry={bt.reload} />
            )
          ) : !bt.data ? (
            <Skeleton className="h-80" />
          ) : (
            <>
              <PerformanceChart
                series={bt.data.series}
                k={bt.data.k}
                liveStart={period === "all" ? LIVE_START : null}
              />
              {bt.data.vnindex_coverage < 1 && (
                <p className="text-xs text-muted-foreground">
                  Thiếu dữ liệu VN-Index cho {Math.round((1 - bt.data.vnindex_coverage) * 100)}% số phiên nên
                  không vẽ đường VN-Index.
                </p>
              )}
              <MetricsTable metrics={bt.data.metrics} />
              <details
                className="text-sm"
                onToggle={(e) => setShowTable((e.target as HTMLDetailsElement).open)}
              >
                <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                  Xem dữ liệu dạng bảng
                </summary>
                {showTable && (
                  <div className="mt-2 max-h-72 overflow-auto rounded border">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-card text-muted-foreground">
                        <tr>
                          <th className="px-2 py-1 text-left font-medium">Phiên</th>
                          <th className="px-2 py-1 text-right font-medium">Top {bt.data.k}</th>
                          <th className="px-2 py-1 text-right font-medium">Ngẫu nhiên</th>
                          <th className="px-2 py-1 text-right font-medium">VN-Index ×{bt.data.k}</th>
                          <th className="px-2 py-1 text-right font-medium">Top {bt.data.k} tích luỹ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bt.data.series.map((s) => (
                          <tr key={s.date} className="border-t">
                            <td className="px-2 py-1">{fmtDate(s.date)}</td>
                            <td className="px-2 py-1 text-right">{fmtPct(s.topk, 2, true)}</td>
                            <td className="px-2 py-1 text-right">{fmtPct(s.random, 2, true)}</td>
                            <td className="px-2 py-1 text-right">{fmtPct(s.vnindex, 2, true)}</td>
                            <td className="px-2 py-1 text-right">{fmtPct(s.topk_cum, 2, true)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </details>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Đọc kết quả thế nào</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            Theo paper, trên năm kiểm thử 2025 cấu hình đang triển khai (giữ nguyên backbone học từ thị trường
            Mỹ, chỉ huấn luyện lại 2,4% tham số) đạt Sharpe theo ngày <strong>0,041</strong> và nDCG@5{" "}
            <strong>0,496</strong>, so với chọn ngẫu nhiên 0,033 và 0,492. Hướng cải thiện nhất quán nhưng{" "}
            <strong>không có ý nghĩa thống kê</strong> (block bootstrap theo ngày, hiệu chỉnh Holm).
          </p>
          <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
            <li>
              Lợi nhuận ngày là <em>tổng</em> lợi suất của {K} mã (quy ước của paper), nên VN-Index được nhân{" "}
              {K} để cùng thang. Sharpe không phụ thuộc hệ số này.
            </li>
            <li>Chưa trừ phí giao dịch; danh mục thay khoảng 96% mỗi ngày.</li>
            <li>
              Chưa mô hình hoá T+2 và biên độ ±7% (HOSE), nên chiến lược không thực hiện được đúng như mô
              phỏng.
            </li>
            <li>Chỉ một năm kiểm thử; dữ liệu 2026 có thể khác phân phối lúc huấn luyện.</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
