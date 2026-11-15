"use client";

import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { TooltipBox } from "@/components/charts/chart-tooltip";
import type { BacktestPoint } from "@/lib/api";
import { fmtDate, fmtPct } from "@/lib/format";
import { useChartColors } from "@/lib/theme-colors";

/** Cumulative (summed) daily return: top-k vs random vs k×VN-Index. Fixed slot order: 1 top-k, 2 random, 3 VN-Index. */
export function PerformanceChart({
  series,
  k,
  liveStart,
  height = 320,
}: {
  series: BacktestPoint[];
  k: number;
  liveStart?: string | null;
  height?: number;
}) {
  const c = useChartColors();
  if (!c) return <div style={{ height }} />;
  const hasIndex = series.some((p) => p.vnindex_cum !== null);
  const lines = [
    { key: "topk_cum", name: `Top ${k} mô hình`, color: c.series1 },
    { key: "random_cum", name: "Chọn ngẫu nhiên", color: c.series2 },
    ...(hasIndex ? [{ key: "vnindex_cum", name: `VN-Index ×${k}`, color: c.series3 }] : []),
  ];
  const showLive = liveStart && series.length && series[0].date < liveStart && series[series.length - 1].date >= liveStart;

  return (
    <div style={{ height }} role="img" aria-label="Lợi nhuận tích luỹ của chiến lược top-k so với chọn ngẫu nhiên và VN-Index">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={(d: string) => fmtDate(d).slice(3)}
            tick={{ fill: c.subtle, fontSize: 11 }}
            stroke={c.axis}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            tickFormatter={(v: number) => fmtPct(v, 0)}
            tick={{ fill: c.subtle, fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={52}
          />
          <ReferenceLine y={0} stroke={c.axis} />
          {showLive && (
            <ReferenceLine
              x={series.find((p) => p.date >= liveStart!)!.date}
              stroke={c.subtle}
              strokeDasharray="4 4"
              label={{ value: "Bắt đầu live", position: "insideTopLeft", fill: c.muted, fontSize: 11 }}
            />
          )}
          <Tooltip
            cursor={{ stroke: c.axis, strokeWidth: 1 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={fmtDate(String(label))}
                  rows={payload.map((p) => ({ label: String(p.name), value: fmtPct(Number(p.value), 2, true), color: String(p.color) }))}
                />
              ) : null
            }
          />
          <Legend
            verticalAlign="top"
            height={28}
            iconType="plainline"
            formatter={(v: string) => <span style={{ color: c.muted, fontSize: 12 }}>{v}</span>}
          />
          {lines.map((l) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.name}
              stroke={l.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: c.card, strokeWidth: 2 }}
              isAnimationActive={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
