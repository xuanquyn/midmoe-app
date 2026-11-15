"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { TooltipBox } from "@/components/charts/chart-tooltip";
import { fmtDate, fmtNum } from "@/lib/format";
import { useChartColors } from "@/lib/theme-colors";

/** Single series (no legend; the card title names it). Y axis inverted: rank 1 at the top. */
export function RankHistoryChart({ points, total, height = 220 }: { points: { date: string; rank: number; score: number }[]; total: number; height?: number }) {
  const c = useChartColors();
  if (!c) return <div style={{ height }} />;
  if (points.length === 0) return <p className="text-sm text-muted-foreground">Chưa có lịch sử hạng.</p>;
  return (
    <div style={{ height }} role="img" aria-label={`Lịch sử hạng ${points.length} phiên, hạng gần nhất ${points[points.length - 1].rank}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 12, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={(d: string) => fmtDate(d).slice(0, 5)}
            tick={{ fill: c.subtle, fontSize: 11 }}
            stroke={c.axis}
            tickLine={false}
            minTickGap={24}
          />
          <YAxis
            reversed
            domain={[1, total]}
            allowDecimals={false}
            ticks={[1, Math.round(total / 4), Math.round(total / 2), Math.round((3 * total) / 4), total]}
            tick={{ fill: c.subtle, fontSize: 11 }}
            stroke={c.axis}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ stroke: c.axis, strokeWidth: 1 }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={fmtDate(String(payload[0].payload.date))}
                  rows={[
                    { label: "Hạng", value: `${payload[0].payload.rank}/${total}`, color: c.series1 },
                    { label: "Điểm", value: fmtNum(Number(payload[0].payload.score), 4) },
                  ]}
                />
              ) : null
            }
          />
          <Line
            type="linear" /* monotone interpolation overshoots past rank 1 / rank N */
            dataKey="rank"
            stroke={c.series1}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, stroke: c.card, strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
