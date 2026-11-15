"use client";

import { Bar, BarChart, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { TooltipBox } from "@/components/charts/chart-tooltip";
import type { ExplanationPayload } from "@/lib/api";
import { fmtNum, SOURCE_LABELS } from "@/lib/format";
import { useChartColors } from "@/lib/theme-colors";

/** Diverging bars around 0: score_full - score_without_source. Positive = the source pushes the ticker up. */
export function SourceChart({ payload, total, height = 200 }: { payload: ExplanationPayload; total: number; height?: number }) {
  const c = useChartColors();
  const data = Object.entries(payload.sources)
    .map(([k, v]) => ({ key: k, label: SOURCE_LABELS[k] ?? k, delta: v.delta, rankWithout: v.rank_without }))
    .sort((a, b) => b.delta - a.delta);
  if (!c) return <div style={{ height }} />;
  const max = Math.max(...data.map((d) => Math.abs(d.delta)), 1e-9);
  return (
    <div style={{ height }} role="img" aria-label={`Đóng góp của 4 nguồn thông tin; lớn nhất: ${data[0]?.label}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }} barCategoryGap={10}>
          <XAxis type="number" domain={[-max * 1.1, max * 1.1]} hide />
          <YAxis
            type="category"
            dataKey="label"
            width={96}
            tick={{ fill: c.muted, fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <ReferenceLine x={0} stroke={c.axis} />
          <Tooltip
            cursor={{ fill: c.grid, opacity: 0.4 }}
            content={({ active, payload: p }) =>
              active && p?.length ? (
                <TooltipBox
                  title={String(p[0].payload.label)}
                  rows={[
                    { label: "Đóng góp vào điểm", value: fmtNum(Number(p[0].payload.delta), 5), color: Number(p[0].payload.delta) >= 0 ? c.pos : c.neg },
                    { label: "Hạng nếu bỏ nguồn này", value: `${p[0].payload.rankWithout}/${total}` },
                    { label: "Hạng hiện tại", value: `${payload.rank}/${total}` },
                  ]}
                />
              ) : null
            }
          />
          <Bar dataKey="delta" radius={4} isAnimationActive={false} maxBarSize={22}>
            {data.map((d) => (
              <Cell key={d.key} fill={d.delta >= 0 ? c.pos : c.neg} />
            ))}
            <LabelList
              dataKey="delta"
              position="right"
              formatter={(v: number) => `${v >= 0 ? "+" : ""}${fmtNum(v, 4)}`}
              style={{ fill: c.muted, fontSize: 11 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
