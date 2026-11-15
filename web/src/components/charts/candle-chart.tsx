"use client";

import { ColorType, createChart, type IChartApi, type MouseEventParams, type Time } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";

import type { PriceBar } from "@/lib/api";
import { fmtDate, fmtPct, fmtPrice, fmtVolume } from "@/lib/format";
import { useChartColors } from "@/lib/theme-colors";

/** Candles + volume histogram. Hovering shows the bar's OHLC in a legend row (crosshair = hover layer). */
export function CandleChart({ bars, height = 340 }: { bars: PriceBar[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const colors = useChartColors();
  const clean = useMemo(
    () => bars.filter((b) => b.open !== null && b.high !== null && b.low !== null && b.close !== null),
    [bars],
  );
  const last = clean[clean.length - 1] ?? null;
  const [hover, setHover] = useState<PriceBar | null>(null);
  const shown = hover ?? last;

  useEffect(() => {
    if (!ref.current || !colors || clean.length === 0) return;
    const chart: IChartApi = createChart(ref.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: colors.card },
        textColor: colors.subtle,
        fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
        fontSize: 11,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: colors.grid } },
      rightPriceScale: { borderColor: colors.axis },
      timeScale: { borderColor: colors.axis, fixLeftEdge: true, fixRightEdge: true },
      localization: { locale: "vi-VN", priceFormatter: (p: number) => fmtPrice(p) },
      handleScroll: { mouseWheel: false },
      handleScale: { mouseWheel: false },
    });
    const candles = chart.addCandlestickSeries({
      upColor: colors.candleUp,
      downColor: colors.candleDown,
      wickUpColor: colors.candleUp,
      wickDownColor: colors.candleDown,
      borderVisible: false,
      priceLineVisible: false,
    });
    candles.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.26 } });
    const volume = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      lastValueVisible: false,
      priceLineVisible: false,
    });
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    candles.setData(
      clean.map((b) => ({
        time: b.date as Time,
        open: b.open!,
        high: b.high!,
        low: b.low!,
        close: b.close!,
      })),
    );
    volume.setData(
      clean.map((b) => ({
        time: b.date as Time,
        value: b.volume ?? 0,
        color: `${b.close! >= b.open! ? colors.candleUp : colors.candleDown}55`,
      })),
    );
    chart.timeScale().fitContent();

    const byDate = new Map(clean.map((b) => [b.date, b]));
    const onMove = (p: MouseEventParams) => setHover(p.time ? (byDate.get(String(p.time)) ?? null) : null);
    chart.subscribeCrosshairMove(onMove);
    return () => {
      chart.unsubscribeCrosshairMove(onMove);
      chart.remove();
    };
  }, [clean, colors]);

  if (clean.length === 0) return <p className="text-sm text-muted-foreground">Chưa có dữ liệu giá.</p>;

  return (
    <div>
      {/* fixed min-height: the legend must not push the chart when it appears */}
      {shown && (
        <dl
          className="mb-2 flex min-h-[2.5rem] flex-wrap content-start gap-x-4 gap-y-1 text-xs text-muted-foreground sm:min-h-[1.25rem]"
          aria-live="polite"
        >
          <div>
            <dt className="sr-only">Ngày</dt>
            <dd className="font-medium text-foreground">{fmtDate(shown.date)}</dd>
          </div>
          {(
            [
              ["Mở", fmtPrice(shown.open)],
              ["Cao", fmtPrice(shown.high)],
              ["Thấp", fmtPrice(shown.low)],
              ["Đóng", fmtPrice(shown.close)],
              ["KL", fmtVolume(shown.volume)],
              ["±", fmtPct(shown.ret_1d, 2, true)],
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="flex gap-1">
              <dt>{k}</dt>
              <dd className="text-foreground">{v}</dd>
            </div>
          ))}
        </dl>
      )}
      <div
        ref={ref}
        style={{ height }}
        role="img"
        aria-label={`Biểu đồ nến và khối lượng ${clean.length} phiên`}
      />
    </div>
  );
}
