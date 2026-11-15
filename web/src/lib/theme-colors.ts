"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export interface ChartColors {
  card: string;
  foreground: string;
  muted: string;
  subtle: string;
  grid: string;
  axis: string;
  series1: string;
  series2: string;
  series3: string;
  pos: string;
  neg: string;
  candleUp: string;
  candleDown: string;
}

const VARS: Record<keyof ChartColors, string> = {
  card: "--card",
  foreground: "--foreground",
  muted: "--muted-foreground",
  subtle: "--subtle",
  grid: "--grid",
  axis: "--axis",
  series1: "--series-1",
  series2: "--series-2",
  series3: "--series-3",
  pos: "--pos",
  neg: "--neg",
  candleUp: "--candle-up",
  candleDown: "--candle-down",
};

function read(): ChartColors {
  const style = getComputedStyle(document.documentElement);
  const out = {} as ChartColors;
  for (const [k, v] of Object.entries(VARS)) out[k as keyof ChartColors] = style.getPropertyValue(v).trim();
  return out;
}

/** Resolved token colors for canvas/SVG libraries that can't read CSS variables; updates on theme change. */
export function useChartColors(): ChartColors | null {
  const { resolvedTheme } = useTheme();
  const [colors, setColors] = useState<ChartColors | null>(null);
  useEffect(() => {
    // wait one frame so the `dark` class is applied before reading
    const id = requestAnimationFrame(() => setColors(read()));
    return () => cancelAnimationFrame(id);
  }, [resolvedTheme]);
  return colors;
}
