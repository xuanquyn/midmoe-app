// Vietnamese number / date formatting.

const nf = (min: number, max: number) =>
  new Intl.NumberFormat("vi-VN", { minimumFractionDigits: min, maximumFractionDigits: max });

export function fmtNum(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return nf(digits, digits).format(x);
}

/** Fraction -> percent string, e.g. 0.0123 -> "1,23%". */
export function fmtPct(x: number | null | undefined, digits = 2, signed = false): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const s = nf(digits, digits).format(x * 100);
  return `${signed && x > 0 ? "+" : ""}${s}%`;
}

export function fmtPrice(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  return nf(0, 2).format(x);
}

export function fmtVolume(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  if (x >= 1e6) return `${nf(1, 1).format(x / 1e6)} tr`;
  if (x >= 1e3) return `${nf(0, 0).format(x / 1e3)} N`;
  return nf(0, 0).format(x);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const [y, m, d] = iso.slice(0, 10).split("-");
  return `${d}/${m}/${y}`;
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    timeZone: "Asia/Ho_Chi_Minh",
  });
}

export function consensus(votes: number, n = 10): string {
  return `${votes}/${n} mô hình đồng thuận`;
}

export const SOURCE_LABELS: Record<string, string> = {
  market: "Thị trường",
  news: "Tin tức",
  graph: "Quan hệ ngành",
  timeseries: "Chuỗi giá",
};

export const FEATURE_LABELS: Record<string, string> = {
  ret_1d: "Lợi suất 1 ngày",
  ret_5d: "Lợi suất 5 ngày",
  ret_10d: "Lợi suất 10 ngày",
  rsi: "RSI (14)",
  macd: "MACD",
  close_open_ratio: "Đóng cửa / Mở cửa",
  high_low_ratio: "Cao / Thấp",
};

export function fmtFeature(name: string, value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (name.startsWith("ret_")) return fmtPct(value, 2, true);
  if (name === "rsi") return fmtNum(value, 1);
  if (name === "macd") return fmtNum(value, 3);
  return fmtNum(value, 3);
}
