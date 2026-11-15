// Typed client for the FastAPI backend (mirrors backend/schemas.py).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type RunStatus = "success" | "degraded" | "failed" | "skipped";

export interface PipelineRun {
  id: number;
  run_date: string;
  status: RunStatus;
  started_at: string;
  finished_at: string | null;
  n_tickers: number | null;
  n_news: number | null;
  error: string | null;
}

export interface Health {
  status: "ok" | "degraded";
  db: "ok" | "error";
  last_run: PipelineRun | null;
}

export interface DatesOut {
  dates: string[];
  latest: string | null;
}

export interface RankingItem {
  ticker: string;
  name: string | null;
  icb_sector: string | null;
  score: number;
  score_std: number;
  rank: number;
  top5_votes: number;
  rank_change: number | null;
}

export interface RecommendationItem extends RankingItem {
  reason: string | null;
}

export interface RecommendationsOut {
  date: string;
  k: number;
  items: RecommendationItem[];
}

export interface RankingsOut {
  date: string;
  sector: string | null;
  items: RankingItem[];
}

export interface PriceBar {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  ret_1d: number | null;
  ret_5d: number | null;
  ret_10d: number | null;
  rsi: number | null;
  macd: number | null;
  close_open_ratio: number | null;
  high_low_ratio: number | null;
}

export interface StockOut {
  ticker: string;
  name: string | null;
  exchange: string | null;
  icb_sector: string | null;
  prices: PriceBar[];
  rank_history: { date: string; rank: number; score: number }[];
}

export type SourceName = "market" | "news" | "graph" | "timeseries";

export interface ExplanationPayload {
  rank: number;
  score: number;
  top5_votes: number;
  score_std: number;
  n_models: number;
  sources: Record<SourceName, { delta: number; rank_without: number }>;
  features: Record<string, { value: number | null; pct: number | null; label: string | null }>;
  sector: { name: string; rank_in_sector: number; n_in_sector: number; top: string[] };
  news: { date: string; title: string; url: string }[];
  summary: string;
}

export interface ExplanationOut {
  ticker: string;
  date: string;
  payload: ExplanationPayload;
}

export interface NewsItem {
  id: number;
  ticker: string;
  date: string;
  title: string;
  url: string;
  source: string | null;
  published_at: string | null;
}

export interface BacktestPoint {
  date: string;
  topk: number;
  random: number;
  vnindex: number | null;
  topk_cum: number;
  random_cum: number;
  vnindex_cum: number | null;
}

export interface BacktestMetrics {
  SR: number;
  IRR: number;
  MaxDD: number;
  nDCG5: number | null;
}

export interface BacktestOut {
  from_date: string;
  to_date: string;
  k: number;
  n_days: number;
  vnindex_coverage: number;
  series: BacktestPoint[];
  metrics: Partial<Record<"topk" | "random" | "vnindex", BacktestMetrics>>;
}

export interface ChatMessageOut {
  id: number;
  role: "user" | "assistant";
  content: string;
  tools: string[];
  created_at: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function qs(params: Record<string, string | number | null | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

export async function apiGet<T>(path: string, params: Record<string, string | number | null | undefined> = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api${path}${qs(params)}`, { cache: "no-store" });
  } catch {
    throw new ApiError(0, "Không kết nối được tới máy chủ API.");
  }
  if (!res.ok) {
    let detail = `Lỗi ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => apiGet<Health>("/health"),
  dates: () => apiGet<DatesOut>("/dates"),
  sectors: () => apiGet<string[]>("/sectors"),
  recommendations: (date?: string | null, k = 5) => apiGet<RecommendationsOut>("/recommendations", { date, k }),
  rankings: (date?: string | null, sector?: string | null) => apiGet<RankingsOut>("/rankings", { date, sector }),
  stock: (ticker: string, from?: string | null, to?: string | null) =>
    apiGet<StockOut>(`/stocks/${encodeURIComponent(ticker)}`, { from, to }),
  explanation: (ticker: string, date?: string | null) =>
    apiGet<ExplanationOut>(`/stocks/${encodeURIComponent(ticker)}/explanation`, { date }),
  news: (ticker: string, days = 7) => apiGet<NewsItem[]>(`/stocks/${encodeURIComponent(ticker)}/news`, { days }),
  backtest: (from?: string | null, to?: string | null, k = 5) => apiGet<BacktestOut>("/backtest", { from, to, k }),
  chatMessages: (sessionId: string) => apiGet<ChatMessageOut[]>(`/chat/${sessionId}/messages`),
};

// ---------------------------------------------------------------------------- chat SSE
export type ChatEvent =
  | { type: "session"; session_id: string }
  | { type: "delta"; text: string }
  | { type: "tool_start"; name: string; label: string }
  | { type: "tool_end"; name: string; ok: boolean }
  | { type: "error"; message: string }
  | { type: "done"; message_id: number | null };

/** POST /api/chat and yield SSE events (EventSource can't POST, so parse the stream by hand). */
export async function* streamChat(
  body: { session_id: string | null; message: string; context: { ticker?: string | null; date?: string | null } },
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    yield { type: "error", message: `Không gửi được câu hỏi (lỗi ${res.status}).` };
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            yield JSON.parse(line.slice(6)) as ChatEvent;
          } catch {
            /* ignore malformed frame */
          }
        }
      }
    }
  }
}
