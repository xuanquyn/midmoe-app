"use client";

import { Bot, Loader2, MessageSquarePlus, PanelRightClose, SendHorizontal, Square, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Markdown } from "@/components/chat/markdown";
import { useChat } from "@/components/chat/chat-context";
import { Button } from "@/components/ui/primitives";
import { api, streamChat } from "@/lib/api";
import { fmtDate } from "@/lib/format";
import { cn } from "@/lib/utils";

const SESSION_KEY = "midmoe.chat.session";

interface Msg {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: string | null; // "Đang tra cứu ..." while a tool runs
  error?: string | null;
  streaming?: boolean;
}

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function suggestions(ticker: string | null): string[] {
  if (ticker)
    return [
      `Vì sao ${ticker} được xếp hạng như vậy?`,
      `Có tin gì mới về ${ticker} không?`,
      `So sánh ${ticker} với mã đứng đầu hôm nay`,
    ];
  return ["Top 5 mã hôm nay là gì?", "Mô hình đáng tin cậy đến mức nào?", "Hiệu quả của mô hình trong năm 2025?"];
}

let idCounter = 0;
const newId = () => `m${Date.now()}-${idCounter++}`;

export function ChatPanel() {
  const { open, setOpen, page, pending, consumePending } = useChat();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // restore the previous session
  useEffect(() => {
    const sid = storage()?.getItem(SESSION_KEY);
    if (!sid) return;
    setSessionId(sid);
    api
      .chatMessages(sid)
      .then((rows) => setMessages(rows.map((r) => ({ id: `h${r.id}`, role: r.role, content: r.content }))))
      .catch(() => {
        storage()?.removeItem(SESSION_KEY);
        setSessionId(null);
      });
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages]);

  const patchLast = useCallback((fn: (m: Msg) => Msg) => {
    setMessages((prev) => (prev.length ? [...prev.slice(0, -1), fn(prev[prev.length - 1])] : prev));
  }, []);

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || busy) return;
      setInput("");
      setBusy(true);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "user", content: question },
        { id: newId(), role: "assistant", content: "", streaming: true, status: "Đang suy nghĩ…" },
      ]);
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      try {
        for await (const ev of streamChat(
          { session_id: sessionId, message: question, context: { ticker: page.ticker, date: page.date } },
          ctrl.signal,
        )) {
          if (ev.type === "session") {
            setSessionId(ev.session_id);
            storage()?.setItem(SESSION_KEY, ev.session_id);
          } else if (ev.type === "delta") {
            patchLast((m) => ({ ...m, content: m.content + ev.text, status: null }));
          } else if (ev.type === "tool_start") {
            patchLast((m) => ({ ...m, status: `${ev.label}…` }));
          } else if (ev.type === "tool_end") {
            patchLast((m) => ({ ...m, status: "Đang tổng hợp câu trả lời…" }));
          } else if (ev.type === "error") {
            patchLast((m) => ({ ...m, error: ev.message }));
          }
        }
      } catch (e) {
        if (!(e instanceof DOMException && e.name === "AbortError")) {
          patchLast((m) => ({ ...m, error: "Mất kết nối tới máy chủ, vui lòng thử lại." }));
        }
      } finally {
        patchLast((m) => ({ ...m, streaming: false, status: null }));
        setBusy(false);
        abortRef.current = null;
      }
    },
    [busy, sessionId, page.ticker, page.date, patchLast],
  );

  // questions queued from elsewhere ("Hỏi trợ lý" buttons)
  useEffect(() => {
    if (pending && !busy) {
      const q = consumePending();
      if (q) void send(q);
    }
  }, [pending, busy, consumePending, send]);

  const reset = () => {
    abortRef.current?.abort();
    storage()?.removeItem(SESSION_KEY);
    setSessionId(null);
    setMessages([]);
  };

  if (!open) {
    return (
      <Button
        className="fixed bottom-4 right-4 z-40 h-12 rounded-full px-4 shadow-lg"
        onClick={() => setOpen(true)}
        aria-label="Mở trợ lý AI"
      >
        <Bot className="h-5 w-5" /> Trợ lý AI
      </Button>
    );
  }

  return (
    <aside
      aria-label="Trợ lý AI"
      className={cn(
        "fixed inset-0 z-40 flex flex-col bg-card",
        "lg:sticky lg:top-[57px] lg:z-10 lg:h-[calc(100vh-57px)] lg:w-[380px] lg:shrink-0 lg:border-l",
      )}
    >
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <Bot className="h-5 w-5 text-primary" aria-hidden />
        <h2 className="text-sm font-semibold">Trợ lý AI</h2>
        <div className="ml-auto flex gap-1">
          <Button variant="ghost" size="icon" onClick={reset} aria-label="Cuộc trò chuyện mới" title="Cuộc trò chuyện mới">
            <MessageSquarePlus className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setOpen(false)} aria-label="Thu gọn trợ lý" title="Thu gọn">
            <PanelRightClose className="hidden h-4 w-4 lg:block" />
            <X className="h-4 w-4 lg:hidden" />
          </Button>
        </div>
      </div>

      <div className="border-b px-3 py-1.5 text-xs text-muted-foreground">
        Đang xem: <span className="font-medium text-foreground">{page.ticker ?? "toàn thị trường"}</span>
        {" · "}
        {page.date ? fmtDate(page.date) : "ngày mới nhất"}
      </div>

      <div ref={listRef} className="scroll-thin flex-1 space-y-3 overflow-y-auto px-3 py-3" aria-live="polite">
        {messages.length === 0 && (
          <div className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Hỏi về khuyến nghị, lý do xếp hạng của một mã, chỉ báo, tin tức hoặc hiệu quả của mô hình.
            </p>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="ml-8 rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
              {m.content}
            </div>
          ) : (
            <div key={m.id} className="mr-4 space-y-1.5">
              {m.content && <Markdown>{m.content}</Markdown>}
              {m.status && (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> {m.status}
                </p>
              )}
              {m.error && <p className="text-xs text-down">{m.error}</p>}
            </div>
          ),
        )}
      </div>

      {!busy && (
        <div className="flex flex-wrap gap-1.5 border-t px-3 pt-2">
          {suggestions(page.ticker).map((s) => (
            <button
              key={s}
              onClick={() => void send(s)}
              className="rounded-full border px-2.5 py-1 text-left text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <form
        className="flex items-end gap-2 p-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <label htmlFor="chat-input" className="sr-only">
          Câu hỏi
        </label>
        <textarea
          id="chat-input"
          rows={2}
          value={input}
          maxLength={4000}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send(input);
            }
          }}
          placeholder="Nhập câu hỏi…"
          className="min-h-[2.5rem] flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
        />
        {busy ? (
          <Button type="button" variant="outline" size="icon" onClick={() => abortRef.current?.abort()} aria-label="Dừng">
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button type="submit" size="icon" disabled={!input.trim()} aria-label="Gửi">
            <SendHorizontal className="h-4 w-4" />
          </Button>
        )}
      </form>
      <p className="px-3 pb-2 text-[11px] text-subtle">
        Trợ lý trả lời dựa trên dữ liệu của hệ thống. Không phải lời khuyên đầu tư.
      </p>
    </aside>
  );
}
