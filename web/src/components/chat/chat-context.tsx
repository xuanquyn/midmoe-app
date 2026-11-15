"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export interface PageContext {
  ticker: string | null;
  date: string | null;
}

interface ChatCtx {
  open: boolean;
  setOpen: (v: boolean) => void;
  page: PageContext;
  setPage: (p: PageContext) => void;
  /** question queued by a button elsewhere on the page ("Hỏi trợ lý"); the panel sends it and clears it */
  pending: string | null;
  ask: (question: string, ctx?: Partial<PageContext>) => void;
  consumePending: () => string | null;
}

const Ctx = createContext<ChatCtx | null>(null);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  // Open by default only on wide screens; on phones the panel is a full-screen overlay opened on demand.
  const [open, setOpen] = useState(false);
  useEffect(() => {
    setOpen(window.matchMedia("(min-width: 1024px)").matches);
  }, []);
  const [page, setPageState] = useState<PageContext>({ ticker: null, date: null });
  const [pending, setPending] = useState<string | null>(null);

  const setPage = useCallback((p: PageContext) => {
    setPageState((prev) => (prev.ticker === p.ticker && prev.date === p.date ? prev : p));
  }, []);

  const ask = useCallback((question: string, ctx?: Partial<PageContext>) => {
    if (ctx) setPageState((prev) => ({ ...prev, ...ctx }));
    setPending(question);
    setOpen(true);
  }, []);

  const consumePending = useCallback(() => {
    const q = pending;
    setPending(null);
    return q;
  }, [pending]);

  const value = useMemo(
    () => ({ open, setOpen, page, setPage, pending, ask, consumePending }),
    [open, page, setPage, pending, ask, consumePending],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Pages call this to tell the assistant what the user is looking at. */
export function usePageContext(ticker: string | null, date: string | null) {
  const { setPage } = useChat();
  useEffect(() => setPage({ ticker, date }), [ticker, date, setPage]);
}

export function useChat(): ChatCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useChat must be used inside ChatProvider");
  return v;
}
