"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

/** The date chosen in the header lives in the URL (?date=YYYY-MM-DD); null means "latest". */
export function useSelectedDate(): [string | null, (d: string | null) => void] {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const date = params.get("date");
  const setDate = useCallback(
    (d: string | null) => {
      const p = new URLSearchParams(params.toString());
      if (d) p.set("date", d);
      else p.delete("date");
      const q = p.toString();
      router.push(q ? `${pathname}?${q}` : pathname, { scroll: false });
    },
    [params, router, pathname],
  );
  return [date, setDate];
}

/** Append the current ?date= to an internal link so navigation keeps the selected day. */
export function useDateHref(): (path: string) => string {
  const params = useSearchParams();
  const date = params.get("date");
  return useCallback((path: string) => (date ? `${path}?date=${date}` : path), [date]);
}
