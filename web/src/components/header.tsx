"use client";

import { CalendarDays, LineChart, Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { Suspense, useEffect, useState } from "react";

import { Button } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { useDateHref, useSelectedDate } from "@/lib/use-selected-date";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Tổng quan" },
  { href: "/performance", label: "Hiệu quả" },
  { href: "/about", label: "Giới thiệu" },
];

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const dark = mounted && resolvedTheme === "dark";
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={dark ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
      onClick={() => setTheme(dark ? "light" : "dark")}
    >
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}

function DateSelect() {
  const [date, setDate] = useSelectedDate();
  const { data } = useApi(() => api.dates(), []);
  const first = data?.dates[0];
  const latest = data?.latest ?? undefined;
  return (
    <div className="flex items-center gap-1.5 text-sm">
      <CalendarDays className="h-4 w-4 text-muted-foreground" aria-hidden />
      <label htmlFor="date-picker" className="sr-only">
        Chọn ngày (ngày không có phiên sẽ dùng phiên gần nhất trước đó)
      </label>
      <input
        id="date-picker"
        type="date"
        className="h-9 rounded-md border bg-card px-2 text-sm"
        min={first}
        max={latest}
        value={date ?? latest ?? ""}
        disabled={!latest}
        onChange={(e) => setDate(e.target.value && e.target.value !== latest ? e.target.value : null)}
      />
      {date && (
        <Button variant="ghost" size="sm" onClick={() => setDate(null)}>
          Mới nhất
        </Button>
      )}
    </div>
  );
}

function NavLinks({ href }: { href: (p: string) => string }) {
  const pathname = usePathname();
  return (
    <>
      {NAV.map((n) => {
        const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
        return (
          <Link
            key={n.href}
            href={href(n.href)}
            aria-current={active ? "page" : undefined}
            className={cn(
              "rounded-md px-2.5 py-1.5 text-sm hover:bg-muted",
              active ? "bg-muted font-medium text-foreground" : "text-muted-foreground",
            )}
          >
            {n.label}
          </Link>
        );
      })}
    </>
  );
}

function DatedNavLinks() {
  const href = useDateHref();
  return <NavLinks href={href} />;
}

export function Header() {
  // Only the parts that read ?date= sit behind Suspense, so the rest of the header is prerendered.
  return (
    <header className="sticky top-0 z-30 border-b bg-card/95 backdrop-blur">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <LineChart className="h-5 w-5 text-primary" aria-hidden />
          <span>MIDMoE</span>
          <span className="hidden text-sm font-normal text-muted-foreground sm:inline">Khuyến nghị cổ phiếu VN</span>
        </Link>
        <nav className="order-3 flex w-full gap-1 sm:order-none sm:w-auto" aria-label="Điều hướng chính">
          <Suspense fallback={<NavLinks href={(p) => p} />}>
            <DatedNavLinks />
          </Suspense>
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <Suspense fallback={<div className="h-9 w-40" />}>
            <DateSelect />
          </Suspense>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

export function Disclaimer() {
  return (
    <div className="border-b bg-warning px-4 py-1.5 text-center text-xs text-warning-foreground" role="note">
      Thông tin tham khảo, không phải lời khuyên đầu tư. Mô hình chưa vượt chọn ngẫu nhiên một cách có ý nghĩa thống
      kê và chưa tính phí giao dịch.
    </div>
  );
}
