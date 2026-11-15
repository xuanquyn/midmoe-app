import type { Metadata, Viewport } from "next";
import { Suspense } from "react";

import { ChatPanel } from "@/components/chat/chat-panel";
import { Disclaimer, Header } from "@/components/header";

import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: { default: "MIDMoE — Khuyến nghị cổ phiếu VN", template: "%s · MIDMoE" },
  description:
    "Hệ thống khuyến nghị đầu tư chứng khoán dựa trên mô hình học sâu phân cấp MIDMoE và trợ lý AI. Thông tin tham khảo, không phải lời khuyên đầu tư.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fcfcfb" },
    { media: "(prefers-color-scheme: dark)", color: "#1a1a19" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body>
        <Providers>
          <Header />
          <Disclaimer />
          <div className="flex min-h-[calc(100vh-57px)]">
            <main className="min-w-0 flex-1 px-4 py-4 lg:px-6">
              {/* pages read ?date= (useSearchParams), which needs a Suspense boundary in the App Router */}
              <Suspense fallback={<div className="mx-auto h-96 max-w-6xl animate-pulse rounded-lg bg-muted" />}>
                {children}
              </Suspense>
            </main>
            <ChatPanel />
          </div>
        </Providers>
      </body>
    </html>
  );
}
