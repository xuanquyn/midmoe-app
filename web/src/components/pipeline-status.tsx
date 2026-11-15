"use client";

import { AlertTriangle, CheckCircle2, CircleSlash, XCircle } from "lucide-react";

import { api, type RunStatus } from "@/lib/api";
import { fmtDate, fmtDateTime } from "@/lib/format";
import { useApi } from "@/lib/use-api";

const STATUS: Record<RunStatus, { label: string; icon: typeof CheckCircle2; cls: string }> = {
  success: { label: "Thành công", icon: CheckCircle2, cls: "text-up" },
  degraded: { label: "Thiếu một phần dữ liệu", icon: AlertTriangle, cls: "text-warning-foreground" },
  failed: { label: "Lỗi", icon: XCircle, cls: "text-down" },
  skipped: { label: "Không có phiên", icon: CircleSlash, cls: "text-muted-foreground" },
};

export function PipelineStatus() {
  const { data, error } = useApi(() => api.health(), []);
  if (error) return <p className="text-xs text-down">Không lấy được trạng thái hệ thống: {error}</p>;
  const run = data?.last_run;
  if (!run) return <p className="text-xs text-muted-foreground">{data ? "Pipeline chưa chạy lần nào." : " "}</p>;
  const s = STATUS[run.status];
  const Icon = s.icon;
  return (
    <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
      <Icon className={`h-3.5 w-3.5 ${s.cls}`} aria-hidden />
      <span>
        Cập nhật lúc {fmtDateTime(run.finished_at ?? run.started_at)} cho phiên {fmtDate(run.run_date)}
      </span>
      <span className={s.cls}>· {s.label}</span>
    </p>
  );
}
