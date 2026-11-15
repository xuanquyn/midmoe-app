import { FileText } from "lucide-react";
import type { Metadata } from "next";

import { buttonVariants, Card, CardContent, CardHeader, CardTitle } from "@/components/ui/primitives";

export const metadata: Metadata = { title: "Giới thiệu" };

const STEPS = [
  { title: "Dữ liệu mỗi phiên", items: ["Giá 78 mã (vnstock)", "Tin CafeF → PhoBERT", "Đồ thị ngành ICB"] },
  { title: "Mã hoá", items: ["Market cube 16 × 78 × 7 → Conv3D", "Chuỗi giá từng mã", "Tin tức · quan hệ ngành"] },
  { title: "Hỗn hợp chuyên gia", items: ["Ngữ cảnh từng mã", "Diffusion transformer sinh tham số", "Chuyên gia 2 tầng"] },
  { title: "Kết quả", items: ["Điểm xếp hạng 78 mã", "Trung bình 10 mô hình", "Top 5 + giải thích"] },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <section className="space-y-1">
        <h1 className="text-xl font-semibold">Về hệ thống</h1>
        <p className="text-sm text-muted-foreground">
          Đồ án: Hệ thống khuyến nghị đầu tư chứng khoán dựa trên mô hình học sâu phân cấp (MIDMoE) và trợ lý ảo AI.
        </p>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Mô hình MIDMoE</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <ol className="grid gap-2 sm:grid-cols-4" aria-label="Luồng xử lý của mô hình">
            {STEPS.map((s, i) => (
              <li key={s.title} className="relative rounded-lg border bg-background p-3">
                <p className="mb-1 text-xs font-medium text-primary">Bước {i + 1}</p>
                <p className="mb-1.5 font-medium">{s.title}</p>
                <ul className="space-y-0.5 text-xs text-muted-foreground">
                  {s.items.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
                {i < STEPS.length - 1 && (
                  <span aria-hidden className="absolute -right-2 top-1/2 hidden -translate-y-1/2 text-subtle sm:block">
                    →
                  </span>
                )}
              </li>
            ))}
          </ol>
          <p>
            Mỗi phiên được biểu diễn thành một <strong>market cube</strong> ba chiều (16 phiên gần nhất × 78 mã sắp theo
            ngành × 7 chỉ báo) và mã hoá bằng tích chập 3D. Kết hợp với chuỗi giá, tin tức và đồ thị ngành, mô hình tạo
            ngữ cảnh cho từng mã; một diffusion transformer sinh tham số cho các chuyên gia, rồi cơ chế attention thưa gộp
            các chuyên gia thành điểm xếp hạng cuối cùng.
          </p>
          <p>
            Bản triển khai dùng <strong>học chuyển giao</strong> từ mô hình huấn luyện trên 147 mã Mỹ: giữ nguyên backbone,
            chỉ huấn luyện lại khoảng 2,4% tham số trên dữ liệu Việt Nam 2022–2024. Mười mô hình (10 seed) chạy song song;
            điểm cuối là trung bình, độ đồng thuận là số mô hình xếp mã vào top 5.
          </p>
          <p>
            <strong>Giải thích</strong> dựa trên đóng góp của 4 nguồn thông tin (thị trường, tin tức, quan hệ ngành, chuỗi
            giá): điểm của mã thay đổi bao nhiêu khi tắt từng nguồn.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Dữ liệu</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <tbody>
              {[
                ["Danh mục", "78 cổ phiếu Việt Nam, phân ngành ICB cấp 2"],
                ["Giá", "vnstock, cập nhật 16:30 các ngày giao dịch"],
                ["Tin tức", "CafeF, mã hoá bằng PhoBERT (khoảng 76 bài/mã trong 4 năm)"],
                ["Huấn luyện", "559 cửa sổ (2022–2024), validation 144 cửa sổ cuối 2024"],
                ["Kiểm thử", "243 phiên năm 2025; từ 2026 là giai đoạn chạy thật"],
              ].map(([k, v]) => (
                <tr key={k} className="border-b last:border-0">
                  <th scope="row" className="w-32 py-2 text-left font-medium">{k}</th>
                  <td className="py-2 text-muted-foreground">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Hạn chế</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            <li>Chưa vượt chọn ngẫu nhiên một cách có ý nghĩa thống kê; kết quả chỉ nhất quán về hướng.</li>
            <li>Chưa tính phí giao dịch; danh mục thay khoảng 96% mỗi ngày.</li>
            <li>Chưa mô hình hoá quy tắc T+2 và biên độ giá (±7% HOSE, ±10% HNX).</li>
            <li>Chỉ một năm kiểm thử; dữ liệu 2026 có thể khác phân phối lúc huấn luyện (mô hình huấn luyện đến hết 2024).</li>
            <li>Tin tức tiếng Việt thưa nên nhánh tin tức đóng góp hạn chế.</li>
          </ul>
        </CardContent>
      </Card>

      <a href="/MIDMoE.pdf" target="_blank" rel="noopener" className={buttonVariants({ variant: "outline" })}>
        <FileText className="h-4 w-4" /> Đọc paper (PDF)
      </a>
    </div>
  );
}
