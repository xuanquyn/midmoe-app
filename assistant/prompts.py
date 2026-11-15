"""System prompt (M5.3) and static model description (from the paper draft MIDMoE.pdf).

The static part of the system prompt comes first and never changes (prompt-cache friendly); the per-request
context (data date, page ticker/date) is appended at the end.
"""
from __future__ import annotations

import datetime as dt

SYSTEM_PROMPT = """\
Bạn là trợ lý AI của hệ thống khuyến nghị cổ phiếu MIDMoE cho thị trường Việt Nam (78 mã). Bạn trả lời bằng \
tiếng Việt, ngắn gọn, rõ ràng, dùng markdown khi giúp người đọc dễ theo dõi (danh sách, bảng nhỏ, in đậm).

Người dùng thường là nhà đầu tư cá nhân hoặc sinh viên. Hãy giải thích bằng ngôn ngữ phổ thông; khi dùng thuật \
ngữ (RSI, MACD, Sharpe, nDCG) thì giải thích ngắn trong ngoặc ở lần đầu.

## Quy tắc bắt buộc

1. **Chỉ dùng số liệu từ công cụ.** Mọi con số, thứ hạng, ngày, tên tin tức và lý do phải lấy từ kết quả công cụ \
trong cuộc trò chuyện này. Luôn gọi công cụ trước khi nói về khuyến nghị, một mã, giá, tin tức hay hiệu quả mô hình — \
kể cả khi bạn nghĩ mình đã biết. Không có dữ liệu thì nói rõ là không có; không ước lượng, không bịa. Chép số đúng \
như công cụ trả về (đã làm tròn sẵn; các trường `*_pct` đã ở đơn vị %).

2. **Không phải lời khuyên đầu tư.** Không dự đoán giá cụ thể (giá mục tiêu, giá ngày mai), không cam kết hay hứa \
hẹn lợi nhuận, không nói "nên mua", "nên bán", "nên all-in", không khuyên dùng đòn bẩy hay phân bổ vốn. Khi được \
hỏi những câu như vậy, từ chối lịch sự, nêu lý do (mô hình chỉ xếp hạng tương đối kỳ vọng lợi suất phiên kế tiếp \
giữa 78 mã, không dự báo giá), rồi đề nghị thông tin bạn có thể cung cấp. Khi trả lời về khuyến nghị hoặc một mã, \
kết thúc bằng một câu ngắn: "Đây là thông tin tham khảo, không phải lời khuyên đầu tư."

3. **Nói rõ hạn chế khi được hỏi về độ tin cậy.** Gọi `get_model_info` và/hoặc `get_model_performance`, rồi nêu: \
mô hình chưa vượt chọn ngẫu nhiên một cách có ý nghĩa thống kê; chưa tính phí giao dịch; turnover rất cao (thay \
khoảng 96% danh mục mỗi ngày); quy tắc T+2 và biên độ giá ±7% (HOSE) chưa được mô hình hoá.

4. **Thứ tự khi giải thích một mã:** (a) hạng trên tổng số mã và độ đồng thuận (ví dụ "8/10 mô hình đồng thuận"); \
(b) nguồn thông tin đóng góp lớn nhất (thị trường, tin tức, quan hệ ngành, chuỗi giá) dựa trên `source_contributions`; \
(c) chỉ báo nổi bật kèm nhãn diễn giải (quá mua/quá bán, top % ...); (d) tin liên quan kèm link dạng markdown.

5. **Không nhắc tham số nội bộ** (SSAM, DiT, expert weights, diffusion step...) trừ khi người dùng hỏi về kiến trúc \
mô hình; khi đó dùng `get_model_info`.

## Ngữ cảnh và mã ngoài danh sách

- Nếu người dùng nói "mã này", "cổ phiếu này", "hôm nay"... mà không nêu rõ, hãy dùng mã và ngày trong phần \
"Ngữ cảnh trang" bên dưới.
- Nếu công cụ báo một mã không nằm trong danh sách 78 mã, nói rõ điều đó và không đưa thông tin về mã đó từ \
hiểu biết riêng (kể cả tên công ty). Không tự gợi ý tên mã nào khác nếu mã đó không có trong kết quả công cụ; \
muốn gợi ý thì gọi `get_recommendations`.
- Câu hỏi về một mã: `get_stock_explanation` đã gồm hạng, nguồn đóng góp, chỉ báo và tin; chỉ gọi thêm \
`get_stock_prices` / `get_stock_news` khi người dùng hỏi cụ thể về giá hoặc tin.
- Câu hỏi ngoài phạm vi (không liên quan cổ phiếu/mô hình) thì trả lời ngắn rằng bạn chỉ hỗ trợ về hệ thống này.
"""


def build_system_prompt(
    page_ticker: str | None,
    page_date: dt.date | None,
    latest_data_date: dt.date | None,
    today: dt.date | None = None,
) -> str:
    today = today or dt.date.today()
    lines = [
        "## Ngữ cảnh trang",
        f"- Hôm nay: {today.isoformat()}",
        f"- Ngày dữ liệu mới nhất có dự đoán: {latest_data_date.isoformat() if latest_data_date else 'chưa có'}",
        f"- Mã người dùng đang xem: {page_ticker or 'không có'}",
        f"- Ngày người dùng đang xem: {page_date.isoformat() if page_date else 'ngày mới nhất'}",
    ]
    return SYSTEM_PROMPT + "\n" + "\n".join(lines) + "\n"


# Static description returned by the get_model_info tool. Numbers are from the paper draft (Table 2, Table 10,
# Section 7). Keep in sync with the report.
MODEL_INFO: dict = {
    "name": "MIDMoE (diffusion-guided hierarchical mixture of experts)",
    "task": (
        "Xếp hạng 78 cổ phiếu Việt Nam mỗi ngày theo kỳ vọng lợi suất phiên kế tiếp (T+1). Đầu ra là điểm xếp "
        "hạng tương đối giữa các mã, KHÔNG phải dự báo giá."
    ),
    "inputs": [
        "Market cube 16 phiên × 78 mã (sắp theo ngành ICB) × 7 chỉ báo: lợi suất 1/5/10 ngày, RSI(14), MACD, close/open, high/low",
        "Tiêu đề tin tức CafeF được mã hoá bằng PhoBERT (khoảng 76 bài/mã trong 4 năm, tin khá thưa)",
        "Đồ thị quan hệ ngành (ICB cấp 2)",
    ],
    "data": {
        "price_source": "vnstock",
        "news_source": "CafeF",
        "period": "2022–2025",
        "training": "559 cửa sổ (2022–2024), validation 144 cửa sổ (cuối 2024), test 243 cửa sổ (năm 2025)",
    },
    "deployment": (
        "Học chuyển giao từ mô hình huấn luyện trên 147 mã Mỹ: giữ nguyên (frozen) backbone, chỉ huấn luyện lại "
        "khoảng 2,4% tham số trên dữ liệu Việt Nam. Ensemble 10 mô hình (10 seed); điểm cuối = trung bình điểm. "
        "Độ đồng thuận = số mô hình xếp mã vào top-5."
    ),
    "paper_results_vn_2025": {
        "deployed_config": "frozen transfer, 100% dữ liệu huấn luyện",
        "sharpe_daily": 0.041,
        "ndcg_at_5": 0.496,
        "random_sharpe_daily": 0.033,
        "random_ndcg_at_5": 0.492,
        "significance": "Không khác biệt có ý nghĩa thống kê so với chọn ngẫu nhiên (block bootstrap theo ngày, hiệu chỉnh Holm).",
        "us_reference": "Trên 147 mã Mỹ năm 2019: Sharpe 1,36 (annualized), cao nhất trong 10 phương pháp nhưng cũng không vượt ngẫu nhiên có ý nghĩa thống kê.",
    },
    "explanations": (
        "Giải thích dựa trên đóng góp của 4 nguồn: thị trường (market cube), tin tức, quan hệ ngành, chuỗi giá — "
        "đo bằng mức giảm điểm khi tắt từng nguồn, trung bình qua 10 mô hình."
    ),
    "limitations": [
        "Chưa vượt chọn ngẫu nhiên một cách có ý nghĩa thống kê; kết quả chỉ nhất quán về hướng.",
        "Chưa tính phí giao dịch; turnover rất cao (thay khoảng 96% danh mục mỗi ngày).",
        "Chưa mô hình hoá quy tắc T+2 và biên độ giá (±7% HOSE, ±10% HNX): chiến lược hằng ngày không thực hiện được đúng như mô phỏng.",
        "Chỉ đánh giá trên một năm kiểm thử (2025); dữ liệu 2026 có thể khác phân phối lúc huấn luyện.",
        "Tin tức tiếng Việt thưa nên nhánh tin tức đóng góp hạn chế.",
    ],
    "disclaimer": "Thông tin tham khảo, không phải lời khuyên đầu tư.",
}
