# Việc còn lại trước demo 23/11/2026

Cập nhật: 15/11/2026. Tiến độ chi tiết theo task: [PLAN.md](../PLAN.md).

## Mốc thời gian

Demo vào thứ Hai 23/11. PLAN (M7.4) yêu cầu **3 phiên liên tiếp chạy thành công trước demo**, nên phiên chạy
được đầu tiên phải là **thứ Tư 18/11** (các phiên 18, 19, 20/11).

| Ngày | Việc | Ai |
|---|---|---|
| **16/11 (CN)** | Chạy cell export trên Kaggle (bước 1), gửi notebook `midmoe-vn-cube-build` (bước 2), lấy API key LLM (bước 3) | Nhóm |
| 16–17/11 | Viết vnstock/CafeF/PhoBERT, job hằng ngày, backfill (M2.1–M2.8); chạy parity test | Claude, khi có đầu vào |
| 17/11 | Thuê VPS, deploy (bước 5), chạy backfill 2025 → nay, chạy eval trợ lý | Nhóm + Claude |
| **18–20/11** | Theo dõi 3 phiên lúc 16:30 (M7.4) | Nhóm |
| 20–22/11 | Test UX với ≥ 10 người (SUS), chuẩn bị slide/kịch bản demo | Nhóm |

## Đã xong

- M1: đóng gói `MarketMoE`, `Predictor` ensemble 10 seed, cell export Kaggle, parity test (đang skip vì chưa có weights).
- M2.2: 7 chỉ báo khớp chính xác pandas-ta 0.4.x mà notebook đã dùng. M2.5: dựng cube một ngày.
- M3: giải thích khuyến nghị (đóng góp 4 nguồn, chỉ báo, ngành, tin, độ đồng thuận).
- M4: database + API + backtest (top-5 vs ngẫu nhiên vs VN-Index). M5: trợ lý AI + bộ eval 50 câu.
- M6: web Next.js. M7.1/M7.2/M7.5: Docker Compose, `.env.example`, README.

Tất cả đã test trên dữ liệu demo tổng hợp và model khởi tạo ngẫu nhiên; **chưa chạy với dữ liệu hay weights thật**.

---

## Bước 1 — Export weights từ Kaggle (M1.2) ⟵ quan trọng nhất

Notebook `midmoe-vn-transfer` đã chạy nhưng **không lưu weights**: cell 20 train 150 model rồi chỉ giữ dự đoán
(`vn_preds.npz`) và bảng kết quả (`*.csv`); model nằm trong RAM và mất khi session kết thúc. File `.ipynb` chỉ chứa
code và phần in ra màn hình. Vì vậy phải train lại 10 model của cấu hình deploy (chỉ 10 lần thay vì 150, nhanh hơn
nhiều so với cell 20).

0. Trước khi chạy lại: vào tab **Output** của phiên bản notebook đã chạy trên Kaggle, tải `vn_preds.npz` (dùng để so
   thứ hạng với paper). Nếu lần chạy đó không được *Save Version* thì file này không còn, có thể bỏ qua.

1. Mở notebook `midmoe-vn-transfer` trên Kaggle, giữ nguyên các input `midmoe-vn-cube-build` và `midmoe-conv3d-weights`.
2. Chạy các cell từ đầu **đến hết Phase 7 phần định nghĩa hàm** (cell 0–19). **Bỏ qua** cell 20 (vòng thí nghiệm
   3 cấu hình × 5 tỉ lệ × 10 seed) và các cell sau đó.
3. Thêm một cell mới ở cuối, dán toàn bộ nội dung file [`notebooks/export_vn_weights.py`](../notebooks/export_vn_weights.py), rồi chạy.
   Cell này huấn luyện lại 10 seed cấu hình `transfer_minimal`, 100% dữ liệu (lâu cỡ thời gian 10 lần train như lúc làm paper).
4. Đọc output của cell, kiểm tra:
   - Có dòng `Saved vn_norm_params.json: {...}`. Nếu thấy `!!! Không tìm thấy normalisation params VN`, nghĩa là
     cell không đoán được tên file tham số chuẩn hoá: mở tab *Output* của notebook `midmoe-vn-cube-build`, tìm file
     chứa `clip_min`/`clip_max`/`mins`/`maxs`, rồi gửi tên file cho Claude (hoặc sửa biến `cands` trong cell).
   - Có file `icb_industry.csv` hoặc `*sector_map*` được copy. Nếu không có, gửi bảng mã → tên công ty → ngành ICB.
5. Tải `/kaggle/working/midmoe_vn_artifacts.zip`, giải nén vào `artifacts/vn/` trong repo.
6. Cuối output có bảng `seed | SR paper | SR export | diff` so với 10 giá trị SR của lần chạy làm paper (trung bình
   0,0410). Trùng hoàn toàn: model train lại giống hệt paper. Lệch nhỏ: do GPU không tất định, chấp nhận được.
   Lệch lớn: báo Claude trước khi deploy.
7. Chép `vn_preds.npz` (bước 0) vào `artifacts/vn/` nếu có.
8. Chạy parity test và gửi kết quả cho Claude:
   ```bash
   .venv/bin/python -m pytest tests/test_parity.py -s
   ```

`artifacts/` đã nằm trong `.gitignore`: **không commit weights lên GitHub**.

## Bước 2 — Gửi notebook `midmoe-vn-cube-build` (M2.1, M2.3, M2.4)

Tải file `.ipynb` từ Kaggle (kèm output đã chạy) và đặt vào thư mục gốc của project. Claude cần lấy từ đó:

- vnstock: lệnh cài (version, nguồn cài) và hàm lấy giá (nguồn `VCI`/`TCBS`, giá điều chỉnh hay không).
  `vnstock` **đã bị gỡ khỏi PyPI**, nên phải biết notebook cài từ đâu.
- CafeF: code crawl và **quy tắc gán ngày** cho tin.
- PhoBERT: tên model, tokenizer, `max_len`, cách pooling, có tách từ (VnCoreNLP / underthesea) hay không.
- Cách lưu tham số chuẩn hoá và danh sách 78 mã + ngành.

## Bước 3 — API key cho trợ lý AI (quyết định D1)

Mặc định dùng **Gemini** vì có free tier:

1. Vào https://aistudio.google.com, đăng nhập Google, chọn *Get API key* → *Create API key*.
2. Tạo file `.env` từ mẫu và điền key:
   ```bash
   cp .env.example .env
   # sửa trong .env:
   # LLM_PROVIDER=gemini
   # LLM_API_KEY=<key vừa tạo>
   ```
3. Không gửi key lên GitHub hay vào chat nhóm công khai (`.env` đã được bỏ qua trong git).

Free tier có giới hạn số request/phút. Nếu khi demo bị báo "quá tải", có thể chuyển sang OpenAI hoặc Anthropic
(trả phí) bằng cách đổi `LLM_PROVIDER` và `LLM_API_KEY`.

## Bước 4 — Hỏi giảng viên (quyết định D6)

- Hạn nộp cuối cho giai đoạn kiểm thử + báo cáo.
- Demo dùng URL public hay chạy trên máy cá nhân được? (Quyết định có cần VPS ở bước 5 không.)

## Bước 5 — Deploy (M7.3, quyết định D4)

Mặc định: Docker Compose trên 1 VPS.

1. Thuê VPS Ubuntu 22.04+, tối thiểu **2 vCPU / 4 GB RAM / 30 GB đĩa** (torch + PhoBERT + PostgreSQL).
   Nên chọn vùng Singapore hoặc Việt Nam để gần nguồn dữ liệu.
2. Trỏ tên miền (nếu có) về IP của VPS, ví dụ `midmoe.<domain>` và `api.midmoe.<domain>`.
3. Trên VPS:
   ```bash
   curl -fsSL https://get.docker.com | sh
   git clone https://github.com/xuanquyn/midmoe-app.git && cd midmoe-app
   cp .env.example .env    # đặt POSTGRES_PASSWORD mạnh, LLM_API_KEY,
                           # NEXT_PUBLIC_API_URL=https://api.midmoe.<domain>, CORS_ORIGINS=https://midmoe.<domain>
   # từ máy cá nhân: scp -r artifacts/vn <user>@<ip>:~/midmoe-app/artifacts/
   docker compose up -d --build
   docker compose ps       # 4 service phải "healthy"
   ```
4. Đặt HTTPS bằng Caddy (Claude có thể viết sẵn `Caddyfile` khi bạn có tên miền). Nhớ tắt buffering cho
   `/api/chat` để chat stream được.
5. Chạy backfill (lệnh sẽ có sau khi xong M2.7), rồi mở web kiểm tra dashboard có dữ liệu.

Nếu không thuê VPS: chạy `docker compose up -d` trên một máy luôn bật từ 18/11 đến hôm demo.

## Bước 6 — Kiểm thử trước demo

| Việc | Lệnh / cách làm | Ngưỡng |
|---|---|---|
| Parity mô hình | `pytest tests/test_parity.py -s` | lệch < 1e-4 |
| Backfill so với paper | (sau M2.7) so điểm trong DB với `vn_preds.npz` năm 2025 | ghi lại mức lệch |
| Eval trợ lý | `.venv/bin/python -m assistant.eval.run --sleep 20` (~50 câu; `--sleep` tránh giới hạn free tier) | ≥ 95% số liệu khớp, 100% từ chối đúng |
| Pipeline | xem `pipeline_runs` hoặc dòng trạng thái trên dashboard 18–20/11 | 3 phiên `success` |
| Web | Lighthouse trên URL đã deploy | performance ≥ 80 |
| UX | ≥ 10 người làm 5 task ở PLAN §7, chấm SUS | SUS ≥ 68 |

Kiểm tra trạng thái pipeline trên server:

```bash
docker compose logs --tail 50 scheduler
docker compose exec db psql -U midmoe midmoe -c "select run_date, status, error from pipeline_runs order by id desc limit 5"
```

## Lưu ý khác

- Các commit hiện đứng tên `xuanquyn <ngocquynh23042005@gmail.com>` theo cấu hình git trên máy. Báo Claude nếu
  cần đổi trước khi có thêm commit.
- Repo public không chứa `MIDMoE.pdf`. Muốn nút "Đọc paper" ở trang Giới thiệu hoạt động, chép file vào
  `web/public/` trên máy deploy (không commit).
- `python -m backend.db.demo` chỉ nạp **dữ liệu giả** để phát triển giao diện. Không chạy lệnh này trên DB dùng cho demo.
