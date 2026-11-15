# PLAN.md — MIDMoE App: Pipeline + Trợ lý AI + Web demo

> File này là spec cho AI coding agent (Claude Code / Copilot / Cursor) trong VS Code.
> Agent: đọc hết file trước khi code. Làm theo thứ tự Milestone. Mỗi task xong thì tick `[x]` và chạy phần "Kiểm tra" của task đó.
> Ngôn ngữ: code, tên biến, commit message bằng tiếng Anh; UI, chat và thông báo cho người dùng bằng tiếng Việt.

---

## 0. Bối cảnh

- Đồ án: **Hệ thống khuyến nghị đầu tư chứng khoán dựa trên mô hình học sâu phân cấp (MIDMoE) và trợ lý ảo AI**.
- Mô hình **đã train xong** trên Kaggle. Paper draft: `MIDMoE.pdf`. Notebooks: `midmoe-pipeline`, `midmoe-data-processed`, `midmoe-cube-build`, `midmoe-model-train`, `midmoe-vn-transfer`, `midmoe-vn-cube-build`.
- Việc còn lại là **phần ứng dụng**: pipeline hằng ngày, đóng gói inference, giải thích khuyến nghị, trợ lý AI, backend, web.
- **Demo chạy trên thị trường VN, 78 mã.** Nguồn tin US (Kaggle) dừng ở 2020 nên không chạy live được.
- **Deadline demo: 23/11/2026.** Sau đó: kiểm thử + báo cáo.

### Mô hình dùng để deploy
- Kiến trúc: class `MarketMoE` (copy nguyên từ notebook `midmoe-vn-transfer`, Phase 2), `market_conv_type='conv3d'`.
- Cấu hình: `transfer_minimal` (frozen US backbone, chỉ train `ssam_b2m` và `ssam_m2t`). Theo paper, đây là cấu hình tốt nhất trên VN.
- Hyperparams lấy từ `model_config.json`: `F_dim=20, dit_dim=32, dit_heads=4, dit_blocks=2, num_middle=3, num_bottom=3, diffusion_K=100`.
- Input: `x (B,1,16,78,7)`, `news_emb (B,78,768)` (PhoBERT), `news_mask (B,78)`, `A_norm (78,78)`.
- Output: `R_final (B,78)` là điểm xếp hạng của từng mã. Điểm cao hơn nghĩa là kỳ vọng lợi suất T+1 cao hơn.
- Eval mode: `k_idx=0`, single forward, không sampling diffusion.
- Ensemble: 10 seed, **score cuối = trung bình score của 10 seed**. Lưu thêm `score_std` và `top5_votes` (số seed xếp mã đó vào top-5).

### Thứ tự 7 chỉ báo (KHÔNG được đổi)
`['ret_1d', 'ret_5d', 'ret_10d', 'rsi', 'macd', 'close_open_ratio', 'high_low_ratio']`
- `ret_Nd = close.pct_change(N)`, `rsi = ta.rsi(close, 14)`, `macd = ta.macd(close)['MACD_12_26_9']`, `close/open`, `high/low`.
- Chuẩn hoá: winsorize p1/p99 rồi min-max về [0,1], **dùng đúng tham số trong `vn_norm_params.json`, không fit lại**.
- Thứ tự mã: đúng theo `vn_tickers_sorted.json` (đã sort theo ngành ICB). Đồ thị ngành: `vn_sector_graph.npy`.

---

## 1. Kiến trúc

```
[vnstock]  [CafeF]
    │         │
    ▼         ▼
 pipeline/daily_job.py  (cron 16:30 ICT, T2–T6)
    │ fetch → indicators → PhoBERT embed → normalize → cube
    ▼
 midmoe/predict.py  (ensemble 10 seed) + midmoe/explain.py
    │
    ▼
 PostgreSQL ◄──── backend/ (FastAPI) ◄──── web/ (Next.js)
                     │
                     └── assistant/ (LLM tool-calling) ──► LLM API
```

## 2. Tech stack

| Phần | Chọn | Ghi chú |
|---|---|---|
| Ngôn ngữ backend/ML | Python 3.11 | |
| ML | PyTorch (CPU là đủ), transformers (PhoBERT `vinai/phobert-base`), pandas-ta | Phải khớp model và pooling đã dùng trong `vn-cube-build` |
| Dữ liệu | `vnstock` (pin version), `requests` + `beautifulsoup4` cho CafeF | |
| DB | PostgreSQL 16 (dev có thể dùng SQLite qua SQLAlchemy) | SQLAlchemy 2 + Alembic |
| API | FastAPI + Uvicorn, Pydantic v2 | |
| Scheduler | APScheduler trong container `scheduler`, hoặc cron | |
| LLM | Provider abstraction: `LLM_PROVIDER` = `gemini` / `openai` / `anthropic` | Cần function calling + streaming |
| Web | Next.js 14 (App Router) + TypeScript + Tailwind + shadcn/ui | Fallback: Streamlit nếu trễ (chốt 19/11) |
| Chart | `lightweight-charts` (nến), Recharts (line/bar) | |
| Deploy | Docker Compose: `db`, `api`, `scheduler`, `web` | |

## 3. Cấu trúc repo

```
midmoe-app/
├── PLAN.md
├── README.md
├── .env.example
├── docker-compose.yml
├── artifacts/vn/                 # KHÔNG commit weights lớn; xem artifacts/README.md
│   ├── weights/model_seed{0..9}.pt
│   ├── model_config.json
│   ├── vn_norm_params.json
│   ├── vn_tickers_sorted.json
│   ├── vn_feature_order.json
│   ├── vn_sector_graph.npy
│   └── icb_industry.csv
├── midmoe/                       # package ML
│   ├── model.py                  # MarketMoE + submodules (copy từ notebook)
│   ├── features.py               # 7 chỉ báo + normalize
│   ├── news.py                   # PhoBERT embed + gộp theo (ticker, date)
│   ├── cube.py                   # dựng x, news_emb, news_mask cho 1 ngày
│   ├── predict.py                # load ensemble, predict(date)
│   └── explain.py                # occlusion + feature snapshot
├── pipeline/
│   ├── sources/vnstock_src.py
│   ├── sources/cafef_src.py
│   ├── daily_job.py
│   └── backfill.py
├── backend/
│   ├── main.py
│   ├── db/models.py, db/session.py, alembic/
│   ├── routers/{recommendations,stocks,backtest,chat,health}.py
│   └── schemas.py
├── assistant/
│   ├── llm.py                    # provider abstraction
│   ├── tools.py                  # tool defs, gọi thẳng service layer (không qua HTTP)
│   ├── prompts.py
│   └── eval/questions.jsonl
├── web/                          # Next.js
└── tests/
```

---

## 4. Milestones & tasks

### M1 — Đóng gói mô hình (15–17/11)

- [x] **M1.1** Tạo `midmoe/model.py`: copy nguyên `MarketMoE` và các class con từ notebook `midmoe-vn-transfer` (Phase 2). Không sửa logic.
- [ ] **M1.2** *(cell đã viết: `notebooks/export_vn_weights.py`, chờ chạy trên Kaggle)* Export artifacts từ Kaggle: chạy lại `build_zeroshot(seed)` + `finetune_vn_model` cho 10 seed với cấu hình deploy (xem Quyết định D2), rồi `torch.save(model.state_dict(), f'model_seed{seed}.pt')`. Thêm cell export vào notebook transfer.
- [x] **M1.3** `midmoe/predict.py`:
  ```python
  class Predictor:
      def __init__(self, artifact_dir: str, device: str = "cpu"): ...
      def predict(self, x, news_emb, news_mask) -> PredictionResult:
          # trả về score_mean (78,), score_std (78,), rank (78,), top5_votes (78,), per_seed (10,78)
  ```
  Load 10 model một lần, `model.eval()`, `torch.no_grad()`.
- [ ] **M1.4** *(test đã viết, skip cho tới khi có artifacts)* **Parity test** (`tests/test_parity.py`): load `vn_test_cube.npz`, chạy Predictor từng seed, so với `vn_preds.npz` key `transfer_minimal__<fraction>` → `max abs diff < 1e-4`. Test này đảm bảo bản đóng gói ra đúng kết quả như trên Kaggle.

**Kiểm tra:** `pytest tests/test_parity.py` pass; predict 1 ngày trên CPU < 5 s.

### M2 — Pipeline dữ liệu hằng ngày (15–18/11)

- [ ] **M2.1** `pipeline/sources/vnstock_src.py`: `fetch_ohlcv(tickers, start, end) -> DataFrame[date,ticker,open,high,low,close,volume]`. Retry 3 lần, có backoff. Lấy ít nhất 60 phiên trước ngày cần dự đoán để MACD/RSI ổn định (thực tế dùng 250, xem `midmoe/features.py`). Lấy thêm chỉ số `VNINDEX` và ghi bằng `backend.services.market_index.upsert_index_bars`.
- [x] **M2.2** `midmoe/features.py`: tính 7 chỉ báo **y hệt** `add_indicators` trong notebook; `normalize(df, norm_params)` theo thứ tự winsorize → min-max → clip [0,1].
- [ ] **M2.3** `pipeline/sources/cafef_src.py`: `fetch_news(ticker, date_from, date_to) -> list[{ticker,date,title,url,published_at}]`. Dùng lại logic crawl trong notebook `vn-cube-build`. Dedupe theo `(ticker, date, title)`. Gán ngày theo **đúng quy tắc** đã dùng khi build cube; nếu notebook không có quy tắc giờ thì dùng ngày lịch, giống paper.
- [ ] **M2.4** `midmoe/news.py`: embed tiêu đề bằng PhoBERT, **cùng tokenizer, max_len, pooling** với `vn-cube-build`. Mean theo `(ticker, date)`. Mã không có tin thì `mask=0`, vector 0.
- [x] **M2.5** `midmoe/cube.py`: `build_inputs(as_of_date) -> (x(1,1,16,78,7), news_emb(1,78,768), news_mask(1,78))`. Lấy đúng 16 phiên giao dịch gần nhất, kết thúc tại `as_of_date`. Thiếu dữ liệu một mã thì forward-fill tối đa 3 phiên và log warning; nhiều hơn thì đánh dấu run `degraded`.
- [ ] **M2.6** `pipeline/daily_job.py`: fetch → features → news → cube → predict → explain (M3) → upsert DB → ghi `pipeline_runs`. Idempotent: chạy lại cùng ngày sẽ ghi đè, không nhân đôi.
- [ ] **M2.7** `pipeline/backfill.py --from 2025-01-01 --to today`: chạy job cho từng ngày giao dịch để demo có lịch sử (năm 2025 là test, 2026 là "live").
- [ ] **M2.8** Lên lịch: 16:30 Asia/Ho_Chi_Minh, T2–T6, bỏ qua ngày lễ (thị trường không có phiên nghĩa là vnstock không trả dòng mới, job kết thúc `skipped`).

**Kiểm tra:** chạy backfill cho 2025, rồi so `score` trong DB với `vn_preds.npz` cho cùng ngày → sai lệch nhỏ (bản dựng lại từ dữ liệu fetch mới có thể lệch nhẹ; ghi lại mức lệch). Job 1 ngày < 10 phút.

### M3 — Giải thích khuyến nghị (16–19/11)

Paper (Table 8) cho thấy lợi thế đến từ **các nguồn thông tin và market cube**, không đến từ tham số do diffusion sinh ra. Vì vậy giải thích dựa trên đóng góp của từng nguồn.

- [x] **M3.1** `midmoe/explain.py::source_contributions`: với mỗi seed, chạy thêm 4 forward pass, mỗi lần tắt một nguồn:
  - `news`: `news_mask = 0`
  - `market`: thay output `market_conv` bằng zeros
  - `graph`: `A_norm = I` (bỏ lan truyền theo ngành)
  - `timeseries`: thay output `ts_proj` bằng zeros
  
  Đóng góp = `score_full − score_ablated`, lấy trung bình 10 seed. Lưu cả `rank_full` và `rank_ablated` của từng nguồn.
- [x] **M3.2** `feature_snapshot`: giá trị raw của 7 chỉ báo hôm nay + percentile so với 78 mã. Thêm nhãn diễn giải: RSI > 70 là "quá mua", < 30 là "quá bán"; ret_5d nằm trong top 10%...
- [x] **M3.3** `sector_context`: hạng của mã trong ngành, top 3 mã cùng ngành.
- [x] **M3.4** Gắn tin: tiêu đề + url của tin trong ngày và 5 phiên trước.
- [x] **M3.5** `confidence`: `top5_votes/10` + `score_std`. Hiển thị dạng "8/10 mô hình đồng thuận".
- [ ] *(payload đã đúng schema mục 5; ghi vào DB thuộc job M2.6)* Lưu toàn bộ vào bảng `explanations` dạng JSON (schema ở mục 5).

**Kiểm tra:** chạy cho 1 ngày, 78 mã < 30 s trên CPU; JSON validate bằng Pydantic.
*(15/11, model ngẫu nhiên 78 mã × 10 seed, CPU: predict 0,83 s, explain 3,9 s; payload validate bằng `ExplanationPayload`.)*

### M4 — Backend API (17–20/11)

- [x] **M4.1** SQLAlchemy models + Alembic migration theo mục 5.
- [x] **M4.2** Service layer `backend/services/*.py`, dùng chung cho router và assistant tools.
- [x] **M4.3** *(xong trừ `/chat` → M5; thêm `/sectors` cho bộ lọc)* Endpoints (prefix `/api`):

| Method | Path | Trả về |
|---|---|---|
| GET | `/health` | status DB + lần chạy pipeline gần nhất |
| GET | `/dates` | danh sách ngày có dự đoán, `latest` |
| GET | `/recommendations?date=&k=5` | top-k: ticker, tên, ngành, score, rank, votes, lý do 1 dòng |
| GET | `/rankings?date=&sector=` | đủ 78 mã |
| GET | `/stocks/{ticker}?from=&to=` | OHLCV + 7 chỉ báo + lịch sử rank |
| GET | `/stocks/{ticker}/explanation?date=` | JSON explanation |
| GET | `/stocks/{ticker}/news?days=7` | danh sách tin |
| GET | `/backtest?from=&to=&k=5` | lợi nhuận ngày + tích luỹ của top-k vs random vs VN-Index; SR, MaxDD, nDCG@5 |
| POST | `/chat` | SSE stream; body `{session_id, message, context:{ticker?, date?}}` |

- [x] **M4.4** *(VN-Index lấy từ bảng `market_index`, nhân k để cùng thang với top-k = tổng k mã)* Backtest: dùng lại đúng hàm `compute_metrics` / `daily_ret` của notebook (SR theo ngày, IRR = tổng lợi suất ngày của top-5).
- [x] **M4.5** CORS cho web, request logging, OpenAPI docs ở `/docs`.

**Kiểm tra:** `pytest tests/api`; p95 latency các GET < 300 ms trên dữ liệu đã backfill.

### M5 — Trợ lý AI (19–22/11)

- [x] **M5.1** `assistant/llm.py`: interface `chat(messages, tools, stream=True)`, có adapter cho provider chọn ở D1. Key lấy từ env.
- [x] **M5.2** `assistant/tools.py`, các tool gọi thẳng service layer:
  - `get_recommendations(date?, k=5)`
  - `get_stock_explanation(ticker, date?)`
  - `get_stock_prices(ticker, days=30)`
  - `get_stock_news(ticker, days=7)`
  - `compare_stocks(tickers[], date?)`
  - `get_model_performance(period="2025"|"live")`
  - `get_model_info()`: mô tả mô hình, dữ liệu, hạn chế (text tĩnh lấy từ paper)
- [x] **M5.3** `assistant/prompts.py`, system prompt tiếng Việt. Quy tắc bắt buộc:
  1. Mọi con số và lý do phải lấy từ kết quả tool. Không có dữ liệu thì nói không có.
  2. Không dự đoán giá cụ thể, không cam kết lợi nhuận, không nói "nên mua/bán". Nói đây là **thông tin tham khảo, không phải lời khuyên đầu tư**.
  3. Khi được hỏi về độ tin cậy, nói rõ hạn chế: mô hình chưa vượt random có ý nghĩa thống kê; chưa tính phí giao dịch; turnover cao; quy tắc T+2 và biên độ ±7% (HOSE) chưa được mô hình hoá.
  4. Giải thích theo thứ tự: hạng + độ đồng thuận → nguồn đóng góp lớn nhất → chỉ báo nổi bật → tin liên quan (kèm link).
  5. Không nhắc đến tham số nội bộ (SSAM, DiT, expert weights) trừ khi người dùng hỏi về kiến trúc.
- [x] **M5.4** Context trang: web gửi `ticker` và `date` đang xem. Câu hỏi kiểu "mã này thì sao?" tự hiểu theo context.
- [x] **M5.5** Lưu `chat_sessions` và `chat_messages`; giữ tối đa 20 lượt gần nhất làm history.
- [ ] **M5.6** *(50 câu + `run.py` xong; chưa chạy thật: cần API key + dữ liệu backfill)* Bộ eval `assistant/eval/questions.jsonl` (≥ 40 câu): giải thích mã, so sánh, câu hỏi về chỉ báo, câu "mai giá bao nhiêu / có nên all-in không" (phải từ chối có lý do), câu về mã ngoài danh sách 78. Viết script `assistant/eval/run.py` để chấm: số liệu trong câu trả lời khớp DB; câu cần từ chối thì có từ chối.

**Kiểm tra:** ≥ 95% câu trả lời có số liệu khớp DB; 100% câu cần từ chối có từ chối; first token < 3 s.

### M6 — Web (18–23/11)

- [x] **M6.1** Layout chung: header (logo, chọn ngày, link các trang), **panel chat cố định bên phải** (thu gọn được), banner cảnh báo "Thông tin tham khảo, không phải lời khuyên đầu tư".
- [x] **M6.2** `/` Dashboard:
  - 5 card top-5: mã, tên, ngành, hạng, "8/10 mô hình đồng thuận", lý do 1 dòng, nút "Hỏi trợ lý".
  - Bảng 78 mã: sort, lọc theo ngành, cột score / rank / thay đổi rank so với hôm trước.
  - Trạng thái pipeline: "Cập nhật lúc 16:42 ngày …".
- [x] **M6.3** `/stocks/[ticker]`: biểu đồ nến + volume (lightweight-charts); bảng 7 chỉ báo kèm percentile; line lịch sử rank 30 phiên; bar chart đóng góp 4 nguồn; danh sách tin (link CafeF); khung chat tự gắn context mã này.
- [x] **M6.4** `/performance`: line lợi nhuận tích luỹ top-5 vs random vs VN-Index, chia 2 giai đoạn "Test 2025" và "Live từ ngày deploy"; bảng SR / MaxDD / nDCG@5; một đoạn ngắn tóm tắt kết quả và hạn chế theo paper.
- [x] **M6.5** `/about`: mô hình MIDMoE (sơ đồ đơn giản), dữ liệu, hạn chế, link paper.
- [x] **M6.6** Chat UI: stream SSE, render markdown, gợi ý câu hỏi mẫu, hiển thị "đang tra cứu dữ liệu…" khi tool đang chạy.
- [x] **M6.7** Responsive tới 375px; dark mode.

**Kiểm tra:** Lighthouse performance ≥ 80; 5 task UX (mục 7) làm được không cần hướng dẫn.
*(15/11: Lighthouse mobile trên dữ liệu demo tổng hợp: `/` 91, `/stocks/[ticker]` 84–91, `/performance` 88, `/about` 99; a11y 100. Chưa test UX với người dùng.)*

### M7 — Tích hợp & deploy (22–23/11)

- [ ] **M7.1** `docker-compose.yml`: `db` (postgres:16), `api`, `scheduler` (dùng chung image với api), `web`. Healthcheck cho từng service.
- [ ] **M7.2** `.env.example`: `DATABASE_URL`, `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `ARTIFACT_DIR`, `TZ=Asia/Ho_Chi_Minh`, `NEXT_PUBLIC_API_URL`.
- [ ] **M7.3** Deploy lên VPS / Railway / Render (D4). Chạy backfill trên môi trường deploy, bật scheduler.
- [ ] **M7.4** Theo dõi 3 phiên liên tiếp chạy thành công trước buổi demo.
- [ ] **M7.5** README: cách chạy local (`docker compose up`), cách backfill, kiến trúc.

---

## 5. Database schema

```sql
tickers(ticker PK, name, exchange, icb_sector, sort_order INT)
prices(ticker, date, open, high, low, close, volume,
       ret_1d, ret_5d, ret_10d, rsi, macd, close_open_ratio, high_low_ratio,
       PK(ticker, date))
news(id PK, ticker, date, title, url UNIQUE, source, published_at)
market_index(index_code, date, open, high, low, close, volume, ret_1d, PK(index_code, date))  -- VNINDEX
predictions(date, ticker, score, score_std, rank INT, top5_votes INT, model_version,
            PK(date, ticker))
explanations(date, ticker, payload JSONB, PK(date, ticker))
pipeline_runs(id PK, run_date, status ENUM(success,degraded,failed,skipped),
              started_at, finished_at, n_tickers, n_news, error TEXT)
chat_sessions(id UUID PK, created_at)
chat_messages(id PK, session_id FK, role, content, tool_calls JSONB, created_at)
```

`explanations.payload`:
```json
{
  "rank": 2, "score": 0.0123, "top5_votes": 8, "score_std": 0.004,
  "sources": {
    "market":     {"delta": 0.0041, "rank_without": 9},
    "news":       {"delta": 0.0020, "rank_without": 4},
    "graph":      {"delta": -0.0005, "rank_without": 2},
    "timeseries": {"delta": 0.0030, "rank_without": 7}
  },
  "features": {"rsi": {"value": 72.1, "pct": 0.93, "label": "quá mua"}, "...": {}},
  "sector": {"name": "Ngân hàng", "rank_in_sector": 1, "n_in_sector": 12},
  "news": [{"date": "2026-11-14", "title": "...", "url": "..."}],
  "summary": "Xếp hạng 2/78, 8/10 mô hình đồng thuận; yếu tố thị trường đóng góp nhiều nhất."
}
```

## 6. Quyết định cần chốt (trước khi code phần liên quan)

| # | Câu hỏi | Mặc định nếu chưa chốt |
|---|---|---|
| D1 | Dùng LLM provider nào, ai trả phí API key | Gemini (có free tier) |
| D2 | Model deploy train trên dữ liệu nào: chỉ phần fit 2022–2024 (giống paper) hay retrain đến hết 2025 với số epoch cố định | Giống paper (fit, early stop trên val) để khớp số liệu báo cáo |
| D3 | Gộp ensemble: mean score hay rank-average | Mean score |
| D4 | Hosting | Docker Compose trên 1 VPS |
| D5 | Next.js hay Streamlit (chốt 19/11 theo tiến độ) | Next.js |
| D6 | Hạn nộp cuối của giai đoạn kiểm thử + báo cáo | Hỏi GVHD |

## 7. Kiểm thử & tiêu chí hoàn thành

| Hạng mục | Cách đo | Ngưỡng đạt (đề xuất) |
|---|---|---|
| Đúng mô hình | Parity test với `vn_preds.npz` | max diff < 1e-4 |
| Độ chính xác | Số liệu paper (SR, nDCG@5, MaxDD, bootstrap) + theo dõi live | Báo cáo trung thực, không đặt ngưỡng |
| Pipeline | Thời gian job, số lần lỗi trong 5 phiên | < 10 phút, 0 lỗi |
| API | p95 latency GET | < 300 ms |
| Trợ lý AI | Eval set ≥ 40 câu | ≥ 95% số liệu khớp DB, 100% từ chối đúng chỗ |
| UX | SUS, ≥ 10 người, 5 task: tìm top-5 hôm nay; xem lý do một mã; hỏi trợ lý so sánh 2 mã; xem hiệu quả mô hình 2025; tìm tin của một mã | SUS ≥ 68 |

## 8. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| CafeF chặn hoặc đổi HTML | Cache, retry; nếu fail thì chạy với `news_mask=0` (mô hình vẫn chạy) và đánh dấu `degraded` |
| vnstock đổi API hoặc rate limit | Pin version, retry, backoff |
| Tiền xử lý live lệch so với lúc train | Parity test M1.4 + so backfill 2025 (M2) |
| Dữ liệu 2026 khác phân phối lúc train | Hiển thị ngày train trên trang About; ghi vào hạn chế; kế hoạch retrain định kỳ |
| LLM bịa số liệu | Tool-calling bắt buộc, eval faithfulness M5.6 |
| Trễ tiến độ web | Fallback Streamlit (D5) |

## 9. Quy ước cho agent

- Một task một commit: `feat(m2.3): cafef news scraper`.
- Không sửa logic `MarketMoE`. Nếu cần hook (ví dụ tắt một nhánh để explain), viết wrapper hoặc truyền flag, không đổi kết quả mặc định. Parity test phải luôn pass.
- Hằng số (`T_WINDOW=16`, thứ tự feature, thứ tự mã) đọc từ `artifacts/`, không hard-code.
- Mọi hàm I/O mạng phải có timeout, retry và log.
- Type hints đầy đủ; `ruff` + `black` cho Python; `eslint` + `prettier` cho web.
- Không commit `.env`, weights `.pt`, hay dữ liệu thô.
