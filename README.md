# MIDMoE App — Khuyến nghị cổ phiếu VN + Trợ lý AI

Ứng dụng của đồ án *Hệ thống khuyến nghị đầu tư chứng khoán dựa trên mô hình học sâu phân cấp (MIDMoE) và trợ lý ảo AI*:
pipeline dữ liệu hằng ngày, đóng gói mô hình (ensemble 10 seed), giải thích khuyến nghị, API, trợ lý AI có tool-calling
và web. Kế hoạch chi tiết và tiến độ: [PLAN.md](PLAN.md).

> Thông tin tham khảo, không phải lời khuyên đầu tư. Mô hình chưa vượt chọn ngẫu nhiên một cách có ý nghĩa thống kê
> (xem paper `MIDMoE.pdf`, Table 10).

## Kiến trúc

```
[vnstock]  [CafeF]
    │         │
    ▼         ▼
 scheduler: pipeline/daily_job.py  (16:30 ICT, T2–T6)
    │ fetch → 7 chỉ báo → PhoBERT → normalize → cube → predict (10 seed) → explain
    ▼
 PostgreSQL ◄── api: FastAPI (backend/) ◄── web: Next.js (web/)
                   │
                   └── assistant/ (tool-calling) ──► Gemini / OpenAI / Anthropic
```

| Thư mục | Nội dung |
|---|---|
| `midmoe/` | `MarketMoE` (copy nguyên từ notebook), 7 chỉ báo + chuẩn hoá, `Predictor` (ensemble), metric của paper |
| `pipeline/` | job hằng ngày + scheduler (APScheduler) |
| `backend/` | FastAPI, SQLAlchemy + Alembic, service layer dùng chung cho API và trợ lý |
| `assistant/` | lớp provider LLM, 7 tool, system prompt, bộ eval 50 câu |
| `web/` | Next.js 14 (App Router) + Tailwind, lightweight-charts, Recharts |
| `artifacts/vn/` | weights + tham số chuẩn hoá (không commit, xem [artifacts/README.md](artifacts/README.md)) |
| `notebooks/` | cell export weights để chạy trên Kaggle |

## Trạng thái (15/11/2026)

| Phần | Trạng thái |
|---|---|
| Mô hình (M1) | Đã đóng gói + test; **chờ chạy `notebooks/export_vn_weights.py` trên Kaggle** để có weights |
| Pipeline (M2) | Chỉ báo xong; **chờ notebook `midmoe-vn-cube-build`** (crawl CafeF, PhoBERT, vnstock). `daily_job.run` chưa có |
| Giải thích (M3), dựng cube (M2.5) | Xong, test trên model ngẫu nhiên; chờ weights thật để kiểm tra kết quả |
| API (M4), trợ lý (M5), web (M6), Docker (M7.1) | Xong; đã test trên dữ liệu demo tổng hợp |

Khi pipeline chưa có, scheduler vẫn chạy đúng lịch và ghi một lần chạy `failed` ("chưa được triển khai") để
dashboard hiển thị trạng thái thật.

## Chạy bằng Docker Compose

Yêu cầu: Docker 24+ với Compose v2, khoảng 4 GB RAM trống.

```bash
cp .env.example .env              # đặt POSTGRES_PASSWORD, LLM_PROVIDER, LLM_API_KEY
# đặt artifacts vào artifacts/vn/ (tuỳ chọn: thiếu thì API vẫn chạy, chỉ chưa có mã)
docker compose up -d --build
docker compose ps                 # cả 4 service phải "healthy"
```

- Web: http://localhost:3000 · API: http://localhost:8000/docs
- Container `api` tự chạy `alembic upgrade head` rồi nạp bảng `tickers` từ `artifacts/vn/` mỗi lần khởi động.
- `NEXT_PUBLIC_API_URL` được nhúng vào bundle **lúc build**: đổi URL public của API thì phải
  `docker compose build web`.

| Service | Image | Healthcheck |
|---|---|---|
| `db` | `postgres:16-alpine` (volume `pgdata`) | `pg_isready` |
| `api` | `Dockerfile` (Python 3.11, torch CPU) | `GET /api/health` |
| `scheduler` | cùng image với `api` | heartbeat ghi mỗi phút |
| `web` | `web/Dockerfile` (Next.js standalone) | `GET /` |

Lệnh hay dùng:

```bash
docker compose logs -f api scheduler
docker compose exec scheduler python -m pipeline.scheduler --once                   # chạy job hôm nay ngay
docker compose exec scheduler python -m pipeline.scheduler --once --date 2026-11-13
docker compose exec api python -m backend.db.seed                                   # nạp lại tickers
docker compose exec db psql -U midmoe midmoe -c "select * from pipeline_runs order by id desc limit 5"
```

### Backfill lịch sử

`pipeline/backfill.py --from 2025-01-01 --to today` (M2.7) chưa có vì phụ thuộc M2. Khi có:

```bash
docker compose exec scheduler python -m pipeline.backfill --from 2025-01-01 --to today
```

## Chạy local để phát triển

```bash
/opt/anaconda3/bin/python3.11 -m venv --system-site-packages .venv   # hoặc python3.11 bất kỳ
.venv/bin/pip install -r requirements.txt
cp .env.example .env                       # local dùng SQLite: DATABASE_URL=sqlite:///./midmoe.db
.venv/bin/alembic upgrade head
.venv/bin/python -m backend.db.demo        # TUỲ CHỌN: dữ liệu demo TỔNG HỢP (mã M00..M77)
.venv/bin/uvicorn backend.main:app --reload            # :8000

cd web && npm ci && cp .env.local.example .env.local && npm run dev   # :3000
```

`backend.db.demo` nạp dữ liệu ngẫu nhiên nhưng nhất quán để phát triển giao diện; lệnh từ chối chạy nếu DB đã có
mã. **Không dùng dữ liệu này cho demo hay báo cáo.** `LLM_PROVIDER=fake` cho phép thử luồng chat không cần API key.

### Kiểm thử

```bash
.venv/bin/python -m pytest                    # backend, mô hình, trợ lý (parity test tự skip khi thiếu weights)
cd web && npm run lint && npm run typecheck && npm run build
.venv/bin/python -m assistant.eval.run        # eval trợ lý: gọi LLM thật ~50 lượt, cần dữ liệu đã backfill
```

## Biến môi trường

| Biến | Ý nghĩa |
|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Tài khoản PostgreSQL (Compose tự ghép `DATABASE_URL`) |
| `DATABASE_URL` | Chỉ cho chạy local, ví dụ `sqlite:///./midmoe.db` |
| `LLM_PROVIDER` / `LLM_API_KEY` / `LLM_MODEL` | `gemini` (mặc định), `openai`, `anthropic`; để trống model = mặc định của provider |
| `ARTIFACT_DIR` | Thư mục weights + tham số (Compose: `/app/artifacts/vn`) |
| `LIVE_START` | Ngày bắt đầu giai đoạn "live" (mặc định 2026-01-01) |
| `TZ` | `Asia/Ho_Chi_Minh` |
| `NEXT_PUBLIC_API_URL` | URL API **mà trình duyệt gọi** |
| `CORS_ORIGINS` | Origin của web được gọi API, phân tách bằng dấu phẩy |
| `API_PORT` / `WEB_PORT` | Cổng publish trên host |

## Deploy lên VPS (D4)

1. VPS Ubuntu 22.04+, 2 vCPU / 4 GB RAM trở lên; cài Docker Engine + Compose plugin.
2. `git clone` repo, chép `artifacts/vn/` lên (scp), tạo `.env` với mật khẩu mạnh và key LLM.
3. Đặt URL public: `NEXT_PUBLIC_API_URL=https://api.<domain>`, `CORS_ORIGINS=https://<domain>`.
4. Đặt reverse proxy có HTTPS (ví dụ Caddy) trước cổng 3000 và 8000. Với API cần tắt buffering để SSE của
   `/api/chat` không bị gom lại (Caddy: `flush_interval -1`; nginx: `proxy_buffering off`).
5. `docker compose up -d --build`, sau đó chạy backfill, rồi theo dõi 3 phiên liên tiếp (M7.4) qua
   `pipeline_runs` hoặc dòng trạng thái trên dashboard.

## Tài liệu

- [PLAN.md](PLAN.md): spec, milestone, tiến độ
- [artifacts/README.md](artifacts/README.md): định dạng artifacts và cách export từ Kaggle
- `MIDMoE.pdf`: paper draft
