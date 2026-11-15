#!/bin/sh
# api container: migrate, seed tickers from artifacts (if present), then serve.
set -e

alembic upgrade head
python -m backend.db.seed --skip-missing

exec uvicorn backend.main:app \
    --host 0.0.0.0 --port 8000 \
    --workers "${API_WORKERS:-2}" \
    --proxy-headers --forwarded-allow-ips='*'
