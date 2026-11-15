# Image shared by the `api` and `scheduler` services.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TZ=Asia/Ho_Chi_Minh

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (the default Linux wheel pulls ~2 GB of CUDA libraries we don't need).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini pyproject.toml ./
COPY midmoe ./midmoe
COPY backend ./backend
COPY assistant ./assistant
COPY pipeline ./pipeline
COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint

RUN chmod +x /usr/local/bin/api-entrypoint \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/.cache/huggingface \
    && chown -R app /app
USER app

EXPOSE 8000
CMD ["api-entrypoint"]
