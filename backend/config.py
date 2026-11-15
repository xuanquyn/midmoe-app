"""Application settings, read from environment / .env (see .env.example)."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./midmoe.db"
    artifact_dir: str = "artifacts/vn"
    tz: str = "Asia/Ho_Chi_Minh"
    cors_origins: str = "http://localhost:3000"
    sql_echo: bool = False

    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_model: str = ""

    # First day of the "live" period (2025 is the paper's test year). Used by /performance and the assistant.
    live_start: dt.date = dt.date(2026, 1, 1)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
