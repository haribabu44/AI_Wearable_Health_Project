"""
core/config.py
--------------
Central configuration loaded from environment variables / .env file.
All settings live here — no magic strings scattered across modules.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "AI Wearable Health API"
    app_version: str = "2.0.0"
    debug: bool = False

    # ── Security ─────────────────────────────────────────────────────────
    api_key: str = "dev-key-change-in-production"
    api_key_header: str = "X-API-Key"

    # ── Rate limiting ─────────────────────────────────────────────────────
    rate_limit_requests: int = 60   # requests per window
    rate_limit_window: int = 60     # seconds

    # ── Database ─────────────────────────────────────────────────────────
    db_path: str = str(
        Path(__file__).parent.parent.parent / "health_data.db"
    )
    db_pool_size: int = 5

    # ── ML ────────────────────────────────────────────────────────────────
    models_dir: str = str(
        Path(__file__).parent.parent.parent.parent / "ml" / "saved_models"
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings — parsed once, reused everywhere."""
    return Settings()
