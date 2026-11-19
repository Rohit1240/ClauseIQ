"""Backend settings, read from environment variables / .env (see .env.example)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUSEIQ_", env_file=str(ROOT / ".env"),
                                      extra="ignore")

    env: str = "development"
    log_level: str = "INFO"
    artifacts_dir: Path = ROOT / "artifacts"
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "app" / "clauseiq.sqlite3"
    max_upload_mb: float = 15.0
    max_document_chars: int = 2_000_000
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    preload_models: bool = True
    max_concurrent_analyses: int = 2
    torch_threads: int = 0
    frontend_dist: Path = ROOT / "frontend" / "dist"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
