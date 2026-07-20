from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "mock"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_timeout_s: float = 60.0
    judge_enabled: bool = True
    judge_threshold: float = Field(default=0.7, ge=0, le=1)
    analysis_timeout_s: float = 300.0
    runs_dir: str = "runs"  # local archive of finished analyses; empty string disables
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
