from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    storage_dir: Path = Path("storage")
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-4o-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()