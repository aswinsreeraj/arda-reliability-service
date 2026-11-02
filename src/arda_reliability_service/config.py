"""Configuration settings for ARDA Reliability Service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for database, embeddings, and vector retrieval."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL & pgvector
    postgres_host: str = Field(default="127.0.0.1", description="PostgreSQL host")
    postgres_port: int = Field(default=5433, description="PostgreSQL port")
    postgres_db: str = Field(default="arda_reliability", description="Database name")
    postgres_user: str = Field(default="arda_user", description="Database user")
    postgres_password: str = Field(
        default="arda_password", description="Database password"
    )

    # Pool settings
    db_pool_size: int = Field(default=10, description="Connection pool size")
    db_max_overflow: int = Field(default=5, description="Max overflow connections")
    db_pool_timeout: int = Field(
        default=30, description="Connection pool timeout in seconds"
    )

    # Embedding settings
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="SentenceTransformers model identifier",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Expected embedding vector dimension",
    )
    embedding_device: str = Field(
        default="auto",
        description="Inference device: 'cuda', 'cpu', or 'auto'",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )

    # Retrieval settings
    default_retrieval_top_k: int = Field(
        default=5,
        description="Default number of nearest neighbors to retrieve",
    )
    hnsw_ef_search: int = Field(
        default=64,
        description="Default HNSW ef_search parameter for approximate queries",
    )

    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL connection string for asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Construct sync PostgreSQL connection string for Alembic/psycopg."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    return Settings()
