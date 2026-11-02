"""Configuration settings for ARDA Reliability Service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from arda_reliability_service.domain.value_objects import ScoringConfig


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

    # gRPC Server Settings
    grpc_host: str = Field(
        default="0.0.0.0",
        description="gRPC server bind host",
    )
    grpc_port: int = Field(
        default=50051,
        description="gRPC server bind port",
    )
    grpc_max_workers: int = Field(
        default=10,
        description="Maximum worker threads for gRPC server",
    )
    grpc_max_message_length_mb: int = Field(
        default=16,
        description="Maximum gRPC message size in megabytes",
    )
    grpc_grace_period_seconds: float = Field(
        default=5.0,
        description="Grace period in seconds for graceful shutdown",
    )

    # Redpanda / Kafka Messaging Settings
    redpanda_brokers: str = Field(
        default="127.0.0.1:19092",
        description="Comma-separated bootstrap broker addresses",
    )
    redpanda_topic: str = Field(
        default="arda.human-resolutions.v1",
        description="Topic for authoritative human feedback resolution events",
    )
    redpanda_consumer_group: str = Field(
        default="arda-reliability-feedback-group",
        description="Consumer group identifier for feedback ingestion",
    )
    redpanda_dlq_topic: str = Field(
        default="arda.human-resolutions.dlq",
        description="Dead-letter queue topic for invalid or exhausted messages",
    )
    redpanda_max_retries: int = Field(
        default=3,
        description="Maximum transient retry attempts before dead-letter routing",
    )
    redpanda_retry_backoff_base_seconds: float = Field(
        default=0.5,
        description="Base backoff duration in seconds for transient retries",
    )
    redpanda_auto_offset_reset: str = Field(
        default="earliest",
        description="Offset reset strategy: 'earliest' or 'latest'",
    )
    redpanda_trusted_producers: list[str] = Field(
        default=[
            "arda.human-review-service",
            "arda.gateway.review-worker",
            "arda.workflow-engine",
        ],
        description="Allowed trusted event producer service identities",
    )
    redpanda_producer_secret: str = Field(
        default="arda-producer-secret-default",
        description="Shared secret for authenticating trusted event producers",
    )
    redpanda_enforce_producer_auth: bool = Field(
        default=True,
        description="Whether to strictly enforce producer authentication and reviewer screening",
    )

    # Provisional Reliability Scoring & Gate Thresholds (Week 3 experimental defaults)
    # NOTE: These values are experimental defaults established for developmental gating.
    # They are provisional and NOT calibrated production thresholds.
    reliability_min_neighbor_count: int = Field(
        default=3,
        description="Minimum relevant neighbors required to avoid cold-start escalation",
    )
    reliability_min_similarity_threshold: float = Field(
        default=0.70,
        description="Cosine similarity floor below which a neighbor is considered irrelevant noise",
    )
    reliability_min_coverage_threshold: float = Field(
        default=0.50,
        description="Minimum evidence coverage factor required to qualify for PASS",
    )
    reliability_min_density_threshold: float = Field(
        default=0.40,
        description="Minimum evidence density factor required to qualify for PASS",
    )
    reliability_min_agreement_threshold: float = Field(
        default=0.60,
        description="Minimum precedent agreement factor required to qualify for PASS",
    )
    reliability_min_threshold: float = Field(
        default=0.65,
        description="Minimum raw reliability score required to qualify for PASS",
    )
    reliability_conflict_penalty_weight: float = Field(
        default=0.50,
        description="Penalty weight applied to coverage when relevant precedents conflict",
    )
    reliability_conflict_veto_similarity: float = Field(
        default=0.95,
        description="Extreme similarity threshold above which any conflicting neighbor triggers veto escalation",
    )
    reliability_density_peak_weight: float = Field(
        default=0.50,
        description="Weight given to top-1 neighbor similarity vs average cohort similarity in density",
    )
    reliability_weighting_strategy: str = Field(
        default="similarity_weighted",
        description="Precedent agreement weighting: 'similarity_weighted' or 'unweighted'",
    )

    def model_post_init(self, context: object, /) -> None:
        """Validate scoring configuration during application startup."""
        super().model_post_init(context)
        # Validate that scoring config parameters satisfy domain invariants
        self.get_scoring_config()

    def get_scoring_config(self) -> ScoringConfig:
        """Construct validated domain ScoringConfig from application settings."""
        return ScoringConfig(
            min_neighbor_count=self.reliability_min_neighbor_count,
            min_similarity_threshold=self.reliability_min_similarity_threshold,
            min_coverage_threshold=self.reliability_min_coverage_threshold,
            min_density_threshold=self.reliability_min_density_threshold,
            min_agreement_threshold=self.reliability_min_agreement_threshold,
            min_reliability_threshold=self.reliability_min_threshold,
            conflict_penalty_weight=self.reliability_conflict_penalty_weight,
            conflict_veto_similarity=self.reliability_conflict_veto_similarity,
            density_peak_weight=self.reliability_density_peak_weight,
            top_k=self.default_retrieval_top_k,
            ef_search=self.hnsw_ef_search,
            weighting_strategy=self.reliability_weighting_strategy,
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
