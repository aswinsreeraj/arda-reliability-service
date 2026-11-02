"""SQLAlchemy 2.0 ORM models for reference examples and vector storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""


class ReferenceExampleModel(Base):
    """Persistent database model for reference decisions with dense vector embeddings."""

    __tablename__ = "reference_examples"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Identifiers & Isolation
    example_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )
    agent_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )

    # Case data
    case_description: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="synthetic")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )

    # Arbitrary structured metadata (JSONB)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )

    # Dense vector embedding (384-dimensional for BGE-small)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    # Model provenance
    model_name: Mapped[str] = mapped_column(
        String(128), nullable=False, default="BAAI/bge-small-en-v1.5"
    )
    model_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="v1.5"
    )

    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        # Ensure tenant + agent scoping uniqueness on business example_id
        UniqueConstraint(
            "tenant_id",
            "agent_id",
            "example_id",
            name="uq_reference_examples_tenant_agent_example_id",
        ),
        # Compound index for tenant-scoped active retrieval filtering
        Index(
            "idx_reference_examples_tenant_agent_status",
            "tenant_id",
            "agent_id",
            "status",
        ),
        # HNSW vector index using cosine distance operator (vector_cosine_ops)
        Index(
            "idx_reference_examples_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class HumanFeedbackEventModel(Base):
    """Persistent audit and idempotency record for ingested human resolution events."""

    __tablename__ = "human_feedback_events"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Unique event tracking
    event_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    event_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # Tenant & Agent scoping
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )
    agent_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )
    case_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Decision details
    human_decision: Mapped[str] = mapped_column(String(32), nullable=False)
    proposed_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Ingestion outcome status ('PROCESSED', 'IDEMPOTENT_DUPLICATE', 'CONFLICT', 'FAILED')
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PROCESSED", index=True
    )

    # Conflict diagnosis and metadata
    conflict_details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Timestamps
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "idx_human_feedback_events_tenant_agent_case",
            "tenant_id",
            "agent_id",
            "case_id",
        ),
    )
