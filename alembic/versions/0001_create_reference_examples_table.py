"""Create reference_examples table with pgvector and HNSW index.

Revision ID: 0001
Revises: None
Create Date: 2026-11-02
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    # Ensure pgvector and UUID extensions are enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # Create reference_examples table
    op.create_table(
        "reference_examples",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("example_id", sa.String(128), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_description", sa.Text(), nullable=False),
        sa.Column("label", sa.String(32), nullable=False),
        sa.Column("source", sa.String(64), nullable=False, server_default="synthetic"),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column(
            "model_name",
            sa.String(128),
            nullable=False,
            server_default="BAAI/bge-small-en-v1.5",
        ),
        sa.Column(
            "model_version", sa.String(32), nullable=False, server_default="v1.5"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "agent_id",
            "example_id",
            name="uq_reference_examples_tenant_agent_example_id",
        ),
    )

    # Scoped filtering index for tenant/agent queries
    op.create_index(
        "idx_reference_examples_tenant_agent_status",
        "reference_examples",
        ["tenant_id", "agent_id", "status"],
    )

    # Approximate nearest neighbor HNSW index on vector embeddings with cosine distance
    op.execute(
        "CREATE INDEX idx_reference_examples_hnsw_cosine "
        "ON reference_examples USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )


def downgrade() -> None:
    op.drop_index("idx_reference_examples_hnsw_cosine", table_name="reference_examples")
    op.drop_index(
        "idx_reference_examples_tenant_agent_status", table_name="reference_examples"
    )
    op.drop_table("reference_examples")
