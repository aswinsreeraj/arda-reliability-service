"""Create human_feedback_events table for idempotency and audit.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-02
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    # Create human_feedback_events table
    op.create_table(
        "human_feedback_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("event_id", sa.String(128), nullable=False),
        sa.Column("event_type", sa.String(128), nullable=False),
        sa.Column("event_version", sa.String(32), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_id", sa.String(128), nullable=False),
        sa.Column("human_decision", sa.String(32), nullable=False),
        sa.Column("proposed_decision", sa.String(32), nullable=True),
        sa.Column("reviewer_id", sa.String(128), nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="PROCESSED",
        ),
        sa.Column(
            "conflict_details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("correlation_id", sa.String(128), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Unique index on event_id for strict database-level idempotency
    op.create_index(
        "idx_human_feedback_events_event_id",
        "human_feedback_events",
        ["event_id"],
        unique=True,
    )

    # Scoped filtering and status indexes
    op.create_index(
        "idx_human_feedback_events_status",
        "human_feedback_events",
        ["status"],
    )
    op.create_index(
        "idx_human_feedback_events_tenant_agent_case",
        "human_feedback_events",
        ["tenant_id", "agent_id", "case_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_human_feedback_events_tenant_agent_case",
        table_name="human_feedback_events",
    )
    op.drop_index(
        "idx_human_feedback_events_status",
        table_name="human_feedback_events",
    )
    op.drop_index(
        "idx_human_feedback_events_event_id",
        table_name="human_feedback_events",
    )
    op.drop_table("human_feedback_events")
