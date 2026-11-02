"""Domain entities for ARDA Reliability Service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    InvalidDescriptionError,
    InvalidEmbeddingDimensionError,
)

VALID_LABELS = frozenset({"APPROVE", "REJECT"})
VALID_STATUSES = frozenset({"active", "archived", "pending"})


@dataclass
class ReferenceExample:
    """Core domain entity representing an adjudicated reference decision."""

    example_id: str
    tenant_id: UUID
    agent_id: UUID
    case_description: str
    label: str
    source: str = "synthetic"
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_version: str = "v1.5"
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate domain invariants upon entity instantiation."""
        if not self.example_id or not self.example_id.strip():
            raise DomainValidationError("example_id must not be empty or whitespace.")

        if not self.case_description or not self.case_description.strip():
            raise InvalidDescriptionError(
                "case_description must not be empty or whitespace."
            )

        if self.label not in VALID_LABELS:
            raise DomainValidationError(
                f"Invalid label '{self.label}'. Must be one of {sorted(VALID_LABELS)}."
            )

        if self.status not in VALID_STATUSES:
            raise DomainValidationError(
                f"Invalid status '{self.status}'. Must be one of {sorted(VALID_STATUSES)}."
            )

    def attach_embedding(
        self, embedding: list[float], expected_dim: int | None = None
    ) -> None:
        """Attach an embedding vector with dimension validation."""
        if expected_dim is not None and len(embedding) != expected_dim:
            raise InvalidEmbeddingDimensionError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
            )
        self.embedding = embedding
        self.updated_at = datetime.now(UTC)
