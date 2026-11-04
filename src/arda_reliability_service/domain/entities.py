"""Domain entities for ARDA Reliability Service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    EventValidationError,
    InvalidDescriptionError,
    InvalidEmbeddingDimensionError,
    UnreviewedAgentDecisionError,
    UnsupportedEventVersionError,
)

VALID_LABELS = frozenset({"APPROVE", "REJECT"})
VALID_STATUSES = frozenset({"active", "archived", "pending", "quarantined", "disputed"})


class HumanResolutionStatus(str, Enum):
    """Lifecycle and outcome status of an ingested human resolution."""

    PROCESSED = "PROCESSED"
    IDEMPOTENT_DUPLICATE = "IDEMPOTENT_DUPLICATE"
    CONFLICT = "CONFLICT"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


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

        if not self.label or not self.label.strip():
            raise DomainValidationError("label must not be empty or whitespace.")

        if self.metadata.get("custom_domain") or self.metadata.get("domain"):
            pass
        elif self.label not in VALID_LABELS:
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


@dataclass
class HumanResolutionEvent:
    """Domain representation of an adjudicated human resolution event."""

    event_id: str
    event_type: str
    event_version: str
    tenant_id: UUID
    agent_id: UUID
    case_id: str
    original_case_description: str
    human_decision: str
    proposed_decision: str
    reviewer_id: str
    resolved_at: datetime
    correlation_id: str = ""
    producer_id: str = "arda.human-review-service"
    key_id: str = "v1"
    signature: str = ""
    timestamp: datetime | None = None

    def get_event_timestamp(self) -> datetime:
        """Return the effective signed event timestamp."""
        return self.timestamp or self.resolved_at

    def __post_init__(self) -> None:
        """Validate domain invariants for human resolution event."""
        if self.timestamp is None:
            self.timestamp = self.resolved_at

        if not self.event_id or not self.event_id.strip():
            raise EventValidationError("event_id must not be empty or whitespace.")

        if self.event_version != "v1":
            raise UnsupportedEventVersionError(
                f"Unsupported event version '{self.event_version}'. Supported versions: ['v1']."
            )

        if not isinstance(self.tenant_id, UUID):
            raise DomainValidationError("tenant_id must be a valid UUID instance.")

        if not isinstance(self.agent_id, UUID):
            raise DomainValidationError("agent_id must be a valid UUID instance.")

        if not self.case_id or not self.case_id.strip():
            raise DomainValidationError("case_id must not be empty or whitespace.")

        if (
            not self.original_case_description
            or not self.original_case_description.strip()
        ):
            raise InvalidDescriptionError(
                "original_case_description must not be empty or whitespace."
            )

        if not self.human_decision or not self.human_decision.strip():
            raise UnreviewedAgentDecisionError(
                "human_decision must not be empty or whitespace."
            )

        if self.human_decision.upper().strip() in {
            "UNREVIEWED",
            "PENDING",
            "UNKNOWN",
            "NONE",
        }:
            raise UnreviewedAgentDecisionError(
                f"human_decision '{self.human_decision}' is invalid. Authoritative human decision required."
            )

        if not self.proposed_decision or not self.proposed_decision.strip():
            raise DomainValidationError(
                "proposed_decision must not be empty or whitespace."
            )

        if not self.reviewer_id or not self.reviewer_id.strip():
            raise DomainValidationError("reviewer_id must not be empty or whitespace.")

        if not self.producer_id or not self.producer_id.strip():
            raise DomainValidationError("producer_id must not be empty or whitespace.")

        if not isinstance(self.resolved_at, datetime):
            raise DomainValidationError(
                "resolved_at must be a valid datetime instance."
            )
