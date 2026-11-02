"""Value objects for retrieval requests and neighbor results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    InvalidDescriptionError,
)


@dataclass(frozen=True)
class RetrievalQuery:
    """Incoming query specification for vector neighbor retrieval."""

    tenant_id: UUID
    agent_id: UUID
    case_description: str
    top_k: int = 5
    ef_search: int = 64

    def __post_init__(self) -> None:
        if not self.case_description or not self.case_description.strip():
            raise InvalidDescriptionError(
                "case_description must not be empty or whitespace."
            )
        if self.top_k <= 0:
            raise DomainValidationError(
                f"top_k must be a positive integer, got {self.top_k}."
            )
        if self.ef_search <= 0:
            raise DomainValidationError(
                f"ef_search must be positive, got {self.ef_search}."
            )


@dataclass(frozen=True)
class ReferenceNeighbor:
    """Retrieved reference example with similarity score, matching gRPC contract."""

    example_id: str
    label: str
    source: str
    case_description: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResultSet:
    """Complete outcome of a vector retrieval operation."""

    neighbors: list[ReferenceNeighbor]
    query_description: str
    top_k: int
    retrieval_mode: str  # "exact" or "approximate_hnsw"
    execution_latency_ms: float
