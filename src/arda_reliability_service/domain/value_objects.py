"""Value objects for retrieval requests, neighbor results, and reliability evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    InvalidDescriptionError,
    InvalidProposedDecisionError,
    InvalidScoringConfigError,
)


class GateDecision(str, Enum):
    """Platform gate decision outcome.

    NOTE: The gate decision is strictly binary: PASS (proceed autonomously)
    or ESCALATE (human-in-the-loop review required).
    Under no circumstances is 'REJECT' a valid gate outcome.
    """

    PASS = "PASS"
    ESCALATE = "ESCALATE"


class ProposedDecision(str, Enum):
    """The incoming agent's proposed action under adjudication."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"


class EvaluationReasonCode(str, Enum):
    """Standardized machine-readable explanation codes for reliability outcomes."""

    RELIABILITY_THRESHOLD_MET = "RELIABILITY_THRESHOLD_MET"
    COLD_START_INSUFFICIENT_EVIDENCE = "COLD_START_INSUFFICIENT_EVIDENCE"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    WEAK_EVIDENCE_DENSITY = "WEAK_EVIDENCE_DENSITY"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    HIGH_SIMILARITY_CONFLICT = "HIGH_SIMILARITY_CONFLICT"
    SCORE_BELOW_THRESHOLD = "SCORE_BELOW_THRESHOLD"


@dataclass(frozen=True)
class ReliabilityScore:
    """Decomposed and combined reliability score value object.

    Formula: raw_score = agreement * density_factor * coverage_factor
    Each component is strictly bounded in [0.0, 1.0].
    """

    agreement: float
    density_factor: float
    coverage_factor: float
    raw_score: float

    def __post_init__(self) -> None:
        for name, val in [
            ("agreement", self.agreement),
            ("density_factor", self.density_factor),
            ("coverage_factor", self.coverage_factor),
            ("raw_score", self.raw_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise DomainValidationError(
                    f"{name} must be within [0.0, 1.0], got {val}"
                )


@dataclass(frozen=True)
class ScoringConfig:
    """Provisional scoring and gating configuration for reliability evaluation.

    Thresholds in this configuration represent baseline defaults for precedent evaluation
    and should be calibrated against domain-specific operational data before production deployment.
    """

    min_neighbor_count: int = 3
    min_similarity_threshold: float = 0.70
    min_coverage_threshold: float = 0.50
    min_density_threshold: float = 0.40
    min_agreement_threshold: float = 0.60
    min_reliability_threshold: float = 0.65
    conflict_penalty_weight: float = 0.50
    conflict_veto_similarity: float = 0.95
    density_peak_weight: float = 0.50
    top_k: int = 5
    ef_search: int = 64
    weighting_strategy: str = "similarity_weighted"

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_similarity_threshold < 1.0):
            raise InvalidScoringConfigError(
                f"min_similarity_threshold must be in [0.0, 1.0), got {self.min_similarity_threshold}"
            )
        if not (0.0 < self.min_coverage_threshold <= 1.0):
            raise InvalidScoringConfigError(
                f"min_coverage_threshold must be in (0.0, 1.0], got {self.min_coverage_threshold}"
            )
        if not (0.0 < self.min_density_threshold <= 1.0):
            raise InvalidScoringConfigError(
                f"min_density_threshold must be in (0.0, 1.0], got {self.min_density_threshold}"
            )
        if not (0.0 < self.min_agreement_threshold <= 1.0):
            raise InvalidScoringConfigError(
                f"min_agreement_threshold must be in (0.0, 1.0], got {self.min_agreement_threshold}"
            )
        if not (0.0 < self.min_reliability_threshold <= 1.0):
            raise InvalidScoringConfigError(
                f"min_reliability_threshold must be in (0.0, 1.0], got {self.min_reliability_threshold}"
            )
        if not (0.0 <= self.conflict_penalty_weight <= 1.0):
            raise InvalidScoringConfigError(
                f"conflict_penalty_weight must be in [0.0, 1.0], got {self.conflict_penalty_weight}"
            )
        if not (0.0 <= self.conflict_veto_similarity <= 1.0):
            raise InvalidScoringConfigError(
                f"conflict_veto_similarity must be in [0.0, 1.0], got {self.conflict_veto_similarity}"
            )
        if not (0.0 <= self.density_peak_weight <= 1.0):
            raise InvalidScoringConfigError(
                f"density_peak_weight must be in [0.0, 1.0], got {self.density_peak_weight}"
            )
        if self.min_neighbor_count <= 0:
            raise InvalidScoringConfigError(
                f"min_neighbor_count must be positive, got {self.min_neighbor_count}"
            )
        if self.top_k < self.min_neighbor_count:
            raise InvalidScoringConfigError(
                f"top_k ({self.top_k}) cannot be less than min_neighbor_count ({self.min_neighbor_count})"
            )
        if self.ef_search <= 0:
            raise InvalidScoringConfigError(
                f"ef_search must be positive, got {self.ef_search}"
            )
        if self.weighting_strategy not in {"similarity_weighted", "unweighted"}:
            raise InvalidScoringConfigError(
                f"weighting_strategy must be 'similarity_weighted' or 'unweighted', got '{self.weighting_strategy}'"
            )


@dataclass(frozen=True)
class ReliabilityEvaluationRequest:
    """Incoming request to evaluate whether a proposed agent decision is reliable."""

    tenant_id: UUID
    agent_id: UUID
    case_description: str
    proposed_decision: str

    def __post_init__(self) -> None:
        if not self.case_description or not self.case_description.strip():
            raise InvalidDescriptionError(
                "case_description must not be empty or whitespace."
            )
        valid_decisions = {
            ProposedDecision.APPROVE.value,
            ProposedDecision.REJECT.value,
        }
        if self.proposed_decision not in valid_decisions:
            raise InvalidProposedDecisionError(
                f"Invalid proposed_decision '{self.proposed_decision}'. Must be one of {sorted(valid_decisions)}."
            )


@dataclass(frozen=True)
class ReliabilityEvaluationResult:
    """Structured and auditable result of a reliability evaluation."""

    decision: GateDecision
    proposed_decision: str
    raw_score: float
    agreement: float
    density_factor: float
    coverage_factor: float
    evidence_count: int
    matching_evidence_count: int
    conflicting_evidence_count: int
    evaluation_reason: str
    reason_code: str
    threshold_used: float
    retrieved_evidence: list[ReferenceNeighbor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    calibrated_confidence: float | None = None
    is_calibrated: bool = False
    calibration_version: str | None = None


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
