"""Domain exceptions for ARDA Reliability Service."""

from __future__ import annotations


class ReliabilityDomainError(Exception):
    """Base class for all domain exceptions in Reliability Service."""


class DomainValidationError(ReliabilityDomainError):
    """Raised when an entity or value object fails domain invariants."""


class InvalidDescriptionError(DomainValidationError):
    """Raised when a case description is empty, whitespace, or invalid."""


class InvalidEmbeddingDimensionError(DomainValidationError):
    """Raised when an embedding vector does not match the configured model dimension."""


class TenantIsolationViolationError(ReliabilityDomainError):
    """Raised when a cross-tenant operation or leaked reference is detected."""


class AgentIsolationViolationError(ReliabilityDomainError):
    """Raised when a cross-agent operation or leaked reference is detected."""


class DuplicateReferenceExampleError(ReliabilityDomainError):
    """Raised when attempting to insert an identical reference example violating uniqueness."""


class ReferenceExampleNotFoundError(ReliabilityDomainError):
    """Raised when a requested reference example does not exist in the tenant/agent scope."""


class InvalidProposedDecisionError(DomainValidationError):
    """Raised when an incoming proposed decision is not supported (e.g. neither APPROVE nor REJECT)."""


class InvalidScoringConfigError(DomainValidationError):
    """Raised when scoring configuration thresholds violate mathematical or domain invariants."""


class ReliabilityEvaluationError(ReliabilityDomainError):
    """Raised when an error occurs during reliability evaluation orchestration."""


class EventValidationError(DomainValidationError):
    """Raised when an incoming human feedback event fails validation."""


class UnsupportedEventVersionError(EventValidationError):
    """Raised when an incoming event has an unsupported schema version."""


class UnreviewedAgentDecisionError(DomainValidationError):
    """Raised when an event lacks an authoritative human decision or attempts to promote an unreviewed decision."""


class ConflictingHumanResolutionError(ReliabilityDomainError):
    """Raised when an incoming human resolution conflicts with an existing authoritative reference."""


class EventProcessingError(ReliabilityDomainError):
    """Raised when an unrecoverable failure occurs during event processing."""


class UnauthorizedProducerError(ReliabilityDomainError):
    """Raised when an event originates from an untrusted or unauthenticated producer."""


class UnauthorizedReviewerError(ReliabilityDomainError):
    """Raised when a human resolution event cannot establish authorized reviewer credentials."""


class DLQPublicationError(ReliabilityDomainError):
    """Raised when forwarding a failed message to the Dead-Letter Queue fails."""
