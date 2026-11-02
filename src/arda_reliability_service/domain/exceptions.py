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
