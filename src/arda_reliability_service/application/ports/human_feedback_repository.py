"""Port defining human feedback persistence and idempotency tracking interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from arda_reliability_service.domain.entities import (
    HumanResolutionEvent,
    ReferenceExample,
)


class HumanFeedbackRepositoryPort(ABC):
    """Abstract port for managing human feedback events and corresponding reference examples."""

    @abstractmethod
    async def get_processed_event(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve existing event record by event_id for idempotency check."""

    @abstractmethod
    async def get_reference_example(
        self, tenant_id: UUID, agent_id: UUID, case_id: str
    ) -> ReferenceExample | None:
        """Retrieve existing reference example by tenant, agent, and case_id."""

    @abstractmethod
    async def record_conflict(
        self,
        event: HumanResolutionEvent,
        existing_label: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a conflicting human resolution and quarantine existing reference example without overwriting its label."""

    @abstractmethod
    async def quarantine_reference_example(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        case_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark an existing reference example as quarantined and exclude it from retrieval."""

    @abstractmethod
    async def record_duplicate(
        self,
        event: HumanResolutionEvent,
    ) -> None:
        """Record duplicate event arrival without re-inserting reference."""

    @abstractmethod
    async def save_reference_and_event(
        self,
        event: HumanResolutionEvent,
        reference: ReferenceExample,
    ) -> bool:
        """Atomically persist a new reference example and its originating event record."""

    @abstractmethod
    async def count_events(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        status: str | None = None,
    ) -> int:
        """Count feedback events matching tenant, agent, and optional status."""

    @abstractmethod
    async def delete_all_for_agent(self, tenant_id: UUID, agent_id: UUID) -> int:
        """Delete all feedback events and reference examples for an agent (primarily for testing)."""
