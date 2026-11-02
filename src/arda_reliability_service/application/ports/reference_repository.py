"""Port defining reference example persistence and vector retrieval interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from arda_reliability_service.domain.entities import ReferenceExample
from arda_reliability_service.domain.value_objects import ReferenceNeighbor


class ReferenceExampleRepositoryPort(ABC):
    """Abstract port for persisting reference examples and executing vector retrieval."""

    @abstractmethod
    async def save(self, example: ReferenceExample) -> ReferenceExample:
        """Persist a single reference example with its embedding."""

    @abstractmethod
    async def save_batch(
        self,
        examples: list[ReferenceExample],
        on_conflict_do_update: bool = True,
    ) -> list[str]:
        """Persist a batch of reference examples idempotently.

        Args:
            examples: Entities to insert or update.
            on_conflict_do_update: If True, update existing records on (tenant_id, agent_id, example_id) collision.

        Returns:
            List of successfully persisted example_ids.
        """

    @abstractmethod
    async def get_by_example_id(
        self, tenant_id: UUID, agent_id: UUID, example_id: str
    ) -> ReferenceExample | None:
        """Retrieve a specific reference example by tenant, agent, and example_id."""

    @abstractmethod
    async def retrieve_exact(
        self,
        query_embedding: list[float],
        tenant_id: UUID,
        agent_id: UUID,
        top_k: int,
    ) -> list[ReferenceNeighbor]:
        """Execute exact (flat scan) cosine similarity vector retrieval.

        Must enforce tenant_id, agent_id, and status = 'active' filters.
        """

    @abstractmethod
    async def retrieve_approximate(
        self,
        query_embedding: list[float],
        tenant_id: UUID,
        agent_id: UUID,
        top_k: int,
        ef_search: int = 64,
    ) -> list[ReferenceNeighbor]:
        """Execute approximate (HNSW index) cosine similarity vector retrieval.

        Must enforce tenant_id, agent_id, and status = 'active' filters,
        and apply the requested ef_search session parameter.
        """

    @abstractmethod
    async def count(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        status: str | None = None,
    ) -> int:
        """Return the number of reference examples matching the given tenant and agent scope."""

    @abstractmethod
    async def delete_all_for_agent(self, tenant_id: UUID, agent_id: UUID) -> int:
        """Delete all reference examples for a specific tenant and agent (primarily for testing)."""
