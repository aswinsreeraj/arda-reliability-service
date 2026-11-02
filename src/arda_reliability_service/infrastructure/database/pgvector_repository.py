"""PostgreSQL and pgvector implementation of the ReferenceExampleRepositoryPort."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from arda_reliability_service.application.ports.reference_repository import (
    ReferenceExampleRepositoryPort,
)
from arda_reliability_service.domain.entities import ReferenceExample
from arda_reliability_service.domain.exceptions import TenantIsolationViolationError
from arda_reliability_service.domain.value_objects import ReferenceNeighbor
from arda_reliability_service.infrastructure.database.models import (
    ReferenceExampleModel,
)


class PgVectorReferenceExampleRepository(ReferenceExampleRepositoryPort):
    """Async repository using SQLAlchemy and pgvector for reference storage and retrieval."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _to_entity(self, model: ReferenceExampleModel) -> ReferenceExample:
        """Convert ORM model to domain entity."""
        return ReferenceExample(
            id=model.id,
            example_id=model.example_id,
            tenant_id=model.tenant_id,
            agent_id=model.agent_id,
            case_description=model.case_description,
            label=model.label,
            source=model.source,
            status=model.status,
            metadata=dict(model.metadata_),
            embedding=[float(x) for x in model.embedding]
            if model.embedding is not None
            else None,
            model_name=model.model_name,
            model_version=model.model_version,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def save(self, example: ReferenceExample) -> ReferenceExample:
        """Persist a single reference example."""
        await self.save_batch([example], on_conflict_do_update=True)
        return example

    async def save_batch(
        self,
        examples: list[ReferenceExample],
        on_conflict_do_update: bool = True,
    ) -> list[str]:
        """Persist a batch of reference examples idempotently."""
        if not examples:
            return []

        records: list[dict[str, Any]] = []
        for ex in examples:
            if ex.embedding is None:
                raise ValueError(
                    f"Cannot persist example '{ex.example_id}' without an embedding vector."
                )

            records.append(
                {
                    "id": ex.id,
                    "example_id": ex.example_id,
                    "tenant_id": ex.tenant_id,
                    "agent_id": ex.agent_id,
                    "case_description": ex.case_description,
                    "label": ex.label,
                    "source": ex.source,
                    "status": ex.status,
                    "metadata": ex.metadata,
                    "embedding": ex.embedding,
                    "model_name": ex.model_name,
                    "model_version": ex.model_version,
                    "created_at": ex.created_at,
                    "updated_at": ex.updated_at,
                }
            )

        stmt = insert(ReferenceExampleModel).values(records)

        if on_conflict_do_update:
            stmt = stmt.on_conflict_do_update(
                index_elements=["tenant_id", "agent_id", "example_id"],
                set_={
                    "case_description": stmt.excluded.case_description,
                    "label": stmt.excluded.label,
                    "source": stmt.excluded.source,
                    "status": stmt.excluded.status,
                    "metadata": stmt.excluded.metadata,
                    "embedding": stmt.excluded.embedding,
                    "model_name": stmt.excluded.model_name,
                    "model_version": stmt.excluded.model_version,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
        else:
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["tenant_id", "agent_id", "example_id"]
            )

        await self.session.execute(stmt)
        await self.session.flush()
        return [ex.example_id for ex in examples]

    async def get_by_example_id(
        self, tenant_id: UUID, agent_id: UUID, example_id: str
    ) -> ReferenceExample | None:
        """Retrieve a specific reference example by tenant, agent, and example_id."""
        stmt = select(ReferenceExampleModel).where(
            ReferenceExampleModel.tenant_id == tenant_id,
            ReferenceExampleModel.agent_id == agent_id,
            ReferenceExampleModel.example_id == example_id,
        )
        res = await self.session.execute(stmt)
        model = res.scalar_one_or_none()
        return self._to_entity(model) if model is not None else None

    async def retrieve_exact(
        self,
        query_embedding: list[float],
        tenant_id: UUID,
        agent_id: UUID,
        top_k: int,
    ) -> list[ReferenceNeighbor]:
        """Execute exact cosine similarity retrieval with strict tenant/agent isolation."""
        # Cosine distance: 0.0 (identical) to 2.0 (opposite)
        # Cosine similarity: 1.0 - distance
        dist_expr = ReferenceExampleModel.embedding.cosine_distance(query_embedding)
        sim_expr = (1.0 - dist_expr).label("similarity")

        stmt = (
            select(
                ReferenceExampleModel.example_id,
                ReferenceExampleModel.label,
                ReferenceExampleModel.source,
                ReferenceExampleModel.case_description,
                sim_expr,
                ReferenceExampleModel.metadata_,
                ReferenceExampleModel.tenant_id,
                ReferenceExampleModel.agent_id,
            )
            .where(
                ReferenceExampleModel.tenant_id == tenant_id,
                ReferenceExampleModel.agent_id == agent_id,
                ReferenceExampleModel.status == "active",
            )
            .order_by(
                dist_expr.asc(),
                ReferenceExampleModel.created_at.desc(),
                ReferenceExampleModel.example_id.asc(),
            )
            .limit(top_k)
        )

        res = await self.session.execute(stmt)
        rows = res.all()

        neighbors: list[ReferenceNeighbor] = []
        for row in rows:
            # Defensive assertion of tenant and agent isolation
            if row.tenant_id != tenant_id or row.agent_id != agent_id:
                raise TenantIsolationViolationError(
                    f"Cross-tenant reference leak detected: query tenant={tenant_id}, row tenant={row.tenant_id}"
                )

            neighbors.append(
                ReferenceNeighbor(
                    example_id=row.example_id,
                    label=row.label,
                    source=row.source,
                    case_description=row.case_description,
                    similarity_score=round(float(row.similarity), 4),
                    metadata=dict(row.metadata_),
                )
            )

        return neighbors

    async def retrieve_approximate(
        self,
        query_embedding: list[float],
        tenant_id: UUID,
        agent_id: UUID,
        top_k: int,
        ef_search: int = 64,
    ) -> list[ReferenceNeighbor]:
        """Execute approximate (HNSW) retrieval with ef_search session tuning."""
        # Set session-local ef_search parameter for HNSW graph traversal
        await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))

        # In PostgreSQL with pgvector, querying ordering by embedding <=> query_vec
        # automatically triggers an HNSW index scan when available.
        return await self.retrieve_exact(
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            agent_id=agent_id,
            top_k=top_k,
        )

    async def count(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        status: str | None = None,
    ) -> int:
        """Count reference examples for the given tenant and agent."""
        stmt = (
            select(func.count())
            .select_from(ReferenceExampleModel)
            .where(
                ReferenceExampleModel.tenant_id == tenant_id,
                ReferenceExampleModel.agent_id == agent_id,
            )
        )
        if status is not None:
            stmt = stmt.where(ReferenceExampleModel.status == status)

        res = await self.session.execute(stmt)
        return int(res.scalar_one())

    async def delete_all_for_agent(self, tenant_id: UUID, agent_id: UUID) -> int:
        """Delete all reference examples for a specific tenant and agent."""
        stmt = delete(ReferenceExampleModel).where(
            ReferenceExampleModel.tenant_id == tenant_id,
            ReferenceExampleModel.agent_id == agent_id,
        )
        res = await self.session.execute(stmt)
        await self.session.flush()
        rowcount = getattr(res, "rowcount", 0)
        return int(rowcount or 0)
