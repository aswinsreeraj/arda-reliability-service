"""PostgreSQL implementation of HumanFeedbackRepositoryPort."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from arda_reliability_service.application.ports.human_feedback_repository import (
    HumanFeedbackRepositoryPort,
)
from arda_reliability_service.domain.entities import (
    HumanResolutionEvent,
    ReferenceExample,
)
from arda_reliability_service.infrastructure.database.models import (
    HumanFeedbackEventModel,
    ReferenceExampleModel,
)


class PgHumanFeedbackRepository(HumanFeedbackRepositoryPort):
    """Async PostgreSQL repository for feedback event audit logging and reference persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _reference_model_to_entity(
        self, model: ReferenceExampleModel
    ) -> ReferenceExample:
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

    async def get_processed_event(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve existing event record by event_id for idempotency check."""
        stmt = select(HumanFeedbackEventModel).where(
            HumanFeedbackEventModel.event_id == event_id
        )
        res = await self.session.execute(stmt)
        model = res.scalar_one_or_none()
        if model is None:
            return None
        return {
            "id": model.id,
            "event_id": model.event_id,
            "event_type": model.event_type,
            "event_version": model.event_version,
            "tenant_id": model.tenant_id,
            "agent_id": model.agent_id,
            "case_id": model.case_id,
            "human_decision": model.human_decision,
            "proposed_decision": model.proposed_decision,
            "reviewer_id": model.reviewer_id,
            "status": model.status,
            "conflict_details": model.conflict_details,
            "correlation_id": model.correlation_id,
            "resolved_at": model.resolved_at,
            "created_at": model.created_at,
        }

    async def get_reference_example(
        self, tenant_id: UUID, agent_id: UUID, case_id: str
    ) -> ReferenceExample | None:
        """Retrieve existing reference example by tenant, agent, and case_id."""
        stmt = select(ReferenceExampleModel).where(
            ReferenceExampleModel.tenant_id == tenant_id,
            ReferenceExampleModel.agent_id == agent_id,
            ReferenceExampleModel.example_id == case_id,
        )
        res = await self.session.execute(stmt)
        model = res.scalar_one_or_none()
        return self._reference_model_to_entity(model) if model is not None else None

    async def record_conflict(
        self,
        event: HumanResolutionEvent,
        existing_label: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a conflicting human resolution without overwriting existing reference."""
        conflict_payload = details or {
            "existing_label": existing_label,
            "new_label": event.human_decision,
            "case_id": event.case_id,
            "reviewer_id": event.reviewer_id,
        }

        stmt = (
            insert(HumanFeedbackEventModel)
            .values(
                event_id=event.event_id,
                event_type=event.event_type,
                event_version=event.event_version,
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                case_id=event.case_id,
                human_decision=event.human_decision,
                proposed_decision=event.proposed_decision,
                reviewer_id=event.reviewer_id,
                status="CONFLICT",
                conflict_details=conflict_payload,
                correlation_id=event.correlation_id,
                resolved_at=event.resolved_at,
            )
            .on_conflict_do_nothing(index_elements=["event_id"])
        )

        await self.session.execute(stmt)
        await self.session.flush()

    async def record_duplicate(
        self,
        event: HumanResolutionEvent,
    ) -> None:
        """Record duplicate event arrival without re-inserting reference."""
        stmt = (
            insert(HumanFeedbackEventModel)
            .values(
                event_id=event.event_id,
                event_type=event.event_type,
                event_version=event.event_version,
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                case_id=event.case_id,
                human_decision=event.human_decision,
                proposed_decision=event.proposed_decision,
                reviewer_id=event.reviewer_id,
                status="IDEMPOTENT_DUPLICATE",
                conflict_details=None,
                correlation_id=event.correlation_id,
                resolved_at=event.resolved_at,
            )
            .on_conflict_do_nothing(index_elements=["event_id"])
        )

        await self.session.execute(stmt)
        await self.session.flush()

    async def save_reference_and_event(
        self,
        event: HumanResolutionEvent,
        reference: ReferenceExample,
    ) -> None:
        """Atomically persist a new reference example and its originating event record."""
        if reference.embedding is None:
            raise ValueError(
                f"Cannot persist reference '{reference.example_id}' without an embedding vector."
            )

        # 1. Insert Reference Example
        ref_stmt = (
            insert(ReferenceExampleModel)
            .values(
                id=reference.id,
                example_id=reference.example_id,
                tenant_id=reference.tenant_id,
                agent_id=reference.agent_id,
                case_description=reference.case_description,
                label=reference.label,
                source=reference.source,
                status=reference.status,
                metadata=reference.metadata,
                embedding=reference.embedding,
                model_name=reference.model_name,
                model_version=reference.model_version,
                created_at=reference.created_at,
                updated_at=reference.updated_at,
            )
            .on_conflict_do_nothing(
                index_elements=["tenant_id", "agent_id", "example_id"]
            )
        )
        await self.session.execute(ref_stmt)

        # 2. Insert Event Record
        event_stmt = (
            insert(HumanFeedbackEventModel)
            .values(
                event_id=event.event_id,
                event_type=event.event_type,
                event_version=event.event_version,
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                case_id=event.case_id,
                human_decision=event.human_decision,
                proposed_decision=event.proposed_decision,
                reviewer_id=event.reviewer_id,
                status="PROCESSED",
                conflict_details=None,
                correlation_id=event.correlation_id,
                resolved_at=event.resolved_at,
            )
            .on_conflict_do_nothing(index_elements=["event_id"])
        )
        await self.session.execute(event_stmt)
        await self.session.flush()

    async def count_events(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        status: str | None = None,
    ) -> int:
        """Count feedback events matching tenant, agent, and optional status."""
        stmt = (
            select(func.count())
            .select_from(HumanFeedbackEventModel)
            .where(
                HumanFeedbackEventModel.tenant_id == tenant_id,
                HumanFeedbackEventModel.agent_id == agent_id,
            )
        )
        if status is not None:
            stmt = stmt.where(HumanFeedbackEventModel.status == status)

        res = await self.session.execute(stmt)
        count = res.scalar()
        return count or 0

    async def delete_all_for_agent(self, tenant_id: UUID, agent_id: UUID) -> int:
        """Delete all feedback events and reference examples for an agent (primarily for testing)."""
        stmt_ev = delete(HumanFeedbackEventModel).where(
            HumanFeedbackEventModel.tenant_id == tenant_id,
            HumanFeedbackEventModel.agent_id == agent_id,
        )
        res_ev = await self.session.execute(stmt_ev)

        stmt_ref = delete(ReferenceExampleModel).where(
            ReferenceExampleModel.tenant_id == tenant_id,
            ReferenceExampleModel.agent_id == agent_id,
        )
        res_ref = await self.session.execute(stmt_ref)

        await self.session.flush()
        count_ev = int(getattr(res_ev, "rowcount", 0) or 0)
        count_ref = int(getattr(res_ref, "rowcount", 0) or 0)
        return count_ev + count_ref
