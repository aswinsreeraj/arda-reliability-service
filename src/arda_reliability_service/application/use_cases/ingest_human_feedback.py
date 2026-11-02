"""Application use case for ingesting authoritative human resolutions and managing reference lifecycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.application.ports.event_authorizer import (
    EventAuthorizerPort,
)
from arda_reliability_service.application.ports.human_feedback_repository import (
    HumanFeedbackRepositoryPort,
)
from arda_reliability_service.domain.entities import (
    HumanResolutionEvent,
    ReferenceExample,
)
from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    EventValidationError,
)

logger = logging.getLogger(__name__)


class HumanFeedbackIngestionOutcome(str, Enum):
    """Outcome of processing a human resolution event."""

    PROCESSED = "PROCESSED"
    IDEMPOTENT_DUPLICATE = "IDEMPOTENT_DUPLICATE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class IngestHumanFeedbackResult:
    """Auditable outcome of human feedback ingestion."""

    event_id: str
    outcome: HumanFeedbackIngestionOutcome
    case_id: str
    tenant_id: UUID
    agent_id: UUID
    message: str
    reference_created: bool = False
    conflict_details: dict[str, Any] | None = None


class IngestHumanFeedbackUseCase:
    """Orchestrates validation, idempotency checks, lifecycle rules, embedding, and atomic persistence."""

    def __init__(
        self,
        repository: HumanFeedbackRepositoryPort,
        embedding_provider: EmbeddingProviderPort,
        authorizer: EventAuthorizerPort | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.authorizer = authorizer

    def _normalize_event(
        self, event_input: HumanResolutionEvent | dict[str, Any] | Any
    ) -> HumanResolutionEvent:
        """Parse and validate incoming event into domain entity."""
        if isinstance(event_input, HumanResolutionEvent):
            return event_input

        # Duck typing if DTO passed with to_domain() method
        if hasattr(event_input, "to_domain") and callable(event_input.to_domain):
            return event_input.to_domain()  # type: ignore[no-any-return]

        if isinstance(event_input, dict):
            try:
                resolved_raw = event_input["resolved_at"]
                if isinstance(resolved_raw, str):
                    resolved_at = datetime.fromisoformat(resolved_raw)
                elif isinstance(resolved_raw, datetime):
                    resolved_at = resolved_raw
                else:
                    raise DomainValidationError(
                        "resolved_at must be an ISO string or datetime"
                    )

                tenant_raw = event_input["tenant_id"]
                tenant_id = (
                    tenant_raw
                    if isinstance(tenant_raw, UUID)
                    else UUID(str(tenant_raw))
                )
                agent_raw = event_input["agent_id"]
                agent_id = (
                    agent_raw if isinstance(agent_raw, UUID) else UUID(str(agent_raw))
                )

                return HumanResolutionEvent(
                    event_id=str(event_input["event_id"]),
                    event_type=str(
                        event_input.get(
                            "event_type", "arda.reliability.human_resolution"
                        )
                    ),
                    event_version=str(event_input.get("event_version", "v1")),
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    case_id=str(event_input["case_id"]),
                    original_case_description=str(
                        event_input["original_case_description"]
                    ),
                    human_decision=str(event_input["human_decision"]),
                    proposed_decision=str(
                        event_input.get("proposed_decision", "APPROVE")
                    ),
                    reviewer_id=str(event_input["reviewer_id"]),
                    resolved_at=resolved_at,
                    correlation_id=str(event_input.get("correlation_id", "")),
                    producer_id=str(
                        event_input.get("producer_id", "arda.human-review-service")
                    ),
                    producer_token=str(
                        event_input.get(
                            "producer_token", "arda-producer-secret-default"
                        )
                    ),
                )
            except KeyError as ke:
                raise EventValidationError(f"Missing required field: {ke}") from ke
            except ValueError as ve:
                raise DomainValidationError(f"Invalid field format: {ve}") from ve

        raise DomainValidationError(
            f"Unsupported event input type: {type(event_input).__name__}"
        )

    async def execute(
        self,
        event_input: HumanResolutionEvent | dict[str, Any] | Any,
    ) -> IngestHumanFeedbackResult:
        """Process an authoritative human resolution event.

        Args:
            event_input: Incoming raw dictionary, DTO, or domain entity.

        Returns:
            IngestHumanFeedbackResult describing the processing outcome.
        """
        # 1. Validation & Domain Normalization
        event = self._normalize_event(event_input)

        # 1b. Producer & Reviewer Authorization
        if self.authorizer is not None:
            self.authorizer.authorize_event(event)

        # 2. Idempotency Check on Event ID
        existing_event = await self.repository.get_processed_event(event.event_id)
        if existing_event is not None:
            logger.info(
                "Idempotent duplicate event detected: event_id=%s, tenant_id=%s, agent_id=%s",
                event.event_id,
                event.tenant_id,
                event.agent_id,
            )
            return IngestHumanFeedbackResult(
                event_id=event.event_id,
                outcome=HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE,
                case_id=event.case_id,
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                message=f"Event {event.event_id} was already processed with status {existing_event.get('status')}.",
                reference_created=False,
            )

        # 3. Check for Existing Reference Example by Business Case ID
        existing_reference = await self.repository.get_reference_example(
            tenant_id=event.tenant_id,
            agent_id=event.agent_id,
            case_id=event.case_id,
        )

        if existing_reference is not None:
            # Case is already quarantined due to prior conflict
            if existing_reference.status in {"quarantined", "disputed"}:
                logger.info(
                    "Case %s is already quarantined; recording duplicate event_id=%s without altering quarantine status",
                    event.case_id,
                    event.event_id,
                )
                await self.repository.record_duplicate(event)
                return IngestHumanFeedbackResult(
                    event_id=event.event_id,
                    outcome=HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE,
                    case_id=event.case_id,
                    tenant_id=event.tenant_id,
                    agent_id=event.agent_id,
                    message=f"Case {event.case_id} is quarantined due to prior conflict; reference remains excluded from retrieval.",
                    reference_created=False,
                )

            # Check for matching label vs conflict
            if existing_reference.label == event.human_decision:
                logger.info(
                    "Case %s already exists with matching label '%s'; recording duplicate event_id=%s",
                    event.case_id,
                    event.human_decision,
                    event.event_id,
                )
                await self.repository.record_duplicate(event)
                return IngestHumanFeedbackResult(
                    event_id=event.event_id,
                    outcome=HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE,
                    case_id=event.case_id,
                    tenant_id=event.tenant_id,
                    agent_id=event.agent_id,
                    message=f"Case {event.case_id} already exists with matching label {event.human_decision}.",
                    reference_created=False,
                )

            # Conflicting resolution detected! Quarantine the reference example
            conflict_details = {
                "existing_label": existing_reference.label,
                "new_label": event.human_decision,
                "existing_source": existing_reference.source,
                "case_id": event.case_id,
                "reviewer_id": event.reviewer_id,
                "quarantined_at": datetime.now(UTC).isoformat(),
                "reason": "Conflicting authoritative human adjudications",
            }
            logger.warning(
                "Conflicting human resolution detected for case %s: existing=%s, new=%s, event_id=%s. Quarantining reference.",
                event.case_id,
                existing_reference.label,
                event.human_decision,
                event.event_id,
            )
            await self.repository.record_conflict(
                event=event,
                existing_label=existing_reference.label,
                details=conflict_details,
            )
            return IngestHumanFeedbackResult(
                event_id=event.event_id,
                outcome=HumanFeedbackIngestionOutcome.CONFLICT,
                case_id=event.case_id,
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                message=(
                    f"Conflicting resolution detected: case {event.case_id} has existing "
                    f"label '{existing_reference.label}', but event proposes '{event.human_decision}'. "
                    "Case quarantined and excluded from retrieval."
                ),
                reference_created=False,
                conflict_details=conflict_details,
            )

        # 4. Generate Embedding from Original Case Description ONLY
        # INVARIANT: Embed only the natural language narrative; never labels, decisions, or reviewer IDs.
        embedding = self.embedding_provider.generate_embedding(
            text=event.original_case_description,
            is_query=False,
        )

        # 5. Construct New Reference Example
        reference = ReferenceExample(
            example_id=event.case_id,
            tenant_id=event.tenant_id,
            agent_id=event.agent_id,
            case_description=event.original_case_description,
            label=event.human_decision,
            source="human_review",
            status="active",
            metadata={
                "event_id": event.event_id,
                "correlation_id": event.correlation_id,
                "proposed_decision": event.proposed_decision,
                "reviewer_id": event.reviewer_id,
                "resolved_at": event.resolved_at.isoformat(),
            },
            model_name=self.embedding_provider.model_name,
            model_version="v1.5",
        )
        reference.attach_embedding(
            embedding=embedding,
            expected_dim=self.embedding_provider.dimension,
        )

        # 6. Atomically Persist Reference and Audit Event Record
        created = await self.repository.save_reference_and_event(
            event=event, reference=reference
        )

        if not created:
            existing_ref = await self.repository.get_reference_example(
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                case_id=event.case_id,
            )
            if existing_ref is not None and existing_ref.status in {
                "quarantined",
                "disputed",
            }:
                return IngestHumanFeedbackResult(
                    event_id=event.event_id,
                    outcome=HumanFeedbackIngestionOutcome.CONFLICT,
                    case_id=event.case_id,
                    tenant_id=event.tenant_id,
                    agent_id=event.agent_id,
                    message=f"Concurrent conflict detected; case {event.case_id} was quarantined.",
                    reference_created=False,
                )
            return IngestHumanFeedbackResult(
                event_id=event.event_id,
                outcome=HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE,
                case_id=event.case_id,
                tenant_id=event.tenant_id,
                agent_id=event.agent_id,
                message=f"Concurrent duplicate detected for case {event.case_id}.",
                reference_created=False,
            )

        logger.info(
            "Successfully ingested human resolution: event_id=%s, case_id=%s, label=%s, tenant_id=%s",
            event.event_id,
            event.case_id,
            event.human_decision,
            event.tenant_id,
        )

        return IngestHumanFeedbackResult(
            event_id=event.event_id,
            outcome=HumanFeedbackIngestionOutcome.PROCESSED,
            case_id=event.case_id,
            tenant_id=event.tenant_id,
            agent_id=event.agent_id,
            message=f"Case {event.case_id} successfully persisted as reference example with label {event.human_decision}.",
            reference_created=True,
        )
