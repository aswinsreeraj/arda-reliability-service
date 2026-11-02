"""Human feedback event schema and serialization for Redpanda/Kafka messaging."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arda_reliability_service.domain.entities import HumanResolutionEvent
from arda_reliability_service.domain.exceptions import (
    EventValidationError,
    InvalidDescriptionError,
    UnreviewedAgentDecisionError,
    UnsupportedEventVersionError,
)

EVENT_TYPE_HUMAN_RESOLUTION = "arda.reliability.human_resolution"
SUPPORTED_EVENT_VERSIONS = frozenset({"v1"})
AUTHORITATIVE_DECISIONS = frozenset({"APPROVE", "REJECT"})


class HumanResolutionEventDTO(BaseModel):
    """Transport-level DTO representing an authoritative human resolution event."""

    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(
        ..., description="Globally unique identifier for the event (UUID or URN)"
    )
    event_type: str = Field(
        default=EVENT_TYPE_HUMAN_RESOLUTION,
        description="Canonical event type identifier",
    )
    event_version: str = Field(default="v1", description="Schema version of this event")
    tenant_id: UUID = Field(..., description="Unique identifier of tenant scope")
    agent_id: UUID = Field(..., description="Unique identifier of agent scope")
    case_id: str = Field(..., description="Business identifier of the case / incident")
    original_case_description: str = Field(
        ...,
        min_length=1,
        description="Natural language case narrative as originally evaluated",
    )
    proposed_decision: str = Field(
        ..., description="Original decision proposed by agent ('APPROVE' or 'REJECT')"
    )
    human_decision: str = Field(
        ...,
        description="Authoritative adjudication outcome decided by human reviewer",
    )
    reviewer_id: str = Field(
        ..., description="Identifier of the human reviewer (e.g. user ID or badge)"
    )
    resolved_at: datetime = Field(
        ..., description="Timestamp when human resolution was finalized"
    )
    correlation_id: str = Field(
        default="", description="Distributed trace or workflow execution ID"
    )

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise EventValidationError("event_id must not be empty or whitespace.")
        return v.strip()

    @field_validator("event_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v not in SUPPORTED_EVENT_VERSIONS:
            raise UnsupportedEventVersionError(
                f"Unsupported event version '{v}'. Supported versions: {sorted(SUPPORTED_EVENT_VERSIONS)}"
            )
        return v

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise EventValidationError("case_id must not be empty or whitespace.")
        return v.strip()

    @field_validator("human_decision")
    @classmethod
    def validate_human_decision(cls, v: str) -> str:
        v_upper = v.upper().strip()
        if v_upper not in AUTHORITATIVE_DECISIONS:
            raise UnreviewedAgentDecisionError(
                f"Invalid human_decision '{v}'. Authoritative human decision must be one of {sorted(AUTHORITATIVE_DECISIONS)}."
            )
        return v_upper

    @field_validator("proposed_decision")
    @classmethod
    def validate_proposed_decision(cls, v: str) -> str:
        v_upper = v.upper().strip()
        if v_upper not in AUTHORITATIVE_DECISIONS:
            raise EventValidationError(
                f"Invalid proposed_decision '{v}'. Must be one of {sorted(AUTHORITATIVE_DECISIONS)}."
            )
        return v_upper

    @field_validator("original_case_description")
    @classmethod
    def validate_case_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise InvalidDescriptionError(
                "original_case_description must not be empty or whitespace."
            )
        return v.strip()

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise EventValidationError("reviewer_id must not be empty or whitespace.")
        return v.strip()

    def to_domain(self) -> HumanResolutionEvent:
        """Convert validated DTO into core domain entity."""
        return HumanResolutionEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            event_version=self.event_version,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            case_id=self.case_id,
            original_case_description=self.original_case_description,
            human_decision=self.human_decision,
            proposed_decision=self.proposed_decision,
            reviewer_id=self.reviewer_id,
            resolved_at=self.resolved_at,
            correlation_id=self.correlation_id,
        )

