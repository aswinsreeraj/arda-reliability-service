"""Unit tests for ReliabilityGateHandler."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import grpc
import pytest

from arda_reliability_service.application.use_cases.evaluate_reliability import (
    EvaluateReliabilityUseCase,
)
from arda_reliability_service.domain.exceptions import DomainValidationError
from arda_reliability_service.domain.value_objects import (
    EvaluationReasonCode,
    GateDecision,
    ReferenceNeighbor,
    ReliabilityEvaluationResult,
    ScoringConfig,
)
from arda_reliability_service.infrastructure.grpc.generated import reliability_pb2
from arda_reliability_service.infrastructure.grpc.handler import ReliabilityGateHandler


class FakeServicerContext:
    """Mock gRPC aio servicer context capturing abort calls."""

    def __init__(self) -> None:
        self.aborted = False
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.aborted = True
        self.code = code
        self.details = details
        raise grpc.RpcError(f"RPC Aborted [{code.name}]: {details}")


@pytest.fixture
def mock_use_case() -> AsyncMock:
    return AsyncMock(spec=EvaluateReliabilityUseCase)


@pytest.fixture
def handler(mock_use_case: AsyncMock) -> ReliabilityGateHandler:
    return ReliabilityGateHandler(
        evaluate_use_case=mock_use_case,
        config=ScoringConfig(min_reliability_threshold=0.65),
    )


class TestReliabilityGateHandlerUnit:
    """Unit tests for gRPC request validation, use-case delegation, and response mapping."""

    @pytest.mark.asyncio
    async def test_valid_approve_request_pass(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        tenant_id = uuid4()
        agent_id = uuid4()

        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="APPROVE",
            raw_score=0.88,
            agreement=1.0,
            density_factor=0.9,
            coverage_factor=0.98,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Reliability threshold met",
            reason_code=EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            threshold_used=0.65,
            retrieved_evidence=[
                ReferenceNeighbor(
                    example_id="ref-101",
                    label="APPROVE",
                    source="synthetic",
                    case_description="Precedent description",
                    similarity_score=0.95,
                )
            ],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Valid disputed debit claim narrative.",
            proposed_decision="APPROVE",
            reliability_threshold=0.65,
        )
        ctx = FakeServicerContext()

        resp = await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert not ctx.aborted
        assert resp.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_PASS
        assert resp.confidence == 0.88
        assert resp.reason_code == "RELIABILITY_THRESHOLD_MET"
        assert len(resp.neighbors) == 1
        assert resp.neighbors[0].reference_example_id == "ref-101"
        assert resp.neighbors[0].label == "APPROVE"
        assert pytest.approx(resp.neighbors[0].similarity, abs=1e-4) == 0.95

    @pytest.mark.asyncio
    async def test_valid_reject_request_pass(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        tenant_id = uuid4()
        agent_id = uuid4()

        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="REJECT",
            raw_score=0.79,
            agreement=1.0,
            density_factor=0.8,
            coverage_factor=0.99,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Reliability threshold met",
            reason_code=EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Expired filing 90 days after transaction.",
            proposed_decision="REJECT",
        )
        ctx = FakeServicerContext()

        resp = await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert not ctx.aborted
        assert resp.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_PASS
        assert resp.confidence == 0.79

    @pytest.mark.asyncio
    async def test_gate_response_confidence_is_raw_score_even_when_calibrated(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        """Verify field 1 confidence in GateResponse remains raw_score and is never overwritten by calibrated_confidence."""
        tenant_id = uuid4()
        agent_id = uuid4()

        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="APPROVE",
            raw_score=0.68,
            calibrated_confidence=0.94,
            is_calibrated=True,
            calibration_version="platt-v1.0.0-seed100",
            agreement=1.0,
            density_factor=0.8,
            coverage_factor=0.99,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Reliability threshold met",
            reason_code=EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Valid claim with precedent evidence.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        resp = await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert not ctx.aborted
        # Field 1 'confidence' must strictly reflect raw_score (0.68), preserving contract stability
        assert resp.confidence == 0.68
        assert resp.confidence != 0.94
        assert resp.raw_score == 0.68
        assert resp.is_calibrated is True
        assert resp.HasField("calibrated_confidence")
        assert resp.calibrated_confidence == 0.94
        assert resp.HasField("calibration_version")
        assert resp.calibration_version == "platt-v1.0.0-seed100"

    @pytest.mark.asyncio
    async def test_uncalibrated_response_behavior_omits_optional_calibration_fields(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        """Verify uncalibrated gate response omits calibrated_confidence and calibration_version while setting raw_score."""
        tenant_id = uuid4()
        agent_id = uuid4()

        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.ESCALATE,
            proposed_decision="APPROVE",
            raw_score=0.35,
            calibrated_confidence=None,
            is_calibrated=False,
            calibration_version=None,
            agreement=0.5,
            density_factor=0.7,
            coverage_factor=1.0,
            evidence_count=2,
            matching_evidence_count=1,
            conflicting_evidence_count=1,
            evaluation_reason="Below threshold",
            reason_code=EvaluationReasonCode.SCORE_BELOW_THRESHOLD.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Uncalibrated cold claim.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        resp = await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert not ctx.aborted
        assert resp.confidence == 0.35
        assert resp.raw_score == 0.35
        assert resp.is_calibrated is False
        assert not resp.HasField("calibrated_confidence")
        assert not resp.HasField("calibration_version")

    @pytest.mark.asyncio
    async def test_stable_confidence_semantics_never_switch_dynamically(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        """Prove that Field 1 confidence produces the exact same value regardless of whether calibration is active or inactive."""
        tenant_id = uuid4()
        agent_id = uuid4()
        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Deterministic test case.",
            proposed_decision="APPROVE",
        )

        # 1. Uncalibrated execution
        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="APPROVE",
            raw_score=0.72,
            calibrated_confidence=None,
            is_calibrated=False,
            calibration_version=None,
            agreement=1.0,
            density_factor=0.8,
            coverage_factor=0.9,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Threshold met",
            reason_code=EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )
        resp_uncal = await handler.Gate(req, FakeServicerContext())  # type: ignore[arg-type]

        # 2. Calibrated execution (calibrator outputs 0.98)
        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="APPROVE",
            raw_score=0.72,
            calibrated_confidence=0.98,
            is_calibrated=True,
            calibration_version="platt-v1.0.0-seed100",
            agreement=1.0,
            density_factor=0.8,
            coverage_factor=0.9,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Threshold met",
            reason_code=EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )
        resp_cal = await handler.Gate(req, FakeServicerContext())  # type: ignore[arg-type]

        # Field 1 'confidence' must be IDENTICAL across both responses
        assert resp_uncal.confidence == 0.72
        assert resp_cal.confidence == 0.72
        assert resp_uncal.confidence == resp_cal.confidence
        # Only the explicit presence-aware fields distinguish calibration state
        assert not resp_uncal.HasField("calibrated_confidence")
        assert resp_cal.HasField("calibrated_confidence")
        assert resp_cal.calibrated_confidence == 0.98

    @pytest.mark.asyncio
    async def test_escalate_response_maps_to_review_required(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        tenant_id = uuid4()
        agent_id = uuid4()

        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.ESCALATE,
            proposed_decision="APPROVE",
            raw_score=0.42,
            agreement=0.6,
            density_factor=0.7,
            coverage_factor=1.0,
            evidence_count=3,
            matching_evidence_count=2,
            conflicting_evidence_count=1,
            evaluation_reason="Reliability score below threshold",
            reason_code=EvaluationReasonCode.SCORE_BELOW_THRESHOLD.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Borderline cancellation dispute.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        resp = await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert not ctx.aborted
        assert resp.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_REVIEW_REQUIRED
        assert resp.confidence == 0.42
        assert resp.reason_code == "SCORE_BELOW_THRESHOLD"

    @pytest.mark.asyncio
    async def test_cold_start_escalation(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        tenant_id = uuid4()
        agent_id = uuid4()

        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.ESCALATE,
            proposed_decision="APPROVE",
            raw_score=0.0,
            agreement=0.0,
            density_factor=0.0,
            coverage_factor=0.0,
            evidence_count=0,
            matching_evidence_count=0,
            conflicting_evidence_count=0,
            evaluation_reason="Cold start: found 0 relevant precedents",
            reason_code=EvaluationReasonCode.COLD_START_INSUFFICIENT_EVIDENCE.value,
            threshold_used=0.65,
            retrieved_evidence=[],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Brand new dispute category with no precedents.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        resp = await handler.Gate(req, ctx)  # type: ignore[arg-type]

        # Cold-start must return valid GateResponse with REVIEW_REQUIRED, NOT an RPC abort!
        assert not ctx.aborted
        assert resp.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_REVIEW_REQUIRED
        assert resp.confidence == 0.0
        assert resp.reason_code == "COLD_START_INSUFFICIENT_EVIDENCE"

    @pytest.mark.asyncio
    async def test_invalid_proposed_decision_aborts_invalid_argument(
        self, handler: ReliabilityGateHandler
    ) -> None:
        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="Valid text.",
            proposed_decision="MAYBE",
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "Invalid proposed_decision 'MAYBE'" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_missing_case_description_aborts_invalid_argument(
        self, handler: ReliabilityGateHandler
    ) -> None:
        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="   ",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "case_description must not be empty" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_invalid_tenant_id_aborts_invalid_argument(
        self, handler: ReliabilityGateHandler
    ) -> None:
        req = reliability_pb2.GateRequest(
            tenant_id="not-a-uuid",
            agent_id=str(uuid4()),
            case_description="Valid text.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "Invalid tenant_id" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_invalid_agent_id_aborts_invalid_argument(
        self, handler: ReliabilityGateHandler
    ) -> None:
        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id="bad-agent-id",
            case_description="Valid text.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "Invalid agent_id" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_invalid_threshold_aborts_invalid_argument(
        self, handler: ReliabilityGateHandler
    ) -> None:
        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="Valid text.",
            proposed_decision="APPROVE",
            reliability_threshold=1.5,  # Exceeds 1.0
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "reliability_threshold must be within [0.0, 1.0]" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_custom_reliability_threshold_applied(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="APPROVE",
            raw_score=0.85,
            agreement=1.0,
            density_factor=0.9,
            coverage_factor=0.95,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Threshold met",
            reason_code="RELIABILITY_THRESHOLD_MET",
            threshold_used=0.80,
            retrieved_evidence=[],
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="High-value corporate dispute.",
            proposed_decision="APPROVE",
            reliability_threshold=0.80,
        )
        ctx = FakeServicerContext()

        await handler.Gate(req, ctx)  # type: ignore[arg-type]

        # Verify that custom_config was passed with min_reliability_threshold=0.80
        mock_use_case.execute.assert_awaited_once()
        call_kwargs = mock_use_case.execute.await_args.kwargs
        custom_cfg: ScoringConfig = call_kwargs["custom_config"]
        assert custom_cfg.min_reliability_threshold == 0.80

    @pytest.mark.asyncio
    async def test_application_exception_aborts_internal(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        mock_use_case.execute.side_effect = RuntimeError(
            "PostgreSQL connection refused"
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="Valid text.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INTERNAL
        # Safe error message without exposing connection strings or traces
        assert "An internal error occurred" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_domain_validation_exception_aborts_invalid_argument(
        self, handler: ReliabilityGateHandler, mock_use_case: AsyncMock
    ) -> None:
        mock_use_case.execute.side_effect = DomainValidationError(
            "Domain invariant violated"
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="Valid text.",
            proposed_decision="APPROVE",
        )
        ctx = FakeServicerContext()

        with pytest.raises(grpc.RpcError):
            await handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert ctx.aborted
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "Domain invariant violated" in (ctx.details or "")

    @pytest.mark.asyncio
    async def test_multi_domain_agent_decision_accepted_via_grpc(
        self, mock_use_case: AsyncMock
    ) -> None:
        """Handler configured with domain-specific decisions accepts custom actions."""
        custom_handler = ReliabilityGateHandler(
            evaluate_use_case=mock_use_case,
            config=ScoringConfig(allowed_decisions=("BILLING", "LEGAL", "KYC")),
        )
        mock_use_case.execute.return_value = ReliabilityEvaluationResult(
            decision=GateDecision.PASS,
            proposed_decision="LEGAL",
            raw_score=0.88,
            agreement=0.90,
            density_factor=0.85,
            coverage_factor=0.95,
            evidence_count=3,
            matching_evidence_count=3,
            conflicting_evidence_count=0,
            evaluation_reason="Precedent threshold met for legal department routing",
            reason_code=EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            threshold_used=0.65,
            retrieved_evidence=[
                ReferenceNeighbor("ref-1", "LEGAL", "synthetic", "Case 1", 0.95),
            ],
            metadata={},
        )

        req = reliability_pb2.GateRequest(
            tenant_id=str(uuid4()),
            agent_id=str(uuid4()),
            case_description="Customer onboarding legal documents.",
            proposed_decision="LEGAL",
            reliability_threshold=0.65,
        )
        ctx = FakeServicerContext()
        resp = await custom_handler.Gate(req, ctx)  # type: ignore[arg-type]

        assert not ctx.aborted
        assert resp.outcome == reliability_pb2.GATE_OUTCOME_PASS
        assert resp.reason_code == "RELIABILITY_THRESHOLD_MET"
