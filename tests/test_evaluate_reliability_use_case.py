"""Unit and integration tests for EvaluateReliabilityUseCase."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from arda_reliability_service.application.use_cases.evaluate_reliability import (
    EvaluateReliabilityUseCase,
)
from arda_reliability_service.application.use_cases.ingest_reference_examples import (
    IngestItemDTO,
    IngestReferenceExamplesUseCase,
)
from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.calibration import (
    PlattCalibrator,
    UnavailableCalibrator,
)
from arda_reliability_service.domain.exceptions import (
    InvalidDescriptionError,
    InvalidProposedDecisionError,
    ReliabilityEvaluationError,
)
from arda_reliability_service.domain.scoring_engine import (
    ReliabilityDecisionEngine,
    ReliabilityScoringEngine,
)
from arda_reliability_service.domain.value_objects import (
    EvaluationReasonCode,
    GateDecision,
    ReferenceNeighbor,
    ReliabilityEvaluationRequest,
    ReliabilityEvaluationResult,
    RetrievalQuery,
    RetrievalResultSet,
    ScoringConfig,
)
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)


class TestEvaluateReliabilityUseCaseUnit:
    """Unit tests for EvaluateReliabilityUseCase using mocked dependencies."""

    @pytest.fixture
    def mock_retrieval(self) -> AsyncMock:
        return AsyncMock(spec=RetrieveReferenceExamplesUseCase)

    @pytest.fixture
    def use_case(self, mock_retrieval: AsyncMock) -> EvaluateReliabilityUseCase:
        return EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            scoring_engine=ReliabilityScoringEngine(),
            decision_engine=ReliabilityDecisionEngine(),
            config=ScoringConfig(
                min_neighbor_count=3,
                min_similarity_threshold=0.70,
                min_coverage_threshold=0.50,
                min_density_threshold=0.40,
                min_agreement_threshold=0.60,
                min_reliability_threshold=0.65,
            ),
        )

    @pytest.mark.asyncio
    async def test_critical_rule_case_description_only_embedded(
        self, use_case: EvaluateReliabilityUseCase, mock_retrieval: AsyncMock
    ) -> None:
        """CRITICAL: Only case_description may be passed to vector retrieval.

        Proposed decision, reference label, rationale, or evaluation results must NEVER be embedded.
        """
        tenant_id = uuid4()
        agent_id = uuid4()
        case_text = "Cardholder reports unauthorized charge of $150 at gas station."
        proposed = "APPROVE"

        # Mock retrieval response
        mock_neighbors = [
            ReferenceNeighbor(
                example_id="ref-1",
                label="APPROVE",
                source="synthetic",
                case_description="Precedent description",
                similarity_score=0.92,
            ),
            ReferenceNeighbor(
                example_id="ref-2",
                label="APPROVE",
                source="synthetic",
                case_description="Precedent description 2",
                similarity_score=0.90,
            ),
            ReferenceNeighbor(
                example_id="ref-3",
                label="APPROVE",
                source="synthetic",
                case_description="Precedent description 3",
                similarity_score=0.88,
            ),
        ]
        mock_retrieval.execute.return_value = RetrievalResultSet(
            neighbors=mock_neighbors,
            query_description=case_text,
            top_k=5,
            retrieval_mode="approximate_hnsw",
            execution_latency_ms=1.5,
        )

        request = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description=case_text,
            proposed_decision=proposed,
        )

        result = await use_case.execute(request)

        # Verify that mock_retrieval was called with a RetrievalQuery containing ONLY case_description
        mock_retrieval.execute.assert_awaited_once()
        call_args = mock_retrieval.execute.await_args
        query_arg: RetrievalQuery = call_args.kwargs.get("query") or call_args.args[0]

        assert query_arg.case_description == case_text
        assert query_arg.tenant_id == tenant_id
        assert query_arg.agent_id == agent_id
        # Ensure proposed_decision is not part of RetrievalQuery
        assert not hasattr(query_arg, "proposed_decision")

        # Verify structured result
        assert isinstance(result, ReliabilityEvaluationResult)
        assert result.decision == GateDecision.PASS
        assert result.proposed_decision == "APPROVE"
        assert (
            result.reason_code == EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value
        )
        assert result.evidence_count == 3
        assert result.matching_evidence_count == 3
        assert result.conflicting_evidence_count == 0
        assert result.raw_score >= 0.65
        assert len(result.retrieved_evidence) == 3

    @pytest.mark.asyncio
    async def test_calibration_independence_from_gate_decision(
        self, mock_retrieval: AsyncMock
    ) -> None:
        """Verify changing, failing, or disabling calibration never changes the gate outcome or reason code."""
        tenant_id = uuid4()
        agent_id = uuid4()
        case_text = "Standard clear dispute precedent query"
        mock_neighbors = [
            ReferenceNeighbor(
                example_id="ref-1",
                label="APPROVE",
                source="synthetic",
                case_description="Precedent 1",
                similarity_score=0.92,
            ),
            ReferenceNeighbor(
                example_id="ref-2",
                label="APPROVE",
                source="synthetic",
                case_description="Precedent 2",
                similarity_score=0.90,
            ),
            ReferenceNeighbor(
                example_id="ref-3",
                label="APPROVE",
                source="synthetic",
                case_description="Precedent 3",
                similarity_score=0.88,
            ),
        ]
        mock_retrieval.execute.return_value = RetrievalResultSet(
            neighbors=mock_neighbors,
            query_description=case_text,
            top_k=5,
            retrieval_mode="approximate_hnsw",
            execution_latency_ms=1.5,
        )
        request = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description=case_text,
            proposed_decision="APPROVE",
        )
        cfg = ScoringConfig(min_reliability_threshold=0.65)

        # 1. Uncalibrated
        uc_none = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval, config=cfg, calibrator=None
        )
        res_none = await uc_none.execute(request)

        # 2. Extreme low calibrator (maps to 0.001)
        uc_low = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            config=cfg,
            calibrator=PlattCalibrator(a=-20.0, b=-10.0, version="low"),
        )
        res_low = await uc_low.execute(request)

        # 3. Extreme high calibrator (maps to 0.999)
        uc_high = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            config=cfg,
            calibrator=PlattCalibrator(a=20.0, b=10.0, version="high"),
        )
        res_high = await uc_high.execute(request)

        # 4. Unavailable calibrator
        uc_unavail = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            config=cfg,
            calibrator=UnavailableCalibrator(),
        )
        res_unavail = await uc_unavail.execute(request)

        # Decision, reason code, and raw score must be 100% IDENTICAL across all four
        for res in [res_low, res_high, res_unavail]:
            assert res.decision == res_none.decision
            assert res.reason_code == res_none.reason_code
            assert res.raw_score == res_none.raw_score

        # Calibration fields correctly reflect calibrator state
        assert res_none.is_calibrated is False
        assert res_none.calibrated_confidence is None
        assert res_low.is_calibrated is True
        assert (
            res_low.calibrated_confidence is not None
            and res_low.calibrated_confidence < 0.05
        )
        assert res_high.is_calibrated is True
        assert (
            res_high.calibrated_confidence is not None
            and res_high.calibrated_confidence > 0.95
        )
        assert res_unavail.is_calibrated is False
        assert res_unavail.calibrated_confidence is None

    @pytest.mark.asyncio
    async def test_invalid_proposed_decision_raises(
        self, use_case: EvaluateReliabilityUseCase
    ) -> None:
        """Incoming proposed decision other than APPROVE or REJECT raises InvalidProposedDecisionError."""
        with pytest.raises(InvalidProposedDecisionError):
            ReliabilityEvaluationRequest(
                tenant_id=uuid4(),
                agent_id=uuid4(),
                case_description="Valid dispute case description.",
                proposed_decision="HOLD",
            )

    @pytest.mark.asyncio
    async def test_empty_case_description_raises(
        self, use_case: EvaluateReliabilityUseCase
    ) -> None:
        """Empty case description raises InvalidDescriptionError."""
        with pytest.raises(InvalidDescriptionError):
            ReliabilityEvaluationRequest(
                tenant_id=uuid4(),
                agent_id=uuid4(),
                case_description="   ",
                proposed_decision="APPROVE",
            )

    @pytest.mark.asyncio
    async def test_retrieval_failure_wrapped_in_reliability_evaluation_error(
        self, use_case: EvaluateReliabilityUseCase, mock_retrieval: AsyncMock
    ) -> None:
        """Unexpected infrastructure failure during retrieval is wrapped in ReliabilityEvaluationError."""
        mock_retrieval.execute.side_effect = RuntimeError(
            "Database connection pool exhausted"
        )

        request = ReliabilityEvaluationRequest(
            tenant_id=uuid4(),
            agent_id=uuid4(),
            case_description="Valid dispute case description.",
            proposed_decision="APPROVE",
        )

        with pytest.raises(ReliabilityEvaluationError) as exc_info:
            await use_case.execute(request)
        assert "Database connection pool exhausted" in str(exc_info.value)


@pytest_asyncio.fixture
async def db_manager() -> AsyncGenerator[DatabaseManager]:
    """Yields a database manager and disposes connections on teardown."""
    manager = DatabaseManager()
    yield manager
    await manager.dispose()


@pytest_asyncio.fixture
async def live_eval_pipeline(
    db_manager: DatabaseManager,
) -> AsyncGenerator[
    tuple[
        EvaluateReliabilityUseCase,
        IngestReferenceExamplesUseCase,
        PgVectorReferenceExampleRepository,
        UUID,
        UUID,
    ]
]:
    """Constructs a fully wired integration pipeline with live database and embedding adapter."""
    async with db_manager.session() as session:
        repo = PgVectorReferenceExampleRepository(session)
        adapter = SentenceTransformerEmbeddingAdapter()
        retrieval_uc = RetrieveReferenceExamplesUseCase(
            repository=repo, embedding_provider=adapter
        )
        ingest_uc = IngestReferenceExamplesUseCase(
            repository=repo, embedding_provider=adapter
        )

        eval_uc = EvaluateReliabilityUseCase(
            retrieval_use_case=retrieval_uc,
            scoring_engine=ReliabilityScoringEngine(),
            decision_engine=ReliabilityDecisionEngine(),
            config=ScoringConfig(
                min_neighbor_count=3,
                min_similarity_threshold=0.60,
                min_coverage_threshold=0.45,
                min_density_threshold=0.35,
                min_agreement_threshold=0.60,
                min_reliability_threshold=0.50,
                top_k=5,
            ),
        )

        tenant_id = uuid4()
        agent_id = uuid4()

        try:
            yield eval_uc, ingest_uc, repo, tenant_id, agent_id
        finally:
            await repo.delete_all_for_agent(tenant_id, agent_id)


class TestEvaluateReliabilityUseCaseIntegration:
    """Integration tests with live PostgreSQL pgvector and SentenceTransformers adapter."""

    @pytest.mark.asyncio
    async def test_end_to_end_pass_on_matching_precedents(
        self,
        live_eval_pipeline: tuple[
            EvaluateReliabilityUseCase,
            IngestReferenceExamplesUseCase,
            PgVectorReferenceExampleRepository,
            UUID,
            UUID,
        ],
    ) -> None:
        """Ingests 4 clear precedent cases, then verifies PASS when agent proposes APPROVE."""
        eval_uc, ingest_uc, _, tenant_id, agent_id = live_eval_pipeline

        # Ingest 4 unanimous APPROVE precedents
        items = [
            IngestItemDTO(
                example_id=f"integ-pass-{i}",
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description=(
                    f"Cardholder reports fraudulent magnetic stripe withdrawal of ${100 + i * 20} "
                    "in foreign country while physically possessing genuine card. "
                    "Filed within 15 days of statement posting date."
                ),
                label="APPROVE",
            )
            for i in range(4)
        ]
        await ingest_uc.execute(items)

        # Incoming request on identical scenario proposing APPROVE
        req = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description=(
                "Cardholder reports fraudulent magnetic stripe debit of $140 while traveling abroad. "
                "Physical card remained in possession; claim filed 12 days post-posting."
            ),
            proposed_decision="APPROVE",
        )

        result = await eval_uc.execute(req, retrieval_mode="exact")

        assert result.decision == GateDecision.PASS
        assert result.proposed_decision == "APPROVE"
        assert (
            result.reason_code == EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value
        )
        assert result.evidence_count >= 3
        assert result.matching_evidence_count >= 3
        assert result.conflicting_evidence_count == 0
        assert result.agreement == 1.0
        assert result.raw_score >= 0.50

    @pytest.mark.asyncio
    async def test_end_to_end_escalate_on_conflicting_precedents(
        self,
        live_eval_pipeline: tuple[
            EvaluateReliabilityUseCase,
            IngestReferenceExamplesUseCase,
            PgVectorReferenceExampleRepository,
            UUID,
            UUID,
        ],
    ) -> None:
        """When precedents support APPROVE but agent proposes REJECT, gate must ESCALATE."""
        eval_uc, ingest_uc, _, tenant_id, agent_id = live_eval_pipeline

        # Ingest precedents that approved the dispute
        items = [
            IngestItemDTO(
                example_id=f"integ-conflict-{i}",
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description=(
                    f"Cardholder duplicate billing of ${50 + i * 10} at merchant terminal within 2 minutes. "
                    "Merchant failed to respond to arbitration notice within 14 days."
                ),
                label="APPROVE",
            )
            for i in range(3)
        ]
        await ingest_uc.execute(items)

        # Incoming request proposes REJECT contrary to precedent
        req = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Duplicate charge at merchant terminal; merchant response window elapsed.",
            proposed_decision="REJECT",
        )

        result = await eval_uc.execute(req, retrieval_mode="exact")

        assert result.decision == GateDecision.ESCALATE
        assert result.proposed_decision == "REJECT"
        assert result.matching_evidence_count == 0
        assert result.conflicting_evidence_count >= 3
        assert result.agreement == 0.0
        assert result.raw_score == 0.0

    @pytest.mark.asyncio
    async def test_strict_tenant_isolation_causes_cold_start_escalation(
        self,
        live_eval_pipeline: tuple[
            EvaluateReliabilityUseCase,
            IngestReferenceExamplesUseCase,
            PgVectorReferenceExampleRepository,
            UUID,
            UUID,
        ],
    ) -> None:
        """Precedents stored under tenant A are invisible to tenant B, causing COLD_START ESCALATE."""
        eval_uc, ingest_uc, _, tenant_id_a, agent_id = live_eval_pipeline
        tenant_id_b = uuid4()  # Empty tenant

        # Ingest for tenant A
        items = [
            IngestItemDTO(
                example_id=f"tenant-a-{i}",
                tenant_id=tenant_id_a,
                agent_id=agent_id,
                case_description=f"Standard clear dispute precedent #{i}",
                label="APPROVE",
            )
            for i in range(5)
        ]
        await ingest_uc.execute(items)

        # Evaluate under tenant B
        req_b = ReliabilityEvaluationRequest(
            tenant_id=tenant_id_b,
            agent_id=agent_id,
            case_description="Standard clear dispute precedent query",
            proposed_decision="APPROVE",
        )

        result = await eval_uc.execute(req_b, retrieval_mode="exact")

        assert result.decision == GateDecision.ESCALATE
        assert (
            result.reason_code
            == EvaluationReasonCode.COLD_START_INSUFFICIENT_EVIDENCE.value
        )
        assert result.evidence_count == 0
        assert result.raw_score == 0.0

    @pytest.mark.asyncio
    async def test_strict_agent_isolation_causes_cold_start_escalation(
        self,
        live_eval_pipeline: tuple[
            EvaluateReliabilityUseCase,
            IngestReferenceExamplesUseCase,
            PgVectorReferenceExampleRepository,
            UUID,
            UUID,
        ],
    ) -> None:
        """Precedents stored under agent A are invisible to agent B within the same tenant."""
        eval_uc, ingest_uc, _, tenant_id, agent_id_a = live_eval_pipeline
        agent_id_b = uuid4()  # New agent with 0 records

        # Ingest for agent A
        items = [
            IngestItemDTO(
                example_id=f"agent-a-{i}",
                tenant_id=tenant_id,
                agent_id=agent_id_a,
                case_description=f"Transaction defect claim precedent #{i}",
                label="APPROVE",
            )
            for i in range(5)
        ]
        await ingest_uc.execute(items)

        # Evaluate under agent B
        req_b = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id_b,
            case_description="Transaction defect claim query",
            proposed_decision="APPROVE",
        )

        result = await eval_uc.execute(req_b, retrieval_mode="exact")

        assert result.decision == GateDecision.ESCALATE
        assert (
            result.reason_code
            == EvaluationReasonCode.COLD_START_INSUFFICIENT_EVIDENCE.value
        )
        assert result.evidence_count == 0
        assert result.raw_score == 0.0
