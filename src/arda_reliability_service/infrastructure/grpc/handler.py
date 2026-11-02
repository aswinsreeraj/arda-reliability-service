"""gRPC handler implementing ReliabilityService.Gate RPC."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from uuid import UUID

import grpc
from arda_reliability_service.application.use_cases.evaluate_reliability import (
    EvaluateReliabilityUseCase,
)
from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.calibration import CalibratorPort
from arda_reliability_service.domain.exceptions import DomainValidationError
from arda_reliability_service.domain.scoring_engine import (
    ReliabilityDecisionEngine,
    ReliabilityScoringEngine,
)
from arda_reliability_service.domain.value_objects import (
    GateDecision,
    ProposedDecision,
    ReliabilityEvaluationRequest,
    ScoringConfig,
)
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from arda_reliability_service.infrastructure.grpc.generated import (
    reliability_pb2,
    reliability_pb2_grpc,
)

logger = logging.getLogger(__name__)


class ReliabilityGateHandler(reliability_pb2_grpc.ReliabilityServiceServicer):
    """gRPC service implementation of arda.v1.ReliabilityService.

    Delegates decision evaluation directly to EvaluateReliabilityUseCase,
    enforcing protocol validation and error mapping.
    """

    def __init__(
        self,
        evaluate_use_case: EvaluateReliabilityUseCase | None = None,
        db_manager: DatabaseManager | None = None,
        embedding_adapter: SentenceTransformerEmbeddingAdapter | None = None,
        config: ScoringConfig | None = None,
        calibrator: CalibratorPort | None = None,
    ) -> None:
        self.evaluate_use_case = evaluate_use_case
        self.db_manager = db_manager
        self.embedding_adapter = embedding_adapter
        self.config = config or ScoringConfig()
        self.calibrator = calibrator
        self.scoring_engine = ReliabilityScoringEngine()
        self.decision_engine = ReliabilityDecisionEngine()

    async def Gate(
        self,
        request: reliability_pb2.GateRequest,
        context: grpc.aio.ServicerContext,
    ) -> reliability_pb2.GateResponse:
        """Handle incoming Gate evaluation request from Go Agent Runtime or clients.

        Flow:
        1. Validate request fields and UUID formats.
        2. Formulate domain ReliabilityEvaluationRequest.
        3. Delegate to EvaluateReliabilityUseCase.
        4. Map domain result into protobuf GateResponse.
        """
        # 1. Validate tenant_id
        if not request.tenant_id or not request.tenant_id.strip():
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "tenant_id must not be empty or whitespace.",
            )
        try:
            tenant_uuid = UUID(request.tenant_id.strip())
        except ValueError:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid tenant_id '{request.tenant_id}': must be a valid UUID string.",
            )

        # 2. Validate agent_id
        if not request.agent_id or not request.agent_id.strip():
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "agent_id must not be empty or whitespace.",
            )
        try:
            agent_uuid = UUID(request.agent_id.strip())
        except ValueError:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid agent_id '{request.agent_id}': must be a valid UUID string.",
            )

        # 3. Validate case_description
        case_description = (
            request.case_description.strip() if request.case_description else ""
        )
        if not case_description:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "case_description must not be empty or whitespace.",
            )

        # 4. Validate proposed_decision
        valid_decisions = {
            ProposedDecision.APPROVE.value,
            ProposedDecision.REJECT.value,
        }
        if request.proposed_decision not in valid_decisions:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid proposed_decision '{request.proposed_decision}'. Must be one of {sorted(valid_decisions)}.",
            )

        # 5. Validate reliability_threshold
        if request.reliability_threshold < 0.0 or request.reliability_threshold > 1.0:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"reliability_threshold must be within [0.0, 1.0], got {request.reliability_threshold}",
            )

        # Prepare domain request and optional custom config
        eval_req = ReliabilityEvaluationRequest(
            tenant_id=tenant_uuid,
            agent_id=agent_uuid,
            case_description=case_description,
            proposed_decision=request.proposed_decision,
        )

        custom_cfg: ScoringConfig | None = None
        if request.reliability_threshold > 0.0:
            custom_cfg = replace(
                self.config, min_reliability_threshold=request.reliability_threshold
            )

        t0 = time.perf_counter()
        try:
            if self.evaluate_use_case is not None:
                result = await self.evaluate_use_case.execute(
                    request=eval_req,
                    custom_config=custom_cfg,
                )
            elif self.db_manager is not None and self.embedding_adapter is not None:
                async with self.db_manager.session() as session:
                    repo = PgVectorReferenceExampleRepository(session)
                    retrieval_uc = RetrieveReferenceExamplesUseCase(
                        repository=repo,
                        embedding_provider=self.embedding_adapter,
                    )
                    use_case = EvaluateReliabilityUseCase(
                        retrieval_use_case=retrieval_uc,
                        scoring_engine=self.scoring_engine,
                        decision_engine=self.decision_engine,
                        config=self.config,
                        calibrator=self.calibrator,
                    )
                    result = await use_case.execute(
                        request=eval_req,
                        custom_config=custom_cfg,
                    )
            else:
                raise RuntimeError(
                    "ReliabilityGateHandler has neither evaluate_use_case nor db_manager configured."
                )

            duration_ms = (time.perf_counter() - t0) * 1000.0

            # Structured audit logging (avoid logging complete case text or embeddings)
            logger.info(
                "Gate evaluation completed",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "agent_id": str(agent_uuid),
                    "proposed_decision": eval_req.proposed_decision,
                    "gate_decision": result.decision.value,
                    "reason_code": result.reason_code,
                    "confidence": round(result.raw_score, 4),
                    "evidence_count": result.evidence_count,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            # Map domain decision to protobuf GateOutcome
            proto_outcome = (
                reliability_pb2.GateOutcome.GATE_OUTCOME_PASS
                if result.decision == GateDecision.PASS
                else reliability_pb2.GateOutcome.GATE_OUTCOME_REVIEW_REQUIRED
            )

            # Map retrieved evidence neighbors
            proto_neighbors = [
                reliability_pb2.ReferenceNeighbor(
                    reference_example_id=n.example_id,
                    label=n.label,
                    similarity=n.similarity_score,
                )
                for n in result.retrieved_evidence
            ]

            effective_confidence = (
                result.calibrated_confidence
                if result.is_calibrated and result.calibrated_confidence is not None
                else result.raw_score
            )

            return reliability_pb2.GateResponse(
                confidence=effective_confidence,
                outcome=proto_outcome,
                reason_code=result.reason_code,
                neighbors=proto_neighbors,
            )

        except DomainValidationError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            logger.exception(
                "Internal error during Gate evaluation for agent %s",
                agent_uuid,
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                "An internal error occurred during reliability evaluation.",
            )
