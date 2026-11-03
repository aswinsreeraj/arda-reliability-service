"""Application use case for reliability evaluation and gate decisioning."""

from __future__ import annotations

import logging
from typing import Any

from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.calibration import CalibratorPort
from arda_reliability_service.domain.exceptions import ReliabilityEvaluationError
from arda_reliability_service.domain.scoring_engine import (
    ReliabilityDecisionEngine,
    ReliabilityScoringEngine,
)
from arda_reliability_service.domain.value_objects import (
    ReliabilityEvaluationRequest,
    ReliabilityEvaluationResult,
    RetrievalQuery,
    ScoringConfig,
)

logger = logging.getLogger(__name__)


class EvaluateReliabilityUseCase:
    """Orchestrates reference retrieval, decomposed reliability scoring, and gate decisioning.

    Flow:
    1. Receive validated ReliabilityEvaluationRequest.
    2. Formulate RetrievalQuery with case_description ONLY (proposed_decision is NEVER embedded).
    3. Retrieve historical reference evidence via RetrieveReferenceExamplesUseCase.
    4. Compute agreement, density, coverage, and raw reliability score via ReliabilityScoringEngine.
    5. Evaluate score and evidence against operational gate policy via ReliabilityDecisionEngine.
    6. Return structured, auditable ReliabilityEvaluationResult.
    """

    def __init__(
        self,
        retrieval_use_case: RetrieveReferenceExamplesUseCase,
        scoring_engine: ReliabilityScoringEngine | None = None,
        decision_engine: ReliabilityDecisionEngine | None = None,
        config: ScoringConfig | None = None,
        calibrator: CalibratorPort | None = None,
    ) -> None:
        self.retrieval_use_case = retrieval_use_case
        self.scoring_engine = scoring_engine or ReliabilityScoringEngine()
        self.decision_engine = decision_engine or ReliabilityDecisionEngine()
        self.config = config or ScoringConfig()
        self.calibrator = calibrator

    async def execute(
        self,
        request: ReliabilityEvaluationRequest,
        retrieval_mode: str = "approximate",
        custom_config: ScoringConfig | None = None,
    ) -> ReliabilityEvaluationResult:
        """Execute reliability evaluation for the given incoming request.

        Args:
            request: Domain evaluation request containing tenant_id, agent_id, case_description,
                     and proposed_decision ('APPROVE' or 'REJECT').
            retrieval_mode: 'exact' (flat cosine scan) or 'approximate' (HNSW index scan).
            custom_config: Optional per-evaluation scoring configuration override.

        Returns:
            Structured ReliabilityEvaluationResult containing gate decision, raw score,
            decomposed factors, and evidence summary.
        """
        cfg = custom_config or self.config

        try:
            # Note: only the case description is embedded; proposed decisions and labels are excluded
            # to prevent target leakage.
            query = RetrievalQuery(
                tenant_id=request.tenant_id,
                agent_id=request.agent_id,
                case_description=request.case_description,
                top_k=cfg.top_k,
                ef_search=cfg.ef_search,
            )

            # Retrieve nearest reference neighbors
            result_set = await self.retrieval_use_case.execute(
                query=query, mode=retrieval_mode
            )
            neighbors = result_set.neighbors

            # Calculate decomposed reliability scores
            score = self.scoring_engine.calculate_score(
                neighbors=neighbors,
                proposed_decision=request.proposed_decision,
                config=cfg,
            )

            # Apply operational decision policy
            decision, reason_code, reason_text = self.decision_engine.evaluate(
                score=score,
                neighbors=neighbors,
                proposed_decision=request.proposed_decision,
                config=cfg,
            )

            # Evidence tallying
            matching_count = sum(
                1 for n in neighbors if n.label == request.proposed_decision
            )
            conflicting_count = len(neighbors) - matching_count

            # Optional confidence calibration
            calibrated_conf: float | None = None
            is_calibrated = False
            calib_version: str | None = None

            if self.calibrator is not None and self.calibrator.is_fitted:
                try:
                    calibrated_conf = self.calibrator.calibrate(score.raw_score)
                    is_calibrated = True
                    calib_version = self.calibrator.version
                except Exception as cal_err:  # noqa: BLE001
                    logger.warning(
                        "Confidence calibration failed for raw score %.4f: %s",
                        score.raw_score,
                        cal_err,
                    )
                    calibrated_conf = None
                    is_calibrated = False

            metadata: dict[str, Any] = {
                "retrieval_mode": result_set.retrieval_mode,
                "retrieval_latency_ms": result_set.execution_latency_ms,
                "top_k": cfg.top_k,
                "weighting_strategy": cfg.weighting_strategy,
                "min_similarity_threshold": cfg.min_similarity_threshold,
                "min_coverage_threshold": cfg.min_coverage_threshold,
                "min_density_threshold": cfg.min_density_threshold,
                "min_agreement_threshold": cfg.min_agreement_threshold,
                "min_reliability_threshold": cfg.min_reliability_threshold,
                "is_calibrated": is_calibrated,
                "calibration_version": calib_version,
            }

            return ReliabilityEvaluationResult(
                decision=decision,
                proposed_decision=request.proposed_decision,
                raw_score=score.raw_score,
                agreement=score.agreement,
                density_factor=score.density_factor,
                coverage_factor=score.coverage_factor,
                evidence_count=len(neighbors),
                matching_evidence_count=matching_count,
                conflicting_evidence_count=conflicting_count,
                evaluation_reason=reason_text,
                reason_code=reason_code,
                threshold_used=cfg.min_reliability_threshold,
                retrieved_evidence=neighbors,
                metadata=metadata,
                calibrated_confidence=calibrated_conf,
                is_calibrated=is_calibrated,
                calibration_version=calib_version,
            )

        except Exception as exc:
            # Re-raise domain validation errors directly without wrapping
            if "DomainValidationError" in type(exc).__name__:
                raise
            logger.exception(
                "Reliability evaluation failed for agent %s", request.agent_id
            )
            raise ReliabilityEvaluationError(
                f"Evaluation failed for agent {request.agent_id}: {exc}"
            ) from exc
