"""Unit and integration tests for Confidence Calibration and Evaluation (Week 4)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import StratifiedGroupKFold

from arda_reliability_service.application.use_cases.evaluate_reliability import (
    EvaluateReliabilityUseCase,
)
from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.calibration import (
    CalibratedConfidence,
    CalibrationUnavailableError,
    InvalidCalibrationInputError,
    IsotonicCalibrator,
    PlattCalibrator,
    UnavailableCalibrator,
)
from arda_reliability_service.domain.exceptions import DomainValidationError
from arda_reliability_service.domain.value_objects import (
    GateDecision,
    ReferenceNeighbor,
    ReliabilityEvaluationRequest,
    RetrievalResultSet,
    ScoringConfig,
)
from research.calibration.calibrator_fitting import CalibrationEvaluator
from research.calibration.metrics import (
    calculate_metrics,
    compute_expected_calibration_error,
)
from research.calibration.pipeline import (
    InstancePrediction,
    load_and_build_evaluation_instances,
)
from research.experiments.config import DATASET_PATH


class TestTargetGenerationAndDatasetSplitting:
    """Verifies symmetric target generation, group assignment, and leakage prevention."""

    def test_symmetric_target_generation(self) -> None:
        """Verify 100 seed examples yield 200 instances: 100 positive (correct=1) and 100 negative (0)."""
        raw_examples, instances = load_and_build_evaluation_instances(
            dataset_path=DATASET_PATH,
            template_key="ratio_verbalized_description",
        )
        assert len(raw_examples) == 100
        assert len(instances) == 200

        correct_count = sum(inst.is_correct for inst in instances)
        incorrect_count = sum(1 - inst.is_correct for inst in instances)

        assert correct_count == 100
        assert incorrect_count == 100

        # Each example must have exactly one correct and one incorrect proposal
        for ex in raw_examples:
            ex_id = ex["example_id"]
            ex_instances = [i for i in instances if i.example_id == ex_id]
            assert len(ex_instances) == 2
            proposals = {i.proposed_decision for i in ex_instances}
            assert proposals == {"APPROVE", "REJECT"}
            assert sum(i.is_correct for i in ex_instances) == 1

    def test_contrastive_pair_group_assignment(self) -> None:
        """Verify contrastive pairs share identical group_id and singletons have unique group_id."""
        _raw_examples, instances = load_and_build_evaluation_instances(
            dataset_path=DATASET_PATH
        )

        cp_instances = [i for i in instances if i.contrastive_pair_id is not None]
        assert len(cp_instances) == 40  # 10 pairs * 2 proposals = 40

        # Verify group_id matches contrastive_pair_id
        for inst in cp_instances:
            assert inst.group_id == inst.contrastive_pair_id

        # Verify singletons have group_id == example_id
        non_cp_instances = [i for i in instances if i.contrastive_pair_id is None]
        for inst in non_cp_instances:
            assert inst.group_id == inst.example_id

    def test_grouped_splitting_prevents_contrastive_leakage(self) -> None:
        """Verify StratifiedGroupKFold keeps both twins of every contrastive pair in the same fold."""
        _raw_examples, instances = load_and_build_evaluation_instances(
            dataset_path=DATASET_PATH
        )

        X = np.zeros(len(instances))
        y = np.array([i.is_correct for i in instances])
        groups = np.array([i.group_id for i in instances])

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in sgkf.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # Intersection between train and test groups must be strictly empty
            intersection = train_groups.intersection(test_groups)
            assert len(intersection) == 0, (
                f"Data leakage detected! Shared groups: {intersection}"
            )

    def test_evaluation_labels_never_enter_embedding_inputs(self) -> None:
        """Verify case descriptions to embed contain only factual narratives, not labels or targets."""
        _raw_examples, instances = load_and_build_evaluation_instances(
            dataset_path=DATASET_PATH
        )
        forbidden_tokens = ["APPROVE", "REJECT", "is_correct", "target="]
        for inst in instances:
            for token in forbidden_tokens:
                assert token not in inst.case_description, (
                    f"Label leakage detected in query text: '{token}' found in {inst.example_id}"
                )


class TestCalibrationMetrics:
    """Verifies metric calculations: Brier score, ECE, log loss, and gating safety metrics."""

    def test_ece_perfect_calibration(self) -> None:
        """ECE must be near 0 for a perfectly calibrated predictor."""
        # 100 samples with 0.8 conf and 80% accuracy, 100 samples with 0.2 conf and 20% accuracy
        probs = [0.8] * 100 + [0.2] * 100
        labels = [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80

        ece, mce, _bins = compute_expected_calibration_error(probs, labels, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-3)
        assert mce == pytest.approx(0.0, abs=1e-3)

    def test_ece_overconfident_predictor(self) -> None:
        """ECE must detect severe overconfidence."""
        # Predicts 1.0 confidence for coin flips (50% true accuracy)
        probs = [1.0] * 100
        labels = [1] * 50 + [0] * 50

        ece, mce, _bins = compute_expected_calibration_error(probs, labels, n_bins=10)
        assert ece == pytest.approx(0.5, abs=1e-2)
        assert mce == pytest.approx(0.5, abs=1e-2)

    def test_brier_and_log_loss_constant_baseline(self) -> None:
        """Constant 0.5 predictor on balanced 50/50 dataset has Brier=0.25 and log_loss=ln(2)=0.6931."""
        y_true = [1] * 50 + [0] * 50
        y_pred = [0.5] * 100

        metrics = calculate_metrics(y_true, y_pred)
        assert metrics.brier_score == pytest.approx(0.25, abs=1e-3)
        assert metrics.log_loss == pytest.approx(math.log(2.0), abs=1e-3)
        assert metrics.roc_auc is None or metrics.roc_auc == pytest.approx(
            0.5, abs=1e-2
        )

    def test_false_pass_and_false_escalation_calculations(self) -> None:
        """Verify False Pass Rate and False Escalation Rate are mathematically sound."""
        y_true = [1, 1, 0, 0]
        y_scores = [0.9, 0.3, 0.8, 0.2]
        gate_decisions = ["PASS", "ESCALATE", "PASS", "ESCALATE"]

        metrics = calculate_metrics(y_true, y_scores, gate_decisions=gate_decisions)

        # Case 0: y=1, PASS -> True Pass
        # Case 1: y=1, ESCALATE -> False Escalation (1 of 2 correct escalated -> FER = 0.50)
        # Case 2: y=0, PASS -> False Pass (1 of 2 incorrect passed -> FPR = 0.50)
        # Case 3: y=0, ESCALATE -> True Escalation
        assert metrics.false_pass_count == 1
        assert metrics.false_pass_rate_among_incorrect == 0.50
        assert metrics.false_escalation_count == 1
        assert metrics.false_escalation_rate_among_correct == 0.50
        assert metrics.coverage == 0.50  # 2 of 4 passed
        assert metrics.risk == 0.50  # 1 of 2 passes was incorrect


class TestCalibratorModels:
    """Verifies Platt and Isotonic calibration models, serialization, and error handling."""

    def test_calibrated_confidence_value_object(self) -> None:
        """Verify CalibratedConfidence invariants."""
        conf = CalibratedConfidence(
            raw_score=0.45,
            confidence=0.72,
            is_calibrated=True,
            calibrator_version="v1.0",
            calibrator_type="platt",
        )
        assert conf.raw_score == 0.45
        assert conf.confidence == 0.72
        assert conf.is_calibrated is True

        # Uncalibrated representation
        uncal = CalibratedConfidence(
            raw_score=0.45,
            confidence=None,
            is_calibrated=False,
        )
        assert uncal.confidence is None
        assert uncal.is_calibrated is False

        # Invariant violations
        with pytest.raises(DomainValidationError):
            CalibratedConfidence(raw_score=1.5, confidence=0.5, is_calibrated=True)

        with pytest.raises(DomainValidationError):
            CalibratedConfidence(raw_score=0.5, confidence=None, is_calibrated=True)

    def test_platt_calibrator_mathematics_and_bounds(self) -> None:
        """Verify Platt scaling sigmoid mapping and bounds."""
        calibrator = PlattCalibrator(a=2.0, b=-1.0, version="platt-test-1")
        # For s = 0.5: logit = 2.0*0.5 - 1.0 = 0.0 -> prob = 1 / (1 + exp(0)) = 0.5
        p = calibrator.calibrate(0.5)
        assert p == pytest.approx(0.5, abs=1e-3)

        # Output must be bounded in [0.0, 1.0]
        assert 0.0 <= calibrator.calibrate(0.0) <= 1.0
        assert 0.0 <= calibrator.calibrate(1.0) <= 1.0

        # Out of bounds input raises InvalidCalibrationInputError
        with pytest.raises(InvalidCalibrationInputError):
            calibrator.calibrate(-0.1)
        with pytest.raises(InvalidCalibrationInputError):
            calibrator.calibrate(1.1)
        with pytest.raises(InvalidCalibrationInputError):
            calibrator.calibrate(float("nan"))

    def test_platt_calibrator_serialization(self) -> None:
        """Verify PlattCalibrator JSON round-trip serialization."""
        calibrator = PlattCalibrator(
            a=1.5,
            b=-0.8,
            version="platt-v2.1",
            metrics={"brier": 0.22, "ece": 0.04},
        )
        json_str = calibrator.to_json()
        restored = PlattCalibrator.from_json(json_str)

        assert restored.a == calibrator.a
        assert restored.b == calibrator.b
        assert restored.version == calibrator.version
        assert restored.metrics == calibrator.metrics
        assert restored.calibrate(0.6) == calibrator.calibrate(0.6)

    def test_isotonic_calibrator_interpolation(self) -> None:
        """Verify IsotonicCalibrator monotonic piecewise linear interpolation."""
        x_knots = [0.0, 0.4, 0.8, 1.0]
        y_knots = [0.1, 0.3, 0.7, 0.9]
        calibrator = IsotonicCalibrator(
            x_knots=x_knots, y_knots=y_knots, version="iso-test"
        )

        # Exact knots
        assert calibrator.calibrate(0.0) == 0.1
        assert calibrator.calibrate(0.4) == 0.3
        assert calibrator.calibrate(1.0) == 0.9

        # Midpoint interpolation between 0.4 and 0.8 (s = 0.6 -> y = 0.5)
        assert calibrator.calibrate(0.6) == pytest.approx(0.5, abs=1e-3)

        # Invalid non-monotonic knots raise DomainValidationError
        with pytest.raises(DomainValidationError):
            IsotonicCalibrator(x_knots=[0.5, 0.2], y_knots=[0.1, 0.2])

    def test_isotonic_calibrator_serialization(self) -> None:
        """Verify IsotonicCalibrator JSON round-trip serialization."""
        calibrator = IsotonicCalibrator(
            x_knots=[0.0, 0.5, 1.0],
            y_knots=[0.2, 0.5, 0.8],
            version="iso-v1.0",
        )
        json_str = calibrator.to_json()
        restored = IsotonicCalibrator.from_json(json_str)

        assert restored.x_knots == calibrator.x_knots
        assert restored.y_knots == calibrator.y_knots
        assert restored.version == calibrator.version
        assert restored.calibrate(0.25) == calibrator.calibrate(0.25)

    def test_unavailable_calibrator_fails_explicitly(self) -> None:
        """Verify UnavailableCalibrator raises CalibrationUnavailableError and never silently succeeds."""
        calibrator = UnavailableCalibrator(reason="Model training deferred")
        assert calibrator.is_fitted is False
        assert calibrator.calibrator_type == "unavailable"

        with pytest.raises(CalibrationUnavailableError) as exc_info:
            calibrator.calibrate(0.5)
        assert "Confidence calibration is unavailable" in str(exc_info.value)


class TestApplicationIntegrationWithCalibrator:
    """Verifies EvaluateReliabilityUseCase integration with calibrator models."""

    @pytest.fixture
    def mock_retrieval(self) -> AsyncMock:
        mock = AsyncMock(spec=RetrieveReferenceExamplesUseCase)
        mock.execute.return_value = RetrievalResultSet(
            neighbors=[
                ReferenceNeighbor(
                    example_id="ref-1",
                    label="APPROVE",
                    source="synthetic",
                    case_description="dispute text",
                    similarity_score=0.88,
                ),
                ReferenceNeighbor(
                    example_id="ref-2",
                    label="APPROVE",
                    source="synthetic",
                    case_description="dispute text",
                    similarity_score=0.85,
                ),
                ReferenceNeighbor(
                    example_id="ref-3",
                    label="APPROVE",
                    source="synthetic",
                    case_description="dispute text",
                    similarity_score=0.82,
                ),
            ],
            query_description="dispute text",
            top_k=3,
            retrieval_mode="approximate",
            execution_latency_ms=1.5,
        )
        return mock

    @pytest.mark.asyncio
    async def test_use_case_with_fitted_calibrator(
        self, mock_retrieval: AsyncMock
    ) -> None:
        """Verify EvaluateReliabilityUseCase populates calibrated_confidence without altering raw_score."""
        platt = PlattCalibrator(a=1.0, b=0.0, version="platt-unit-test")
        use_case = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            config=ScoringConfig(min_similarity_threshold=0.70),
            calibrator=platt,
        )

        request = ReliabilityEvaluationRequest(
            tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
            agent_id=UUID("00000000-0000-0000-0000-000000000002"),
            case_description="Valid dispute case description",
            proposed_decision="APPROVE",
        )

        result = await use_case.execute(request)

        assert result.raw_score > 0.0
        assert result.is_calibrated is True
        assert result.calibrated_confidence is not None
        assert result.calibrated_confidence == platt.calibrate(result.raw_score)
        assert result.calibration_version == "platt-unit-test"

    @pytest.mark.asyncio
    async def test_use_case_without_calibrator(self, mock_retrieval: AsyncMock) -> None:
        """Verify uncalibrated use case leaves calibrated_confidence as None."""
        use_case = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            config=ScoringConfig(min_similarity_threshold=0.70),
            calibrator=None,
        )

        request = ReliabilityEvaluationRequest(
            tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
            agent_id=UUID("00000000-0000-0000-0000-000000000002"),
            case_description="Valid dispute case description",
            proposed_decision="APPROVE",
        )

        result = await use_case.execute(request)

        assert result.raw_score > 0.0
        assert result.is_calibrated is False
        assert result.calibrated_confidence is None
        assert result.calibration_version is None

    @pytest.mark.asyncio
    async def test_use_case_with_unavailable_calibrator_falls_back_gracefully(
        self, mock_retrieval: AsyncMock
    ) -> None:
        """Verify UnavailableCalibrator does not crash evaluation; keeps raw score intact."""
        unavail = UnavailableCalibrator(reason="Model calibration deferred")
        use_case = EvaluateReliabilityUseCase(
            retrieval_use_case=mock_retrieval,
            config=ScoringConfig(min_similarity_threshold=0.70),
            calibrator=unavail,
        )

        request = ReliabilityEvaluationRequest(
            tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
            agent_id=UUID("00000000-0000-0000-0000-000000000002"),
            case_description="Valid dispute case description",
            proposed_decision="APPROVE",
        )

        result = await use_case.execute(request)

        assert result.raw_score > 0.0
        assert result.is_calibrated is False
        assert result.calibrated_confidence is None
        assert result.decision in (GateDecision.PASS, GateDecision.ESCALATE)


class TestCalibrationReproducibility:
    """Verifies deterministic seeds and evaluation consistency."""

    def test_deterministic_evaluation_metrics(self) -> None:
        """Verify running the evaluation pipeline twice with seed 42 produces identical results."""
        dummy_predictions = [
            InstancePrediction(
                example_id=f"ex-{i}",
                group_id=f"grp-{i // 2}",
                case_type="straightforward",
                category="goods_not_received",
                true_label="APPROVE" if i % 2 == 0 else "REJECT",
                proposed_decision="APPROVE",
                is_correct=1 if i % 2 == 0 else 0,
                raw_score=0.1 + (i % 5) * 0.08,
                agreement=0.5,
                density_factor=0.4,
                coverage_factor=0.6,
                gate_decision="ESCALATE",
                reason_code="SCORE_BELOW_THRESHOLD",
            )
            for i in range(20)
        ]

        evaluator1 = CalibrationEvaluator(n_splits=3, random_state=42)
        res1 = evaluator1.evaluate_all(dummy_predictions)

        evaluator2 = CalibrationEvaluator(n_splits=3, random_state=42)
        res2 = evaluator2.evaluate_all(dummy_predictions)

        assert (
            res1["platt_oof_metrics"]["brier_score"]
            == res2["platt_oof_metrics"]["brier_score"]
        )
        assert (
            res1["platt_oof_metrics"]["expected_calibration_error"]
            == res2["platt_oof_metrics"]["expected_calibration_error"]
        )
        assert (
            res1["isotonic_oof_metrics"]["brier_score"]
            == res2["isotonic_oof_metrics"]["brier_score"]
        )
