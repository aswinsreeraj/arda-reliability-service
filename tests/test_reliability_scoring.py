"""Unit tests for Reliability Scoring Engine and Decision Engine.

Verifies agreement, density, coverage, multiplicative score aggregation,
decision policy gates (PASS / ESCALATE), and configuration validation.
"""

from __future__ import annotations

import pytest

from arda_reliability_service.config import Settings
from arda_reliability_service.domain.exceptions import (
    InvalidProposedDecisionError,
    InvalidScoringConfigError,
)
from arda_reliability_service.domain.scoring_engine import (
    ReliabilityDecisionEngine,
    ReliabilityScoringEngine,
)
from arda_reliability_service.domain.value_objects import (
    EvaluationReasonCode,
    GateDecision,
    ReferenceNeighbor,
    ReliabilityScore,
    ScoringConfig,
)


def _make_neighbor(
    example_id: str,
    label: str,
    similarity_score: float,
    case_desc: str = "Test dispute case description",
) -> ReferenceNeighbor:
    """Helper to construct a ReferenceNeighbor with realistic test parameters."""
    return ReferenceNeighbor(
        example_id=example_id,
        label=label,
        source="synthetic",
        case_description=case_desc,
        similarity_score=similarity_score,
    )


class TestAgreementCalculation:
    """Tests for Agreement: consistency of retrieved labels with the proposed decision."""

    @pytest.fixture
    def engine(self) -> ReliabilityScoringEngine:
        return ReliabilityScoringEngine()

    @pytest.fixture
    def config(self) -> ScoringConfig:
        return ScoringConfig()

    def test_all_agree(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """When 100% of retrieved neighbors match the proposed decision, agreement must be 1.0."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.95),
            _make_neighbor("ref-2", "APPROVE", 0.90),
            _make_neighbor("ref-3", "APPROVE", 0.85),
        ]
        agreement = engine.calculate_agreement(neighbors, "APPROVE", config)
        assert agreement == 1.0

    def test_all_conflict(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """When 100% of retrieved neighbors conflict with proposed decision, agreement must be 0.0."""
        neighbors = [
            _make_neighbor("ref-1", "REJECT", 0.95),
            _make_neighbor("ref-2", "REJECT", 0.90),
            _make_neighbor("ref-3", "REJECT", 0.85),
        ]
        agreement = engine.calculate_agreement(neighbors, "APPROVE", config)
        assert agreement == 0.0

    def test_mixed_labels(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Similarity-weighted agreement reflects proportional similarity mass."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.90),  # Matching
            _make_neighbor("ref-2", "REJECT", 0.60),  # Conflicting
        ]
        # Total weight = 0.90 + 0.60 = 1.50. Matching = 0.90. Agreement = 0.90 / 1.50 = 0.6000
        agreement = engine.calculate_agreement(neighbors, "APPROVE", config)
        assert pytest.approx(agreement, abs=1e-4) == 0.6000

    def test_similarity_weighted_evidence_vs_unweighted(
        self, engine: ReliabilityScoringEngine
    ) -> None:
        """A strong match outweighs a weak conflict in similarity-weighted mode."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.95),  # Strong match
            _make_neighbor("ref-2", "REJECT", 0.05),  # Very weak conflict
        ]
        # Weighted: 0.95 / 1.00 = 0.9500
        cfg_weighted = ScoringConfig(weighting_strategy="similarity_weighted")
        weighted_score = engine.calculate_agreement(neighbors, "APPROVE", cfg_weighted)
        assert pytest.approx(weighted_score, abs=1e-4) == 0.9500

        # Unweighted: 1 / 2 = 0.5000
        cfg_unweighted = ScoringConfig(weighting_strategy="unweighted")
        unweighted_score = engine.calculate_agreement(
            neighbors, "APPROVE", cfg_unweighted
        )
        assert pytest.approx(unweighted_score, abs=1e-4) == 0.5000

    def test_empty_evidence(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Empty retrieval evidence must return 0.0 agreement."""
        assert engine.calculate_agreement([], "APPROVE", config) == 0.0

    def test_equal_strength_conflicting_evidence(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Equal similarity matching and conflicting neighbors result in exactly 0.50 agreement."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.88),
            _make_neighbor("ref-2", "REJECT", 0.88),
        ]
        agreement = engine.calculate_agreement(neighbors, "APPROVE", config)
        assert pytest.approx(agreement, abs=1e-4) == 0.5000

    def test_invalid_proposed_decision(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Invalid proposed decision must raise InvalidProposedDecisionError."""
        neighbors = [_make_neighbor("ref-1", "APPROVE", 0.90)]
        with pytest.raises(InvalidProposedDecisionError):
            engine.calculate_agreement(neighbors, "UNKNOWN_DECISION", config)


class TestDensityCalculation:
    """Tests for Density factor: proximity and concentration of evidence."""

    @pytest.fixture
    def engine(self) -> ReliabilityScoringEngine:
        return ReliabilityScoringEngine()

    @pytest.fixture
    def config(self) -> ScoringConfig:
        return ScoringConfig(
            min_similarity_threshold=0.70,
            min_neighbor_count=3,
            density_peak_weight=0.50,
        )

    def test_strong_and_concentrated_evidence(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """High similarity precedents yield high density."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.95),
            _make_neighbor("ref-2", "APPROVE", 0.95),
            _make_neighbor("ref-3", "APPROVE", 0.95),
        ]
        # s_min = 0.70, norm = 0.30. r_i = (0.95 - 0.70) / 0.30 = 0.8333
        # q = 3/3 = 1.0. blend = 0.8333. density = 0.8333
        density = engine.calculate_density(neighbors, config)
        assert pytest.approx(density, abs=1e-3) == 0.8333

    def test_weak_evidence(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Evidence hovering right above the similarity floor produces near-zero density."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.71),
            _make_neighbor("ref-2", "APPROVE", 0.71),
            _make_neighbor("ref-3", "APPROVE", 0.71),
        ]
        # r_i = (0.71 - 0.70) / 0.30 = 0.0333. density ~ 0.0333
        density = engine.calculate_density(neighbors, config)
        assert density < 0.05
        assert density > 0.0

    def test_sparse_evidence_penalized(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Having fewer than min_neighbor_count relevant precedents penalizes density linearly."""
        # Only 1 relevant neighbor when min_neighbor_count = 3
        neighbors = [_make_neighbor("ref-1", "APPROVE", 0.95)]
        # r_1 = 0.8333. q = 1/3 = 0.3333. density = 0.3333 * 0.8333 = 0.2778
        density = engine.calculate_density(neighbors, config)
        assert pytest.approx(density, abs=1e-3) == 0.2778

    def test_similarity_distribution_variation(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Variation between peak similarity and cohort average is blended as configured."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 1.00),  # r_1 = 1.0
            _make_neighbor("ref-2", "APPROVE", 0.70),  # r_2 = 0.0
            _make_neighbor("ref-3", "APPROVE", 0.70),  # r_3 = 0.0
        ]
        # r_top = 1.0. r_bar = 1.0 / 3 = 0.3333.
        # blend = 0.5 * 1.0 + 0.5 * 0.3333 = 0.6667. q = 1.0. density = 0.6667
        density = engine.calculate_density(neighbors, config)
        assert pytest.approx(density, abs=1e-3) == 0.6667

    def test_boundary_values(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Perfect similarity (1.0) on all neighbors yields maximum density 1.0."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 1.00),
            _make_neighbor("ref-2", "APPROVE", 1.00),
            _make_neighbor("ref-3", "APPROVE", 1.00),
        ]
        density = engine.calculate_density(neighbors, config)
        assert density == 1.0

    def test_empty_or_all_below_similarity_floor(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """When no neighbors reach the similarity floor, density must be 0.0."""
        assert engine.calculate_density([], config) == 0.0
        below_floor = [
            _make_neighbor("ref-1", "APPROVE", 0.69),
            _make_neighbor("ref-2", "APPROVE", 0.50),
        ]
        assert engine.calculate_density(below_floor, config) == 0.0


class TestCoverageCalculation:
    """Tests for Coverage factor: breadth and clarity of evidence."""

    @pytest.fixture
    def engine(self) -> ReliabilityScoringEngine:
        return ReliabilityScoringEngine()

    @pytest.fixture
    def config(self) -> ScoringConfig:
        return ScoringConfig(
            min_neighbor_count=3,
            min_similarity_threshold=0.70,
            conflict_penalty_weight=0.50,
        )

    def test_insufficient_neighbors_reduces_coverage(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Having only 1 relevant neighbor out of 1 retrieved yields adequacy = 1/3."""
        neighbors = [_make_neighbor("ref-1", "APPROVE", 0.85)]
        # adequacy = 1/3, relevance_ratio = 1/1 = 1.0, clarity = 1.0. coverage = 0.3333
        coverage = engine.calculate_coverage(neighbors, "APPROVE", config)
        assert pytest.approx(coverage, abs=1e-3) == 0.3333

    def test_sufficient_and_unanimous_neighbors(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """When all retrieved neighbors are relevant and agree, coverage is 1.0."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.85),
            _make_neighbor("ref-2", "APPROVE", 0.82),
            _make_neighbor("ref-3", "APPROVE", 0.80),
        ]
        # adequacy = 1.0, relevance_ratio = 1.0, clarity = 1.0. coverage = 1.0
        coverage = engine.calculate_coverage(neighbors, "APPROVE", config)
        assert coverage == 1.0

    def test_weakly_related_neighbors_zero_coverage(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """A retrieval result containing only weakly related examples receives 0.0 coverage."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.65),
            _make_neighbor("ref-2", "APPROVE", 0.60),
            _make_neighbor("ref-3", "APPROVE", 0.55),
        ]
        coverage = engine.calculate_coverage(neighbors, "APPROVE", config)
        assert coverage == 0.0

    def test_conflicting_evidence_penalized(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """Conflicting labels within the relevant set reduce coverage via the clarity factor."""
        # 2 matching, 2 conflicting, all relevant, k=4
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.85),
            _make_neighbor("ref-2", "APPROVE", 0.82),
            _make_neighbor("ref-3", "REJECT", 0.80),
            _make_neighbor("ref-4", "REJECT", 0.78),
        ]
        # adequacy = min(1.0, 4/3) = 1.0
        # relevance_ratio = 4/4 = 1.0
        # conflict_rate = 2 * min(2, 2) / 4 = 1.0
        # clarity = 1.0 - 0.50 * 1.0 = 0.50
        # coverage = 1.0 * 1.0 * 0.50 = 0.5000
        coverage = engine.calculate_coverage(neighbors, "APPROVE", config)
        assert pytest.approx(coverage, abs=1e-4) == 0.5000


class TestScoreAggregation:
    """Tests for raw_score = agreement * density * coverage."""

    @pytest.fixture
    def engine(self) -> ReliabilityScoringEngine:
        return ReliabilityScoringEngine()

    @pytest.fixture
    def config(self) -> ScoringConfig:
        return ScoringConfig(min_similarity_threshold=0.70, min_neighbor_count=3)

    def test_multiplicative_gate_property(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """If any single component is zero, the raw score must be strictly zero."""
        # Empty evidence -> agreement=0, density=0, coverage=0 -> raw_score=0
        score = engine.calculate_score([], "APPROVE", config)
        assert score.raw_score == 0.0
        assert score.agreement == 0.0
        assert score.density_factor == 0.0
        assert score.coverage_factor == 0.0

        # All conflicting evidence -> agreement=0 -> raw_score=0
        conflicting = [
            _make_neighbor("ref-1", "REJECT", 0.95),
            _make_neighbor("ref-2", "REJECT", 0.90),
            _make_neighbor("ref-3", "REJECT", 0.88),
        ]
        score_conf = engine.calculate_score(conflicting, "APPROVE", config)
        assert score_conf.agreement == 0.0
        assert score_conf.raw_score == 0.0
        assert score_conf.density_factor > 0.0
        assert score_conf.coverage_factor > 0.0

    def test_score_components_in_bounds(
        self, engine: ReliabilityScoringEngine, config: ScoringConfig
    ) -> None:
        """All score components must satisfy 0.0 <= x <= 1.0 invariant."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.92),
            _make_neighbor("ref-2", "APPROVE", 0.88),
            _make_neighbor("ref-3", "REJECT", 0.75),
        ]
        score = engine.calculate_score(neighbors, "APPROVE", config)
        assert 0.0 <= score.agreement <= 1.0
        assert 0.0 <= score.density_factor <= 1.0
        assert 0.0 <= score.coverage_factor <= 1.0
        assert 0.0 <= score.raw_score <= 1.0
        assert pytest.approx(score.raw_score, abs=1e-4) == round(
            score.agreement * score.density_factor * score.coverage_factor, 4
        )


class TestDecisionEngine:
    """Tests for ReliabilityDecisionEngine gate policies (PASS vs ESCALATE)."""

    @pytest.fixture
    def decision_engine(self) -> ReliabilityDecisionEngine:
        return ReliabilityDecisionEngine()

    @pytest.fixture
    def config(self) -> ScoringConfig:
        return ScoringConfig(
            min_neighbor_count=3,
            min_similarity_threshold=0.70,
            min_coverage_threshold=0.50,
            min_density_threshold=0.40,
            min_agreement_threshold=0.60,
            min_reliability_threshold=0.65,
            conflict_veto_similarity=0.95,
        )

    def test_pass_on_high_reliability_sufficient_evidence(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """When evidence is dense, unanimous, and exceeds reliability threshold, decision is PASS."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.95),
            _make_neighbor("ref-2", "APPROVE", 0.92),
            _make_neighbor("ref-3", "APPROVE", 0.90),
        ]
        # agreement = 1.0, density = 0.7444, coverage = 1.0 -> raw_score = 0.7444 >= 0.65
        score = ReliabilityScore(
            agreement=1.0,
            density_factor=0.75,
            coverage_factor=1.0,
            raw_score=0.75,
        )
        decision, code, reason = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.PASS
        assert code == EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value
        assert "meets operational threshold" in reason

    def test_escalate_on_cold_start(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """Cold-start condition (fewer than min_neighbor_count relevant precedents) must ESCALATE."""
        # Only 1 relevant neighbor
        neighbors = [_make_neighbor("ref-1", "APPROVE", 0.92)]
        score = ReliabilityScore(
            agreement=1.0, density_factor=0.5, coverage_factor=0.33, raw_score=0.165
        )
        decision, code, reason = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.ESCALATE
        assert code == EvaluationReasonCode.COLD_START_INSUFFICIENT_EVIDENCE.value
        assert "Cold start" in reason

    def test_escalate_on_insufficient_coverage(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """When coverage factor is below min_coverage_threshold (0.50), must ESCALATE."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.85),
            _make_neighbor("ref-2", "APPROVE", 0.80),
            _make_neighbor("ref-3", "APPROVE", 0.75),
            _make_neighbor("ref-4", "APPROVE", 0.50),  # irrelevant
            _make_neighbor("ref-5", "APPROVE", 0.40),  # irrelevant
        ]
        # Low coverage score
        score = ReliabilityScore(
            agreement=1.0, density_factor=0.50, coverage_factor=0.45, raw_score=0.225
        )
        decision, code, _ = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.ESCALATE
        assert code == EvaluationReasonCode.INSUFFICIENT_COVERAGE.value

    def test_escalate_on_weak_density(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """When evidence density factor is below min_density_threshold (0.40), must ESCALATE."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.72),
            _make_neighbor("ref-2", "APPROVE", 0.71),
            _make_neighbor("ref-3", "APPROVE", 0.70),
        ]
        score = ReliabilityScore(
            agreement=1.0, density_factor=0.35, coverage_factor=0.80, raw_score=0.28
        )
        decision, code, _ = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.ESCALATE
        assert code == EvaluationReasonCode.WEAK_EVIDENCE_DENSITY.value

    def test_escalate_on_conflict_veto(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """A hyper-similar conflicting precedent (e.g. 0.98 contrastive twin) must trigger veto ESCALATE."""
        neighbors = [
            _make_neighbor(
                "ref-twin", "REJECT", 0.98
            ),  # Contrastive twin with opposing label
            _make_neighbor("ref-2", "APPROVE", 0.92),
            _make_neighbor("ref-3", "APPROVE", 0.90),
            _make_neighbor("ref-4", "APPROVE", 0.88),
        ]
        # Even if aggregate agreement is 0.75, the 0.98 conflict indicates a boundary condition
        score = ReliabilityScore(
            agreement=0.74, density_factor=0.80, coverage_factor=0.70, raw_score=0.414
        )
        decision, code, reason = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.ESCALATE
        assert code == EvaluationReasonCode.HIGH_SIMILARITY_CONFLICT.value
        assert "ref-twin" in reason

    def test_escalate_on_low_agreement(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """When agreement is below min_agreement_threshold (0.60), must ESCALATE."""
        neighbors = [
            _make_neighbor("ref-1", "REJECT", 0.88),
            _make_neighbor("ref-2", "REJECT", 0.85),
            _make_neighbor("ref-3", "APPROVE", 0.80),
        ]
        score = ReliabilityScore(
            agreement=0.35, density_factor=0.60, coverage_factor=0.60, raw_score=0.126
        )
        decision, code, _ = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.ESCALATE
        assert code == EvaluationReasonCode.CONFLICTING_EVIDENCE.value

    def test_escalate_on_low_raw_score(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """When individual factors pass but their product is below min_reliability_threshold (0.65), ESCALATE."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.88),
            _make_neighbor("ref-2", "APPROVE", 0.85),
            _make_neighbor("ref-3", "APPROVE", 0.80),
        ]
        # agreement=0.70 (>=0.60), density=0.50 (>=0.40), coverage=0.60 (>=0.50), but product = 0.210 (< 0.65)
        score = ReliabilityScore(
            agreement=0.70, density_factor=0.50, coverage_factor=0.60, raw_score=0.210
        )
        decision, code, _ = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.ESCALATE
        assert code == EvaluationReasonCode.SCORE_BELOW_THRESHOLD.value

    def test_threshold_boundary_conditions(
        self, decision_engine: ReliabilityDecisionEngine, config: ScoringConfig
    ) -> None:
        """Exact threshold matches (boundary values) qualify for PASS."""
        neighbors = [
            _make_neighbor("ref-1", "APPROVE", 0.85),
            _make_neighbor("ref-2", "APPROVE", 0.82),
            _make_neighbor("ref-3", "APPROVE", 0.80),
        ]
        # All factors exactly meeting thresholds: agreement=0.60, density=0.40, coverage=0.50, raw_score=0.65
        score = ReliabilityScore(
            agreement=0.60,
            density_factor=0.40,
            coverage_factor=0.50,
            raw_score=0.65,
        )
        decision, code, _ = decision_engine.evaluate(
            score, neighbors, "APPROVE", config
        )
        assert decision == GateDecision.PASS
        assert code == EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value


class TestScoringConfigValidation:
    """Tests for ScoringConfig domain invariants and Settings startup validation."""

    def test_default_config_valid(self) -> None:
        cfg = ScoringConfig()
        assert cfg.min_neighbor_count == 3
        assert cfg.min_similarity_threshold == 0.70
        assert cfg.min_reliability_threshold == 0.65

    def test_invalid_similarity_threshold_raises(self) -> None:
        with pytest.raises(InvalidScoringConfigError):
            ScoringConfig(min_similarity_threshold=1.5)

    def test_invalid_neighbor_count_raises(self) -> None:
        with pytest.raises(InvalidScoringConfigError):
            ScoringConfig(min_neighbor_count=0)

    def test_top_k_smaller_than_min_neighbors_raises(self) -> None:
        with pytest.raises(InvalidScoringConfigError):
            ScoringConfig(top_k=2, min_neighbor_count=5)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(InvalidScoringConfigError):
            ScoringConfig(weighting_strategy="invalid_mode")

    def test_settings_startup_validation(self) -> None:
        settings = Settings()
        scoring_cfg = settings.get_scoring_config()
        assert scoring_cfg.min_neighbor_count == settings.reliability_min_neighbor_count
        assert (
            scoring_cfg.min_reliability_threshold == settings.reliability_min_threshold
        )

    def test_multi_domain_agent_decision_support(self) -> None:
        """Arbitrary domain agents (e.g. onboarding department routing) are supported."""
        engine = ReliabilityScoringEngine()
        decision_engine = ReliabilityDecisionEngine()
        config = ScoringConfig(
            allowed_decisions=("BILLING", "LEGAL", "KYC", "SALES"),
            min_reliability_threshold=0.60,
        )

        neighbors = [
            _make_neighbor(
                "ref-1",
                "LEGAL",
                0.94,
                "Customer uploaded German commercial register extract",
            ),
            _make_neighbor(
                "ref-2",
                "LEGAL",
                0.91,
                "Corporate charter and power of attorney verification",
            ),
            _make_neighbor(
                "ref-3", "LEGAL", 0.88, "EU cross-border entity compliance review"
            ),
            _make_neighbor(
                "ref-4", "LEGAL", 0.85, "Foreign ownership compliance certificate"
            ),
        ]

        score = engine.calculate_score(neighbors, "LEGAL", config)
        assert score.agreement == 1.0
        assert score.density_factor > 0.60
        assert score.raw_score >= 0.60

        decision, code, _reason = decision_engine.evaluate(
            score, neighbors, "LEGAL", config
        )
        assert decision == GateDecision.PASS
        assert code == EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value
