"""Domain services for reliability scoring and decision gating.

Implements the decomposed reliability scoring formula:
    raw_score = agreement * density_factor * coverage_factor

and the operational gate decision policy (PASS vs ESCALATE).
"""

from __future__ import annotations

from arda_reliability_service.domain.exceptions import InvalidProposedDecisionError
from arda_reliability_service.domain.value_objects import (
    EvaluationReasonCode,
    GateDecision,
    ReferenceNeighbor,
    ReliabilityScore,
    ScoringConfig,
)


class ReliabilityScoringEngine:
    """Calculates decomposed and aggregate reliability scores from retrieved reference evidence."""

    def calculate_agreement(
        self,
        neighbors: list[ReferenceNeighbor],
        proposed_decision: str,
        config: ScoringConfig,
    ) -> float:
        """Calculate how consistently retrieved reference labels agree with the proposed decision.

        Distinguishes matching and conflicting reference labels, weighting evidence
        by cosine similarity rather than assuming uniform importance.

        Mathematical Definition:
            Let M = {n in neighbors | n.label == proposed_decision}
            Let w_i = max(0.0, n_i.similarity_score)
            If sum(w_i) > 0:
                agreement = sum_{i in M}(w_i) / sum_{i}(w_i)
            Else:
                agreement = 0.0

        Args:
            neighbors: Retrieved reference neighbors with similarity scores and labels.
            proposed_decision: The agent's proposed action ('APPROVE' or 'REJECT').
            config: Scoring configuration specifying weighting strategy.

        Returns:
            Agreement value bounded in [0.0, 1.0].
        """
        if not proposed_decision or not proposed_decision.strip():
            raise InvalidProposedDecisionError(
                "proposed_decision cannot be empty or whitespace."
            )

        if (
            config.allowed_decisions is not None
            and proposed_decision not in config.allowed_decisions
        ):
            raise InvalidProposedDecisionError(
                f"Invalid proposed_decision '{proposed_decision}'. Must be one of {sorted(config.allowed_decisions)}."
            )

        if not neighbors:
            return 0.0

        matching = [n for n in neighbors if n.label == proposed_decision]

        if config.weighting_strategy == "unweighted":
            return round(len(matching) / len(neighbors), 4)

        # Default: similarity_weighted
        total_weight = sum(max(0.0, n.similarity_score) for n in neighbors)
        if total_weight <= 0.0:
            return 0.0

        matching_weight = sum(max(0.0, n.similarity_score) for n in matching)
        agreement = matching_weight / total_weight
        return min(1.0, max(0.0, round(agreement, 4)))

    def calculate_density(
        self,
        neighbors: list[ReferenceNeighbor],
        config: ScoringConfig,
    ) -> float:
        """Measure the strength and proximity of retrieved reference evidence.

        Quantifies whether the query lies in a well-represented, high-density precedent
        region of the embedding space, or in a sparse, out-of-distribution margin.

        Mathematical Definition:
            Let R = {n in neighbors | n.similarity_score >= min_similarity_threshold}
            If |R| == 0:
                density = 0.0
            Else:
                r_i = clamp((s_i - s_min) / (1.0 - s_min), 0.0, 1.0)
                r_top = max(r_i)
                r_bar = mean(r_i)
                q = min(1.0, |R| / min_neighbor_count)
                blend = alpha * r_top + (1.0 - alpha) * r_bar
                density = q * blend

        Args:
            neighbors: Retrieved reference neighbors.
            config: Scoring configuration specifying similarity floors and neighbor minimums.

        Returns:
            Density factor bounded in [0.0, 1.0].
        """
        if not neighbors:
            return 0.0

        s_min = config.min_similarity_threshold
        relevant = [n for n in neighbors if n.similarity_score >= s_min]

        if not relevant:
            return 0.0

        # Normalized similarity above the relevance floor
        norm_factor = 1.0 - s_min
        if norm_factor <= 0.0:
            return 0.0

        rel_scores = [
            min(1.0, max(0.0, (n.similarity_score - s_min) / norm_factor))
            for n in relevant
        ]

        r_top = max(rel_scores)
        r_bar = sum(rel_scores) / len(rel_scores)

        # Quantity saturation factor
        q = min(1.0, len(relevant) / config.min_neighbor_count)

        # Blended density of top precedent and cohort average
        alpha = config.density_peak_weight
        blended = alpha * r_top + (1.0 - alpha) * r_bar

        density = q * blended
        return min(1.0, max(0.0, round(density, 4)))

    def calculate_coverage(
        self,
        neighbors: list[ReferenceNeighbor],
        proposed_decision: str,
        config: ScoringConfig,
    ) -> float:
        """Measure whether retrieved evidence provides sufficient and unambiguous precedent coverage.

        Penalizes retrievals with few relevant neighbors, low relevance proportions,
        or high precedent conflict (e.g., contrastive counterfactuals near policy boundaries).

        Mathematical Definition:
            Let R = {n in neighbors | n.similarity_score >= min_similarity_threshold}
            If len(neighbors) == 0 or |R| == 0:
                coverage = 0.0
            Else:
                adequacy = min(1.0, |R| / min_neighbor_count)
                relevance_ratio = |R| / len(neighbors)
                Let R_M = {n in R | n.label == proposed_decision}
                Let R_C = {n in R | n.label != proposed_decision}
                conflict_rate = 2 * min(|R_M|, |R_C|) / |R|
                clarity = 1.0 - beta * conflict_rate
                coverage = adequacy * relevance_ratio * clarity

        Args:
            neighbors: Retrieved reference neighbors.
            proposed_decision: The agent's proposed action ('APPROVE' or 'REJECT').
            config: Scoring configuration.

        Returns:
            Coverage factor bounded in [0.0, 1.0].
        """
        if not neighbors:
            return 0.0

        s_min = config.min_similarity_threshold
        relevant = [n for n in neighbors if n.similarity_score >= s_min]

        if not relevant:
            return 0.0

        # 1. Sample size adequacy
        adequacy = min(1.0, len(relevant) / config.min_neighbor_count)

        # 2. Relevance ratio across retrieved window
        relevance_ratio = len(relevant) / len(neighbors)

        # 3. Precedent clarity (penalizing conflicting relevant labels)
        r_matching = sum(1 for n in relevant if n.label == proposed_decision)
        r_conflicting = len(relevant) - r_matching

        conflict_rate = (2.0 * min(r_matching, r_conflicting)) / len(relevant)
        clarity = 1.0 - (config.conflict_penalty_weight * conflict_rate)

        coverage = adequacy * relevance_ratio * clarity
        return min(1.0, max(0.0, round(coverage, 4)))

    def calculate_score(
        self,
        neighbors: list[ReferenceNeighbor],
        proposed_decision: str,
        config: ScoringConfig,
    ) -> ReliabilityScore:
        """Calculate all components and the raw multiplicative reliability score.

        Formula: raw_score = agreement * density_factor * coverage_factor
        """
        agreement = self.calculate_agreement(neighbors, proposed_decision, config)
        density = self.calculate_density(neighbors, config)
        coverage = self.calculate_coverage(neighbors, proposed_decision, config)

        raw_score = min(1.0, max(0.0, round(agreement * density * coverage, 4)))

        return ReliabilityScore(
            agreement=agreement,
            density_factor=density,
            coverage_factor=coverage,
            raw_score=raw_score,
        )


class ReliabilityDecisionEngine:
    """Evaluates reliability scores and precedent evidence against gate policy to decide PASS or ESCALATE."""

    def evaluate(
        self,
        score: ReliabilityScore,
        neighbors: list[ReferenceNeighbor],
        proposed_decision: str,
        config: ScoringConfig,
    ) -> tuple[GateDecision, str, str]:
        """Apply operational gate policy to determine PASS or ESCALATE.

        Checks are executed in strict policy hierarchy:
        1. Cold-start / minimum relevant evidence count.
        2. Minimum coverage threshold.
        3. Minimum evidence density threshold.
        4. High-similarity conflicting precedent check (contrastive veto).
        5. Minimum agreement threshold.
        6. Operational reliability score threshold.

        Args:
            score: The decomposed ReliabilityScore value object.
            neighbors: Retrieved reference neighbors.
            proposed_decision: Proposed agent action ('APPROVE' or 'REJECT').
            config: Operational scoring configuration.

        Returns:
            Tuple of (GateDecision, reason_code, evaluation_reason).
        """
        s_min = config.min_similarity_threshold
        relevant = [n for n in neighbors if n.similarity_score >= s_min]
        matching_count = sum(1 for n in relevant if n.label == proposed_decision)
        conflicting_count = len(relevant) - matching_count

        # Check 1: Cold start / Insufficient relevant evidence count
        if len(relevant) < config.min_neighbor_count:
            return (
                GateDecision.ESCALATE,
                EvaluationReasonCode.COLD_START_INSUFFICIENT_EVIDENCE.value,
                (
                    f"Cold start or insufficient relevant evidence: found {len(relevant)} relevant precedents "
                    f"meeting similarity threshold {s_min:.2f} (minimum required is {config.min_neighbor_count})."
                ),
            )

        # Check 2: Minimum coverage threshold
        if score.coverage_factor < config.min_coverage_threshold:
            return (
                GateDecision.ESCALATE,
                EvaluationReasonCode.INSUFFICIENT_COVERAGE.value,
                (
                    f"Evidence coverage factor ({score.coverage_factor:.4f}) is below minimum threshold "
                    f"({config.min_coverage_threshold:.4f})."
                ),
            )

        # Check 3: Minimum evidence density threshold
        if score.density_factor < config.min_density_threshold:
            return (
                GateDecision.ESCALATE,
                EvaluationReasonCode.WEAK_EVIDENCE_DENSITY.value,
                (
                    f"Evidence density factor ({score.density_factor:.4f}) is below minimum threshold "
                    f"({config.min_density_threshold:.4f})."
                ),
            )

        # Check 4: High-similarity conflicting precedent veto
        # If a conflicting precedent exists with extreme similarity (e.g. >= 0.95),
        # this indicates the case lies directly on a policy boundary (contrastive twin).
        conflicting_neighbors = [n for n in relevant if n.label != proposed_decision]
        for cn in conflicting_neighbors:
            if cn.similarity_score >= config.conflict_veto_similarity:
                return (
                    GateDecision.ESCALATE,
                    EvaluationReasonCode.HIGH_SIMILARITY_CONFLICT.value,
                    (
                        f"High-similarity conflicting precedent detected ({cn.example_id}, "
                        f"similarity={cn.similarity_score:.4f}, label='{cn.label}') "
                        f"violating conflict veto threshold ({config.conflict_veto_similarity:.4f})."
                    ),
                )

        # Check 5: Minimum agreement threshold
        if score.agreement < config.min_agreement_threshold:
            return (
                GateDecision.ESCALATE,
                EvaluationReasonCode.CONFLICTING_EVIDENCE.value,
                (
                    f"Precedent agreement ({score.agreement:.4f}) is below minimum threshold "
                    f"({config.min_agreement_threshold:.4f}) with {conflicting_count} conflicting precedents."
                ),
            )

        # Check 6: Operational reliability score threshold
        if score.raw_score < config.min_reliability_threshold:
            return (
                GateDecision.ESCALATE,
                EvaluationReasonCode.SCORE_BELOW_THRESHOLD.value,
                (
                    f"Reliability score ({score.raw_score:.4f}) is below operational gate threshold "
                    f"({config.min_reliability_threshold:.4f})."
                ),
            )

        # All checks passed
        return (
            GateDecision.PASS,
            EvaluationReasonCode.RELIABILITY_THRESHOLD_MET.value,
            (
                f"Reliability score ({score.raw_score:.4f}) meets operational threshold "
                f"({config.min_reliability_threshold:.4f}) with {matching_count} supporting precedents."
            ),
        )
