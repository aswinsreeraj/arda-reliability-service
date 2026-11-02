"""Evaluation pipeline for reliability scoring and confidence calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from arda_reliability_service.domain.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
)
from arda_reliability_service.domain.scoring_engine import (
    ReliabilityDecisionEngine,
    ReliabilityScoringEngine,
)
from arda_reliability_service.domain.value_objects import (
    ReferenceNeighbor,
    ScoringConfig,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)


@dataclass(frozen=True)
class EvaluationInstance:
    """A single evaluation query presented to the Reliability Gate."""

    example_id: str
    group_id: str
    case_type: str
    category: str
    true_label: str
    proposed_decision: str
    is_correct: int  # 1 if proposed_decision == true_label else 0
    case_description: str
    contrastive_pair_id: str | None


@dataclass(frozen=True)
class InstancePrediction:
    """Predictions and scores generated for a single evaluation query."""

    example_id: str
    group_id: str
    case_type: str
    category: str
    true_label: str
    proposed_decision: str
    is_correct: int
    raw_score: float
    agreement: float
    density_factor: float
    coverage_factor: float
    gate_decision: str
    reason_code: str
    calibrated_confidence: float | None = None
    calibrator_type: str | None = None
    platt_oof: float | None = None
    isotonic_oof: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConnectedGroupAssigner:
    """Assigns evaluation instances to connected groups and partitions groups into balanced folds.

    Connected group rules:
    1. Contrastive pairs (2 original examples with opposing labels) form a single connected group.
    2. Singleton cases (80 examples) form their own unique connected groups.
    3. All evaluation proposals for each original case (matching and opposing) belong to the same group.

    Resulting structure:
    - 10 contrastive pair groups (2 original cases each, 4 evaluation instances)
    - 80 singleton groups (1 original case each, 2 evaluation instances)
    - Total: 90 connected groups across 100 cases (200 evaluation queries)
    - 5 folds: Each fold contains exactly 18 groups (2 contrastive pairs + 16 singletons),
      which corresponds to exactly 20 original cases and 40 evaluation queries.
    """

    @staticmethod
    def get_group_map(raw_examples: list[dict[str, Any]]) -> dict[str, str]:
        group_map: dict[str, str] = {}
        for ex in raw_examples:
            cp_id = ex["metadata"].get("contrastive_pair_id")
            group_map[ex["example_id"]] = cp_id if cp_id else ex["example_id"]
        return group_map

    @classmethod
    def get_unique_groups(
        cls, raw_examples: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """Return sorted lists of (cp_groups, singleton_groups)."""
        cp_groups = sorted(
            {
                ex["metadata"]["contrastive_pair_id"]
                for ex in raw_examples
                if ex["metadata"].get("contrastive_pair_id")
            }
        )
        singleton_groups = sorted(
            [
                ex["example_id"]
                for ex in raw_examples
                if not ex["metadata"].get("contrastive_pair_id")
            ]
        )
        return cp_groups, singleton_groups

    @classmethod
    def partition_groups(
        cls,
        raw_examples: list[dict[str, Any]],
        n_splits: int = 5,
    ) -> list[list[str]]:
        """Partition the 90 connected groups deterministically into n_splits folds.

        Guarantees:
        - 2 contrastive pair groups per fold (if 10 CPs, 5 folds)
        - 16 singleton groups per fold (if 80 singletons, 5 folds)
        - Exactly 18 groups, 20 original cases, and 40 queries per fold.
        """
        cp_groups, singleton_groups = cls.get_unique_groups(raw_examples)

        folds_groups: list[list[str]] = [[] for _ in range(n_splits)]
        for i, g in enumerate(cp_groups):
            folds_groups[i % n_splits].append(g)
        for i, g in enumerate(singleton_groups):
            folds_groups[i % n_splits].append(g)

        return folds_groups


def load_and_build_evaluation_instances(
    dataset_path: Path,
    template_key: str = "ratio_verbalized_description",
) -> tuple[list[dict[str, Any]], list[EvaluationInstance]]:
    """Load reference dataset and construct symmetric evaluation instances.

    Each underlying example is evaluated under both possible proposed decisions:
    1. proposed_decision == true_label (is_correct = 1)
    2. proposed_decision != true_label (is_correct = 0)

    This yields a strictly balanced evaluation dataset (50% correct, 50% incorrect)
    without introducing class imbalance artifacts.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_examples: list[dict[str, Any]] = raw_data["examples"]
    instances: list[EvaluationInstance] = []

    for ex in raw_examples:
        ex_id = ex["example_id"]
        meta = ex["metadata"]
        cp_id = meta.get("contrastive_pair_id")
        group_id = cp_id if cp_id else ex_id
        case_type = meta.get("case_type", "unknown")
        category = meta.get("category", "unknown")
        true_label = ex["label"]

        if template_key == "ratio_verbalized_description":
            desc = meta.get("ratio_verbalized_description", ex["case_description"])
        elif template_key == "raw_value_description":
            desc = meta.get("raw_value_description", ex["case_description"])
        else:
            desc = ex["case_description"]

        # Evaluate both proposed outcomes
        for proposed in ["APPROVE", "REJECT"]:
            is_correct = 1 if proposed == true_label else 0
            instances.append(
                EvaluationInstance(
                    example_id=ex_id,
                    group_id=group_id,
                    case_type=case_type,
                    category=category,
                    true_label=true_label,
                    proposed_decision=proposed,
                    is_correct=is_correct,
                    case_description=desc,
                    contrastive_pair_id=cp_id,
                )
            )

    return raw_examples, instances


class EvaluationPipeline:
    """Executes vector retrieval, reliability scoring, gating, and cross-validation."""

    def __init__(
        self,
        embedding_adapter: SentenceTransformerEmbeddingAdapter | None = None,
        scoring_engine: ReliabilityScoringEngine | None = None,
        decision_engine: ReliabilityDecisionEngine | None = None,
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        self.adapter = embedding_adapter or SentenceTransformerEmbeddingAdapter()
        self.scoring_engine = scoring_engine or ReliabilityScoringEngine()
        self.decision_engine = decision_engine or ReliabilityDecisionEngine()
        self.config = scoring_config or ScoringConfig()

    def generate_embeddings(
        self,
        raw_examples: list[dict[str, Any]],
        template_key: str = "ratio_verbalized_description",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate normalized query and passage embeddings for all dataset examples."""
        texts: list[str] = []
        for ex in raw_examples:
            meta = ex["metadata"]
            if template_key == "ratio_verbalized_description":
                texts.append(
                    meta.get("ratio_verbalized_description", ex["case_description"])
                )
            elif template_key == "raw_value_description":
                texts.append(meta.get("raw_value_description", ex["case_description"]))
            else:
                texts.append(ex["case_description"])

        query_embs = np.array(
            self.adapter.generate_embeddings_batch(texts, is_query=True)
        )
        passage_embs = np.array(
            self.adapter.generate_embeddings_batch(texts, is_query=False)
        )
        return query_embs, passage_embs

    def run_partitioned_cross_validation(
        self,
        raw_examples: list[dict[str, Any]],
        query_embs: np.ndarray,
        passage_embs: np.ndarray,
        n_splits: int = 5,
    ) -> list[InstancePrediction]:
        """Execute strict N-fold partitioned cross-validation.

        Methodology:
        - 90 connected groups partitioned into n_splits folds (18 groups per fold).
        - For each fold k:
          - Candidate retrieval pool for test queries is STRICTLY the training cases R_k (80 cases).
          - Test queries NEVER retrieve from themselves or any fold k cases.
          - Training predictions on R_k are generated using Leave-One-Connected-Group-Out within R_k.
          - Fold-level Platt and Isotonic calibrators are fitted strictly on R_k predictions.
          - Test queries are scored against R_k and calibrated using fold k's fitted calibrators.
        """
        group_map = ConnectedGroupAssigner.get_group_map(raw_examples)
        folds_groups = ConnectedGroupAssigner.partition_groups(
            raw_examples, n_splits=n_splits
        )

        group_to_fold: dict[str, int] = {}
        for fold_idx, fg in enumerate(folds_groups):
            for g in fg:
                group_to_fold[g] = fold_idx

        # Cosine similarity matrix (query dot passage)
        sim_matrix = np.matmul(query_embs, passage_embs.T)
        n_examples = len(raw_examples)

        oof_predictions: list[InstancePrediction] = []

        for test_fold in range(n_splits):
            test_ex_indices = [
                i
                for i, ex in enumerate(raw_examples)
                if group_to_fold[group_map[ex["example_id"]]] == test_fold
            ]
            train_ex_indices = [
                i
                for i, ex in enumerate(raw_examples)
                if group_to_fold[group_map[ex["example_id"]]] != test_fold
            ]

            # 1. Generate training predictions within training set R_k (Leave-One-Connected-Group-Out)
            train_scores: list[float] = []
            train_y: list[int] = []

            for tr_idx in train_ex_indices:
                tr_ex = raw_examples[tr_idx]
                tr_group = group_map[tr_ex["example_id"]]

                sims = sim_matrix[tr_idx].copy()
                for j in range(n_examples):
                    if (
                        j not in train_ex_indices
                        or group_map[raw_examples[j]["example_id"]] == tr_group
                    ):
                        sims[j] = -np.inf

                top_k = np.argsort(-sims)[: self.config.top_k]
                neighbors = [
                    ReferenceNeighbor(
                        example_id=raw_examples[j]["example_id"],
                        label=raw_examples[j]["label"],
                        source=raw_examples[j]["source"],
                        case_description=raw_examples[j]["case_description"],
                        similarity_score=round(float(sims[j]), 4),
                    )
                    for j in top_k
                    if not np.isneginf(sims[j])
                ]

                for proposed in ["APPROVE", "REJECT"]:
                    is_corr = 1 if proposed == tr_ex["label"] else 0
                    sc = self.scoring_engine.calculate_score(
                        neighbors, proposed, self.config
                    )
                    train_scores.append(sc.raw_score)
                    train_y.append(is_corr)

            # Fit fold-level calibrators on training fold predictions
            X_tr = np.array(train_scores, dtype=np.float64).reshape(-1, 1)
            y_tr = np.array(train_y, dtype=np.int32)

            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(X_tr, y_tr)

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(X_tr.ravel(), y_tr)

            # 2. Evaluate held-out test queries against training reference pool R_k ONLY
            for te_idx in test_ex_indices:
                te_ex = raw_examples[te_idx]
                te_meta = te_ex["metadata"]
                te_group = group_map[te_ex["example_id"]]

                sims = sim_matrix[te_idx].copy()
                for j in range(n_examples):
                    if j not in train_ex_indices:
                        sims[j] = -np.inf

                top_k = np.argsort(-sims)[: self.config.top_k]
                neighbors = [
                    ReferenceNeighbor(
                        example_id=raw_examples[j]["example_id"],
                        label=raw_examples[j]["label"],
                        source=raw_examples[j]["source"],
                        case_description=raw_examples[j]["case_description"],
                        similarity_score=round(float(sims[j]), 4),
                    )
                    for j in top_k
                    if not np.isneginf(sims[j])
                ]

                for proposed in ["APPROVE", "REJECT"]:
                    is_corr = 1 if proposed == te_ex["label"] else 0
                    score = self.scoring_engine.calculate_score(
                        neighbors=neighbors,
                        proposed_decision=proposed,
                        config=self.config,
                    )
                    decision, reason_code, _ = self.decision_engine.evaluate(
                        score=score,
                        neighbors=neighbors,
                        proposed_decision=proposed,
                        config=self.config,
                    )

                    platt_prob = float(lr.predict_proba([[score.raw_score]])[0, 1])
                    iso_prob = float(iso.predict([score.raw_score])[0])

                    oof_predictions.append(
                        InstancePrediction(
                            example_id=te_ex["example_id"],
                            group_id=te_group,
                            case_type=te_meta.get("case_type", "unknown"),
                            category=te_meta.get("category", "unknown"),
                            true_label=te_ex["label"],
                            proposed_decision=proposed,
                            is_correct=is_corr,
                            raw_score=score.raw_score,
                            agreement=score.agreement,
                            density_factor=score.density_factor,
                            coverage_factor=score.coverage_factor,
                            gate_decision=decision.value,
                            reason_code=reason_code,
                            platt_oof=platt_prob,
                            isotonic_oof=iso_prob,
                        )
                    )

        return oof_predictions

    def fit_reference_calibrators(
        self,
        raw_examples: list[dict[str, Any]],
        query_embs: np.ndarray,
        passage_embs: np.ndarray,
        version_suffix: str = "seed100",
    ) -> tuple[PlattCalibrator, IsotonicCalibrator]:
        """Fit reference calibrator artifacts on full dataset using Leave-One-Connected-Group-Out."""
        group_map = ConnectedGroupAssigner.get_group_map(raw_examples)
        sim_matrix = np.matmul(query_embs, passage_embs.T)
        n_examples = len(raw_examples)

        full_scores: list[float] = []
        full_y: list[int] = []

        for i, ex in enumerate(raw_examples):
            grp = group_map[ex["example_id"]]
            sims = sim_matrix[i].copy()
            for j in range(n_examples):
                if group_map[raw_examples[j]["example_id"]] == grp:
                    sims[j] = -np.inf

            top_k = np.argsort(-sims)[: self.config.top_k]
            neighbors = [
                ReferenceNeighbor(
                    example_id=raw_examples[j]["example_id"],
                    label=raw_examples[j]["label"],
                    source=raw_examples[j]["source"],
                    case_description=raw_examples[j]["case_description"],
                    similarity_score=round(float(sims[j]), 4),
                )
                for j in top_k
                if not np.isneginf(sims[j])
            ]

            for proposed in ["APPROVE", "REJECT"]:
                is_corr = 1 if proposed == ex["label"] else 0
                sc = self.scoring_engine.calculate_score(
                    neighbors, proposed, self.config
                )
                full_scores.append(sc.raw_score)
                full_y.append(is_corr)

        X_full = np.array(full_scores, dtype=np.float64).reshape(-1, 1)
        y_full = np.array(full_y, dtype=np.int32)

        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(X_full, y_full)
        platt = PlattCalibrator(
            a=float(lr.coef_[0][0]),
            b=float(lr.intercept_[0]),
            version=f"platt-v1.0.0-{version_suffix}",
            metrics={"n_samples": len(y_full)},
        )

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(X_full.ravel(), y_full)
        isotonic = IsotonicCalibrator(
            x_knots=list(map(float, iso.X_thresholds_)),
            y_knots=list(map(float, iso.y_thresholds_)),
            version=f"isotonic-v1.0.0-{version_suffix}",
            metrics={"n_samples": len(y_full)},
        )

        return platt, isotonic

    def run_evaluation(
        self,
        raw_examples: list[dict[str, Any]],
        instances: list[EvaluationInstance],
        query_embs: np.ndarray,
        passage_embs: np.ndarray,
        protocol: str = "group_exclusion",
    ) -> list[InstancePrediction]:
        """Execute evaluation queries against reference candidates across the full corpus.

        Protocols:
        - 'group_exclusion': For query i, candidate pool excludes all examples in the same group_id.
                             Guarantees no self-retrieval AND no contrastive twin leakage.
        - 'standard_loo': For query i, candidate pool excludes only example i itself.
        """
        # Precompute dot-product cosine similarity matrix
        sim_matrix = np.matmul(query_embs, passage_embs.T)

        example_id_to_idx = {ex["example_id"]: i for i, ex in enumerate(raw_examples)}
        group_ids = [
            ex["metadata"].get("contrastive_pair_id") or ex["example_id"]
            for ex in raw_examples
        ]

        predictions: list[InstancePrediction] = []

        for inst in instances:
            q_idx = example_id_to_idx[inst.example_id]
            sims = sim_matrix[q_idx].copy()

            if protocol == "group_exclusion":
                # Exclude all candidates sharing this instance's group_id
                for j, gid in enumerate(group_ids):
                    if gid == inst.group_id:
                        sims[j] = -np.inf
            elif protocol == "standard_loo":
                # Exclude only self
                sims[q_idx] = -np.inf
            else:
                raise ValueError(f"Unknown evaluation protocol: '{protocol}'")

            # Retrieve top-K candidates
            top_k_indices = np.argsort(-sims)[: self.config.top_k]

            neighbors = [
                ReferenceNeighbor(
                    example_id=raw_examples[idx]["example_id"],
                    label=raw_examples[idx]["label"],
                    source=raw_examples[idx]["source"],
                    case_description=raw_examples[idx]["case_description"],
                    similarity_score=round(float(sims[idx]), 4),
                )
                for idx in top_k_indices
                if not np.isneginf(sims[idx])
            ]

            # Compute decomposed reliability score
            score = self.scoring_engine.calculate_score(
                neighbors=neighbors,
                proposed_decision=inst.proposed_decision,
                config=self.config,
            )

            # Apply operational gate policy
            decision, reason_code, _ = self.decision_engine.evaluate(
                score=score,
                neighbors=neighbors,
                proposed_decision=inst.proposed_decision,
                config=self.config,
            )

            predictions.append(
                InstancePrediction(
                    example_id=inst.example_id,
                    group_id=inst.group_id,
                    case_type=inst.case_type,
                    category=inst.category,
                    true_label=inst.true_label,
                    proposed_decision=inst.proposed_decision,
                    is_correct=inst.is_correct,
                    raw_score=score.raw_score,
                    agreement=score.agreement,
                    density_factor=score.density_factor,
                    coverage_factor=score.coverage_factor,
                    gate_decision=decision.value,
                    reason_code=reason_code,
                )
            )

        return predictions
