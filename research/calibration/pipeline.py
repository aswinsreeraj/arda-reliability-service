"""Evaluation pipeline for reliability scoring and confidence calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """Executes vector retrieval, reliability scoring, and gating across evaluation instances."""

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

    def run_evaluation(
        self,
        raw_examples: list[dict[str, Any]],
        instances: list[EvaluationInstance],
        query_embs: np.ndarray,
        passage_embs: np.ndarray,
        protocol: str = "group_exclusion",
    ) -> list[InstancePrediction]:
        """Execute evaluation queries against reference candidates.

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
