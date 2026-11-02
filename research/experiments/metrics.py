"""Evaluation metrics for ARDA Reliability-Gate Leave-One-Out Retrieval experiments.

Calculates:
- Top-k retrieval accuracy and agreement
- Precision, Recall, F1 (per-class and macro)
- Confusion matrices
- Mean same-label vs different-label cosine similarity and separation
- Contrastive pair mutual similarity, ranks, and neighbor collision analysis
- Inference latency and peak memory usage
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class RetrievedNeighbor:
    """Individual retrieved reference neighbor."""

    example_id: str
    label: str
    similarity: float
    rank: int


@dataclass
class LOOResult:
    """Retrieval evaluation outcome for a single query example."""

    example_id: str
    true_label: str
    retrieved_neighbors: list[RetrievedNeighbor]
    predicted_labels: dict[int, str] = field(default_factory=dict)
    is_correct: dict[int, bool] = field(default_factory=dict)


@dataclass
class ClassificationMetrics:
    """Standard classification and agreement metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    approve_precision: float
    approve_recall: float
    approve_f1: float
    reject_precision: float
    reject_recall: float
    reject_f1: float
    true_positives: int  # True APPROVE, Pred APPROVE
    true_negatives: int  # True REJECT, Pred REJECT
    false_positives: int  # True REJECT, Pred APPROVE
    false_negatives: int  # True APPROVE, Pred REJECT


@dataclass
class SimilarityMetrics:
    """Distributional similarity metrics evaluating embedding separation."""

    mean_same_label_similarity: float
    std_same_label_similarity: float
    mean_diff_label_similarity: float
    std_diff_label_similarity: float
    similarity_separation: float  # mean(same) - mean(diff)


@dataclass
class ContrastivePairMetric:
    """Evaluation of model behavior on a contrastive pair."""

    pair_id: str
    example_a_id: str
    example_b_id: str
    label_a: str
    label_b: str
    pairwise_similarity: float
    rank_b_in_a: int
    rank_a_in_b: int
    b_in_top1_of_a: bool
    b_in_top3_of_a: bool
    b_in_top5_of_a: bool
    a_in_top1_of_b: bool
    a_in_top3_of_b: bool
    a_in_top5_of_b: bool


@dataclass
class PerformanceMetrics:
    """Telemetry metrics covering latency and compute footprint."""

    model_load_time_seconds: float
    embedding_generation_seconds: float
    per_example_embedding_ms: float
    per_query_retrieval_ms: float
    total_pipeline_seconds: float
    peak_memory_mb: float
    inference_device: str


@dataclass
class ExperimentResult:
    """Aggregated output for a single experiment run."""

    experiment_id: str
    model_name: str
    template_name: str
    total_examples: int
    k_metrics: dict[int, ClassificationMetrics]
    similarity_metrics: SimilarityMetrics
    contrastive_pairs: list[ContrastivePairMetric]
    performance: PerformanceMetrics
    detailed_loo_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result object into a serializable dictionary."""
        return asdict(self)


def calculate_classification_metrics(
    y_true: list[str], y_pred: list[str]
) -> ClassificationMetrics:
    """Compute precision, recall, F1, and confusion matrix values for binary APPROVE/REJECT."""
    assert len(y_true) == len(y_pred), "Mismatched label lengths"
    n = len(y_true)
    assert n > 0, "Empty label set"

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "APPROVE" and yp == "APPROVE")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REJECT" and yp == "REJECT")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REJECT" and yp == "APPROVE")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "APPROVE" and yp == "REJECT")

    accuracy = (tp + tn) / n

    # APPROVE metrics (Positive class)
    app_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    app_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    app_f1 = (
        (2 * app_prec * app_rec) / (app_prec + app_rec)
        if (app_prec + app_rec) > 0
        else 0.0
    )

    # REJECT metrics (Negative class)
    rej_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rej_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rej_f1 = (
        (2 * rej_prec * rej_rec) / (rej_prec + rej_rec)
        if (rej_prec + rej_rec) > 0
        else 0.0
    )

    prec_macro = (app_prec + rej_prec) / 2.0
    rec_macro = (app_rec + rej_rec) / 2.0
    f1_macro = (app_f1 + rej_f1) / 2.0

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        precision_macro=round(prec_macro, 4),
        recall_macro=round(rec_macro, 4),
        f1_macro=round(f1_macro, 4),
        approve_precision=round(app_prec, 4),
        approve_recall=round(app_rec, 4),
        approve_f1=round(app_f1, 4),
        reject_precision=round(rej_prec, 4),
        reject_recall=round(rej_rec, 4),
        reject_f1=round(rej_f1, 4),
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )


def calculate_similarity_metrics(loo_results: list[LOOResult]) -> SimilarityMetrics:
    """Compute distributional similarity separation across top retrieved neighbors."""
    same_sims: list[float] = []
    diff_sims: list[float] = []

    for res in loo_results:
        q_label = res.true_label
        for neighbor in res.retrieved_neighbors:
            if neighbor.label == q_label:
                same_sims.append(neighbor.similarity)
            else:
                diff_sims.append(neighbor.similarity)

    same_mean = float(np.mean(same_sims)) if same_sims else 0.0
    same_std = float(np.std(same_sims)) if same_sims else 0.0
    diff_mean = float(np.mean(diff_sims)) if diff_sims else 0.0
    diff_std = float(np.std(diff_sims)) if diff_sims else 0.0
    separation = same_mean - diff_mean

    return SimilarityMetrics(
        mean_same_label_similarity=round(same_mean, 4),
        std_same_label_similarity=round(same_std, 4),
        mean_diff_label_similarity=round(diff_mean, 4),
        std_diff_label_similarity=round(diff_std, 4),
        similarity_separation=round(separation, 4),
    )


def evaluate_contrastive_pairs(
    dataset_examples: list[dict[str, Any]],
    similarity_matrix: np.ndarray,
    id_to_idx: dict[str, int],
) -> list[ContrastivePairMetric]:
    """Evaluate pairwise similarity and nearest neighbor ranks for all contrastive pairs."""
    pair_map: dict[str, list[dict[str, Any]]] = {}
    for ex in dataset_examples:
        meta = ex["metadata"]
        if meta.get("case_type") == "contrastive_pair":
            pid = meta.get("contrastive_pair_id")
            if pid:
                pair_map.setdefault(pid, []).append(ex)

    results: list[ContrastivePairMetric] = []
    for pid, pair_exs in sorted(pair_map.items()):
        if len(pair_exs) != 2:
            continue
        ex_a, ex_b = pair_exs[0], pair_exs[1]
        idx_a, idx_b = id_to_idx[ex_a["example_id"]], id_to_idx[ex_b["example_id"]]

        # Pairwise cosine similarity between A and B
        sim_ab = float(similarity_matrix[idx_a, idx_b])

        # Ranks in LOO retrieval (excluding self):
        # Sort all j != idx_a by similarity descending
        sims_from_a = [
            (j, similarity_matrix[idx_a, j])
            for j in range(len(dataset_examples))
            if j != idx_a
        ]
        sims_from_a.sort(key=lambda x: x[1], reverse=True)
        rank_b_in_a = next(i + 1 for i, (j, _) in enumerate(sims_from_a) if j == idx_b)

        sims_from_b = [
            (j, similarity_matrix[idx_b, j])
            for j in range(len(dataset_examples))
            if j != idx_b
        ]
        sims_from_b.sort(key=lambda x: x[1], reverse=True)
        rank_a_in_b = next(i + 1 for i, (j, _) in enumerate(sims_from_b) if j == idx_a)

        metric = ContrastivePairMetric(
            pair_id=pid,
            example_a_id=ex_a["example_id"],
            example_b_id=ex_b["example_id"],
            label_a=ex_a["label"],
            label_b=ex_b["label"],
            pairwise_similarity=round(sim_ab, 4),
            rank_b_in_a=rank_b_in_a,
            rank_a_in_b=rank_a_in_b,
            b_in_top1_of_a=(rank_b_in_a <= 1),
            b_in_top3_of_a=(rank_b_in_a <= 3),
            b_in_top5_of_a=(rank_b_in_a <= 5),
            a_in_top1_of_b=(rank_a_in_b <= 1),
            a_in_top3_of_b=(rank_a_in_b <= 3),
            a_in_top5_of_b=(rank_a_in_b <= 5),
        )
        results.append(metric)

    return results
