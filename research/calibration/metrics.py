"""Evaluation and calibration metrics for ARDA Reliability Service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinMetrics:
    """Statistics for a single confidence bin in a reliability diagram."""

    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_error: float


@dataclass(frozen=True)
class CalibrationEvaluationMetrics:
    """Comprehensive evaluation metrics assessing predictive quality and calibration."""

    sample_count: int
    positive_count: int
    negative_count: int
    roc_auc: float | None
    precision: float
    recall: float
    f1: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    false_pass_count: int
    false_pass_rate_among_incorrect: float  # FP / (FP + TN)
    false_pass_ratio_among_passes: float  # FP / (TP + FP) if (TP + FP) > 0 else 0.0
    false_escalation_count: int
    false_escalation_rate_among_correct: float  # FN / (TP + FN)
    coverage: float  # Total PASS / N
    risk: float  # FP / Total PASS if Total PASS > 0 else 0.0
    bins: list[BinMetrics]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, list[BinMetrics]]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Uses M uniform bins over [0.0, 1.0]:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|

    Args:
        probabilities: Predicted confidence / probabilities in [0.0, 1.0].
        labels: Binary ground-truth correctness targets (1 or 0).
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        Tuple of (ece, mce, bin_metrics_list).
    """
    probs_arr = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    labels_arr = np.asarray(labels, dtype=np.int32)
    total_samples = len(probs_arr)

    if total_samples == 0:
        return 0.0, 0.0, []

    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    bin_metrics: list[BinMetrics] = []
    weighted_ece = 0.0
    max_ce = 0.0

    for i in range(n_bins):
        b_low = float(bin_limits[i])
        b_high = float(bin_limits[i + 1])

        if i == n_bins - 1:
            in_bin = (probs_arr >= b_low) & (probs_arr <= b_high)
        else:
            in_bin = (probs_arr >= b_low) & (probs_arr < b_high)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(probs_arr[in_bin]))
            emp_acc = float(np.mean(labels_arr[in_bin]))
            error = abs(emp_acc - mean_conf)
            weighted_ece += (count / total_samples) * error
            max_ce = max(max_ce, error)
        else:
            mean_conf = (b_low + b_high) / 2.0
            emp_acc = 0.0
            error = 0.0

        bin_metrics.append(
            BinMetrics(
                bin_index=i + 1,
                bin_lower=round(b_low, 3),
                bin_upper=round(b_high, 3),
                sample_count=count,
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(emp_acc, 4),
                calibration_error=round(error, 4),
            )
        )

    return round(weighted_ece, 4), round(max_ce, 4), bin_metrics


def calculate_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred_score: Sequence[float] | np.ndarray,
    gate_decisions: Sequence[str] | None = None,
    decision_threshold: float = 0.5,
    n_bins: int = 10,
) -> CalibrationEvaluationMetrics:
    """Calculate comprehensive predictive, calibration, and safety gating metrics.

    Args:
        y_true: Ground truth binary correctness targets (1 = decision was correct, 0 = incorrect).
        y_pred_score: Continuous score or probability in [0.0, 1.0].
        gate_decisions: Optional operational decisions ('PASS' or 'ESCALATE'). If None,
                        derived by thresholding y_pred_score >= decision_threshold.
        decision_threshold: Threshold to convert score to binary prediction (default 0.5).
        n_bins: Number of bins for ECE calculation (default 10).

    Returns:
        CalibrationEvaluationMetrics object.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.clip(np.asarray(y_pred_score, dtype=np.float64), 0.0, 1.0)
    n = len(y_true_arr)

    pos_count = int(np.sum(y_true_arr == 1))
    neg_count = int(np.sum(y_true_arr == 0))

    # ROC-AUC (requires at least one sample from each class)
    if pos_count > 0 and neg_count > 0:
        roc_auc = float(round(roc_auc_score(y_true_arr, y_score_arr), 4))
    else:
        roc_auc = None

    # Binary classification metrics at decision threshold
    binary_preds = (y_score_arr >= decision_threshold).astype(int)
    prec = float(round(precision_score(y_true_arr, binary_preds, zero_division=0.0), 4))
    rec = float(round(recall_score(y_true_arr, binary_preds, zero_division=0.0), 4))
    f1 = float(round(f1_score(y_true_arr, binary_preds, zero_division=0.0), 4))

    # Scoring rules
    brier = float(round(brier_score_loss(y_true_arr, y_score_arr), 4))

    # Log loss requires probabilities strictly in (0, 1)
    clipped_probs = np.clip(y_score_arr, 1e-6, 1.0 - 1e-6)
    if pos_count > 0 and neg_count > 0:
        ll = float(round(log_loss(y_true_arr, clipped_probs), 4))
    else:
        ll = 0.0

    # Calibration error
    ece, mce, bins = compute_expected_calibration_error(
        y_score_arr, y_true_arr, n_bins=n_bins
    )

    # Gate decision safety metrics
    # Decision semantics:
    # PASS: autonomous execution
    # ESCALATE: human review
    if gate_decisions is not None:
        pass_mask = np.array([d == "PASS" for d in gate_decisions])
    else:
        pass_mask = binary_preds == 1

    escalate_mask = ~pass_mask

    # False pass: Gate passed an INCORRECT decision (y_true == 0 and passed)
    false_pass_count = int(np.sum((y_true_arr == 0) & pass_mask))
    total_incorrect = max(1, neg_count)
    fp_rate = round(false_pass_count / total_incorrect, 4)

    total_passes = int(np.sum(pass_mask))
    fp_ratio_among_passes = (
        round(false_pass_count / total_passes, 4) if total_passes > 0 else 0.0
    )

    # False escalation: Gate escalated a CORRECT decision (y_true == 1 and escalated)
    false_escalation_count = int(np.sum((y_true_arr == 1) & escalate_mask))
    total_correct = max(1, pos_count)
    fe_rate = round(false_escalation_count / total_correct, 4)

    # Selective classification coverage & risk
    coverage = round(total_passes / max(1, n), 4)
    risk = fp_ratio_among_passes

    return CalibrationEvaluationMetrics(
        sample_count=n,
        positive_count=pos_count,
        negative_count=neg_count,
        roc_auc=roc_auc,
        precision=prec,
        recall=rec,
        f1=f1,
        brier_score=brier,
        log_loss=ll,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        false_pass_count=false_pass_count,
        false_pass_rate_among_incorrect=fp_rate,
        false_pass_ratio_among_passes=fp_ratio_among_passes,
        false_escalation_count=false_escalation_count,
        false_escalation_rate_among_correct=fe_rate,
        coverage=coverage,
        risk=risk,
        bins=bins,
    )
