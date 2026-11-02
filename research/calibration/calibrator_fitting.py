"""Calibration model fitting, cross-validation, and comparative evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from arda_reliability_service.domain.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
)
from research.calibration.metrics import (
    CalibrationEvaluationMetrics,
    calculate_metrics,
)
from research.calibration.pipeline import InstancePrediction


class CalibrationEvaluator:
    """Evaluates uncalibrated raw scores against Platt scaling and Isotonic regression.

    Strictly enforces out-of-fold (OOF) cross-validation so calibration models
    are never evaluated on the same predictions they were fitted on.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate_all(
        self,
        predictions: list[InstancePrediction],
        decision_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Run complete comparative calibration benchmark.

        Returns:
            Dictionary containing metrics for:
            - 'raw_score': Uncalibrated raw reliability score baseline
            - 'platt_oof': Out-of-fold Platt scaling
            - 'isotonic_oof': Out-of-fold Isotonic regression
            - 'random_baseline': Uniform random baseline
            - 'fitted_models': Fully fitted serializable calibrator artifacts
        """
        X_raw = np.array([p.raw_score for p in predictions], dtype=np.float64)
        y = np.array([p.is_correct for p in predictions], dtype=np.int32)
        groups = np.array([p.group_id for p in predictions])
        gate_decisions = [p.gate_decision for p in predictions]

        # 1. Uncalibrated Raw Score Metrics
        raw_metrics = calculate_metrics(
            y_true=y,
            y_pred_score=X_raw,
            gate_decisions=gate_decisions,
            decision_threshold=decision_threshold,
        )

        # 2. Cross-Validated Out-of-Fold Calibration
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        platt_oof = np.zeros(len(y), dtype=np.float64)
        iso_oof = np.zeros(len(y), dtype=np.float64)

        X_2d = X_raw.reshape(-1, 1)

        for train_idx, test_idx in sgkf.split(X_2d, y, groups):
            X_tr, y_tr = X_2d[train_idx], y[train_idx]
            X_te = X_2d[test_idx]

            # Platt Scaling (Logistic Regression on raw score)
            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(X_tr, y_tr)
            platt_oof[test_idx] = lr.predict_proba(X_te)[:, 1]

            # Isotonic Regression (monotonic step function)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(X_tr.ravel(), y_tr)
            iso_oof[test_idx] = iso.predict(X_te.ravel())

        # Platt OOF Metrics
        platt_metrics = calculate_metrics(
            y_true=y,
            y_pred_score=platt_oof,
            gate_decisions=gate_decisions,
            decision_threshold=decision_threshold,
        )

        # Isotonic OOF Metrics
        iso_metrics = calculate_metrics(
            y_true=y,
            y_pred_score=iso_oof,
            gate_decisions=gate_decisions,
            decision_threshold=decision_threshold,
        )

        # 3. Random Guessing Baseline (predicts constant p = 0.50)
        random_probs = np.full(len(y), 0.50, dtype=np.float64)
        random_metrics = calculate_metrics(
            y_true=y,
            y_pred_score=random_probs,
            gate_decisions=gate_decisions,
            decision_threshold=decision_threshold,
        )

        # 4. Fit production-ready versioned calibrators on complete dataset
        # (for export and persistence)
        full_lr = LogisticRegression(C=1.0, solver="lbfgs")
        full_lr.fit(X_2d, y)
        platt_calibrator = PlattCalibrator(
            a=float(full_lr.coef_[0][0]),
            b=float(full_lr.intercept_[0]),
            version="platt-v1.0.0-seed100",
            metrics={
                "brier_score_oof": platt_metrics.brier_score,
                "ece_oof": platt_metrics.expected_calibration_error,
                "log_loss_oof": platt_metrics.log_loss,
                "roc_auc_oof": platt_metrics.roc_auc,
                "n_samples": len(y),
            },
        )

        full_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        full_iso.fit(X_raw, y)
        iso_calibrator = IsotonicCalibrator(
            x_knots=list(map(float, full_iso.X_thresholds_)),
            y_knots=list(map(float, full_iso.y_thresholds_)),
            version="isotonic-v1.0.0-seed100",
            metrics={
                "brier_score_oof": iso_metrics.brier_score,
                "ece_oof": iso_metrics.expected_calibration_error,
                "log_loss_oof": iso_metrics.log_loss,
                "roc_auc_oof": iso_metrics.roc_auc,
                "n_samples": len(y),
            },
        )

        # 5. Cohort-Level Evaluation (disaggregated across case types)
        cohort_metrics = self.evaluate_cohorts(
            predictions=predictions,
            platt_oof=platt_oof,
            iso_oof=iso_oof,
        )

        return {
            "raw_score_metrics": raw_metrics.to_dict(),
            "platt_oof_metrics": platt_metrics.to_dict(),
            "isotonic_oof_metrics": iso_metrics.to_dict(),
            "random_baseline_metrics": random_metrics.to_dict(),
            "cohort_metrics": cohort_metrics,
            "fitted_calibrators": {
                "platt": platt_calibrator.to_dict(),
                "isotonic": iso_calibrator.to_dict(),
            },
            "oof_predictions": {
                "raw_scores": [float(x) for x in X_raw],
                "labels": [int(l) for l in y],
                "platt_oof": [float(p) for p in platt_oof],
                "isotonic_oof": [float(p) for p in iso_oof],
            },
        }

    def evaluate_cohorts(
        self,
        predictions: list[InstancePrediction],
        platt_oof: np.ndarray,
        iso_oof: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        """Calculate disaggregated metrics per case type cohort."""
        cohort_groups: dict[str, list[int]] = {}
        for idx, p in enumerate(predictions):
            ct = p.case_type
            if ct not in cohort_groups:
                cohort_groups[ct] = []
            cohort_groups[ct].append(idx)

        cohort_results: dict[str, dict[str, Any]] = {}
        y_all = np.array([p.is_correct for p in predictions], dtype=np.int32)
        raw_all = np.array([p.raw_score for p in predictions], dtype=np.float64)
        gate_all = [p.gate_decision for p in predictions]

        for ct, indices in cohort_groups.items():
            y_sub = y_all[indices]
            raw_sub = raw_all[indices]
            platt_sub = platt_oof[indices]
            iso_sub = iso_oof[indices]
            gate_sub = [gate_all[i] for i in indices]

            raw_m: CalibrationEvaluationMetrics = calculate_metrics(
                y_sub, raw_sub, gate_decisions=gate_sub
            )
            platt_m: CalibrationEvaluationMetrics = calculate_metrics(
                y_sub, platt_sub, gate_decisions=gate_sub
            )
            iso_m: CalibrationEvaluationMetrics = calculate_metrics(
                y_sub, iso_sub, gate_decisions=gate_sub
            )

            c_scores = [raw_all[i] for i in indices if y_all[i] == 1]
            inc_scores = [raw_all[i] for i in indices if y_all[i] == 0]

            cohort_results[ct] = {
                "sample_count": len(indices),
                "mean_raw_score_correct": round(float(np.mean(c_scores)), 4)
                if c_scores
                else 0.0,
                "mean_raw_score_incorrect": round(float(np.mean(inc_scores)), 4)
                if inc_scores
                else 0.0,
                "raw_score": raw_m.to_dict(),
                "platt_oof": platt_m.to_dict(),
                "isotonic_oof": iso_m.to_dict(),
            }

        return cohort_results
