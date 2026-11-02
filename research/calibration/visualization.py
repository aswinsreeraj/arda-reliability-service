"""Visualizations for confidence calibration, reliability diagrams, and cohort performance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagrams_and_distributions(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate comprehensive 4-panel calibration figure.

    Panel 1: Reliability Diagram (Calibration Curves with ECE)
    Panel 2: Confidence Score Distributions (Histograms)
    Panel 3: Selective Risk vs Coverage Tradeoff
    Panel 4: Disaggregated Cohort ROC-AUC Performance
    """
    _fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.26)

    raw_m = results["raw_score_metrics"]
    platt_m = results["platt_oof_metrics"]
    iso_m = results["isotonic_oof_metrics"]
    cohort_m = results["cohort_metrics"]

    # -------------------------------------------------------------
    # Panel 1: Reliability Diagram (Calibration Curves)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration (y = x)",
        alpha=0.8,
    )

    # Plot Bins for Raw Score
    raw_confs = [b["mean_confidence"] for b in raw_m["bins"] if b["sample_count"] > 0]
    raw_accs = [b["empirical_accuracy"] for b in raw_m["bins"] if b["sample_count"] > 0]
    if raw_confs:
        ax1.plot(
            raw_confs,
            raw_accs,
            marker="o",
            color="#d95f02",
            label=f"Raw Score (ECE={raw_m['expected_calibration_error']:.3f}, Brier={raw_m['brier_score']:.3f})",
            linewidth=1.8,
        )

    # Plot Bins for Platt Scaling
    platt_confs = [
        b["mean_confidence"] for b in platt_m["bins"] if b["sample_count"] > 0
    ]
    platt_accs = [
        b["empirical_accuracy"] for b in platt_m["bins"] if b["sample_count"] > 0
    ]
    if platt_confs:
        ax1.plot(
            platt_confs,
            platt_accs,
            marker="s",
            color="#1b9e77",
            label=f"Platt Scaling OOF (ECE={platt_m['expected_calibration_error']:.3f}, Brier={platt_m['brier_score']:.3f})",
            linewidth=1.8,
        )

    # Plot Bins for Isotonic Regression
    iso_confs = [b["mean_confidence"] for b in iso_m["bins"] if b["sample_count"] > 0]
    iso_accs = [b["empirical_accuracy"] for b in iso_m["bins"] if b["sample_count"] > 0]
    if iso_confs:
        ax1.plot(
            iso_confs,
            iso_accs,
            marker="^",
            color="#7570b3",
            label=f"Isotonic OOF (ECE={iso_m['expected_calibration_error']:.3f}, Brier={iso_m['brier_score']:.3f})",
            linewidth=1.8,
        )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax1.set_ylabel("Empirical Decision Accuracy", fontsize=11)
    ax1.set_title(
        "Reliability Diagram (Calibration Curves)", fontsize=12, fontweight="bold"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: Predicted Confidence Distribution
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    raw_scores = results["oof_predictions"]["raw_scores"]
    platt_probs = results["oof_predictions"]["platt_oof"]
    iso_probs = results["oof_predictions"]["isotonic_oof"]

    bins = np.linspace(0, 1, 21)
    ax2.hist(
        raw_scores,
        bins=bins,
        alpha=0.5,
        color="#d95f02",
        label="Raw Reliability Score",
        edgecolor="black",
        density=True,
    )
    ax2.hist(
        platt_probs,
        bins=bins,
        alpha=0.5,
        color="#1b9e77",
        label="Platt Calibrated Confidence",
        edgecolor="black",
        density=True,
    )
    ax2.hist(
        iso_probs,
        bins=bins,
        alpha=0.4,
        color="#7570b3",
        label="Isotonic Calibrated Confidence",
        edgecolor="black",
        density=True,
    )

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence Value", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.set_title("Confidence Value Distributions", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 3: Risk vs Coverage Across Operating Thresholds
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    thresholds = np.linspace(0.05, 0.95, 30)
    y = np.array(results["oof_predictions"]["labels"])
    raw_arr = np.array(raw_scores)

    coverages = []
    risks = []
    for th in thresholds:
        passed = raw_arr >= th
        cov = np.mean(passed)
        coverages.append(cov)
        if np.sum(passed) > 0:
            err_in_pass = np.mean(y[passed] == 0)
            risks.append(err_in_pass)
        else:
            risks.append(0.0)

    ax3.plot(
        coverages,
        risks,
        marker="o",
        color="#386cb0",
        linewidth=2,
        label="Raw Reliability Gating",
    )
    ax3.axhline(
        0.50,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="Unfiltered Error Rate (50%)",
    )

    ax3.set_xlim(0, 1.05)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_xlabel("Coverage (Autonomous Pass Fraction)", fontsize=11)
    ax3.set_ylabel("Risk (Error Rate in Passed Cohort)", fontsize=11)
    ax3.set_title(
        "Selective Gating: Risk vs Coverage Tradeoff", fontsize=12, fontweight="bold"
    )
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 4: Disaggregated Cohort ROC-AUC
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    cohort_names = list(cohort_m.keys())
    # Format readable cohort names
    display_names = [c.replace("_", "\n").title() for c in cohort_names]
    aucs = [cohort_m[c]["raw_score"]["roc_auc"] or 0.50 for c in cohort_names]

    colors = [
        "#4daf4a" if a >= 0.60 else "#e41a1c" if a < 0.50 else "#ff7f00" for a in aucs
    ]
    bars = ax4.bar(display_names, aucs, color=colors, edgecolor="black", width=0.55)
    ax4.axhline(
        0.50, color="gray", linestyle="--", linewidth=1.2, label="Random Guess (0.50)"
    )

    for bar, val in zip(bars, aucs):
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax4.set_ylim(0, 0.90)
    ax4.set_ylabel("ROC-AUC Score", fontsize=11)
    ax4.set_title(
        "Disaggregated Predictive Quality by Case Type", fontsize=12, fontweight="bold"
    )
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"  -> Successfully generated calibration visualization at {output_path}")
