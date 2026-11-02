"""Plotting and visualization generators for ARDA Reliability-Gate Research Spike.

Produces:
- Confusion matrices for all four experiments
- Similarity distribution and separation plots
- Top-k retrieval accuracy comparisons
- Contrastive pair similarity comparisons
- Performance (latency and memory) visualizations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from research.experiments.metrics import ExperimentResult


def plot_accuracy_comparison(
    results: dict[str, ExperimentResult], output_path: Path
) -> None:
    """Generate grouped bar chart comparing Top-1, Top-3, and Top-5 accuracy across experiments."""
    exp_ids = list(results.keys())
    k_vals = [1, 3, 5]

    x = np.arange(len(exp_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, k in enumerate(k_vals):
        accs = [results[eid].k_metrics[k].accuracy * 100.0 for eid in exp_ids]
        offset = (i - 1) * width
        rects = ax.bar(x + offset, accs, width, label=f"Top-{k} Agreement", alpha=0.85)
        # Add labels above bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    labels = [
        f"{eid}\n{results[eid].model_name.split('/')[-1]}\n({results[eid].template_name})"
        for eid in exp_ids
    ]
    ax.set_ylabel("Accuracy / Label Agreement (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "ARDA Reliability Gate: LOO Retrieval Accuracy Across Experiments",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(
    results: dict[str, ExperimentResult], output_path: Path, k: int = 1
) -> None:
    """Generate 2x2 grid of confusion matrices for Top-k predictions."""
    exp_ids = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, eid in enumerate(exp_ids):
        ax = axes[idx]
        metrics = results[eid].k_metrics[k]
        cm = np.array(
            [
                [metrics.true_positives, metrics.false_negatives],
                [metrics.false_positives, metrics.true_negatives],
            ]
        )

        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(
            f"{eid}: {results[eid].model_name.split('/')[-1]} ({results[eid].template_name})\nTop-{k} Acc: {metrics.accuracy * 100:.1f}%",
            fontsize=11,
            fontweight="bold",
        )

        classes = ["APPROVE", "REJECT"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10)

        # Print cell counts and percentages
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm[i, j] / 50.0 * 100:.0f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=11,
                )

        ax.set_ylabel("True Reference Label", fontsize=10, fontweight="bold")
        ax.set_xlabel("Retrieved Majority Vote", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"ARDA Reliability Gate: Top-{k} Confusion Matrices (N=100 LOO Evaluation)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_similarity_separation(
    results: dict[str, ExperimentResult], output_path: Path
) -> None:
    """Generate bar chart comparing mean same-label vs different-label similarities and separation."""
    exp_ids = list(results.keys())
    x = np.arange(len(exp_ids))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 6))

    same_means = [
        results[eid].similarity_metrics.mean_same_label_similarity for eid in exp_ids
    ]
    same_stds = [
        results[eid].similarity_metrics.std_same_label_similarity for eid in exp_ids
    ]
    diff_means = [
        results[eid].similarity_metrics.mean_diff_label_similarity for eid in exp_ids
    ]
    diff_stds = [
        results[eid].similarity_metrics.std_diff_label_similarity for eid in exp_ids
    ]
    separations = [
        results[eid].similarity_metrics.similarity_separation for eid in exp_ids
    ]

    ax.bar(
        x - width / 2,
        same_means,
        width,
        yerr=same_stds,
        capsize=4,
        label="Same-Label Neighbors (Mean ± Std)",
        color="#2ca02c",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        diff_means,
        width,
        yerr=diff_stds,
        capsize=4,
        label="Different-Label Neighbors (Mean ± Std)",
        color="#d62728",
        alpha=0.85,
    )

    # Annotate separation delta
    for i, sep in enumerate(separations):
        max_h = max(same_means[i], diff_means[i])
        ax.annotate(
            f"Δ = {sep:+.3f}",
            xy=(x[i], max_h + 0.05),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "yellow", "alpha": 0.3},
        )

    labels = [
        f"{eid}\n{results[eid].model_name.split('/')[-1]}\n({results[eid].template_name})"
        for eid in exp_ids
    ]
    ax.set_ylabel("Cosine Similarity Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "ARDA Reliability Gate: Neighbor Similarity Separation (Same vs Different Label)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_contrastive_pairs_comparison(
    results: dict[str, ExperimentResult], output_path: Path
) -> None:
    """Generate bar chart comparing pairwise similarities across the 10 contrastive pairs."""
    exp_ids = list(results.keys())
    sample_pairs = results[exp_ids[0]].contrastive_pairs
    pair_labels = [p.pair_id.replace("pair_", "P").split("_")[0] for p in sample_pairs]

    x = np.arange(len(pair_labels))
    n_exps = len(exp_ids)
    width = 0.8 / n_exps

    fig, ax = plt.subplots(figsize=(13, 6))

    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

    for i, eid in enumerate(exp_ids):
        sims = [p.pairwise_similarity for p in results[eid].contrastive_pairs]
        offset = (i - (n_exps - 1) / 2) * width
        ax.bar(
            x + offset,
            sims,
            width,
            label=f"{eid} ({results[eid].model_name.split('/')[-1]}, {results[eid].template_name})",
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_ylabel("Pairwise Cosine Similarity", fontsize=11, fontweight="bold")
    ax.set_title(
        "ARDA Reliability Gate: Contrastive Pair Similarities Across Experiments",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latency_and_memory(
    results: dict[str, ExperimentResult], output_path: Path
) -> None:
    """Generate side-by-side plots for latency and memory usage."""
    exp_ids = list(results.keys())
    labels = [
        f"{eid}\n{results[eid].model_name.split('/')[-1]}\n({results[eid].template_name})"
        for eid in exp_ids
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Latency per example
    latencies = [results[eid].performance.per_example_embedding_ms for eid in exp_ids]
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]

    rects1 = ax1.bar(labels, latencies, color=colors, alpha=0.85, width=0.5)
    ax1.set_ylabel("Per-Example Embedding Latency (ms)", fontsize=10, fontweight="bold")
    ax1.set_title("Embedding Generation Latency", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.1f} ms",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Memory usage
    mems = [results[eid].performance.peak_memory_mb for eid in exp_ids]
    rects2 = ax2.bar(labels, mems, color=colors, alpha=0.85, width=0.5)
    ax2.set_ylabel("Peak Memory (MB)", fontsize=10, fontweight="bold")
    ax2.set_title("Peak Model Memory Footprint", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(
            f"{height:.1f} MB",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
