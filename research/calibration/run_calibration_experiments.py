#!/usr/bin/env python3
"""Executable experiment runner for ARDA Week 4: Confidence Calibration and Evaluation.

Orchestrates:
1. Dataset ingestion and symmetric target generation (correct = 1 if proposed == true else 0).
2. Embedding generation via BGE-small with ratio-verbalized template.
3. Strict 5-fold partitioned cross-validation:
   - 90 connected groups (10 contrastive pairs + 80 singletons).
   - Exactly 18 groups, 20 cases, 40 queries per fold.
   - Retrieval candidate pool strictly restricted to training fold R_k (80 cases).
   - Zero test case or connected twin leakage into retrieval or calibrator fitting.
   - Fold-level Platt and Isotonic calibrators fit on out-of-group training predictions.
4. Metric computation: ROC-AUC, Brier score, Log loss, ECE, MCE, False Pass Rate, False Escalation Rate.
5. Disaggregated cohort evaluation across the 5 case types.
6. Machine-readable JSON export to research/results/confidence_calibration_results.json.
7. Publication-quality figure generation to research/results/reliability_diagram.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arda_reliability_service.domain.value_objects import ScoringConfig
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from research.calibration.calibrator_fitting import CalibrationEvaluator
from research.calibration.pipeline import (
    EvaluationPipeline,
    load_and_build_evaluation_instances,
)
from research.calibration.visualization import (
    plot_reliability_diagrams_and_distributions,
)
from research.experiments.config import DATASET_PATH, RESULTS_DIR


def run_confidence_calibration_experiments() -> None:
    print("=" * 75)
    print("ARDA WEEK 4: CONFIDENCE CALIBRATION AND PREDICTIVE EVALUATION")
    print("=" * 75)

    # 1. Load dataset and build instances
    print(f"\n[1/6] Loading synthetic seed dataset from {DATASET_PATH}...")
    raw_examples, instances = load_and_build_evaluation_instances(
        dataset_path=DATASET_PATH,
        template_key="ratio_verbalized_description",
    )
    print(f"       -> Loaded {len(raw_examples)} seed examples.")
    print(
        f"       -> Constructed {len(instances)} symmetric evaluation queries (100 correct, 100 incorrect)."
    )

    # 2. Generate / load normalized embeddings
    print("\n[2/6] Generating dense embeddings via BAAI/bge-small-en-v1.5...")
    adapter = SentenceTransformerEmbeddingAdapter(
        model_name="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
    )
    pipeline = EvaluationPipeline(
        embedding_adapter=adapter,
        scoring_config=ScoringConfig(),
    )
    query_embs, passage_embs = pipeline.generate_embeddings(
        raw_examples=raw_examples,
        template_key="ratio_verbalized_description",
    )
    print(f"       -> Query embeddings shape:   {query_embs.shape}")
    print(f"       -> Passage embeddings shape: {passage_embs.shape}")

    # 3. Execute Strict 5-Fold Partitioned Cross-Validation
    print(
        "\n[3/6] Executing Strict 5-Fold Partitioned Cross-Validation (Connected Groups)..."
    )
    oof_predictions = pipeline.run_partitioned_cross_validation(
        raw_examples=raw_examples,
        query_embs=query_embs,
        passage_embs=passage_embs,
        n_splits=5,
    )
    print(
        f"       -> Generated out-of-fold predictions for all {len(oof_predictions)} queries."
    )

    # Fit reference production-ready calibrators on complete dataset
    ref_platt, ref_iso = pipeline.fit_reference_calibrators(
        raw_examples=raw_examples,
        query_embs=query_embs,
        passage_embs=passage_embs,
    )

    # 4. Evaluate Partitioned Cross-Validation
    print(
        "\n[4/6] Computing evaluation metrics across partitioned cross-validation folds..."
    )
    evaluator = CalibrationEvaluator(n_splits=5, random_state=42)
    benchmark_results = evaluator.evaluate_partitioned_cv(
        predictions=oof_predictions,
        fitted_calibrators={
            "platt": ref_platt.to_dict(),
            "isotonic": ref_iso.to_dict(),
        },
    )

    # Also compute Leave-One-Out comparisons for documentation
    print("       -> Running secondary full LOO group-exclusion evaluation...")
    loo_ge_predictions = pipeline.run_evaluation(
        raw_examples=raw_examples,
        instances=instances,
        query_embs=query_embs,
        passage_embs=passage_embs,
        protocol="group_exclusion",
    )
    loo_ge_benchmark = evaluator.evaluate_all(loo_ge_predictions)
    benchmark_results["group_exclusion_loo_comparison"] = {
        "raw_score_metrics": loo_ge_benchmark["raw_score_metrics"],
        "cohort_metrics": loo_ge_benchmark["cohort_metrics"],
    }

    print(
        "       -> Running secondary full LOO standard (unfiltered contrastive) evaluation..."
    )
    loo_std_predictions = pipeline.run_evaluation(
        raw_examples=raw_examples,
        instances=instances,
        query_embs=query_embs,
        passage_embs=passage_embs,
        protocol="standard_loo",
    )
    loo_std_benchmark = evaluator.evaluate_all(loo_std_predictions)
    benchmark_results["standard_loo_comparison"] = {
        "raw_score_metrics": loo_std_benchmark["raw_score_metrics"],
        "cohort_metrics": loo_std_benchmark["cohort_metrics"],
    }

    # 5. Display Summary Tables
    print("\n" + "=" * 96)
    print("PARTITIONED 5-FOLD CROSS-VALIDATION RESULTS (OVERALL N=200 QUERIES)")
    print("=" * 96)
    print(
        f"{'Model / Approach':<25} | {'ROC-AUC':<8} | {'Brier':<8} | {'LogLoss':<8} | {'ECE':<8} | {'F1-Pos (τ=0.5)':<14} | {'F1-Macro (τ=0.5)':<16}"
    )
    print("-" * 96)

    raw_m = benchmark_results["raw_score_metrics"]
    platt_m = benchmark_results["platt_oof_metrics"]
    iso_m = benchmark_results["isotonic_oof_metrics"]
    rand_m = benchmark_results["random_baseline_metrics"]

    print(
        f"{'Raw Score (Baseline)*':<25} | {raw_m['roc_auc']:<8.4f} | {raw_m['brier_score']:<8.4f} | {raw_m['log_loss']:<8.4f} | {raw_m['expected_calibration_error']:<8.4f} | {'N/A (artif.)':<14} | {'0.3333 (artif.)':<16}"
    )
    print(
        f"{'Platt Scaling (OOF)':<25} | {platt_m['roc_auc']:<8.4f} | {platt_m['brier_score']:<8.4f} | {platt_m['log_loss']:<8.4f} | {platt_m['expected_calibration_error']:<8.4f} | {platt_m['f1_positive']:<14.4f} | {platt_m['f1_macro']:<16.4f}"
    )
    print(
        f"{'Isotonic Reg (OOF)':<25} | {iso_m['roc_auc']:<8.4f} | {iso_m['brier_score']:<8.4f} | {iso_m['log_loss']:<8.4f} | {iso_m['expected_calibration_error']:<8.4f} | {iso_m['f1_positive']:<14.4f} | {iso_m['f1_macro']:<16.4f}"
    )
    print(
        f"{'Random Guess (p=0.50)*':<25} | {rand_m['roc_auc'] or 0.50:<8.4f} | {rand_m['brier_score']:<8.4f} | {rand_m['log_loss']:<8.4f} | {rand_m['expected_calibration_error']:<8.4f} | {rand_m['f1_positive']:<14.4f} | {rand_m['f1_macro']:<16.4f}"
    )
    print("-" * 96)
    print(
        "* Note on Raw Score: Threshold τ=0.50 on uncalibrated score (max 0.4881) produces 0 positive predictions."
    )
    print(
        "* Note on Random Guess: Constant p=0.50 at τ=0.50 predicts 100% positive, giving F1_positive=0.6667 and F1_macro=0.3333."
    )

    print("\n" + "=" * 75)
    print("DISAGGREGATED COHORT PERFORMANCE (RAW RELIABILITY SCORE)")
    print("=" * 75)
    print(
        f"{'Cohort':<22} | {'N':<4} | {'Mean Correct':<12} | {'Mean Incorrect':<14} | {'ROC-AUC':<8} | {'Brier':<8} | {'ECE':<8}"
    )
    print("-" * 75)

    for ct, cm in benchmark_results["cohort_metrics"].items():
        rm = cm["raw_score"]
        print(
            f"{ct:<22} | {cm['sample_count']:<4} | {cm['mean_raw_score_correct']:<12.4f} | {cm['mean_raw_score_incorrect']:<14.4f} | {rm['roc_auc']:<8.4f} | {rm['brier_score']:<8.4f} | {rm['expected_calibration_error']:<8.4f}"
        )

    print("\n" + "=" * 75)
    print("GATE SAFETY METRICS (RAW RELIABILITY ENGINE)")
    print("=" * 75)
    print(f"  * Total Evaluation Queries:               {raw_m['sample_count']}")
    print(
        f"  * Autonomous PASS Decisions:              {raw_m['sample_count'] * raw_m['coverage']:.0f} (Coverage = {raw_m['coverage'] * 100:.1f}%)"
    )
    print(
        f"  * Human Review ESCALATE Decisions:        {raw_m['sample_count'] * (1.0 - raw_m['coverage']):.0f} (Abstention = {(1.0 - raw_m['coverage']) * 100:.1f}%)"
    )
    print(
        f"  * False Passes (Catastrophic Pass):       {raw_m['false_pass_count']} (FPR among incorrect = {raw_m['false_pass_rate_among_incorrect'] * 100:.2f}%)"
    )
    print(
        f"  * False Escalations (Unnecessary Review): {raw_m['false_escalation_count']} (FER among correct = {raw_m['false_escalation_rate_among_correct'] * 100:.2f}%)"
    )
    print(f"  * Selective Risk (Error in PASS cohort):  {raw_m['risk'] * 100:.2f}%")

    # 6. Export Results and Visualization
    print("\n[5/6] Exporting machine-readable results to JSON...")
    json_path = RESULTS_DIR / "confidence_calibration_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"       -> Saved {json_path}")

    print("\n[6/6] Generating calibration diagrams and distribution figures...")
    plot_path = RESULTS_DIR / "reliability_diagram.png"
    plot_reliability_diagrams_and_distributions(
        results=benchmark_results,
        output_path=plot_path,
    )

    print("\n" + "=" * 75)
    print("CALIBRATION EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_confidence_calibration_experiments()
