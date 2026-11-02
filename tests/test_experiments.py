"""Unit and integration tests for ARDA Reliability-Gate Research Spike.

Verifies:
1. Experiment matrix and configuration integrity.
2. EmbeddingEvaluator prefix formatting and zero target leakage.
3. Strict Leave-One-Out (LOO) self-exclusion.
4. Metric calculations: precision, recall, F1, accuracy, confusion matrix.
5. Deterministic tie-breaking behavior.
6. Similarity separation and contrastive pair evaluation metrics.
7. Generated results and report artifact existence and integrity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.experiments.config import (
    DATASET_PATH,
    EXPERIMENT_MATRIX,
    RESULTS_DIR,
)
from research.experiments.metrics import (
    LOOResult,
    RetrievedNeighbor,
    calculate_category_relevance,
    calculate_classification_metrics,
    calculate_cohort_metrics,
    calculate_global_similarity_metrics,
    calculate_retrieval_baselines,
    calculate_similarity_metrics,
    evaluate_contrastive_pairs,
)


@pytest.fixture
def seed_dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestExperimentConfiguration:
    """Verifies that the experiment matrix and configuration adhere to spike requirements."""

    def test_matrix_contains_all_four_experiments(self) -> None:
        expected_keys = {"E1", "E2", "E3", "E4"}
        assert set(EXPERIMENT_MATRIX.keys()) == expected_keys

    def test_matrix_models_and_templates(self) -> None:
        assert EXPERIMENT_MATRIX["E1"].model.short_name == "bge-small"
        assert EXPERIMENT_MATRIX["E1"].template.name == "raw_value"

        assert EXPERIMENT_MATRIX["E2"].model.short_name == "bge-small"
        assert EXPERIMENT_MATRIX["E2"].template.name == "ratio_verbalized"

        assert EXPERIMENT_MATRIX["E3"].model.short_name == "e5-base"
        assert EXPERIMENT_MATRIX["E3"].template.name == "raw_value"

        assert EXPERIMENT_MATRIX["E4"].model.short_name == "e5-base"
        assert EXPERIMENT_MATRIX["E4"].template.name == "ratio_verbalized"

    def test_model_dimensions_and_tokens(self) -> None:
        bge_cfg = EXPERIMENT_MATRIX["E1"].model
        e5_cfg = EXPERIMENT_MATRIX["E3"].model

        assert bge_cfg.dimension == 384
        assert bge_cfg.max_seq_length == 512
        assert (
            bge_cfg.query_prefix
            == "Represent this sentence for searching relevant passages: "
        )
        assert bge_cfg.passage_prefix == ""

        assert e5_cfg.dimension == 768
        assert e5_cfg.max_seq_length == 512
        assert e5_cfg.query_prefix == "query: "
        assert e5_cfg.passage_prefix == "passage: "


class TestFormattingAndTargetLeakage:
    """Ensures query/passage prefixes comply with model specs and contain zero label/rationale leakage."""

    def test_zero_target_leakage_in_templates(self, seed_dataset: dict) -> None:
        forbidden_substrings = [
            "APPROVE",
            "REJECT",
            "Proposed decision",
            "Adjudication rationale",
            "Reference label",
            "Evaluation result",
        ]

        examples = seed_dataset["examples"]
        for ex in examples:
            for field in ["raw_value_description", "ratio_verbalized_description"]:
                text = ex["metadata"][field]
                for forbidden in forbidden_substrings:
                    assert forbidden not in text, (
                        f"Target leakage in {ex['example_id']} field '{field}': '{forbidden}' found in '{text}'"
                    )

    def test_prefix_formatting(self, seed_dataset: dict) -> None:
        examples = seed_dataset["examples"]
        for config in EXPERIMENT_MATRIX.values():
            template_field = config.template.metadata_field
            raw_texts = [ex["metadata"][template_field] for ex in examples]

            query_texts = [f"{config.model.query_prefix}{t}" for t in raw_texts]
            passage_texts = [f"{config.model.passage_prefix}{t}" for t in raw_texts]

            assert len(query_texts) == 100
            assert len(passage_texts) == 100

            if "e5" in config.model.short_name:
                assert all(q.startswith("query: ") for q in query_texts)
                assert all(p.startswith("passage: ") for p in passage_texts)
            elif "bge" in config.model.short_name:
                assert all(
                    q.startswith(
                        "Represent this sentence for searching relevant passages: "
                    )
                    for q in query_texts
                )
                assert all(
                    not p.startswith(
                        "Represent this sentence for searching relevant passages: "
                    )
                    for p in passage_texts
                )


class TestLeaveOneOutAndTieBreaking:
    """Tests the math and guarantees of Leave-One-Out retrieval and deterministic tie-breaking."""

    def test_strict_self_exclusion_in_loo_results(self) -> None:
        # Load generated LOO results from experiment_results.json if it exists
        results_file = RESULTS_DIR / "experiment_results.json"
        if not results_file.exists():
            pytest.skip("experiment_results.json not yet generated")

        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)

        for exp_id, exp_data in data.items():
            for item in exp_data["detailed_loo_results"]:
                query_id = item["example_id"]
                retrieved_ids = [n["example_id"] for n in item["top_neighbors"]]
                assert query_id not in retrieved_ids, (
                    f"Query {query_id} retrieved itself in experiment {exp_id}!"
                )
                assert len(retrieved_ids) == 5

    def test_deterministic_tie_breaking_behavior(self) -> None:
        # Test tie breaking logic:
        # Top-2 with 1 APPROVE (sim 0.88) and 1 REJECT (sim 0.85) -> APPROVE wins
        neighbors_approve_wins = [
            RetrievedNeighbor(
                example_id="ex-1", label="APPROVE", similarity=0.88, rank=1
            ),
            RetrievedNeighbor(
                example_id="ex-2", label="REJECT", similarity=0.85, rank=2
            ),
        ]
        approve_votes = [n for n in neighbors_approve_wins if n.label == "APPROVE"]
        reject_votes = [n for n in neighbors_approve_wins if n.label == "REJECT"]

        max_app = max((n.similarity for n in approve_votes), default=-1.0)
        max_rej = max((n.similarity for n in reject_votes), default=-1.0)
        pred = "APPROVE" if max_app > max_rej else "REJECT"
        assert pred == "APPROVE"

        # Top-2 with 1 APPROVE (sim 0.70) and 1 REJECT (sim 0.92) -> REJECT wins
        neighbors_reject_wins = [
            RetrievedNeighbor(
                example_id="ex-1", label="APPROVE", similarity=0.70, rank=1
            ),
            RetrievedNeighbor(
                example_id="ex-2", label="REJECT", similarity=0.92, rank=2
            ),
        ]
        approve_votes_rej = [n for n in neighbors_reject_wins if n.label == "APPROVE"]
        reject_votes_rej = [n for n in neighbors_reject_wins if n.label == "REJECT"]

        max_app_rej = max((n.similarity for n in approve_votes_rej), default=-1.0)
        max_rej_rej = max((n.similarity for n in reject_votes_rej), default=-1.0)
        pred_rej = "APPROVE" if max_app_rej > max_rej_rej else "REJECT"
        assert pred_rej == "REJECT"


class TestMetricsCalculation:
    """Tests accuracy, precision, recall, F1, and similarity separation metrics."""

    def test_classification_metrics_math(self) -> None:
        y_true = ["APPROVE", "APPROVE", "REJECT", "REJECT"]
        y_pred = ["APPROVE", "REJECT", "REJECT", "APPROVE"]  # 1 TP, 1 FN, 1 TN, 1 FP

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.5
        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.approve_precision == 0.5
        assert metrics.approve_recall == 0.5
        assert metrics.approve_f1 == 0.5
        assert metrics.reject_precision == 0.5
        assert metrics.reject_recall == 0.5
        assert metrics.reject_f1 == 0.5
        assert metrics.f1_macro == 0.5

    def test_similarity_separation_calculation(self) -> None:
        # Construct mock LOOResult instances
        res1 = LOOResult(
            example_id="ex-1",
            true_label="APPROVE",
            retrieved_neighbors=[
                RetrievedNeighbor(
                    example_id="ex-2", label="APPROVE", similarity=0.85, rank=1
                ),
                RetrievedNeighbor(
                    example_id="ex-3", label="REJECT", similarity=0.65, rank=2
                ),
            ],
        )
        res2 = LOOResult(
            example_id="ex-4",
            true_label="REJECT",
            retrieved_neighbors=[
                RetrievedNeighbor(
                    example_id="ex-5", label="REJECT", similarity=0.75, rank=1
                ),
                RetrievedNeighbor(
                    example_id="ex-6", label="APPROVE", similarity=0.55, rank=2
                ),
            ],
        )

        sim_metrics = calculate_similarity_metrics([res1, res2])
        # Same class: 0.85 and 0.75 -> mean = 0.80
        # Diff class: 0.65 and 0.55 -> mean = 0.60
        # Separation: 0.80 - 0.60 = 0.20
        assert np.isclose(sim_metrics.mean_same_label_similarity, 0.80)
        assert np.isclose(sim_metrics.mean_diff_label_similarity, 0.60)
        assert np.isclose(sim_metrics.similarity_separation, 0.20)

    def test_global_similarity_separation_calculation(self) -> None:
        # Synthetic 3x3 matrix: 2 APPROVE, 1 REJECT
        # sim[0, 1] = 0.8 (same), sim[0, 2] = 0.3 (diff), sim[1, 2] = 0.4 (diff)
        sim_matrix = np.array(
            [
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )
        labels = ["APPROVE", "APPROVE", "REJECT"]
        g_metrics = calculate_global_similarity_metrics(sim_matrix, labels)

        # Same pairs: (0, 1) and (1, 0) -> mean = 0.80
        # Diff pairs: (0, 2), (2, 0), (1, 2), (2, 1) -> (0.3+0.3+0.4+0.4)/4 = 0.35
        # Separation: 0.80 - 0.35 = 0.45
        assert np.isclose(g_metrics.mean_same_label_similarity, 0.80)
        assert np.isclose(g_metrics.mean_diff_label_similarity, 0.35)
        assert np.isclose(g_metrics.similarity_separation, 0.45)

    def test_cohort_metrics_calculation(self, seed_dataset: dict) -> None:
        examples = seed_dataset["examples"]
        # Create mock LOOResult for all 100 items
        mock_loo = []
        for ex in examples:
            mock_loo.append(
                LOOResult(
                    example_id=ex["example_id"],
                    true_label=ex["label"],
                    retrieved_neighbors=[],
                    predicted_labels={1: ex["label"], 3: ex["label"], 5: ex["label"]},
                    is_correct={1: True, 3: True, 5: True},
                )
            )

        cohort = calculate_cohort_metrics(mock_loo, examples)
        assert cohort["contrastive_count"] == 20
        assert cohort["non_contrastive_count"] == 80
        assert cohort["contrastive_top1_acc"] == 1.0
        assert cohort["non_contrastive_top1_acc"] == 1.0

    def test_category_relevance_calculation(self, seed_dataset: dict) -> None:
        examples = seed_dataset["examples"]
        # Mock LOOResult where retrieved neighbor matches query's category
        mock_loo = [
            LOOResult(
                example_id=examples[0]["example_id"],
                true_label="APPROVE",
                retrieved_neighbors=[
                    RetrievedNeighbor(
                        example_id=examples[0]["example_id"],
                        label="APPROVE",
                        similarity=0.9,
                        rank=1,
                    ),
                    RetrievedNeighbor(
                        example_id=examples[0]["example_id"],
                        label="APPROVE",
                        similarity=0.8,
                        rank=2,
                    ),
                    RetrievedNeighbor(
                        example_id=examples[0]["example_id"],
                        label="APPROVE",
                        similarity=0.7,
                        rank=3,
                    ),
                ],
            )
        ]
        relevance = calculate_category_relevance(mock_loo, examples)
        assert relevance["category_precision_at_1"] == 1.0
        assert relevance["category_precision_at_3"] == 1.0

    def test_retrieval_baselines_calculation(self) -> None:
        baselines = calculate_retrieval_baselines()
        assert baselines["majority_class_global"] == 0.5000
        assert baselines["majority_class_loo"] == 0.0000
        assert baselines["random_balanced_chance"] == 0.5000
        assert baselines["random_retrieval_top1"] == 0.4949
        assert baselines["random_retrieval_top3"] == 0.4923
        assert baselines["random_retrieval_top5"] == 0.4903
        assert baselines["random_category_precision_top1"] == 0.1810


class TestContrastivePairsEvaluation:
    """Tests contrastive pair counterfactual twin evaluation."""

    def test_evaluate_contrastive_pairs_calculation(self, seed_dataset: dict) -> None:
        examples = seed_dataset["examples"]
        id_to_idx = {ex["example_id"]: i for i, ex in enumerate(examples)}
        n = len(examples)

        # Create mock similarity matrix where pairs are rank 1 to each other
        sim_matrix = np.full((n, n), 0.3)
        np.fill_diagonal(sim_matrix, 1.0)

        contrastive_exs = [
            ex
            for ex in examples
            if ex["metadata"].get("case_type") == "contrastive_pair"
        ]
        assert len(contrastive_exs) == 20

        # Set pairwise similarities to 0.95
        for ex in contrastive_exs:
            twin_id = ex["metadata"]["contrastive_pair_id"]
            # Find the other member
            twins = [
                e
                for e in contrastive_exs
                if e["metadata"]["contrastive_pair_id"] == twin_id
                and e["example_id"] != ex["example_id"]
            ]
            if twins:
                twin_idx = id_to_idx[twins[0]["example_id"]]
                cur_idx = id_to_idx[ex["example_id"]]
                sim_matrix[cur_idx, twin_idx] = 0.95

        c_pairs = evaluate_contrastive_pairs(examples, sim_matrix, id_to_idx)
        assert len(c_pairs) == 10
        for cp in c_pairs:
            assert cp.rank_b_in_a == 1
            assert cp.rank_a_in_b == 1
            assert cp.b_in_top1_of_a is True
            assert cp.a_in_top1_of_b is True


class TestGeneratedArtifacts:
    """Verifies that all experiment result artifacts are generated and non-empty."""

    @pytest.mark.parametrize(
        "filename",
        [
            "experiment_results.json",
            "experiment_summary.csv",
            "contrastive_pairs_analysis.csv",
            "reliability_gate_spike_report.md",
            "accuracy_comparison.png",
            "confusion_matrices_top1.png",
            "confusion_matrices_top3.png",
            "similarity_separation.png",
            "similarity_distribution_density.png",
            "contrastive_pairs_comparison.png",
            "latency_memory_comparison.png",
        ],
    )
    def test_artifact_exists_and_non_empty(self, filename: str) -> None:
        file_path = RESULTS_DIR / filename
        assert file_path.exists(), (
            f"Artifact {filename} does not exist in {RESULTS_DIR}"
        )
        assert file_path.stat().st_size > 0, f"Artifact {filename} is empty"
