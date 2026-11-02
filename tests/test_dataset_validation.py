"""Unit and integration tests for ARDA Reliability Service synthetic reference dataset.

Verifies:
1. File existence and schema validation.
2. Exact 100 examples count.
3. Strict 50/50 balanced label distribution.
4. Single tenant and single agent scoping.
5. Case typology coverage (straightforward, borderline, incomplete evidence, conflicting signals, contrastive pairs).
6. Critical embedding isolation (strictly no label leakage or rationale embedding in case_description).
7. Contrastive pair symmetry and outcome inversion.
8. Template variation attributes (raw-value and ratio-verbalized descriptions).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for direct imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from research.datasets.generate_dataset import (
    build_ratio_template,
    build_raw_template,
    compute_ratio_features,
    generate_dataset,
    save_dataset,
)
from research.datasets.validate_dataset import (
    DatasetValidationError,
    validate_dataset_file,
    validate_embedding_isolation,
)

DATASET_DIR = Path(__file__).resolve().parent.parent / "research" / "datasets"
DATASET_PATH = DATASET_DIR / "synthetic_seed_examples.json"
SCHEMA_PATH = DATASET_DIR / "dataset_schema.json"


@pytest.fixture(scope="session")
def dataset_data() -> dict[str, Any]:
    """Load and parse the synthetic reference dataset."""
    assert DATASET_PATH.exists(), f"Dataset file missing at {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def examples(dataset_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of 100 reference examples."""
    return dataset_data["examples"]


# ---------------------------------------------------------------------------
# Test Group 1: Dataset Structure and Top-level Invariants
# ---------------------------------------------------------------------------


def test_dataset_files_exist():
    """Verify that dataset and schema files exist in the expected directory."""
    assert DATASET_PATH.is_file(), f"Missing dataset file: {DATASET_PATH}"
    assert SCHEMA_PATH.is_file(), f"Missing schema file: {SCHEMA_PATH}"


def test_top_level_metadata(dataset_data: dict[str, Any]):
    """Verify top-level dataset metadata attributes."""
    assert dataset_data["dataset_name"] == "arda_reliability_synthetic_seed_v1"
    assert dataset_data["dataset_version"] == "1.0.0"
    assert "Financial Transaction Dispute" in dataset_data["decision_domain"]
    assert dataset_data["total_examples"] == 100
    assert dataset_data["label_distribution"] == {"APPROVE": 50, "REJECT": 50}


def test_exact_example_count(examples: list[dict[str, Any]]):
    """Quality Check 1: Exactly 100 reference examples must exist."""
    assert len(examples) == 100, f"Expected exactly 100 examples, found {len(examples)}"


def test_unique_example_ids(examples: list[dict[str, Any]]):
    """Quality Check 2: Every example must possess a unique identifier."""
    ids = [ex["example_id"] for ex in examples]
    assert len(ids) == len(set(ids)), "Duplicate example_id values detected"
    for i, ex_id in enumerate(ids, start=1):
        assert ex_id == f"ref-example-{i:03d}", (
            f"Unexpected ID format for index {i}: {ex_id}"
        )


def test_valid_labels_and_distribution(examples: list[dict[str, Any]]):
    """Quality Checks 3 & 6: Valid labels and exact 50/50 balance."""
    labels = [ex["label"] for ex in examples]
    valid_labels = {"APPROVE", "REJECT"}
    for label in labels:
        assert label in valid_labels, f"Invalid label: {label}"

    approve_count = labels.count("APPROVE")
    reject_count = labels.count("REJECT")
    assert approve_count == 50, f"Expected 50 APPROVE, got {approve_count}"
    assert reject_count == 50, f"Expected 50 REJECT, got {reject_count}"


def test_single_tenant_and_agent_scope(
    examples: list[dict[str, Any]], dataset_data: dict[str, Any]
):
    """Verify single tenant and agent scope across all examples."""
    expected_tenant = dataset_data["tenant_id"]
    expected_agent = dataset_data["agent_id"]

    assert expected_tenant == "00000000-0000-0000-0000-000000000001"
    assert expected_agent == "00000000-0000-0000-0000-000000000002"

    for ex in examples:
        assert ex["tenant_id"] == expected_tenant, (
            f"Mismatched tenant_id in {ex['example_id']}"
        )
        assert ex["agent_id"] == expected_agent, (
            f"Mismatched agent_id in {ex['example_id']}"
        )
        assert ex["source"] == "synthetic", f"Invalid source in {ex['example_id']}"
        assert ex["status"] == "active", f"Invalid status in {ex['example_id']}"


def test_required_fields_present(examples: list[dict[str, Any]]):
    """Quality Check 4: No required fields are missing from any example."""
    required_fields = {
        "example_id",
        "tenant_id",
        "agent_id",
        "case_description",
        "label",
        "source",
        "status",
        "metadata",
        "rationale",
    }
    for ex in examples:
        missing = required_fields - set(ex.keys())
        assert not missing, f"Missing fields in {ex.get('example_id')}: {missing}"


# ---------------------------------------------------------------------------
# Test Group 2: Descriptions and Critical Embedding Separation
# ---------------------------------------------------------------------------


def test_descriptions_non_empty_and_diverse(examples: list[dict[str, Any]]):
    """Quality Checks 5 & 7: Descriptions are non-empty, substantial, and unique."""
    descriptions = [ex["case_description"].strip() for ex in examples]

    # Non-empty and substantial
    for i, desc in enumerate(descriptions):
        assert len(desc) >= 50, (
            f"Example {i + 1} has an overly short description ({len(desc)} chars)"
        )

    # No accidental duplicates
    normalized_descs = [" ".join(d.lower().split()) for d in descriptions]
    assert len(normalized_descs) == len(set(normalized_descs)), (
        "Duplicate case descriptions detected"
    )


def test_inspectable_rationale_present(examples: list[dict[str, Any]]):
    """Quality Check 8: Every example has an auditable, inspectable rationale."""
    for ex in examples:
        rationale = ex["rationale"].strip()
        assert len(rationale) >= 20, (
            f"Rationale in {ex['example_id']} is too brief: '{rationale}'"
        )
        assert ex["example_id"] not in rationale, (
            "Rationale should not redundantly contain example ID"
        )


def test_critical_embedding_isolation(examples: list[dict[str, Any]]):
    """Quality Check 9: ONLY case_description must be used for embeddings.

    Verify that NO proposed decision, reference label, expected outcome, or rationale
    text leaks into case_description.
    """
    for ex in examples:
        case_desc = ex["case_description"]
        label = ex["label"]
        rationale = ex["rationale"]
        ex_id = ex["example_id"]

        # 1. No standalone label keywords
        assert not re.search(r"\b(APPROVE|REJECT)\b", case_desc), (
            f"Explicit decision token found in case_description of {ex_id}: {case_desc}"
        )

        # 2. No evaluation markers
        forbidden_substrings = [
            "PROPOSED DECISION",
            "EXPECTED OUTCOME",
            "EVALUATION RESULT",
            "LABEL:",
            "DECISION:",
            "OUTCOME:",
        ]
        for sub in forbidden_substrings:
            assert sub not in case_desc.upper(), (
                f"Forbidden evaluation marker '{sub}' in {ex_id}"
            )

        # 3. Rationale isolation
        validate_embedding_isolation(case_desc, label, rationale, ex_id)


def test_embedding_isolation_rejects_leakage():
    """Verify that validate_embedding_isolation raises DatasetValidationError on synthetic leaks."""
    # Label leakage
    with pytest.raises(DatasetValidationError, match="explicit label token"):
        validate_embedding_isolation(
            case_desc="Customer filed dispute and system should APPROVE the request.",
            label="APPROVE",
            rationale="Valid return.",
            example_id="leak-test-1",
        )

    # Rationale leakage
    long_rationale_sentence = (
        "Adjudication policy mandates that the merchant must supply valid tracking."
    )
    with pytest.raises(DatasetValidationError, match="rationale text snippet leaked"):
        validate_embedding_isolation(
            case_desc=f"Customer claims non-receipt. {long_rationale_sentence}",
            label="REJECT",
            rationale=f"{long_rationale_sentence} Failure requires approval.",
            example_id="leak-test-2",
        )


# ---------------------------------------------------------------------------
# Test Group 3: Case Typology and Contrastive Pairs
# ---------------------------------------------------------------------------


def test_case_type_distribution(examples: list[dict[str, Any]]):
    """Verify comprehensive representation of required case types."""
    type_counts: dict[str, int] = {}
    for ex in examples:
        ctype = ex["metadata"]["case_type"]
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

    expected_types = {
        "straightforward": 20,
        "borderline": 20,
        "incomplete_evidence": 20,
        "conflicting_signals": 20,
        "contrastive_pair": 20,
    }
    assert type_counts == expected_types, (
        f"Mismatched case type distribution: {type_counts}"
    )


def test_contrastive_pairs_integrity(examples: list[dict[str, Any]]):
    """Verify the 10 contrastive pairs (20 examples) where a single change flips the outcome."""
    contrastive_examples = [
        ex for ex in examples if ex["metadata"]["case_type"] == "contrastive_pair"
    ]
    assert len(contrastive_examples) == 20

    pairs: dict[str, list[dict[str, Any]]] = {}
    for ex in contrastive_examples:
        pid = ex["metadata"]["contrastive_pair_id"]
        assert pid is not None, (
            f"Example {ex['example_id']} missing contrastive_pair_id"
        )
        pairs.setdefault(pid, []).append(ex)

    assert len(pairs) == 10, (
        f"Expected 10 distinct contrastive pairs, found {len(pairs)}"
    )

    for pid, pair in pairs.items():
        assert len(pair) == 2, f"Pair {pid} does not have exactly 2 examples"
        labels = {pair[0]["label"], pair[1]["label"]}
        assert labels == {"APPROVE", "REJECT"}, (
            f"Pair {pid} must have exactly one APPROVE and one REJECT, got {labels}"
        )
        # Ensure merchants and amounts are closely aligned between the pair
        assert pair[0]["metadata"]["merchant"] == pair[1]["metadata"]["merchant"], (
            f"Merchant should match across contrastive pair {pid}"
        )
        assert (
            pair[0]["metadata"]["transaction_amount"]
            == pair[1]["metadata"]["transaction_amount"]
        ), f"Transaction amount should match across contrastive pair {pid}"


def test_metadata_templates_prepared_for_future_experiments(
    examples: list[dict[str, Any]],
):
    """Verify that metadata includes both raw-value and ratio-verbalized descriptions for future experiments."""
    for ex in examples:
        meta = ex["metadata"]
        raw_desc = meta.get("raw_value_description")
        ratio_desc = meta.get("ratio_verbalized_description")
        ratio_feats = meta.get("ratio_features")

        assert raw_desc and len(raw_desc) >= 30, (
            f"Missing or short raw_value_description in {ex['example_id']}"
        )
        assert ratio_desc and len(ratio_desc) >= 30, (
            f"Missing or short ratio_verbalized_description in {ex['example_id']}"
        )
        assert isinstance(ratio_feats, dict)
        assert "dispute_to_lifetime_tx_ratio" in ratio_feats
        assert "dispute_filing_to_statutory_window_ratio" in ratio_feats
        assert "amount_to_median_transaction_ratio" in ratio_feats
        assert "tenure_years" in ratio_feats


# ---------------------------------------------------------------------------
# Test Group 4: Integrated Validation Script Test
# ---------------------------------------------------------------------------


def test_validator_script_execution():
    """Verify that validate_dataset_file passes cleanly on the actual file."""
    summary = validate_dataset_file(DATASET_PATH)
    assert summary["status"] == "PASS"
    assert summary["total_examples"] == 100
    assert summary["label_distribution"] == {"APPROVE": 50, "REJECT": 50}
    assert summary["quality_checks_passed"] == 9


# ---------------------------------------------------------------------------
# Test Group 5: Generator Reproducibility and Negative Validation Checks
# ---------------------------------------------------------------------------


def test_generator_reproducibility(dataset_data: dict[str, Any]):
    """Verify that programmatic generation produces identical data deterministically."""
    freshly_generated = generate_dataset()
    assert freshly_generated == dataset_data, (
        "Programmatic generation is not 100% deterministic"
    )


def test_template_helpers():
    """Verify template helper formatting and feature computation."""
    raw = build_raw_template(
        amount=150.0,
        tenure=24,
        prior_disputes=1,
        delay_days=10,
        merchant="Test Merchant",
        category="retail",
        evidence_summary="Tracking verified.",
    )
    assert "Transaction amount is $150.00" in raw
    assert "Test Merchant (retail)" in raw

    ratio = build_ratio_template(
        amount=150.0,
        tenure=24,
        prior_disputes=1,
        delay_days=10,
        merchant="Test Merchant",
        category="retail",
        evidence_summary="Tracking verified.",
    )
    assert "$150.00 charge" in ratio
    assert "lifetime dispute rate" in ratio

    feats = compute_ratio_features(
        amount=250.0, tenure=24, prior_disputes=2, delay_days=15
    )
    assert feats["amount_to_median_transaction_ratio"] == 2.0
    assert feats["tenure_years"] == 2.0
    assert 0 <= feats["dispute_filing_to_statutory_window_ratio"] <= 1.0


def test_save_dataset_function(tmp_path: Path):
    """Verify save_dataset writes valid JSON to specified output path."""
    test_output = tmp_path / "subdir" / "test_seed.json"
    save_dataset(test_output)
    assert test_output.is_file()
    summary = validate_dataset_file(test_output)
    assert summary["status"] == "PASS"
    assert summary["total_examples"] == 100


def test_validation_catches_invalid_file(tmp_path: Path):
    """Verify validation raises error when file does not exist."""
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(DatasetValidationError, match="file not found"):
        validate_dataset_file(non_existent)


def test_validation_catches_truncated_examples(
    tmp_path: Path, dataset_data: dict[str, Any]
):
    """Verify validation raises error when example count != 100."""
    mutated = dict(dataset_data)
    mutated["examples"] = dataset_data["examples"][:99]
    bad_file = tmp_path / "bad_count.json"
    with open(bad_file, "w") as f:
        json.dump(mutated, f)
    with pytest.raises(DatasetValidationError, match="Expected exactly 100 examples"):
        validate_dataset_file(bad_file)


def test_validation_catches_imbalanced_labels(
    tmp_path: Path, dataset_data: dict[str, Any]
):
    """Verify validation raises error when label distribution is not 50/50."""
    import copy

    mutated = copy.deepcopy(dataset_data)
    # Flip one REJECT to APPROVE -> 51 APPROVE, 49 REJECT
    for ex in mutated["examples"]:
        if ex["label"] == "REJECT":
            ex["label"] = "APPROVE"
            break
    bad_file = tmp_path / "bad_balance.json"
    with open(bad_file, "w") as f:
        json.dump(mutated, f)
    with pytest.raises(
        DatasetValidationError, match="Expected 50 APPROVE and 50 REJECT"
    ):
        validate_dataset_file(bad_file)


def test_validation_catches_duplicate_ids(tmp_path: Path, dataset_data: dict[str, Any]):
    """Verify validation raises error when duplicate IDs exist."""
    import copy

    mutated = copy.deepcopy(dataset_data)
    mutated["examples"][1]["example_id"] = mutated["examples"][0]["example_id"]
    bad_file = tmp_path / "duplicate_id.json"
    with open(bad_file, "w") as f:
        json.dump(mutated, f)
    with pytest.raises(DatasetValidationError, match="Duplicate example_id"):
        validate_dataset_file(bad_file)
