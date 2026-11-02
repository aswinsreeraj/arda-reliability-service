"""Validation suite for ARDA Reliability Service synthetic reference dataset.

Verifies all 9 dataset quality checks:
1. Exactly 100 examples exist.
2. Every example has a unique ID.
3. Every example has a valid label (APPROVE or REJECT).
4. No required field is missing.
5. Descriptions are non-empty and well-formed.
6. The dataset contains the intended balanced label distribution (50 APPROVE, 50 REJECT).
7. There are no accidental duplicate descriptions.
8. Each example has an inspectable rationale.
9. Critical embedding isolation: No rationale or label is accidentally included in the embedding input.

Also enforces Pydantic structural validation, UUID format compliance, and contrastive pair consistency.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError


class ExampleMetadataModel(BaseModel):
    category: str
    case_type: Literal[
        "straightforward",
        "borderline",
        "incomplete_evidence",
        "conflicting_signals",
        "contrastive_pair",
    ]
    contrastive_pair_id: str | None = None
    transaction_id: str
    transaction_amount: float = Field(gt=0)
    currency: Literal["USD"] = "USD"
    merchant: str
    merchant_category: str
    customer_tenure_months: int = Field(ge=0)
    prior_dispute_count: int = Field(ge=0)
    dispute_filing_delay_days: int = Field(ge=0)
    evidence_signals: list[str] = Field(min_length=1)
    raw_value_description: str = Field(min_length=20)
    ratio_verbalized_description: str = Field(min_length=20)
    ratio_features: dict[str, float]


class ReferenceExampleModel(BaseModel):
    example_id: str = Field(pattern=r"^ref-example-\d{3}$")
    tenant_id: str
    agent_id: str
    case_description: str = Field(min_length=20)
    label: Literal["APPROVE", "REJECT"]
    source: Literal["synthetic"]
    status: Literal["active", "archived", "deprecated"]
    metadata: ExampleMetadataModel
    rationale: str = Field(min_length=15)


class ReferenceDatasetModel(BaseModel):
    dataset_name: str
    dataset_version: str
    description: str
    decision_domain: str
    tenant_id: str
    agent_id: str
    total_examples: int = 100
    label_distribution: dict[str, int]
    case_type_distribution: dict[str, int]
    examples: list[ReferenceExampleModel]


class DatasetValidationError(Exception):
    """Raised when dataset validation checks fail."""


def validate_uuid(value: str, field_name: str) -> None:
    """Ensure string is a valid UUID."""
    try:
        UUID(value)
    except ValueError as exc:
        raise DatasetValidationError(
            f"Invalid UUID in {field_name}: '{value}'"
        ) from exc


def validate_embedding_isolation(
    case_desc: str, label: str, rationale: str, example_id: str
) -> None:
    """Check that case_description has no label leakage or rationale embedding.

    Embedding input must solely reflect the raw case facts without leaking:
    - The reference label (APPROVE / REJECT)
    - Metadata keywords like 'proposed decision', 'expected outcome'
    - Text from the rationale
    """
    desc_upper = case_desc.upper()

    # Disallow explicit adjudication labels as isolated decision words
    # Note: words like 'approved' or 'rejected' should not appear in case_description as determinations
    forbidden_tokens = [
        "LABEL: APPROVE",
        "LABEL: REJECT",
        "DECISION: APPROVE",
        "DECISION: REJECT",
        "OUTCOME: APPROVE",
        "OUTCOME: REJECT",
        "RECOMMENDED DECISION",
        "PROPOSED DECISION",
        "EVALUATION RESULT",
    ]
    for token in forbidden_tokens:
        if token in desc_upper:
            raise DatasetValidationError(
                f"Embedding isolation violation in {example_id}: forbidden marker '{token}' in case_description"
            )

    # Disallow standalone label declaration
    if re.search(r"\b(APPROVE|REJECT)\b", case_desc):
        raise DatasetValidationError(
            f"Embedding isolation violation in {example_id}: explicit label token '{label}' found in case_description"
        )

    # Ensure no sentence from rationale is pasted into case_description
    rationale_sentences = [
        s.strip() for s in rationale.split(".") if len(s.strip()) > 20
    ]
    for s in rationale_sentences:
        if s.lower() in case_desc.lower():
            raise DatasetValidationError(
                f"Embedding isolation violation in {example_id}: rationale text snippet leaked into case_description: '{s[:40]}...'"
            )


def validate_dataset_file(file_path: Path) -> dict[str, Any]:
    """Execute all quality and consistency checks on the dataset file."""
    if not file_path.exists():
        raise DatasetValidationError(f"Dataset file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"JSON decode failure in {file_path}: {exc}"
            ) from exc

    # 1. Pydantic schema validation
    try:
        parsed_dataset = ReferenceDatasetModel.model_validate(data)
    except ValidationError as exc:
        raise DatasetValidationError(f"Schema validation error: {exc}") from exc

    examples = parsed_dataset.examples

    # Check 1: Exactly 100 examples exist
    if len(examples) != 100:
        raise DatasetValidationError(
            f"Check 1 Failed: Expected exactly 100 examples, found {len(examples)}"
        )

    # Check 2: Every example has a unique ID
    seen_ids = set()
    for ex in examples:
        if ex.example_id in seen_ids:
            raise DatasetValidationError(
                f"Check 2 Failed: Duplicate example_id '{ex.example_id}'"
            )
        seen_ids.add(ex.example_id)

    # Check 3: Every example has a valid label
    valid_labels = {"APPROVE", "REJECT"}
    for ex in examples:
        if ex.label not in valid_labels:
            raise DatasetValidationError(
                f"Check 3 Failed: Invalid label '{ex.label}' in {ex.example_id}"
            )

    # Check 4: No required field is missing (enforced by Pydantic, plus explicit tenant/agent checks)
    for ex in examples:
        validate_uuid(ex.tenant_id, f"{ex.example_id}.tenant_id")
        validate_uuid(ex.agent_id, f"{ex.example_id}.agent_id")
        if ex.source != "synthetic":
            raise DatasetValidationError(
                f"Check 4 Failed: Expected source='synthetic', got '{ex.source}'"
            )
        if ex.status != "active":
            raise DatasetValidationError(
                f"Check 4 Failed: Expected status='active', got '{ex.status}'"
            )

    # Check 5: Descriptions are non-empty and substantial
    for ex in examples:
        if not ex.case_description or len(ex.case_description.strip()) < 30:
            raise DatasetValidationError(
                f"Check 5 Failed: Empty or overly short description in {ex.example_id} (len={len(ex.case_description.strip())})"
            )

    # Check 6: Intended label distribution (50 APPROVE, 50 REJECT)
    label_counts = {"APPROVE": 0, "REJECT": 0}
    for ex in examples:
        label_counts[ex.label] += 1
    if label_counts["APPROVE"] != 50 or label_counts["REJECT"] != 50:
        raise DatasetValidationError(
            f"Check 6 Failed: Expected 50 APPROVE and 50 REJECT, got {label_counts}"
        )

    # Check 7: No accidental duplicate descriptions
    seen_descriptions = set()
    for ex in examples:
        cleaned_desc = " ".join(ex.case_description.lower().split())
        if cleaned_desc in seen_descriptions:
            raise DatasetValidationError(
                f"Check 7 Failed: Duplicate case_description detected in {ex.example_id}"
            )
        seen_descriptions.add(cleaned_desc)

    # Check 8: Each example has an inspectable rationale
    for ex in examples:
        if not ex.rationale or len(ex.rationale.strip()) < 20:
            raise DatasetValidationError(
                f"Check 8 Failed: Empty or trivial rationale in {ex.example_id}"
            )

    # Check 9: Critical embedding isolation
    for ex in examples:
        validate_embedding_isolation(
            case_desc=ex.case_description,
            label=ex.label,
            rationale=ex.rationale,
            example_id=ex.example_id,
        )

    # Additional Check: Contrastive pairs consistency
    contrastive_pairs: dict[str, list[ReferenceExampleModel]] = {}
    for ex in examples:
        if ex.metadata.case_type == "contrastive_pair":
            pid = ex.metadata.contrastive_pair_id
            if not pid:
                raise DatasetValidationError(
                    f"Contrastive pair example {ex.example_id} missing contrastive_pair_id"
                )
            contrastive_pairs.setdefault(pid, []).append(ex)

    if len(contrastive_pairs) != 10:
        raise DatasetValidationError(
            f"Expected 10 contrastive pairs (20 examples), found {len(contrastive_pairs)}"
        )

    for pid, pair_examples in contrastive_pairs.items():
        if len(pair_examples) != 2:
            raise DatasetValidationError(
                f"Contrastive pair {pid} has {len(pair_examples)} examples, expected exactly 2"
            )
        labels = {e.label for e in pair_examples}
        if labels != {"APPROVE", "REJECT"}:
            raise DatasetValidationError(
                f"Contrastive pair {pid} must contain exactly 1 APPROVE and 1 REJECT, got {labels}"
            )

    # Report results summary
    summary = {
        "status": "PASS",
        "file": str(file_path),
        "total_examples": len(examples),
        "label_distribution": label_counts,
        "case_type_distribution": {
            "straightforward": sum(
                1 for e in examples if e.metadata.case_type == "straightforward"
            ),
            "borderline": sum(
                1 for e in examples if e.metadata.case_type == "borderline"
            ),
            "incomplete_evidence": sum(
                1 for e in examples if e.metadata.case_type == "incomplete_evidence"
            ),
            "conflicting_signals": sum(
                1 for e in examples if e.metadata.case_type == "conflicting_signals"
            ),
            "contrastive_pair": sum(
                1 for e in examples if e.metadata.case_type == "contrastive_pair"
            ),
        },
        "contrastive_pairs_count": len(contrastive_pairs),
        "quality_checks_passed": 9,
    }
    return summary


def main() -> None:
    """CLI execution for dataset validation."""
    current_dir = Path(__file__).resolve().parent
    dataset_file = current_dir / "synthetic_seed_examples.json"

    print("=================================================================")
    print("  ARDA Reliability Service — Synthetic Dataset Validator")
    print(f"  Target File: {dataset_file}")
    print("=================================================================")

    try:
        summary = validate_dataset_file(dataset_file)
        print("\n[OK] All 9 Dataset Quality Checks PASSED successfully!\n")
        print(f"  * Total Examples:          {summary['total_examples']}")
        print(f"  * Label Distribution:      {summary['label_distribution']}")
        print(f"  * Case Types:              {summary['case_type_distribution']}")
        print(
            f"  * Contrastive Pairs:       {summary['contrastive_pairs_count']} pairs (20 examples)"
        )
        print(
            "  * Embedding Isolation:     PASSED (Zero label or rationale leakage in case_description)"
        )
        print("=================================================================\n")
        sys.exit(0)
    except DatasetValidationError as exc:
        print(f"\n[FAIL] Dataset Validation FAILED: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
