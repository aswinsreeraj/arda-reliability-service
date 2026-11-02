"""Configuration definitions for ARDA Reliability-Gate Research Spike.

Defines model metadata, prefix formatting rules, template accessors,
and experiment execution parameters for the four-way matrix:
- E1: BGE-small + Raw-value
- E2: BGE-small + Ratio-verbalized
- E3: E5-base + Raw-value
- E4: E5-base + Ratio-verbalized
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = EXPERIMENTS_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parent
DATASET_PATH = RESEARCH_DIR / "datasets" / "synthetic_seed_examples.json"
RESULTS_DIR = RESEARCH_DIR / "results"
CACHE_DIR = EXPERIMENTS_DIR / ".cache"


@dataclass(frozen=True)
class ModelConfig:
    """Specification and prefix conventions for an embedding model."""

    name: str
    short_name: str
    dimension: int
    max_seq_length: int
    normalize_embeddings: bool = True
    query_prefix: str = ""
    passage_prefix: str = ""
    notes: str = ""


# BGE-small v1.5: Symmetric sentence similarity or general retrieval does not require prefixes.
# If asymmetric retrieval query instruction is used, BGE recommends query prefix:
# "Represent this sentence for searching relevant passages: "
# For fair comparison and document similarity, passages have no prefix.
BGE_SMALL_CONFIG = ModelConfig(
    name="BAAI/bge-small-en-v1.5",
    short_name="bge-small",
    dimension=384,
    max_seq_length=512,
    normalize_embeddings=True,
    query_prefix="Represent this sentence for searching relevant passages: ",
    passage_prefix="",
    notes="BAAI BGE-small v1.5 with recommended asymmetric retrieval query instruction.",
)

# E5-base-v2: Strictly requires "query: " and "passage: " prefixes to activate its dual-encoder behavior.
E5_BASE_CONFIG = ModelConfig(
    name="intfloat/e5-base-v2",
    short_name="e5-base",
    dimension=768,
    max_seq_length=512,
    normalize_embeddings=True,
    query_prefix="query: ",
    passage_prefix="passage: ",
    notes="intfloat E5-base-v2 with mandatory 'query: ' and 'passage: ' prefixes.",
)


@dataclass(frozen=True)
class TemplateConfig:
    """Specification of case description template."""

    name: Literal["raw_value", "ratio_verbalized"]
    metadata_field: str
    display_name: str
    description: str


RAW_VALUE_TEMPLATE = TemplateConfig(
    name="raw_value",
    metadata_field="raw_value_description",
    display_name="Raw-Value Description",
    description="Case description featuring concrete numerical quantities, calendar timelines, and raw merchant details.",
)

RATIO_VERBALIZED_TEMPLATE = TemplateConfig(
    name="ratio_verbalized",
    metadata_field="ratio_verbalized_description",
    display_name="Ratio-Verbalized Description",
    description="Case description verbalizing normalized lifetime dispute rates, elapsed statutory percentages, and relative thresholds.",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for an individual LOO retrieval experiment."""

    experiment_id: str
    model: ModelConfig
    template: TemplateConfig
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    tie_breaker: Literal["max_similarity", "sum_similarity"] = "max_similarity"
    batch_size: int = 32


EXPERIMENT_MATRIX: dict[str, ExperimentConfig] = {
    "E1": ExperimentConfig(
        experiment_id="E1",
        model=BGE_SMALL_CONFIG,
        template=RAW_VALUE_TEMPLATE,
    ),
    "E2": ExperimentConfig(
        experiment_id="E2",
        model=BGE_SMALL_CONFIG,
        template=RATIO_VERBALIZED_TEMPLATE,
    ),
    "E3": ExperimentConfig(
        experiment_id="E3",
        model=E5_BASE_CONFIG,
        template=RAW_VALUE_TEMPLATE,
    ),
    "E4": ExperimentConfig(
        experiment_id="E4",
        model=E5_BASE_CONFIG,
        template=RATIO_VERBALIZED_TEMPLATE,
    ),
}
