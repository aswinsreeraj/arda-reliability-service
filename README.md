# ARDA Reliability Service

The **ARDA Reliability Service** provides calibrated confidence gating, semantic similarity retrieval, and leave-one-out (LOO) reliability evaluation for AI agents executing high-stakes decisions on the ARDA platform.

## Development Week 3: Reliability Service & Reference Datasets

### Synthetic Seed Dataset
The initial reference seed dataset of 100 reference examples is available under [`research/datasets/`](file:///home/aswin/second-project/arda-reliability-service/research/datasets):
- [`research/datasets/synthetic_seed_examples.json`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/synthetic_seed_examples.json): 100 curated, balanced reference cases (50 `APPROVE`, 50 `REJECT`).
- [`research/datasets/dataset_schema.json`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/dataset_schema.json): Formal JSON Schema Draft-07 specification.
- [`research/datasets/synthetic_adjudication_policy.md`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/synthetic_adjudication_policy.md): Synthetic demonstration dispute adjudication policy and threshold definitions.
- [`research/datasets/semantic_audit_report.md`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/semantic_audit_report.md): Domain consistency and semantic audit report.
- [`research/datasets/generate_dataset.py`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/generate_dataset.py): Deterministic generator script.
- [`research/datasets/validate_dataset.py`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/validate_dataset.py): 9-point quality validation suite.
- [`research/datasets/README.md`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/README.md): Detailed dataset documentation, field definitions, and research considerations.

### Quick Start & Verification

```bash
# Validate the dataset
.venv/bin/python research/datasets/validate_dataset.py

# Run the test suite
.venv/bin/pytest tests/ -v
```

