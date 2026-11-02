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
# Start local PostgreSQL (pgvector) and Redpanda
docker compose up -d postgres redpanda

# Run database migrations
.venv/bin/alembic upgrade head

# Validate the dataset
.venv/bin/python research/datasets/validate_dataset.py

# Run the complete test suite (193 tests)
.venv/bin/pytest tests/ -v
```

## Development Week 5: Asynchronous Human Feedback Ingestion

The **Asynchronous Human Feedback Ingestion Pipeline** consumes authoritative human-adjudicated resolutions from Redpanda (`arda.human-resolutions.v1`), validates schemas, enforces database-level idempotency via `human_feedback_events`, generates 384-dimensional dense embeddings exclusively from original case descriptions (`BAAI/bge-small-en-v1.5`), and durably stores active reference examples into PostgreSQL (`reference_examples` with pgvector).

- **Architecture & Design Document**: [`research/human_feedback_ingestion_design.md`](file:///home/aswin/second-project/arda-reliability-service/research/human_feedback_ingestion_design.md)
- **Dead-Letter Queue (DLQ)**: Poison and permanently invalid events are routed to `arda.human-resolutions.dlq`.
- **Conflict Handling**: Opposing human decisions on existing cases are recorded with status `CONFLICT` and prevented from silently corrupting reference data.
- **Verification Suite**: [`tests/test_human_feedback_ingestion.py`](file:///home/aswin/second-project/arda-reliability-service/tests/test_human_feedback_ingestion.py) covers all 15 operational failure-recovery and lifecycle scenarios.
- **Live Integration Suite**: [`tests/test_redpanda_real_integration.py`](file:///home/aswin/second-project/arda-reliability-service/tests/test_redpanda_real_integration.py) runs against live Redpanda and PostgreSQL containers.


