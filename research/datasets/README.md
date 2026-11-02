# ARDA Reliability Service — Synthetic Seed Dataset (v1.0.0)

## 1. Overview & Decision Domain

This dataset constitutes the foundational **synthetic reference dataset** for the **ARDA Reliability Service** (Development Week 3). It contains exactly **100 reference examples** curated to evaluate vector embeddings, semantic retrieval, calibrated confidence scoring, and leave-one-out (LOO) evaluation.

- **Decision Domain**: **Financial Transaction Dispute & Refund Adjudication**
- **Scoped Agent**: `Transaction Dispute Investigator` (`agent_id: 00000000-0000-0000-0000-000000000002`)
- **Scoped Tenant**: Acme Fintech / ARDA Synthetic Tenant (`tenant_id: 00000000-0000-0000-0000-000000000001`)
- **Decision Labels**: `APPROVE` (50 examples), `REJECT` (50 examples)

All examples represent adjudicated historical cases within a single tenant and agent scope, providing ground truth for similarity-based reliability gating.

---

## 2. Critical Separation Rule: Embedding vs. Decision Isolation

> [!IMPORTANT]
> **Only `case_description` must be used to generate embeddings.**

To prevent target leakage and retrieval bias, the following fields are strictly quarantined from the embedding input text:
- Proposed decision (`proposed_decision` in gRPC `GateRequest`)
- Reference label (`label`: `APPROVE` / `REJECT`)
- Expected outcome
- Business rationale (`rationale`)
- Gate evaluation result (`outcome`, `confidence`)

This architectural separation ensures that vector retrieval evaluates the semantic and factual similarity of the underlying **case situation**, rather than trivially clustering on outcome labels. The proposed decision is supplied independently at query time to determine whether nearby historical cases support or contradict the agent's intent.

---

## 3. Reference-Example Schema

The dataset adheres to the schema specified in [`dataset_schema.json`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/dataset_schema.json):

| Field | Type | Description |
|---|---|---|
| `example_id` | `string` | Unique identifier (e.g. `ref-example-001` through `ref-example-100`). |
| `tenant_id` | `string (UUID)` | Consistent tenant boundary scope (`00000000-0000-0000-0000-000000000001`). |
| `agent_id` | `string (UUID)` | Consistent agent boundary scope (`00000000-0000-0000-0000-000000000002`). |
| `case_description` | `string` | **The sole text used for embedding generation.** Natural language description of facts, merchant details, timelines, and evidence signals. |
| `label` | `enum` | Adjudication label: `APPROVE` or `REJECT`. |
| `source` | `enum` | Origin marker: `"synthetic"`. |
| `status` | `enum` | Lifecycle status: `"active"`. |
| `metadata` | `object` | Structured features, domain attributes, and experimental template representations. |
| `rationale` | `string` | Auditable explanation of why policy dictates the outcome. **Inspection and evaluation only; never embedded.** |

### Metadata Object Structure

```json
{
  "category": "unauthorized_transaction | goods_not_received | defective_or_not_as_described | duplicate_billing | subscription_cancellation | credit_not_processed",
  "case_type": "straightforward | borderline | incomplete_evidence | conflicting_signals | contrastive_pair",
  "contrastive_pair_id": "pair_01_filing_timeline | null",
  "transaction_id": "tx_syn_0001",
  "transaction_amount": 249.99,
  "currency": "USD",
  "merchant": "Apex Electronics Supply",
  "merchant_category": "electronics",
  "customer_tenure_months": 36,
  "prior_dispute_count": 0,
  "dispute_filing_delay_days": 18,
  "evidence_signals": ["carrier_tracking_lost_in_transit", "merchant_unresponsive_14d"],
  "raw_value_description": "Transaction amount is $249.99 at Apex Electronics Supply (electronics)...",
  "ratio_verbalized_description": "Claim on Apex Electronics Supply (electronics) involves a $249.99 charge...",
  "ratio_features": {
    "dispute_to_lifetime_tx_ratio": 0.0,
    "dispute_filing_to_statutory_window_ratio": 0.3,
    "amount_to_median_transaction_ratio": 1.9999,
    "tenure_years": 3.0
  }
}
```

---

## 4. Dataset Composition & Typology

The dataset contains a balanced 50/50 label split across five distinct case typologies (20 examples each):

```
+-----------------------------------------------------------------------------------+
|                        100 Reference Examples (50 APPROVE / 50 REJECT)             |
+---------------------+--------------------+--------------------+-------------------+
| Typology            | Total | APPROVE | REJECT | Purpose                           |
+---------------------+-------+---------+--------+-----------------------------------+
| Straightforward     | 20    | 10      | 10     | Core benchmark for high-confidence|
| Borderline          | 20    | 10      | 10     | Threshold & SLA edge cases        |
| Incomplete Evidence | 20    | 10      | 10     | Default rules & burden-of-proof   |
| Conflicting Signals | 20    | 10      | 10     | Multi-modal evidence trade-offs   |
| Contrastive Pairs   | 20    | 10      | 10     | Sensitivity to critical details   |
+---------------------+-------+---------+--------+-----------------------------------+
```

### Typology Breakdown

1. **Straightforward Cases (Examples 1–20)**:
   - Clear facts, undisputed evidence, explicit adherence or violation of core adjudication policy.
   - *Example*: Delivery tracking confirms package lost in transit; merchant non-responsive for 14 days (`APPROVE`).
   - *Example*: Verified EMV chip-and-PIN cash withdrawal with zero PIN failures (`REJECT`).

2. **Borderline Cases (Examples 21–40)**:
   - Factual parameters close to decision boundaries (e.g. filing delay on day 58 vs. 62 of 60-day window, partial damage, delivery to unattended building lobbies).
   - *Example*: Filing on day 58 of 60 with documented 4-week merchant troubleshooting history (`APPROVE`).
   - *Example*: Filing on day 62 (exceeding statutory cutoff by 48 hours) without extenuating medical proof (`REJECT`).

3. **Incomplete Evidence Cases (Examples 41–60)**:
   - Scenarios where key pieces of information are unverified or missing, testing default burden-of-proof rules.
   - *Example*: Merchant failed to respond to dispute notice within mandatory 14-day window (`APPROVE` by procedural default).
   - *Example*: Cardholder claims $750 defect but refuses to provide photos, error logs, or inspection reports (`REJECT` due to unfulfilled burden of proof).

4. **Conflicting Signals Cases (Examples 61–80)**:
   - Strong evidence pointing in opposite directions, requiring deep forensic weighting.
   - *Example*: IP matches customer home subnet, but cybersecurity forensics prove an infostealer trojan reverse-proxied traffic (`APPROVE`).
   - *Example*: Customer claims unfamiliar with purchase, but transaction authenticated via primary iPhone Face ID and shipped to biological sibling (`REJECT`).

5. **Contrastive Pairs (Examples 81–100, 10 pairs)**:
   - Semantically similar case descriptions where a single material factual distinction flips the correct outcome:
     - **Pair 1 (`pair_01_filing_timeline`)**: Day 59 filing (`APPROVE`) vs. Day 61 filing (`REJECT`).
     - **Pair 2 (`pair_02_return_tracking`)**: Return with authorized RMA and carrier delivery receipt (`APPROVE`) vs. Return without RMA dropped at unstaffed box with no tracking (`REJECT`).
     - **Pair 3 (`pair_03_3ds_authentication`)**: CNP transaction without 3DS from unknown IP (`APPROVE`) vs. Full 3DS v2.2 biometric authentication on trusted phone (`REJECT`).
     - **Pair 4 (`pair_04_delivery_location`)**: Photo shows red brick porch whereas customer home has wooden siding (`APPROVE`) vs. Photo shows customer house number 742 on matching porch (`REJECT`).
     - **Pair 5 (`pair_05_cancellation_timing`)**: Cancellation 25 minutes prior to billing cycle executed due to race condition (`APPROVE`) vs. Cancellation 45 minutes after cycle renewal (`REJECT`).
     - **Pair 6 (`pair_06_duplicate_vs_distinct`)**: Two identical charges 2 seconds apart for 1 order (`APPROVE`) vs. Two separate orders 25 minutes apart with distinct items and packages (`REJECT`).
     - **Pair 7 (`pair_07_digital_activation`)**: License key never activated or downloaded, canceled within 3 hours (`APPROVE`) vs. Key activated with 18 hours active rendering use (`REJECT`).
     - **Pair 8 (`pair_08_service_work_order`)**: Generic POS slip without signed work order or parts itemization (`APPROVE`) vs. Signed customer work order certifying satisfaction and deadbolt photos (`REJECT`).
     - **Pair 9 (`pair_09_hotel_cancellation_window`)**: Canceled 49 hours prior (within 48h free window) (`APPROVE`) vs. Canceled 23 hours prior (missed 48h deadline) (`REJECT`).
     - **Pair 10 (`pair_10_restocking_fee_disclosure`)**: Undisclosed surprise 25% restocking fee (`APPROVE`) vs. Explicitly acknowledged mandatory restocking fee checkbox (`REJECT`).

---

## 5. Future Research Roadmap

This dataset is pre-configured to support two upcoming experimental tracks in Week 3:

### Track A: Embedding Model Comparison
Evaluate retrieval quality and calibrated confidence separation between:
1. `BAAI/bge-small-en-v1.5` (384-dimensional dense embeddings)
2. `intfloat/e5-base-v2` (768-dimensional dense embeddings)

### Track B: Description Verbalization Templates
Compare semantic retrieval sensitivity using two standardized representations stored in `metadata`:
1. `raw_value_description`: Direct numerical quantities and raw merchant attributes.
2. `ratio_verbalized_description`: Verbalized percentages, normalized ratios, and relative elapsed timeframes.

---

## 6. Reproduction & Quality Validation

### Regenerate the Dataset
```bash
.venv/bin/python research/datasets/generate_dataset.py
```

### Run Standalone Validation Script
```bash
.venv/bin/python research/datasets/validate_dataset.py
```

### Run Pytest Test Suite
```bash
.venv/bin/pytest tests/test_dataset_validation.py -v
```

---

## 7. Assumptions & Limitations

1. **Synthetic Nature**: This dataset is generated programmatically for engineering experiments, vector retrieval benchmarking, and reliability score calibration. It does not represent live production banking data.
2. **Scope Boundary**: Designed specifically for the `Transaction Dispute Investigator` agent. Extrapolating confidence calibration to other agent domains (e.g. loan origination or medical triage) requires separate reference seed datasets.
3. **Deterministic Seed**: The generation script produces a deterministic, version-controlled output without external API dependencies or stochastic LLM drift.

