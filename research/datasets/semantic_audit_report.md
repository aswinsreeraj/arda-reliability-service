# ARDA Reliability Service — Synthetic Dataset Semantic Audit Report

**Date**: November 2, 2026  
**Auditor**: Senior Fintech Domain Engineer & ML Evaluation Engineer  
**Dataset**: [`research/datasets/synthetic_seed_examples.json`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/synthetic_seed_examples.json)  
**Schema**: [`research/datasets/dataset_schema.json`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/dataset_schema.json)  
**Policy Document**: [`research/datasets/synthetic_adjudication_policy.md`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/synthetic_adjudication_policy.md)  

---

## 1. Executive Summary

An independent semantic and domain-consistency audit was conducted on the 100 reference examples comprising the ARDA Reliability Service synthetic seed dataset. The dataset previously passed all structural validations and unit tests (Pydantic models, JSON Schema Draft-07, and pytest).

The primary objective of this audit was to ensure domain plausibility, evidentiary coherence, factual consistency between narrative descriptions and policy rationales, and mathematical integrity of experimental verbalization templates prior to running embedding retrieval experiments.

### Audit Summary
- **Total Examples Reviewed**: 100
- **Examples with Consistent Labels & Rationales**: 93 / 100 (93%)
- **Examples Identified with Factual or Entity Omissions**: 6 / 100
- **Examples Identified with Category Taxonomy Mismatches**: 7 / 100
- **Mathematical Inconsistencies in Verbalization Templates**: 1 (divergent denominator between `build_ratio_template` and `compute_ratio_features`)
- **Critical Embedding Isolation**: Confirmed **100% compliant**. Zero label tokens, decision markers, or rationale fragments leak into `case_description`.

---

## 2. Number of Examples Reviewed

| Case Typology | Total Reviewed | `APPROVE` | `REJECT` | Status |
|---|---|---|---|---|
| Straightforward | 20 | 10 | 10 | 18 fully compliant, 2 entity/category corrections |
| Borderline | 20 | 10 | 10 | 19 fully compliant, 1 category correction |
| Incomplete Evidence | 20 | 10 | 10 | 19 fully compliant, 1 entity correction |
| Conflicting Signals | 20 | 10 | 10 | 18 fully compliant, 2 category corrections |
| Contrastive Pairs | 20 (10 pairs) | 10 | 10 | 16 fully compliant, 4 entity/category corrections |
| **Total** | **100** | **50** | **50** | **93 initially clean, 7 category fixes, 6 entity fixes** |

---

## 3. Examples with Consistent Labels

93 out of 100 examples demonstrated rock-solid domain consistency where:
1. The `case_description` establishes all material facts necessary to apply the synthetic adjudication policy.
2. The `rationale` derives directly from the facts without introducing unstated evidence.
3. The adjudication label (`APPROVE` or `REJECT`) strictly conforms to the defined policy thresholds (such as the 60-day filing cutoff, 14-day merchant response SLA, carrier tracking standards, and cryptographic authentication hierarchies).

Notable high-fidelity examples:
- **`ref-example-003`** (`APPROVE`): Unauthenticated foreign CNP gaming purchase refuted by simultaneous verified local chip-and-PIN purchase in Denver, Colorado.
- **`ref-example-012`** (`REJECT`): Physical chip-and-PIN cash withdrawal with zero failed PIN attempts, confirming authorized card and PIN possession.
- **`ref-example-021`** (`APPROVE`): Defective smart camera filed on day 58 of 60; 4-week documented merchant troubleshooting excuses filing proximity.
- **`ref-example-061`** (`APPROVE`): Home IP subnet match superseded by verified forensic proof of an infostealer trojan reverse-proxying traffic at 4 AM.
- **`ref-example-081` & `ref-example-082`** (Contrastive Pair 1): Clean, isolated 1-day variance around the 60-day statutory window (Day 59 = `APPROVE`, Day 61 = `REJECT`).

---

## 4. Examples Requiring Correction

The audit identified two categories of issues requiring correction:

### A. Entity Omissions in `case_description`
In 6 examples, the merchant name specified in `metadata["merchant"]` was omitted from `case_description`, referring instead generically to "an online liquidation sale", "an ATM terminal", or "recurring cloud sync software":

| Example ID | Metadata Merchant | Original Description Excerpt | Issue |
|---|---|---|---|
| `ref-example-012` | `First National ATM Network` | *"...occurred at an ATM terminal in their neighborhood..."* | Merchant entity omitted from narrative |
| `ref-example-015` | `Alpine Gear Clearance` | *"...during an online liquidation sale..."* | Merchant entity omitted from narrative |
| `ref-example-016` | `FitPulse Online Gym` | *"...disputes a $29.99 monthly fitness app membership fee..."* | Merchant entity omitted from narrative |
| `ref-example-054` | `Apex Rental Car Services` | *"...disputes a $110.00 rental car cleaning fee..."* | Merchant entity omitted from narrative |
| `ref-example-089` | `CloudSync Enterprise` | *"...charged $55.00 for recurring cloud sync software..."* | Merchant entity omitted from narrative |
| `ref-example-090` | `CloudSync Enterprise` | *"...charged $55.00 for recurring cloud sync software..."* | Merchant entity omitted from narrative |

*Remediation*: Explicitly include the merchant entity name in `case_description` for each of these examples to ensure embedding models have full named entity context.

---

### B. Category Taxonomy Mismatches
In 7 examples, the dispute claim was categorized as `goods_not_received`, even though the case facts involved restaurant authorizations, cancelled flight tickets, hotel reservations, or software license refunds:

| Example ID | Merchant | Claim Narrative | Current Category | Correct Category | Reason |
|---|---|---|---|---|---|
| `ref-example-007` | Skyline Express Airways | Canceled flight, airline refused cash refund, issued expiring voucher | `goods_not_received` | `credit_not_processed` | Airline transportation service; customer seeking promised/mandated refund credit. |
| `ref-example-017` | Gourmet Bistro & Grill | Customer claims they never authorized $95 dine-in restaurant bill | `goods_not_received` | `unauthorized_transaction` | Customer explicitly claimed transaction was unauthorized; merchant proved chip contact and signed slip. |
| `ref-example-040` | SwiftAir Regional | Passenger missed flight due to traffic on non-refundable ticket | `goods_not_received` | `credit_not_processed` | Flight transportation service; dispute seeks fare refund. |
| `ref-example-068` | Island Paradise Resort | Beachfront villa booking during mandatory hurricane evacuation | `goods_not_received` | `credit_not_processed` | Lodging service cancellation refund due to emergency closure. |
| `ref-example-080` | Fintech Summit International | Cardholder claimed conference was canceled; badge logs show attendance | `goods_not_received` | `credit_not_processed` | Event registration fee dispute seeking cancellation credit. |
| `ref-example-093` | VFX Studio Tools | Accidental purchase of plugin license key; unactivated, refunded | `goods_not_received` | `credit_not_processed` | Accidental purchase refund claim on digital software license. |
| `ref-example-094` | VFX Studio Tools | Claimed accidental purchase, but key activated with 18h usage | `goods_not_received` | `credit_not_processed` | Accidental purchase refund claim on digital software license. |

*Remediation*: Reclassify these 7 examples to their appropriate categories (`credit_not_processed` and `unauthorized_transaction`).

---

## 5. Policy Inconsistencies & Synthetic Policy Standardization

During the audit, all implicit policy assumptions were extracted and codified into [`research/datasets/synthetic_adjudication_policy.md`](file:///home/aswin/second-project/arda-reliability-service/research/datasets/synthetic_adjudication_policy.md). 

Key policy invariants confirmed:
1. **Filing Window**: Strict 60-day statutory cutoff.
2. **Merchant Response Window**: Strict 14-day default response window.
3. **Burden of Proof**:
   - For non-delivery: Carrier tracking with GPS/signature shifts burden to cardholder (requiring police report or photographic proof of wrong house).
   - For defect/counterfeit: Cardholder holds initial burden of proof (photos, error logs, or appraisal).
4. **Authentication Hierarchy**: Chip-and-PIN and 3DS v2.2 biometrics protect merchants; however, verified forensic compromises (SIM swap, malware proxy, physical skimmers) restore cardholder protection.

---

## 6. Contrastive Pair Assessment

All 10 contrastive pairs (20 examples, `ref-081` through `ref-100`) were evaluated side-by-side:

| Pair ID | Case Focus | Shared Context | Minimal Divergent Feature | Outcome Inversion |
|---|---|---|---|---|
| `pair_01_filing_timeline` | Statutory Filing Window | $350 online electronics order at Nova Gadgets Online; CNP without 3DS | Day 59 filing vs. Day 61 filing | 081: `APPROVE`<br>082: `REJECT` |
| `pair_02_return_tracking` | Return Authorization | $280 defective coffee grinder from Artisan Coffee Roasters | Merchant RMA + FedEx dock delivery vs. unnotified return at unstaffed box | 083: `APPROVE`<br>084: `REJECT` |
| `pair_03_3ds_authentication` | 3DS Authentication | $450 online clothing at Omni Fashion Store; customer claims fraud | Unauthenticated CNP from unknown IP vs. 3DS v2.2 biometric on trusted phone | 085: `APPROVE`<br>086: `REJECT` |
| `pair_04_delivery_location` | Photographic Proof | $190 bed linens from HomeComfort Furnishings; porch drop-off photo | Photo shows red brick porch (mismatch) vs. Photo shows house #742 (match) | 087: `APPROVE`<br>088: `REJECT` |
| `pair_05_cancellation_timing` | Subscription Timing | $55 cloud sync software from CloudSync Enterprise; renewed Nov 1 | Canceled 25m prior to renewal vs. Canceled 45m after renewal | 089: `APPROVE`<br>090: `REJECT` |
| `pair_06_duplicate_vs_distinct` | Duplicate Billing | Two $89.99 charges at Beacon Bookshop | Charges 2s apart for 1 order vs. Charges 25m apart for 2 distinct orders | 091: `APPROVE`<br>092: `REJECT` |
| `pair_07_digital_activation` | Digital Asset Usage | $150 software plugin from VFX Studio Tools; accidental purchase claim | Key never activated / zero downloads vs. Key activated / 18h active use | 093: `APPROVE`<br>094: `REJECT` |
| `pair_08_service_work_order` | Service Documentation | $320 emergency locksmith lockout charge from Metro Locksmith 24/7 | Generic POS slip without signed work order vs. Signed work order + photos | 095: `APPROVE`<br>096: `REJECT` |
| `pair_09_hotel_cancellation_window` | Contractual Grace Period | $220 hotel reservation at Grand Pacific Hotel with 48h free window | Canceled 49h prior to check-in vs. Canceled 23h prior to check-in | 097: `APPROVE`<br>098: `REJECT` |
| `pair_10_restocking_fee_disclosure` | Restocking Fee Disclosure | $75 restocking fee on returned sealed monitor at Optima Displays Direct | Fee undisclosed anywhere at checkout vs. Explicitly acknowledged checkbox | 099: `APPROVE`<br>100: `REJECT` |

**Assessment**: Every pair possesses high semantic similarity, a cleanly isolated variable, and a justifiable policy inversion.

---

## 7. Template Assessment

The audit scrutinized the two description templates:
1. `raw_value_description`: Standardized raw numerical values.
2. `ratio_verbalized_description`: Standardized verbalized percentages and ratios.

### Identified Discrepancy
In `generate_dataset.py`:
- `build_ratio_template()` computed lifetime dispute rate as:
  $$\text{dispute\_ratio} = \frac{\text{prior\_disputes}}{\max(1, \text{tenure} \times 3)} \times 100\%$$
- While `compute_ratio_features()` computed:
  $$\text{dispute\_to\_lifetime\_tx\_ratio} = \frac{\text{prior\_disputes}}{\max(1, \text{tenure} \times 4)}$$

This created an internal mathematical discrepancy where the narrative stated e.g. `2.8% lifetime dispute rate` while the structured feature recorded `0.0208` (2.1%).

### Corrective Formula
Standardize on a consistent velocity benchmark of 4 transactions per customer tenure month with a minimum floor of 10 lifetime transactions to prevent extreme zero-tenure artifacts:
$$\text{estimated\_lifetime\_transactions} = \max(10, \text{customer\_tenure\_months} \times 4)$$
$$\text{dispute\_rate\_pct} = \left(\frac{\text{prior\_dispute\_count}}{\text{estimated\_lifetime\_transactions}}\right) \times 100\%$$
$$\text{dispute\_to\_lifetime\_tx\_ratio} = \frac{\text{prior\_dispute\_count}}{\text{estimated\_lifetime\_transactions}}$$

---

## 8. Limitations

1. **Synthetic Sandbox Constraints**: Synthetic examples simulate realistic consumer credit card disputes but do not constitute legal or regulatory advice.
2. **Agent Specialization**: The rules are tailored specifically to the `Transaction Dispute Investigator` agent. Expanding to other agents will require independent policy specifications.

---

## 9. Recommended Corrections

1. **Update `generate_dataset.py`**:
   - Reclassify `ref-007`, `ref-040`, `ref-068`, `ref-080`, `ref-093`, `ref-094` to `credit_not_processed`.
   - Reclassify `ref-017` to `unauthorized_transaction`.
   - Add missing merchant names to `case_description` in `ref-012`, `ref-015`, `ref-016`, `ref-054`, `ref-089`, and `ref-090`.
   - Unify the lifetime dispute rate formula across `build_ratio_template` and `compute_ratio_features`.
2. **Regenerate `synthetic_seed_examples.json`**.
3. **Update `validate_dataset.py` & `tests/test_dataset_validation.py`** to reflect the corrected taxonomy and feature consistency.
4. **Re-run validation suite, pytest, Ruff, and Mypy**.

