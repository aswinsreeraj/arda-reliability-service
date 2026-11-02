# ARDA Reliability Service — Synthetic Dispute Adjudication Policy

This document defines the formal synthetic adjudication policy governing reference example evaluations for the **ARDA Reliability Service** (scoped to the `Transaction Dispute Investigator` agent).

> [!NOTE]
> These rules represent a **synthetic demonstration policy** designed for AI agent governance, vector retrieval benchmarking, and calibrated confidence evaluation. They model realistic payment card industry and consumer protection dynamics (e.g., card network chargeback guidelines, statutory cooling-off periods, and merchant evidentiary standards) without claiming to represent binding regulatory requirements.

---

## 1. Statutory Deadlines & Procedural Timelines

| Parameter | Policy Threshold | Effect of Non-Compliance | Reference Examples |
|---|---|---|---|
| **Dispute Filing Window** | **60 calendar days** from statement posting date | Filing after day 60 strictly extinguishes chargeback rights, requiring immediate **`REJECT`** regardless of fraud or defect merits. | `ref-example-020` (Day 80, `REJECT`), `ref-example-031` (Day 62, `REJECT`), `ref-example-081` (Day 59, `APPROVE`), `ref-example-082` (Day 61, `REJECT`) |
| **Merchant Arbitration Response Window** | **14 calendar days** from formal notice | Failure by the merchant to respond with evidence within 14 calendar days results in immediate procedural default **`APPROVE`** in favor of the cardholder. | `ref-example-001`, `ref-example-041`, `ref-example-050` |
| **Return Credit SLA** | **10–14 business days** post warehouse intake | If returned merchandise is delivered to the merchant warehouse (or accepted by carrier on a prepaid label) and the merchant fails to issue credit within SLA, the dispute is **`APPROVE`**d. | `ref-example-004`, `ref-example-046`, `ref-example-083` |
| **Cooling-Off / Booking Grace Period** | **24 hours** post booking / **48 hours** pre check-in | Cancellations submitted within statutory cooling-off windows or contractual free-cancellation periods must be refunded without penalty (**`APPROVE`**). | `ref-example-027`, `ref-example-097` (`APPROVE`) vs. `ref-example-098` (`REJECT`) |

---

## 2. Authentication & Unauthorized Transaction Hierarchy

The policy evaluates unauthorized transaction claims by weighing cryptographic authentication, physical card presence, and telemetry forensics:

```
+-----------------------------------------------------------------------------------------+
|                  AUTHENTICATION & UNAUTHORIZED TRANSACTION HIERARCHY                     |
+-----------------------------------------------------------------------------------------+
| Level 1: Conclusive Physical / Biometric Proof (Mandates REJECT)                        |
|   - Physical EMV Chip & PIN with 0 prior failed attempts (ref-012)                      |
|   - 3DS v2.2 with hardware token biometrics on primary trusted device (ref-086)          |
|   - Biometric Apple Pay / Face ID ordering to family member address (ref-071)           |
+-----------------------------------------------------------------------------------------+
| Level 2: High-Severity Forensic Rebuttals (Overrides Normal Authentication -> APPROVE)   |
|   - Carrier-verified unauthorized SIM swap prior to SMS OTP entry (ref-064)             |
|   - Verified forensic malware / infostealer trojan reverse-proxy attack (ref-061)        |
|   - Regulatory bulletin confirming physical POS skimmer at terminal + alibi (ref-010,070)|
|   - Documented OS biometric gate bypass vulnerability (ref-069)                         |
+-----------------------------------------------------------------------------------------+
| Level 3: Card-Not-Present (CNP) Defaults                                                |
|   - CNP without 3D Secure / biometric verification from unrecognized IP/device -> APPROVE|
|     (ref-003, ref-085)                                                                  |
|   - Domestic household use on shared family devices / friendly fraud -> REJECT           |
|     (ref-018, ref-078)                                                                  |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Fulfillment & Non-Receipt Evidence Rules

### Cardholder Burden vs. Merchant Burden

1. **Carrier Tracking as Prima Facie Proof**:
   - A carrier tracking record showing "Delivered" with GPS coordinates matching the cardholder's residential perimeter and/or recipient signature constitutes conclusive fulfillment (**`REJECT`**, e.g., `ref-example-011`, `ref-example-088`).
   - Digital goods are considered fulfilled upon documented license key activation or media streaming from the cardholder's known IP/device (**`REJECT`**, e.g., `ref-example-013`, `ref-example-037`).

2. **Rebuttals to Carrier Delivery Scans**:
   - **Photographic Mismatch**: Carrier drop-off photo clearly depicting architecturally mismatched premises (e.g. brick porch vs. wooden siding) refutes the scan (**`APPROVE`**, `ref-example-087`).
   - **Physical Impossibility**: Delivery status asserting placement in an impossible location (e.g. 45-lb lawn mower in a 6-inch mailbox) invalidates carrier tracking (**`APPROVE`**, `ref-example-062`).
   - **Predatory Porch Piracy**: Continuous home security video proving theft within minutes of delivery accompanied by an official police report warrants approval (**`APPROVE`**, `ref-example-063`).
   - **Unattended Common Areas**: Delivery dropped in an unsecured multi-unit apartment lobby without signature or lockbox verification fails delivery standards when supported by tenant affidavit (**`APPROVE`**, `ref-example-022`).
   - **Uncorroborated Porch Theft**: When carrier photo shows delivery to the front door and customer fails to file a police report, merchant delivery stands (**`REJECT`**, `ref-example-032`).

---

## 4. Defective Merchandise & Misrepresentation Rules

1. **Substantiation Requirement**:
   - Cardholders hold the initial evidentiary burden to substantiate claims of defect or counterfeit status using photographs, error logs, or third-party expert appraisals (`ref-example-006`, `ref-example-008`, `ref-example-024`).
   - Cardholder refusal to provide photos, diagnostics, or return the merchandise mandates rejection (**`REJECT`**, `ref-example-015`, `ref-example-051`, `ref-example-055`).

2. **Counterfeit Certification**:
   - Authorized manufacturer or certified third-party appraisal confirming counterfeit status overrides all merchant "all sales final" disclaimers (**`APPROVE`**, `ref-example-006`).

3. **Non-Material Defect Limitations**:
   - Minor packaging cosmetic blemishes that do not impair functionality do not constitute actionable defects (**`REJECT`**, `ref-example-034`).
   - Subjective appraisal valuation differences where physical gemstone/metal specifications strictly match invoice certificates are not grounds for chargeback (**`REJECT`**, `ref-example-038`).

---

## 5. Subscriptions, Cancellations, & Restocking Fees

1. **Subscription Termination Timing**:
   - Cancellations initiated **prior** to the renewal billing timestamp must be honored; system latency or batch processing race conditions do not justify renewal (**`APPROVE`**, `ref-example-005`, `ref-example-089`).
   - Cancellations initiated **after** renewal execution apply to the subsequent cycle (**`REJECT`**, `ref-example-033`, `ref-example-090`).
   - Ongoing active consumption of digital services after alleged cancellation refutes the dispute (**`REJECT`**, `ref-example-016`, `ref-example-033`, `ref-example-076`).
   - Immediate cancellation within hours of renewal with verified zero usage justifies equitable reversal (**`APPROVE`**, `ref-example-023`).

2. **Restocking Fees**:
   - Restocking fees are legally enforceable only if clearly, prominently, and conspicuously disclosed to the customer prior to purchase completion (**`REJECT`**, `ref-example-036`, `ref-example-100`).
   - Undisclosed, surprise restocking fees deducted from returned items are prohibited (**`APPROVE`**, `ref-example-099`).

---

## 6. Mathematical Ratios & Reference Benchmarks

For experimental ratio-verbalized templates, the synthetic policy defines the following normalized benchmarks:

- **Baseline Transaction Velocity**: Assumed at 4 transactions per customer tenure month.
  $$\text{Estimated Lifetime Transactions} = \max(10, \text{customer\_tenure\_months} \times 4)$$
- **Lifetime Dispute Rate**:
  $$\text{dispute\_to\_lifetime\_tx\_ratio} = \frac{\text{prior\_dispute\_count}}{\text{Estimated Lifetime Transactions}}$$
- **Statutory Window Elapsed Ratio**:
  $$\text{dispute\_filing\_to\_statutory\_window\_ratio} = \frac{\text{dispute\_filing\_delay\_days}}{60.0}$$
- **Portfolio Median Transaction Benchmark**: Defined as $\$125.00$.
  $$\text{amount\_to\_median\_transaction\_ratio} = \frac{\text{transaction\_amount}}{125.00}$$

