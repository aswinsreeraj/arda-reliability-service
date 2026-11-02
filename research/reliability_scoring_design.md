# ARDA Reliability Service — Reliability Scoring & Decision Engine Design

**Document Version**: 1.0.0  
**Status**: Implemented & Verified  
**Scope**: Development Week 3 — Reliability Scoring Engine & Gate Decisioning  
**Target Service**: `arda-reliability-service`  

---

## 1. Executive Overview

The **ARDA Reliability Service** provides an autonomous, auditable verification gate for AI agent decisions. In high-stakes fintech domains (such as payment dispute adjudication), autonomous AI agents must not be allowed to execute irreversible financial actions without verifiable evidentiary backing from historical precedents.

When an AI agent (e.g., the *Transaction Dispute Investigator*) proposes an outcome (`APPROVE` or `REJECT`) on an incoming case, the Reliability Service executes a two-phase adjudication:
1. **Semantic Vector Retrieval**: Retrieves the top-$K$ nearest historical reference decisions within the tenant and agent scope using pgvector cosine indexing.
2. **Reliability Scoring & Gate Decisioning**: Evaluates whether retrieved reference evidence consistently and strongly supports the agent's proposed action, returning either **`PASS`** (proceed autonomously) or **`ESCALATE`** (route to human-in-the-loop review).

```
+------------------------+      +---------------------------+      +---------------------------+
| Incoming Agent Request |      | Phase 1: Vector Retrieval |      | Phase 2: Scoring & Gating |
|------------------------|      |---------------------------|      |---------------------------|
| tenant_id              | ===> | Embed case_description    | ===> | Agreement                 |
| agent_id               |      | pgvector HNSW (tenant/agt)|      | Density Factor            |
| case_description       |      | Retrieve top-K precedents |      | Coverage Factor           |
| proposed_decision      |      +---------------------------+      | raw_score = A * D * C     |
+------------------------+                                         +---------------------------+
                                                                                 ||
                                                                                 \/
                                                                   +---------------------------+
                                                                   | Gate Outcome:             |
                                                                   | PASS or ESCALATE          |
                                                                   +---------------------------+
```

> [!CRITICAL]
> **Strict Semantic Boundary**: Only `case_description` is converted into an embedding vector. The agent's `proposed_decision`, historical reference labels, rationales, and evaluation results are **NEVER** embedded or passed to vector search. Retrieval searches solely for factual precedent relevance.

---

## 2. Mathematical Scoring Formulation

The raw reliability score is computed via a multiplicative three-factor decomposition:

$$\text{raw\_score} = \text{Agreement} \times \text{Density Factor} \times \text{Coverage Factor}$$

Each component is independently bounded in the closed interval $[0.0, 1.0]$, ensuring that $\text{raw\_score} \in [0.0, 1.0]$. The multiplicative formulation guarantees a **strict gating property**: if any single dimension fails entirely (e.g., zero agreement, zero density, or zero coverage), the aggregate reliability score collapses to zero.

```
       Agreement (Directional Consensus)
                    X
      Density Factor (Proximity & Strength)
                    X
     Coverage Factor (Breadth & Decisiveness)
                    ||
            Raw Reliability Score
```

---

### 2.1 Agreement ($A$)

**Purpose**: Measures the degree to which retrieved precedent outcomes agree with the agent's proposed decision.

#### Mathematical Definition
Let $N = \{ (y_i, s_i) \}_{i=1}^k$ be the sequence of retrieved reference neighbors, where $y_i \in \{\text{APPROVE}, \text{REJECT}\}$ is the adjudicated historical label, and $s_i \in [-1, 1]$ is the cosine similarity score.  
Let $y_{\text{prop}} \in \{\text{APPROVE}, \text{REJECT}\}$ be the agent's proposed decision.  
Let $M = \{ i \in \{1, \dots, k\} \mid y_i = y_{\text{prop}} \}$ be the set of matching neighbors.

In **similarity-weighted mode** (`weighting_strategy = "similarity_weighted"`, default):
$$w_i = \max(0.0, s_i)$$

$$\text{Agreement} = \begin{cases} 
\dfrac{\sum_{i \in M} w_i}{\sum_{i=1}^k w_i} & \text{if } \sum_{i=1}^k w_i > 0 \\
0.0 & \text{if } k = 0 \text{ or } \sum_{i=1}^k w_i = 0 
\end{cases}$$

In **unweighted mode** (`weighting_strategy = "unweighted"`):
$$\text{Agreement} = \frac{|M|}{k} \quad (\text{for } k > 0)$$

#### Properties
1. **Boundedness**: Strictly within $[0.0, 1.0]$.
2. **Unanimous Consensus**: If all precedents match ($|M| = k$), $A = 1.0$.
3. **Unanimous Conflict**: If all precedents conflict ($|M| = 0$), $A = 0.0$.
4. **Weighted Voting**: A high-similarity precedent ($s_i = 0.95$) exerts greater voting mass than a marginal neighbor ($s_i = 0.71$).
5. **Non-Probabilistic**: Similarity scores represent geometrical proximity on a unit hypersphere, not class posterior probabilities. Agreement is an evidence vote, not a Bayes estimate.

---

### 2.2 Density Factor ($D$)

**Purpose**: Measures the proximity and concentration of precedent evidence around the query in embedding space. Distinguishes cases situated in dense precedent regions from those in sparse, unexplored, or out-of-distribution margins.

#### Mathematical Definition
Let $s_{\min}$ be the configured similarity floor (`min_similarity_threshold`, e.g., $0.70$) below which an example is considered irrelevant noise.  
Let $R = \{ i \in \{1, \dots, k\} \mid s_i \ge s_{\min} \}$ be the subset of relevant neighbors.

If $|R| = 0$, then $D = 0.0$.

If $|R| > 0$:
1. **Normalized Precedent Relevance**:
   $$r_i = \text{clamp}\left( \frac{s_i - s_{\min}}{1.0 - s_{\min}}, 0.0, 1.0 \right)$$
2. **Peak Precedent Fidelity**:
   $$r_{\text{top}} = \max_{i \in R}(r_i)$$
3. **Cohort Mean Fidelity**:
   $$\bar{r} = \frac{1}{|R|} \sum_{i \in R} r_i$$
4. **Quantity Saturation**:
   $$q = \min\left(1.0, \frac{|R|}{k_{\min}}\right)$$
5. **Blended Density**:
   $$D = q \cdot \left[ \alpha \cdot r_{\text{top}} + (1.0 - \alpha) \cdot \bar{r} \right]$$
   where $\alpha \in [0.0, 1.0]$ (`density_peak_weight`, default $0.50$) balances top-precedent match against cohort consensus.

#### Properties
- If all neighbors have $s_i \le s_{\min}$, $D = 0.0$.
- If $|R| < k_{\min}$, density is discounted proportionally ($q < 1.0$).
- Weak evidence hovering near the floor ($s_i \approx 0.71$) yields $r_i \approx 0.03$, resulting in negligible density ($D < 0.05$).
- Highly concentrated evidence ($s_i \ge 0.95$, $|R| \ge k_{\min}$) yields $D > 0.80$.

---

### 2.3 Coverage Factor ($C$)

**Purpose**: Measures whether the retrieved evidence provides adequate sample breadth and decisiveness. Severe precedent conflict (such as retrieving counterfactual twins at policy boundaries) degrades coverage.

#### Mathematical Definition
Let $k = \text{len}(N)$ be the total retrieved neighbors.  
Let $R = \{ i \in \{1, \dots, k\} \mid s_i \ge s_{\min} \}$ be the relevant neighbors.

If $k = 0$ or $|R| = 0$, then $C = 0.0$.

If $|R| > 0$:
1. **Sample Size Adequacy ($A_{\text{sample}}$)**:
   $$A_{\text{sample}} = \min\left(1.0, \frac{|R|}{k_{\min}}\right)$$
2. **Relevance Ratio ($V_{\text{rel}}$)**:
   $$V_{\text{rel}} = \frac{|R|}{k}$$
3. **Precedent Conflict Rate ($CR$)**:
   Let $R_M = \{ i \in R \mid y_i = y_{\text{prop}} \}$ and $R_C = \{ i \in R \mid y_i \neq y_{\text{prop}} \}$.
   $$CR = \frac{2 \cdot \min(|R_M|, |R_C|)}{|R|} \in [0.0, 1.0]$$
4. **Precedent Clarity ($C_{\text{clarity}}$)**:
   $$C_{\text{clarity}} = 1.0 - \beta \cdot CR$$
   where $\beta \in [0.0, 1.0]$ (`conflict_penalty_weight`, default $0.50$) governs the severity of the conflict discount.
5. **Final Coverage Factor**:
   $$C = A_{\text{sample}} \times V_{\text{rel}} \times C_{\text{clarity}}$$

#### Properties
- **Noise Rejection**: If 5 neighbors are retrieved but only 1 is relevant ($s \ge s_{\min}$), $V_{\text{rel}} = 1/5 = 0.20$ and $A_{\text{sample}} = 1/3 = 0.33$, driving $C \approx 0.067$.
- **Conflict Sensitivity**: When relevant precedents are split 50/50 ($|R_M| = |R_C|$), $CR = 1.0$, reducing clarity to $1.0 - \beta$ ($0.50$).
- **Sufficient & Unanimous**: If $|R| \ge k_{\min}$, all $k$ are relevant, and zero conflict exists, $C = 1.0$.

---

## 3. Operational Decision Policy & Gate Architecture

The platform gate decision is strictly binary:
- **`PASS`**: The agent's proposed action is supported by dense, unanimous, and sufficient historical precedent. The agent proceeds autonomously without human intervention.
- **`ESCALATE`**: Precedent evidence is absent, weak, sparse, conflicting, or below operational reliability standards. The case is routed to human-in-the-loop review.

> [!WARNING]
> Under no circumstances does the platform issue a `REJECT` gate outcome. The agent proposes `APPROVE` or `REJECT`. The platform evaluates whether that proposal can `PASS` or must `ESCALATE`.

```
                                  [Reliability Score Evaluated]
                                                |
                                                v
                              +-----------------------------------+
                              | 1. Relevant Neighbors >= k_min?   |--- NO ---> ESCALATE (COLD_START)
                              +-----------------------------------+
                                                | YES
                                                v
                              +-----------------------------------+
                              | 2. Coverage Factor >= c_min?      |--- NO ---> ESCALATE (INSUFFICIENT_COVERAGE)
                              +-----------------------------------+
                                                | YES
                                                v
                              +-----------------------------------+
                              | 3. Density Factor >= d_min?       |--- NO ---> ESCALATE (WEAK_EVIDENCE_DENSITY)
                              +-----------------------------------+
                                                | YES
                                                v
                              +-----------------------------------+
                              | 4. Any Conflict s_i >= s_veto?    |--- YES --> ESCALATE (HIGH_SIMILARITY_CONFLICT)
                              +-----------------------------------+
                                                | NO
                                                v
                              +-----------------------------------+
                              | 5. Agreement >= a_min?            |--- NO ---> ESCALATE (CONFLICTING_EVIDENCE)
                              +-----------------------------------+
                                                | YES
                                                v
                              +-----------------------------------+
                              | 6. raw_score >= tau_reliability?  |--- NO ---> ESCALATE (SCORE_BELOW_THRESHOLD)
                              +-----------------------------------+
                                                | YES
                                                v
                                         [ PASS DECISION ]
```

### Policy Evaluation Hierarchy & Reason Codes

Every evaluation outcome is tagged with a deterministic machine-readable reason code:

| Hierarchy Level | Gate Condition | Reason Code | Explanation |
|---|---|---|---|
| **1. Cold Start** | $|R| < k_{\min}$ | `COLD_START_INSUFFICIENT_EVIDENCE` | Fewer than $k_{\min}$ relevant precedents exist within the agent's historical corpus. |
| **2. Coverage Floor** | $C < c_{\min}$ | `INSUFFICIENT_COVERAGE` | Precedent coverage factor is inadequate due to noise or low sample adequacy. |
| **3. Density Floor** | $D < d_{\min}$ | `WEAK_EVIDENCE_DENSITY` | Precedents are geographically distant or weakly aligned in embedding space. |
| **4. Conflict Veto** | $\exists n \in R_C : s_n \ge s_{\text{veto}}$ | `HIGH_SIMILARITY_CONFLICT` | A conflicting precedent with near-identical semantics ($\ge 0.95$) was retrieved, signaling a policy boundary condition. |
| **5. Agreement Floor** | $A < a_{\min}$ | `CONFLICTING_EVIDENCE` | Precedent outcome consensus falls below the minimum agreement threshold. |
| **6. Score Floor** | $\text{raw\_score} < \tau$ | `SCORE_BELOW_THRESHOLD` | Aggregate reliability score is below the operational pass threshold. |
| **7. Qualified Pass** | All checks pass | `RELIABILITY_THRESHOLD_MET` | Precedent evidence is dense, unanimous, and satisfies all reliability constraints. |

---

## 4. Cold-Start Dynamics & Sparse Partitions

A fundamental requirement of ARDA reliability gating is that **insufficient evidence must never silently PASS**.

1. **New Tenant / New Agent Deployment**:
   - When a tenant or agent has zero reference examples, vector retrieval returns 0 rows.
   - Agreement $= 0.0$, Density $= 0.0$, Coverage $= 0.0$, and Raw Score $= 0.0$.
   - Check 1 immediately triggers `GateDecision.ESCALATE` with `COLD_START_INSUFFICIENT_EVIDENCE`.
2. **Sparse Partitions ($1 \le |R| < k_{\min}$)**:
   - If an agent only has 1 or 2 reference examples, it cannot satisfy the statistical significance requirement ($k_{\min} = 3$).
   - The engine automatically routes the case to human review.
   - As human operators adjudicate these cases, verified decisions are promoted into the reference repository, gradually transitioning the agent out of cold-start mode.

---

## 5. Structured Configuration & Provisional Defaults

All operational scoring and gating thresholds are managed via `ScoringConfig`:

| Parameter | Type | Default | Domain Invariant | Purpose |
|---|:---:|:---:|:---:|---|
| `min_neighbor_count` ($k_{\min}$) | `int` | **3** | $> 0$ | Minimum relevant precedents required to avoid cold-start escalation. |
| `min_similarity_threshold` ($s_{\min}$) | `float` | **0.70** | $[0.0, 1.0)$ | Cosine similarity floor below which neighbors are considered noise. |
| `min_coverage_threshold` ($c_{\min}$) | `float` | **0.50** | $(0.0, 1.0]$ | Minimum evidence coverage factor required to qualify for PASS. |
| `min_density_threshold` ($d_{\min}$) | `float` | **0.40** | $(0.0, 1.0]$ | Minimum evidence density factor required to qualify for PASS. |
| `min_agreement_threshold` ($a_{\min}$) | `float` | **0.60** | $(0.0, 1.0]$ | Minimum precedent agreement factor required to qualify for PASS. |
| `min_reliability_threshold` ($\tau$) | `float` | **0.65** | $(0.0, 1.0]$ | Minimum raw reliability score required to qualify for PASS. |
| `conflict_penalty_weight` ($\beta$) | `float` | **0.50** | $[0.0, 1.0]$ | Penalty weight applied to coverage when relevant precedents conflict. |
| `conflict_veto_similarity` ($s_{\text{veto}}$) | `float` | **0.95** | $[0.0, 1.0]$ | Similarity threshold above which any conflicting neighbor triggers veto. |
| `density_peak_weight` ($\alpha$) | `float` | **0.50** | $[0.0, 1.0]$ | Weight of top-1 neighbor similarity vs average cohort similarity. |
| `top_k` | `int` | **5** | $\ge k_{\min}$ | Number of nearest neighbors to retrieve from vector database. |
| `ef_search` | `int` | **64** | $> 0$ | HNSW index graph exploration parameter. |
| `weighting_strategy` | `str` | `"similarity_weighted"` | `{"similarity_weighted", "unweighted"}` | Precedent vote weighting mode. |

> [!IMPORTANT]
> **Provisional Defaults Disclaimer**: All thresholds listed above are developmental defaults established during Week 3 implementation. They are **NOT** calibrated production thresholds. Formal calibration against empirical risk distributions is scheduled for subsequent milestones.

---

## 6. Why Raw Reliability Score is Not Calibrated Confidence

The raw reliability score $\text{raw\_score} \in [0.0, 1.0]$ must **never** be interpreted as a Bayesian posterior probability or a calibrated confidence metric:

1. **Geometric, Not Probabilistic**:
   Cosine similarities in SentenceTransformers reflect relative angles on a unit sphere. A similarity of $0.85$ does not imply an 85% probability of correctness.
2. **Cohort Density vs. Risk**:
   High precedent density means a case looks similar to past examples, but it does not account for label noise or shifts in merchant dispute policies.
3. **Multiplicative Distortion**:
   Multiplying $A \times D \times C$ creates a nonlinear metric designed for conservative gating. For example, $0.90 \times 0.80 \times 0.90 = 0.648$. An evaluation with strong evidence across all dimensions produces a raw score of ~0.65, which is structurally lower than any constituent factor.
4. **Subsequent Calibration Plan**:
   In Week 4, a calibration mapping (e.g., isotonic regression or temperature-scaled sigmoid) will be fitted to map $\text{raw\_score} \mapsto \hat{P}(\text{Decision Correct} \mid \text{raw\_score})$. Until that milestone, the raw score serves solely as an ordinal gate index.

---

## 7. Five Worked Synthetic Case Evaluations

The following worked examples demonstrate the exact numerical execution of the scoring engine and decision policy across diverse scenario types.

---

### Example 1: Clear Unanimous Fraud Precedent ($\implies \text{PASS}$)

#### Scenario
- **Agent Proposed Decision**: `APPROVE`
- **Case Description**: Cardholder reports unauthorized magnetic stripe debit of \$240 at gas station in Mexico while cardholder was in Texas. Claim filed 12 days post-statement.
- **Config**: $k_{\min} = 3, s_{\min} = 0.70, c_{\min} = 0.50, d_{\min} = 0.40, a_{\min} = 0.60, \tau = 0.65, \beta = 0.50, \alpha = 0.50$.

#### Retrieved Precedents ($k=5$)
| Precedent ID | Label | Cosine Similarity ($s_i$) | Relevance ($s_i \ge 0.70$) | Normalized Relevance ($r_i$) |
|---|:---:|:---:|:---:|:---:|
| `ref-001` | `APPROVE` | 0.96 | Yes | $(0.96 - 0.70)/0.30 = 0.8667$ |
| `ref-010` | `APPROVE` | 0.94 | Yes | $(0.94 - 0.70)/0.30 = 0.8000$ |
| `ref-042` | `APPROVE` | 0.91 | Yes | $(0.91 - 0.70)/0.30 = 0.7000$ |
| `ref-070` | `APPROVE` | 0.89 | Yes | $(0.89 - 0.70)/0.30 = 0.6333$ |
| `ref-081` | `APPROVE` | 0.85 | Yes | $(0.85 - 0.70)/0.30 = 0.5000$ |

#### Step-by-Step Scoring
1. **Agreement ($A$)**:
   - Matching: 5 ($w_M = 0.96 + 0.94 + 0.91 + 0.89 + 0.85 = 4.55$).
   - Total weight: $4.55$.
   - $A = 4.55 / 4.55 = \mathbf{1.0000}$.
2. **Density ($D$)**:
   - Relevant count: $|R| = 5 \ge 3 \implies q = 1.0$.
   - $r_{\text{top}} = 0.8667$.
   - $\bar{r} = (0.8667 + 0.8000 + 0.7000 + 0.6333 + 0.5000) / 5 = 0.7000$.
   - Blended: $0.5 \times 0.8667 + 0.5 \times 0.7000 = 0.7833$.
   - $D = 1.0 \times 0.7833 = \mathbf{0.7833}$.
3. **Coverage ($C$)**:
   - Adequacy: $\min(1.0, 5/3) = 1.0$.
   - Relevance ratio: $5/5 = 1.0$.
   - Conflicting relevant: $0 \implies CR = 0.0 \implies C_{\text{clarity}} = 1.0 - 0.5(0) = 1.0$.
   - $C = 1.0 \times 1.0 \times 1.0 = \mathbf{1.0000}$.
4. **Raw Score**:
   $$\text{raw\_score} = 1.0000 \times 0.7833 \times 1.0000 = \mathbf{0.7833}$$

#### Policy Evaluation
- Check 1: $|R| = 5 \ge 3$ (Pass)
- Check 2: $C = 1.0000 \ge 0.50$ (Pass)
- Check 3: $D = 0.7833 \ge 0.40$ (Pass)
- Check 4: No conflict $\ge 0.95$ (Pass)
- Check 5: $A = 1.0000 \ge 0.60$ (Pass)
- Check 6: $\text{raw\_score} = 0.7833 \ge 0.65$ (Pass)

**Outcome**: **`PASS`**  
**Reason Code**: `RELIABILITY_THRESHOLD_MET`  
**Evaluation Reason**: Reliability score (0.7833) meets operational threshold (0.6500) with 5 supporting precedents.

---

### Example 2: Ambiguous Policy Boundary with Contrastive Twin ($\implies \text{ESCALATE}$)

#### Scenario
- **Agent Proposed Decision**: `APPROVE`
- **Case Description**: Cardholder disputes hotel reservation cancellation fee (\$350) claimed 47 hours prior to check-in.
- **Context**: The policy defines a strict 48-hour cancellation deadline. A counterfactual twin exists where cancellation was 49 hours prior (`APPROVE`) and 47 hours prior (`REJECT`).

#### Retrieved Precedents ($k=5$)
| Precedent ID | Label | Cosine Similarity ($s_i$) | Relevance ($s_i \ge 0.70$) | Normalized Relevance ($r_i$) |
|---|:---:|:---:|:---:|:---:|
| `ref-098` | `REJECT` | **0.97** | Yes | $(0.97 - 0.70)/0.30 = 0.9000$ |
| `ref-097` | `APPROVE` | 0.96 | Yes | $(0.96 - 0.70)/0.30 = 0.8667$ |
| `ref-027` | `APPROVE` | 0.88 | Yes | $(0.88 - 0.70)/0.30 = 0.6000$ |
| `ref-033` | `REJECT` | 0.81 | Yes | $(0.81 - 0.70)/0.30 = 0.3667$ |
| `ref-016` | `APPROVE` | 0.74 | Yes | $(0.74 - 0.70)/0.30 = 0.1333$ |

#### Step-by-Step Scoring
1. **Agreement ($A$)**:
   - Matching: 3 (`ref-097`: 0.96, `ref-027`: 0.88, `ref-016`: 0.74) $\implies w_M = 2.58$.
   - Conflicting: 2 (`ref-098`: 0.97, `ref-033`: 0.81) $\implies w_C = 1.78$.
   - Total weight: $4.36$.
   - $A = 2.58 / 4.36 = \mathbf{0.5917}$.
2. **Density ($D$)**:
   - $|R| = 5 \ge 3 \implies q = 1.0$.
   - $r_{\text{top}} = 0.9000$.
   - $\bar{r} = (0.90 + 0.8667 + 0.60 + 0.3667 + 0.1333) / 5 = 0.5733$.
   - Blended: $0.5 \times 0.90 + 0.5 \times 0.5733 = 0.7367 \implies D = \mathbf{0.7367}$.
3. **Coverage ($C$)**:
   - $A_{\text{sample}} = 1.0, V_{\text{rel}} = 1.0$.
   - Conflicting relevant: $R_C = 2, R_M = 3$.
   - $CR = (2 \times 2) / 5 = 0.80$.
   - $C_{\text{clarity}} = 1.0 - 0.5(0.80) = 0.6000 \implies C = \mathbf{0.6000}$.
4. **Raw Score**:
   $$\text{raw\_score} = 0.5917 \times 0.7367 \times 0.6000 = \mathbf{0.2615}$$

#### Policy Evaluation
- Check 1: $|R| = 5 \ge 3$ (Pass)
- Check 2: $C = 0.6000 \ge 0.50$ (Pass)
- Check 3: $D = 0.7367 \ge 0.40$ (Pass)
- **Check 4: Conflict Veto Triggered!**
  Conflicting neighbor `ref-098` has $s_i = 0.97 \ge 0.95$.

**Outcome**: **`ESCALATE`**  
**Reason Code**: `HIGH_SIMILARITY_CONFLICT`  
**Evaluation Reason**: High-similarity conflicting precedent detected (`ref-098`, similarity=0.9700, label='REJECT') violating conflict veto threshold (0.9500).

---

### Example 3: Cold-Start / Sparse Precedents ($\implies \text{ESCALATE}$)

#### Scenario
- **Agent Proposed Decision**: `APPROVE`
- **Case Description**: Novel cryptocurrency off-ramp dispute involving smart contract escrow timeout.
- **Context**: Only 1 relevant historical example exists in the agent partition.

#### Retrieved Precedents ($k=5$)
| Precedent ID | Label | Cosine Similarity ($s_i$) | Relevance ($s_i \ge 0.70$) | Normalized Relevance ($r_i$) |
|---|:---:|:---:|:---:|:---:|
| `ref-crypto-1` | `APPROVE` | 0.82 | Yes | $(0.82 - 0.70)/0.30 = 0.4000$ |
| `ref-gen-1` | `REJECT` | 0.62 | No (noise) | 0.0 |
| `ref-gen-2` | `APPROVE` | 0.58 | No (noise) | 0.0 |
| `ref-gen-3` | `REJECT` | 0.51 | No (noise) | 0.0 |
| `ref-gen-4` | `REJECT` | 0.45 | No (noise) | 0.0 |

#### Step-by-Step Scoring
1. **Agreement ($A$)**:
   - $w_M = 0.82 + 0.58 = 1.40$. Total weight $= 0.82 + 0.62 + 0.58 + 0.51 + 0.45 = 2.98$.
   - $A = 1.40 / 2.98 = \mathbf{0.4698}$.
2. **Density ($D$)**:
   - Relevant count: $|R| = 1 < 3 \implies q = 1/3 = 0.3333$.
   - $r_{\text{top}} = 0.4000, \bar{r} = 0.4000 \implies \text{blend} = 0.4000$.
   - $D = 0.3333 \times 0.4000 = \mathbf{0.1333}$.
3. **Coverage ($C$)**:
   - $A_{\text{sample}} = 1/3 = 0.3333$.
   - $V_{\text{rel}} = 1/5 = 0.2000$.
   - Conflicting relevant: $0 \implies C_{\text{clarity}} = 1.0$.
   - $C = 0.3333 \times 0.2000 \times 1.0 = \mathbf{0.0667}$.
4. **Raw Score**:
   $$\text{raw\_score} = 0.4698 \times 0.1333 \times 0.0667 = \mathbf{0.0042}$$

#### Policy Evaluation
- **Check 1: Cold Start Triggered!**
  $|R| = 1 < 3$.

**Outcome**: **`ESCALATE`**  
**Reason Code**: `COLD_START_INSUFFICIENT_EVIDENCE`  
**Evaluation Reason**: Cold start or insufficient relevant evidence: found 1 relevant precedents meeting similarity threshold 0.70 (minimum required is 3).

---

### Example 4: Distant / Weak Precedent Cluster ($\implies \text{ESCALATE}$)

#### Scenario
- **Agent Proposed Decision**: `REJECT`
- **Case Description**: Dispute regarding custom tailoring blemish, cardholder refuses photographic submission.
- **Context**: 5 precedents retrieved, all agree (`REJECT`), but all hover barely above the similarity floor.

#### Retrieved Precedents ($k=5$)
| Precedent ID | Label | Cosine Similarity ($s_i$) | Relevance ($s_i \ge 0.70$) | Normalized Relevance ($r_i$) |
|---|:---:|:---:|:---:|:---:|
| `ref-051` | `REJECT` | 0.72 | Yes | $(0.72 - 0.70)/0.30 = 0.0667$ |
| `ref-055` | `REJECT` | 0.71 | Yes | $(0.71 - 0.70)/0.30 = 0.0333$ |
| `ref-015` | `REJECT` | 0.71 | Yes | $(0.71 - 0.70)/0.30 = 0.0333$ |
| `ref-034` | `REJECT` | 0.70 | Yes | $(0.70 - 0.70)/0.30 = 0.0000$ |
| `ref-038` | `REJECT` | 0.70 | Yes | $(0.70 - 0.70)/0.30 = 0.0000$ |

#### Step-by-Step Scoring
1. **Agreement ($A$)**:
   - All 5 agree $\implies A = \mathbf{1.0000}$.
2. **Density ($D$)**:
   - $|R| = 5 \ge 3 \implies q = 1.0$.
   - $r_{\text{top}} = 0.0667$.
   - $\bar{r} = (0.0667 + 0.0333 + 0.0333 + 0.0 + 0.0) / 5 = 0.0267$.
   - Blended: $0.5 \times 0.0667 + 0.5 \times 0.0267 = 0.0467$.
   - $D = 1.0 \times 0.0467 = \mathbf{0.0467}$.
3. **Coverage ($C$)**:
   - $A_{\text{sample}} = 1.0, V_{\text{rel}} = 1.0$. Zero conflict $\implies C_{\text{clarity}} = 1.0$.
   - $C = \mathbf{1.0000}$.
4. **Raw Score**:
   $$\text{raw\_score} = 1.0000 \times 0.0467 \times 1.0000 = \mathbf{0.0467}$$

#### Policy Evaluation
- Check 1: $|R| = 5 \ge 3$ (Pass)
- Check 2: $C = 1.0000 \ge 0.50$ (Pass)
- **Check 3: Weak Evidence Density Triggered!**
  $D = 0.0467 < 0.40$.

**Outcome**: **`ESCALATE`**  
**Reason Code**: `WEAK_EVIDENCE_DENSITY`  
**Evaluation Reason**: Evidence density factor (0.0467) is below minimum threshold (0.4000).

---

### Example 5: Moderate Density with Substantial Precedent Conflict ($\implies \text{ESCALATE}$)

#### Scenario
- **Agent Proposed Decision**: `APPROVE`
- **Case Description**: Cardholder claims goods lost in transit; tracking status shows "Delivered" but customer provided neighbor affidavit of common area package theft.
- **Context**: Precedents are divided on uncorroborated carrier delivery vs. tenant affidavit corroboration.

#### Retrieved Precedents ($k=5$)
| Precedent ID | Label | Cosine Similarity ($s_i$) | Relevance ($s_i \ge 0.70$) | Normalized Relevance ($r_i$) |
|---|:---:|:---:|:---:|:---:|
| `ref-022` | `APPROVE` | 0.88 | Yes | 0.6000 |
| `ref-088` | `REJECT` | 0.86 | Yes | 0.5333 |
| `ref-063` | `APPROVE` | 0.85 | Yes | 0.5000 |
| `ref-032` | `REJECT` | 0.84 | Yes | 0.4667 |
| `ref-011` | `REJECT` | 0.81 | Yes | 0.3667 |

#### Step-by-Step Scoring
1. **Agreement ($A$)**:
   - Matching: 2 (`ref-022`: 0.88, `ref-063`: 0.85) $\implies w_M = 1.73$.
   - Conflicting: 3 (`ref-088`: 0.86, `ref-032`: 0.84, `ref-011`: 0.81) $\implies w_C = 2.51$.
   - Total weight: $4.24$.
   - $A = 1.73 / 4.24 = \mathbf{0.4080}$.
2. **Density ($D$)**:
   - $|R| = 5 \ge 3 \implies q = 1.0$.
   - $r_{\text{top}} = 0.6000$.
   - $\bar{r} = (0.60 + 0.5333 + 0.50 + 0.4667 + 0.3667) / 5 = 0.4933$.
   - Blended: $0.5 \times 0.60 + 0.5 \times 0.4933 = 0.5467 \implies D = \mathbf{0.5467}$.
3. **Coverage ($C$)**:
   - $A_{\text{sample}} = 1.0, V_{\text{rel}} = 1.0$.
   - $R_M = 2, R_C = 3 \implies CR = (2 \times 2) / 5 = 0.80$.
   - $C_{\text{clarity}} = 1.0 - 0.5(0.80) = 0.6000 \implies C = \mathbf{0.6000}$.
4. **Raw Score**:
   $$\text{raw\_score} = 0.4080 \times 0.5467 \times 0.6000 = \mathbf{0.1338}$$

#### Policy Evaluation
- Check 1: $|R| = 5 \ge 3$ (Pass)
- Check 2: $C = 0.6000 \ge 0.50$ (Pass)
- Check 3: $D = 0.5467 \ge 0.40$ (Pass)
- Check 4: Max conflicting similarity is $0.86 < 0.95$ (Pass)
- **Check 5: Conflicting Evidence Triggered!**
  $A = 0.4080 < 0.60$.

**Outcome**: **`ESCALATE`**  
**Reason Code**: `CONFLICTING_EVIDENCE`  
**Evaluation Reason**: Precedent agreement (0.4080) is below minimum threshold (0.6000) with 3 conflicting precedents.

---

## 8. Architectural Alignment with Synthetic Adjudication Policy

The scoring and decision engine operates in exact harmony with the rules codified in `research/datasets/synthetic_adjudication_policy.md`:

1. **Statutory 60-Day Window Enforcement**:
   Cases filed after 60 days strictly match `ref-020` and `ref-031` (`REJECT`). If an agent attempts to `APPROVE` a late-filed case, agreement drops to 0.0, immediately escalating to human review.
2. **Procedural Defaults (Merchant Response Window)**:
   Merchant non-response cases match `ref-001` and `ref-041` (`APPROVE`). An agent proposing `APPROVE` receives 1.0 agreement, high density, and a clean `PASS`.
3. **Counterfactual Twin Detection (Contrastive Pairs)**:
   Borderline disputes (such as cooling-off periods or 3DS token exceptions) retrieve counterfactual twins with opposing labels and near-identical similarity ($\ge 0.95$). Check 4 (Conflict Veto) prevents the agent from guessing across policy boundaries, mandating human oversight.

---

## 9. Known Limitations & Roadmap

1. **Uncalibrated Thresholds**:
   The current operational thresholds ($k_{\min}=3, s_{\min}=0.70, c_{\min}=0.50, d_{\min}=0.40, a_{\min}=0.60, \tau=0.65$) are developmental defaults. They will be calibrated in Week 4 against empirical chargeback reversal risk distributions.
2. **Embedding Model Sensitivity**:
   Density factor relies on cosine distances generated by `BAAI/bge-small-en-v1.5`. If the embedding model is upgraded to `e5-base` or another architecture, $s_{\min}$ and $d_{\min}$ must be retuned to account for different embedding space dispersion.
3. **Subsequent Milestones**:
   - Week 3 Gate Handler: Implement gRPC service endpoint (`reliability.proto`).
   - Week 4 Confidence Calibration: Map `raw_score` to empirical error rates via Platt scaling or isotonic regression.
   - Week 4 Redpanda Streaming: Ingest asynchronous human review feedback and update reference example datasets.

