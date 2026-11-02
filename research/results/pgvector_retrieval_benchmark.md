# ARDA Reliability Service — Vector Retrieval Benchmark Report

**Benchmark Date**: 2026-11-02 13:46:07 UTC
**Database Engine**: PostgreSQL 16.15 with pgvector v0.8.7
**Host/Port**: `127.0.0.1:5433` (Docker container: `arda-reliability-postgres`)
**Provisional Embedding Model**: `BAAI/bge-small-en-v1.5` (Dimension: 384)
**Operational Input Representation**: `ratio_verbalized_description`

---

## 1. Executive Summary & Verification Outcomes

This report documents verified retrieval performance across dataset scales from $N=100$ to $N=10,000$, distinguishing exact ground truth, actual HNSW graph traversal, forced diagnostic execution, and normal PostgreSQL cost-based optimization.

### Core Verification Discoveries
- **Scale-Dependent Planner Selection**: For small partitions ($N=100$), PostgreSQL's optimizer chooses a B-tree index scan + in-memory Top-N heapsort because reading 19 heap pages is lower cost than the HNSW graph entry cost (43 vs 116). At scale ($N=10,000$), the optimizer **naturally and unconditionally chooses the HNSW index**.
- **pgvector 0.8.x Iterative Scan Requirement**: In shared multi-agent tables, `hnsw.iterative_scan = strict_order` is mandatory to prevent graph search truncation on smaller partitions.
- **HNSW Fidelity**: At $N=10,000$, HNSW with $m=16, ef\_construction=64, ef\_search=64$ achieves **98.8% Recall@5** and **98.2% Recall@10** with **0.75 ms** query execution time.

---

## 2. Multi-Scale Retrieval Performance Summary

| Partition Scale | Planner Mode | Index Utilized | Recall@1 | Recall@5 | Recall@10 | Database Execution | Total Pipeline (p50) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$N=100$** | Default Planner | B-Tree `idx_..._tenant_agent_status` | 100.0% | 100.0% | 100.0% | 0.28 ms | 11.2 ms |
| **$N=100$** | Forced HNSW (Diag) | HNSW `idx_..._hnsw_cosine` | 100.0% | 100.0% | 100.0% | 0.12 ms | 10.9 ms |
| **$N=1,000$** | Default (`iterative=off`) | HNSW `idx_..._hnsw_cosine` | 0.0%* | 0.0%* | 0.0%* | 2.44 ms | 13.5 ms |
| **$N=1,000$** | HNSW (`iterative=strict`) | HNSW `idx_..._hnsw_cosine` | 99.1% | 98.9% | 98.4% | 1.39 ms | 12.6 ms |
| **$N=10,000$** | Default Planner | HNSW `idx_..._hnsw_cosine` | 99.4% | 98.8% | 98.2% | 0.75 ms | 11.8 ms |
| **$N=10,000$** | Forced Exact (Baseline) | Full Scan / B-tree Sort | 100.0% | 100.0% | 100.0% | 12.80 ms | 23.9 ms |

\* *Note: With `hnsw.iterative_scan = off`, candidate queue exhaustion across the shared 12,650-row graph results in 0 returned rows for sparse partitions. Solved by setting `hnsw.iterative_scan = strict_order`.*

---

## 3. HNSW Parameter Tuning Analysis ($N=10,000$)

| Configuration | $m$ | $ef\_c$ | $ef\_s$ | Recall@1 | Recall@5 | Recall@10 | DB Latency (p50) | Index Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Config-A (m=16, ef_c=32) | 16 | 32 | 16 | 96.0% | 94.0% | 92.8% | 1.16 ms | 25 MB |
| Config-A (m=16, ef_c=32) | 16 | 32 | 32 | 98.0% | 96.0% | 94.8% | 0.86 ms | 25 MB |
| Config-A (m=16, ef_c=32) | 16 | 32 | 64 | 98.0% | 96.0% | 94.8% | 1.06 ms | 25 MB |
| Config-A (m=16, ef_c=32) | 16 | 32 | 128 | 98.0% | 96.0% | 94.8% | 1.37 ms | 25 MB |
| Config-B (m=16, ef_c=64) [Default] | 16 | 64 | 16 | 96.0% | 96.0% | 96.0% | 1.09 ms | 25 MB |
| Config-B (m=16, ef_c=64) [Default] | 16 | 64 | 32 | 100.0% | 100.0% | 100.0% | 0.92 ms | 25 MB |
| Config-B (m=16, ef_c=64) [Default] | 16 | 64 | 64 | 100.0% | 100.0% | 100.0% | 0.87 ms | 25 MB |
| Config-B (m=16, ef_c=64) [Default] | 16 | 64 | 128 | 100.0% | 100.0% | 100.0% | 1.42 ms | 25 MB |
| Config-C (m=24, ef_c=128) | 24 | 128 | 16 | 100.0% | 100.0% | 100.0% | 0.98 ms | 25 MB |
| Config-C (m=24, ef_c=128) | 24 | 128 | 32 | 100.0% | 100.0% | 100.0% | 0.70 ms | 25 MB |
| Config-C (m=24, ef_c=128) | 24 | 128 | 64 | 100.0% | 100.0% | 100.0% | 0.88 ms | 25 MB |
| Config-C (m=24, ef_c=128) | 24 | 128 | 128 | 100.0% | 100.0% | 100.0% | 1.49 ms | 25 MB |
| Config-D (m=32, ef_c=128) | 32 | 128 | 16 | 100.0% | 100.0% | 100.0% | 1.12 ms | 26 MB |
| Config-D (m=32, ef_c=128) | 32 | 128 | 32 | 100.0% | 100.0% | 100.0% | 0.97 ms | 26 MB |
| Config-D (m=32, ef_c=128) | 32 | 128 | 64 | 100.0% | 100.0% | 100.0% | 1.11 ms | 26 MB |
| Config-D (m=32, ef_c=128) | 32 | 128 | 128 | 100.0% | 100.0% | 100.0% | 1.25 ms | 26 MB |

---

## 4. Latency Decomposition Breakdown

- **Embedding Inference (CUDA)**: ~9.5 ms (73% of total retrieval time)
- **Database Query Execution (pgvector)**: ~0.75 ms (5.8% of total retrieval time)
- **Connection Pool Checkout**: ~0.05 ms (<0.5%)
- **Result Mapping / Deserialization**: ~0.04 ms (<0.5%)
- **Total End-to-End Latency**: **~12.98 ms (p50: 11.76 ms, p95: 23.38 ms)**

---

## 5. Architectural Recommendations for Production Gate Service

1. **Operational Connection Settings**:
   Configure the async connection pool in `DatabaseManager` to initialize sessions with:
   ```sql
   SET hnsw.iterative_scan = strict_order;
   SET hnsw.ef_search = 64;
   ```
2. **Index Parameters**:
   Adopt `m = 16, ef_construction = 64` as the production index configuration.
3. **Multi-Tenant Scaling**:
   Implement declarative table partitioning by `tenant_id` once any tenant reaches $>50,000$ examples to eliminate graph traversal filtering overhead.

