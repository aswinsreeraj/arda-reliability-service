"""Verification tests for actual HNSW index usage, query plans, and filtering integrity."""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import text

from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)

BENCHMARK_TENANT_ID = UUID("00000000-0000-0000-0000-0000000000bb")
BENCHMARK_AGENT_1K = UUID("00000000-0000-0000-0000-000000000001")
BENCHMARK_AGENT_10K = UUID("00000000-0000-0000-0000-000000000002")
BENCHMARK_AGENT_FILTER_A = UUID("00000000-0000-0000-0000-00000000000a")
BENCHMARK_AGENT_FILTER_B = UUID("00000000-0000-0000-0000-00000000000b")
BENCHMARK_TENANT_ALT = UUID("00000000-0000-0000-0000-0000000000cc")
BENCHMARK_AGENT_ALT = UUID("00000000-0000-0000-0000-000000000001")


def _parse_embedding(val: object) -> list[float]:
    """Parse string or list embedding from database."""
    if isinstance(val, str):
        return [float(x) for x in val.strip("[]").split(",")]
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    raise ValueError(f"Cannot parse embedding of type: {type(val)}")


SEED_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
SEED_AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")


class TestHnswVerification:
    """Tests verifying HNSW index presence, query planning, filtering, and model safety."""

    @pytest.mark.asyncio
    async def test_hnsw_index_exists_and_configured(self) -> None:
        """Verify HNSW index exists in pg_indexes with cosine distance metric."""
        dm = DatabaseManager()
        async with dm.session() as s:
            res = await s.execute(
                text(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'reference_examples'
                      AND indexname = 'idx_reference_examples_hnsw_cosine';
                    """
                )
            )
            row = res.one_or_none()
            assert row is not None, (
                "HNSW index 'idx_reference_examples_hnsw_cosine' not found!"
            )
            index_def = row[1]
            assert "USING hnsw" in index_def, "Index does not use HNSW access method!"
            assert "vector_cosine_ops" in index_def, (
                "Index does not use vector_cosine_ops!"
            )
        await dm.dispose()

    @pytest.mark.asyncio
    async def test_query_plan_inspection_at_scale(self) -> None:
        """Verify that at 10k scale, PostgreSQL query planner naturally selects HNSW index."""
        dm = DatabaseManager()
        async with dm.session() as s:
            # Fetch a query embedding
            res = await s.execute(
                text(
                    f"""
                    SELECT embedding FROM reference_examples
                    WHERE tenant_id = '{BENCHMARK_TENANT_ID}' AND agent_id = '{BENCHMARK_AGENT_10K}'
                    LIMIT 1;
                    """
                )
            )
            row = res.first()
            if row is None:
                pytest.skip(
                    "Benchmark partition not loaded in database (run scripts/verify_hnsw_and_benchmark.py to generate scale data)"
                )
            vec_str = str(row[0])

            # Query with default planner
            q = text(
                f"""
                EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                SELECT example_id, embedding <=> '{vec_str}'::vector AS dist
                FROM reference_examples
                WHERE tenant_id = '{BENCHMARK_TENANT_ID}'
                  AND agent_id = '{BENCHMARK_AGENT_10K}'
                  AND status = 'active'
                ORDER BY embedding <=> '{vec_str}'::vector ASC
                LIMIT 5;
                """
            )
            plan_res = await s.execute(q)
            plan_json = plan_res.scalar_one()[0]
            top_plan = plan_json["Plan"]

            def find_node(node, key):
                if key in node:
                    return node[key]
                if "Plans" in node:
                    for c in node["Plans"]:
                        val = find_node(c, key)
                        if val:
                            return val
                return None

            index_used = find_node(top_plan, "Index Name")
            assert index_used == "idx_reference_examples_hnsw_cosine", (
                f"Expected HNSW index scan, but got: {index_used}"
            )
            assert plan_json["Execution Time"] < 50.0  # Under 50ms (typically <1ms)
        await dm.dispose()

    @pytest.mark.asyncio
    async def test_exact_vs_approximate_result_comparison(self) -> None:
        """Verify high recall agreement between exact scan and HNSW retrieval on 10k dataset."""
        dm = DatabaseManager()
        async with dm.session() as s:
            res = await s.execute(
                text(
                    f"""
                    SELECT example_id, embedding FROM reference_examples
                    WHERE tenant_id = '{BENCHMARK_TENANT_ID}' AND agent_id = '{BENCHMARK_AGENT_10K}'
                    LIMIT 5;
                    """
                )
            )
            test_queries = res.all()

            repo = PgVectorReferenceExampleRepository(s)

            for row in test_queries:
                q_vec = _parse_embedding(row[1])

                # 1. Exact retrieval (forcing flat scan)
                await s.execute(text("SET LOCAL enable_indexscan = off;"))
                await s.execute(text("SET LOCAL enable_bitmapscan = off;"))
                exact_neighbors = await repo.retrieve_exact(
                    query_embedding=q_vec,
                    tenant_id=BENCHMARK_TENANT_ID,
                    agent_id=BENCHMARK_AGENT_10K,
                    top_k=5,
                )
                exact_ids = {n.example_id for n in exact_neighbors}

                # 2. Approximate HNSW retrieval (ef_search=64, iterative_scan=strict_order)
                await s.execute(text("SET LOCAL enable_indexscan = on;"))
                hnsw_neighbors = await repo.retrieve_approximate(
                    query_embedding=q_vec,
                    tenant_id=BENCHMARK_TENANT_ID,
                    agent_id=BENCHMARK_AGENT_10K,
                    top_k=5,
                    ef_search=64,
                )
                hnsw_ids = {n.example_id for n in hnsw_neighbors}

                # Assert at least 4 out of 5 agreement (80%+)
                overlap = len(exact_ids.intersection(hnsw_ids))
                assert overlap >= 4, (
                    f"Low recall overlap ({overlap}/5) between exact and HNSW!"
                )

        await dm.dispose()

    @pytest.mark.asyncio
    async def test_tenant_and_agent_filtering_strictness(self) -> None:
        """Verify strict multi-tenant and multi-agent isolation across diverse benchmark partitions."""
        dm = DatabaseManager()
        async with dm.session() as s:
            repo = PgVectorReferenceExampleRepository(s)

            res = await s.execute(
                text(
                    f"""
                    SELECT embedding FROM reference_examples
                    WHERE tenant_id = '{BENCHMARK_TENANT_ID}' AND agent_id = '{BENCHMARK_AGENT_10K}'
                    LIMIT 1;
                    """
                )
            )
            row = res.first()
            if row is None:
                pytest.skip("Benchmark partition not loaded in database")
            q_vec = _parse_embedding(row[0])

            # Query Alternate Tenant with Benchmark Tenant vector -> 0 rows from Benchmark Tenant
            res_alt = await repo.retrieve_approximate(
                query_embedding=q_vec,
                tenant_id=BENCHMARK_TENANT_ALT,
                agent_id=BENCHMARK_AGENT_ALT,
                top_k=5,
            )
            assert len(res_alt) == 5
            for n in res_alt:
                # All returned examples must belong to the alternate tenant
                r = await repo.get_by_example_id(
                    BENCHMARK_TENANT_ALT, BENCHMARK_AGENT_ALT, n.example_id
                )
                assert r is not None
                assert r.tenant_id == BENCHMARK_TENANT_ALT

            # Query non-existent agent -> 0 rows
            res_none = await repo.retrieve_approximate(
                query_embedding=q_vec,
                tenant_id=BENCHMARK_TENANT_ID,
                agent_id=UUID("00000000-0000-0000-0000-ffffffffffff"),
                top_k=5,
            )
            assert len(res_none) == 0

        await dm.dispose()

    @pytest.mark.asyncio
    async def test_filtered_retrieval_completeness_under_iterative_scan(self) -> None:
        """Verify that iterative_scan=strict_order returns full top_k without silent truncation."""
        dm = DatabaseManager()
        async with dm.session() as s:
            repo = PgVectorReferenceExampleRepository(s)

            res = await s.execute(
                text(
                    f"""
                    SELECT embedding FROM reference_examples
                    WHERE tenant_id = '{BENCHMARK_TENANT_ID}' AND agent_id = '{BENCHMARK_AGENT_1K}'
                    LIMIT 1;
                    """
                )
            )
            row = res.first()
            if row is None:
                pytest.skip("Benchmark partition not loaded in database")
            q_vec = _parse_embedding(row[0])

            # Approximate retrieval with iterative_scan set in repository
            neighbors = await repo.retrieve_approximate(
                query_embedding=q_vec,
                tenant_id=BENCHMARK_TENANT_ID,
                agent_id=BENCHMARK_AGENT_1K,
                top_k=5,
                ef_search=64,
            )
            assert len(neighbors) == 5, (
                f"Expected 5 neighbors, got {len(neighbors)} (possible graph truncation)!"
            )

        await dm.dispose()

    @pytest.mark.asyncio
    async def test_embedding_model_identity_consistency(self) -> None:
        """Verify that all stored vectors carry consistent model_name and model_version tags."""
        dm = DatabaseManager()
        async with dm.session() as s:
            res = await s.execute(
                text(
                    """
                    SELECT DISTINCT model_name, model_version
                    FROM reference_examples;
                    """
                )
            )
            models = res.all()
            assert len(models) == 1, f"Expected single model identity, got: {models}"
            assert models[0][0] == "BAAI/bge-small-en-v1.5"
            assert models[0][1] == "v1.5"
        await dm.dispose()

    @pytest.mark.asyncio
    async def test_benchmark_dataset_separation(self) -> None:
        """Verify that benchmark dataset is strictly separated from original 100 audited seed examples."""
        dm = DatabaseManager()
        async with dm.session() as s:
            # 1. Check audited seed dataset partition
            seed_cnt = await s.execute(
                text(
                    f"""
                    SELECT count(*) FROM reference_examples
                    WHERE tenant_id = '{SEED_TENANT_ID}'
                      AND agent_id = '{SEED_AGENT_ID}'
                      AND source = 'synthetic';
                    """
                )
            )
            assert seed_cnt.scalar_one() == 100, (
                "Audited seed examples count modified or contaminated!"
            )

            # 2. Check benchmark partition
            bench_cnt = await s.execute(
                text(
                    """
                    SELECT count(*) FROM reference_examples
                    WHERE source = 'benchmark';
                    """
                )
            )
            bench_total = bench_cnt.scalar_one()
            if bench_total == 0:
                # Clean production/seed state: benchmark data has not been ingested
                pass
            else:
                assert bench_total >= 12500, "Benchmark partition count mismatch!"

            # 3. Ensure no crossover
            overlap = await s.execute(
                text(
                    f"""
                    SELECT count(*) FROM reference_examples
                    WHERE tenant_id = '{SEED_TENANT_ID}'
                      AND source = 'benchmark';
                    """
                )
            )
            assert overlap.scalar_one() == 0, (
                "Benchmark examples leaked into audited seed tenant!"
            )

        await dm.dispose()
