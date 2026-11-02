"""Integration tests for pgvector reference repository and retrieval pipelines."""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from arda_reliability_service.application.use_cases.ingest_reference_examples import (
    IngestItemDTO,
    IngestReferenceExamplesUseCase,
)
from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.entities import ReferenceExample
from arda_reliability_service.domain.value_objects import RetrievalQuery
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)


def _make_unit_vector(dim: int, non_zero_idx: int, val: float = 1.0) -> list[float]:
    """Helper to generate a unit vector of given dimension with a single non-zero index."""
    vec = [0.0] * dim
    vec[non_zero_idx] = val
    return vec


def _make_normalized_vector(dim: int, values_dict: dict[int, float]) -> list[float]:
    """Helper to generate a normalized dense vector from specific coordinate values."""
    vec = [0.0] * dim
    for idx, v in values_dict.items():
        vec[idx] = v
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest_asyncio.fixture
async def db_manager() -> AsyncGenerator[DatabaseManager]:
    """Yields a database manager and disposes connections on teardown."""
    manager = DatabaseManager()
    yield manager
    await manager.dispose()


@pytest_asyncio.fixture
async def test_scope(
    db_manager: DatabaseManager,
) -> AsyncGenerator[tuple[PgVectorReferenceExampleRepository, UUID, UUID]]:
    """Provides an isolated (tenant_id, agent_id) scope, cleaning up after each test."""
    tenant_id = uuid4()
    agent_id = uuid4()

    async with db_manager.session() as session:
        repo = PgVectorReferenceExampleRepository(session)
        yield repo, tenant_id, agent_id

        # Cleanup test scope
        await repo.delete_all_for_agent(tenant_id=tenant_id, agent_id=agent_id)


class TestPgVectorRepositoryIntegration:
    """Integration test suite executing against real PostgreSQL and pgvector."""

    @pytest.mark.asyncio
    async def test_save_and_get_by_example_id(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope
        example = ReferenceExample(
            example_id="test-ex-001",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Dispute over unauthorized food delivery charge.",
            label="APPROVE",
            source="synthetic",
            status="active",
            metadata={"category": "unauthorized_transaction", "amount": 42.50},
        )
        example.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)

        await repo.save(example)

        fetched = await repo.get_by_example_id(tenant_id, agent_id, "test-ex-001")
        assert fetched is not None
        assert fetched.example_id == "test-ex-001"
        assert fetched.tenant_id == tenant_id
        assert fetched.agent_id == agent_id
        assert (
            fetched.case_description
            == "Dispute over unauthorized food delivery charge."
        )
        assert fetched.label == "APPROVE"
        assert fetched.source == "synthetic"
        assert fetched.status == "active"
        assert fetched.metadata["category"] == "unauthorized_transaction"
        assert fetched.embedding is not None
        assert len(fetched.embedding) == 384
        assert abs(fetched.embedding[0] - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_save_batch_and_count(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        ex1 = ReferenceExample(
            example_id="batch-001",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Description 1",
            label="APPROVE",
            status="active",
        )
        ex1.attach_embedding(_make_unit_vector(384, 1), expected_dim=384)

        ex2 = ReferenceExample(
            example_id="batch-002",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Description 2",
            label="REJECT",
            status="archived",
        )
        ex2.attach_embedding(_make_unit_vector(384, 2), expected_dim=384)

        saved_ids = await repo.save_batch([ex1, ex2])
        assert saved_ids == ["batch-001", "batch-002"]

        total_cnt = await repo.count(tenant_id, agent_id)
        assert total_cnt == 2

        active_cnt = await repo.count(tenant_id, agent_id, status="active")
        assert active_cnt == 1

        archived_cnt = await repo.count(tenant_id, agent_id, status="archived")
        assert archived_cnt == 1

    @pytest.mark.asyncio
    async def test_idempotent_upsert(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        # Initial save
        ex = ReferenceExample(
            example_id="idem-001",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Initial description",
            label="APPROVE",
        )
        ex.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)
        await repo.save(ex)

        cnt_before = await repo.count(tenant_id, agent_id)
        assert cnt_before == 1

        # Updated re-save of the same business key
        ex_updated = ReferenceExample(
            example_id="idem-001",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Updated description reflecting arbitration outcome",
            label="REJECT",
        )
        ex_updated.attach_embedding(_make_unit_vector(384, 5), expected_dim=384)
        await repo.save_batch([ex_updated], on_conflict_do_update=True)

        cnt_after = await repo.count(tenant_id, agent_id)
        assert cnt_after == 1  # No duplicate row created

        fetched = await repo.get_by_example_id(tenant_id, agent_id, "idem-001")
        assert fetched is not None
        assert fetched.label == "REJECT"
        assert (
            fetched.case_description
            == "Updated description reflecting arbitration outcome"
        )
        assert fetched.embedding is not None
        assert abs(fetched.embedding[5] - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_exact_retrieval_ranking_and_similarity(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        # Query vector along dimension 0: [1.0, 0, 0, ...]
        query_vec = _make_unit_vector(384, 0)

        # Candidate 1: Exactly identical (cosine dist = 0.0 -> similarity = 1.0)
        c1 = ReferenceExample(
            example_id="cand-identical",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Identical passage",
            label="APPROVE",
        )
        c1.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)

        # Candidate 2: 45 degree angle (cosine similarity = 1/sqrt(2) ~ 0.7071)
        c2 = ReferenceExample(
            example_id="cand-similar",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Similar passage",
            label="REJECT",
        )
        c2.attach_embedding(
            _make_normalized_vector(384, {0: 1.0, 1: 1.0}), expected_dim=384
        )

        # Candidate 3: Orthogonal (cosine dist = 1.0 -> similarity = 0.0)
        c3 = ReferenceExample(
            example_id="cand-orthogonal",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Orthogonal passage",
            label="REJECT",
        )
        c3.attach_embedding(_make_unit_vector(384, 1), expected_dim=384)

        await repo.save_batch([c3, c1, c2])

        neighbors = await repo.retrieve_exact(query_vec, tenant_id, agent_id, top_k=2)

        assert len(neighbors) == 2
        # Nearest neighbor should be c1
        assert neighbors[0].example_id == "cand-identical"
        assert abs(neighbors[0].similarity_score - 1.0) < 1e-3
        assert neighbors[0].label == "APPROVE"

        # Second neighbor should be c2
        assert neighbors[1].example_id == "cand-similar"
        assert abs(neighbors[1].similarity_score - 0.7071) < 1e-3
        assert neighbors[1].label == "REJECT"

    @pytest.mark.asyncio
    async def test_deterministic_tie_breaking(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        # Two candidates with identical embeddings (same cosine distance)
        # Created at distinct times
        t_base = datetime.now(UTC)
        c_older = ReferenceExample(
            example_id="tie-b",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Older record",
            label="APPROVE",
            created_at=t_base - timedelta(hours=1),
        )
        c_older.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)

        c_newer = ReferenceExample(
            example_id="tie-a",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Newer record",
            label="REJECT",
            created_at=t_base,
        )
        c_newer.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)

        await repo.save_batch([c_older, c_newer])

        query_vec = _make_unit_vector(384, 0)
        neighbors = await repo.retrieve_exact(query_vec, tenant_id, agent_id, top_k=2)

        assert len(neighbors) == 2
        # Tie-breaker order: created_at DESC -> c_newer must come first
        assert neighbors[0].example_id == "tie-a"
        assert neighbors[1].example_id == "tie-b"

    @pytest.mark.asyncio
    async def test_approximate_hnsw_retrieval(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        ex = ReferenceExample(
            example_id="hnsw-001",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="HNSW candidate",
            label="APPROVE",
        )
        ex.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)
        await repo.save(ex)

        query_vec = _make_unit_vector(384, 0)

        # Retrieve with ef_search=32
        res32 = await repo.retrieve_approximate(
            query_vec, tenant_id, agent_id, top_k=1, ef_search=32
        )
        assert len(res32) == 1
        assert res32[0].example_id == "hnsw-001"

        # Retrieve with ef_search=128
        res128 = await repo.retrieve_approximate(
            query_vec, tenant_id, agent_id, top_k=1, ef_search=128
        )
        assert len(res128) == 1
        assert res128[0].example_id == "hnsw-001"

    @pytest.mark.asyncio
    async def test_strict_tenant_isolation(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_a, agent_id = test_scope
        tenant_b = uuid4()

        # Insert example into Tenant A
        ex_a = ReferenceExample(
            example_id="tenant-a-ex",
            tenant_id=tenant_a,
            agent_id=agent_id,
            case_description="Tenant A confidential dispute",
            label="APPROVE",
        )
        ex_a.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)
        await repo.save(ex_a)

        query_vec = _make_unit_vector(384, 0)

        # Query Tenant B with the same query vector
        res_b = await repo.retrieve_exact(
            query_vec, tenant_id=tenant_b, agent_id=agent_id, top_k=5
        )
        assert len(res_b) == 0, (
            "Tenant B must never retrieve Tenant A's reference data!"
        )

        # Query Tenant A returns the record
        res_a = await repo.retrieve_exact(
            query_vec, tenant_id=tenant_a, agent_id=agent_id, top_k=5
        )
        assert len(res_a) == 1
        assert res_a[0].example_id == "tenant-a-ex"

    @pytest.mark.asyncio
    async def test_strict_agent_isolation(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_1 = test_scope
        agent_2 = uuid4()

        # Insert example into Agent 1
        ex_1 = ReferenceExample(
            example_id="agent-1-ex",
            tenant_id=tenant_id,
            agent_id=agent_1,
            case_description="Agent 1 specific dispute policy case",
            label="APPROVE",
        )
        ex_1.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)
        await repo.save(ex_1)

        query_vec = _make_unit_vector(384, 0)

        # Query Agent 2 within the same tenant
        res_2 = await repo.retrieve_exact(
            query_vec, tenant_id=tenant_id, agent_id=agent_2, top_k=5
        )
        assert len(res_2) == 0, "Agent 2 must never retrieve Agent 1's reference data!"

        # Query Agent 1 returns the record
        res_1 = await repo.retrieve_exact(
            query_vec, tenant_id=tenant_id, agent_id=agent_1, top_k=5
        )
        assert len(res_1) == 1
        assert res_1[0].example_id == "agent-1-ex"

    @pytest.mark.asyncio
    async def test_status_filtering_ignores_archived(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        active_ex = ReferenceExample(
            example_id="active-ex",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Active policy reference",
            label="APPROVE",
            status="active",
        )
        active_ex.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)

        archived_ex = ReferenceExample(
            example_id="archived-ex",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Deprecated policy reference",
            label="REJECT",
            status="archived",
        )
        archived_ex.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)

        await repo.save_batch([active_ex, archived_ex])

        query_vec = _make_unit_vector(384, 0)
        neighbors = await repo.retrieve_exact(query_vec, tenant_id, agent_id, top_k=5)

        assert len(neighbors) == 1
        assert neighbors[0].example_id == "active-ex"

    @pytest.mark.asyncio
    async def test_top_k_boundaries(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope

        ex = ReferenceExample(
            example_id="single-ex",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Only record available",
            label="APPROVE",
        )
        ex.attach_embedding(_make_unit_vector(384, 0), expected_dim=384)
        await repo.save(ex)

        query_vec = _make_unit_vector(384, 0)

        # Request top_k = 10 when only 1 exists
        neighbors = await repo.retrieve_exact(query_vec, tenant_id, agent_id, top_k=10)
        assert len(neighbors) == 1
        assert neighbors[0].example_id == "single-ex"

        # Request top_k on non-existent tenant/agent
        empty_neighbors = await repo.retrieve_exact(
            query_vec, uuid4(), uuid4(), top_k=5
        )
        assert len(empty_neighbors) == 0

    @pytest.mark.asyncio
    async def test_end_to_end_use_cases_with_live_adapter(
        self,
        test_scope: tuple[PgVectorReferenceExampleRepository, UUID, UUID],
    ) -> None:
        repo, tenant_id, agent_id = test_scope
        adapter = SentenceTransformerEmbeddingAdapter(
            model_name="BAAI/bge-small-en-v1.5"
        )

        ingest_use_case = IngestReferenceExamplesUseCase(
            repository=repo, embedding_provider=adapter
        )
        retrieve_use_case = RetrieveReferenceExamplesUseCase(
            repository=repo, embedding_provider=adapter
        )

        # Ingest 2 reference cases
        items = [
            IngestItemDTO(
                example_id="e2e-001",
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description="Customer disputes duplicate gym subscription charge on the same calendar day.",
                label="APPROVE",
                metadata={
                    "ratio_verbalized_description": "Dispute filing ratio is 0.05, 1 prior dispute, duplicate subscription charge on identical day."
                },
            ),
            IngestItemDTO(
                example_id="e2e-002",
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description="Cardholder authorized transaction via SMS OTP with biometric authentication confirmed.",
                label="REJECT",
                metadata={
                    "ratio_verbalized_description": "Merchant response ratio is 1.0, 3D Secure OTP verification verified with zero fraud flags."
                },
            ),
        ]

        ingested_ids = await ingest_use_case.execute(items)
        assert ingested_ids == ["e2e-001", "e2e-002"]

        # Retrieve for a query similar to e2e-001
        query = RetrievalQuery(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Customer was double-billed for health club membership on the same day.",
            top_k=2,
            ef_search=64,
        )

        result_set = await retrieve_use_case.execute(query, mode="exact")

        assert len(result_set.neighbors) == 2
        assert result_set.neighbors[0].example_id == "e2e-001"
        assert result_set.neighbors[0].label == "APPROVE"
        assert result_set.neighbors[0].similarity_score > 0.5
        assert result_set.execution_latency_ms > 0.0
