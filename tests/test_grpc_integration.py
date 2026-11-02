"""Live integration tests for ReliabilityService.Gate over async gRPC loopback."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

import grpc
import pytest
import pytest_asyncio

from arda_reliability_service.application.use_cases.ingest_reference_examples import (
    IngestItemDTO,
    IngestReferenceExamplesUseCase,
)
from arda_reliability_service.domain.value_objects import ScoringConfig
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from arda_reliability_service.infrastructure.grpc.generated import (
    reliability_pb2,
    reliability_pb2_grpc,
)
from arda_reliability_service.infrastructure.grpc.handler import ReliabilityGateHandler
from arda_reliability_service.infrastructure.grpc.server import ReliabilityGrpcServer


@pytest_asyncio.fixture
async def db_manager() -> AsyncGenerator[DatabaseManager]:
    """Yields a database manager and disposes connections on teardown."""
    manager = DatabaseManager()
    yield manager
    await manager.dispose()


@pytest_asyncio.fixture
async def embedding_adapter() -> SentenceTransformerEmbeddingAdapter:
    """Yields an embedding adapter instance."""
    return SentenceTransformerEmbeddingAdapter()


@pytest_asyncio.fixture
async def grpc_test_environment(
    db_manager: DatabaseManager,
    embedding_adapter: SentenceTransformerEmbeddingAdapter,
) -> AsyncGenerator[
    tuple[
        reliability_pb2_grpc.ReliabilityServiceStub,
        UUID,
        UUID,
        DatabaseManager,
        SentenceTransformerEmbeddingAdapter,
    ]
]:
    """Starts live gRPC server on dynamic port, returns connected async stub and test IDs."""
    tenant_id = uuid4()
    agent_id = uuid4()

    scoring_config = ScoringConfig(
        min_neighbor_count=3,
        min_similarity_threshold=0.60,
        min_coverage_threshold=0.45,
        min_density_threshold=0.35,
        min_agreement_threshold=0.60,
        min_reliability_threshold=0.50,
        top_k=5,
    )

    handler = ReliabilityGateHandler(
        db_manager=db_manager,
        embedding_adapter=embedding_adapter,
        config=scoring_config,
    )

    server = ReliabilityGrpcServer(
        handler=handler,
        host="127.0.0.1",
        port=0,
        grace_period_seconds=0.1,
    )

    port = await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = reliability_pb2_grpc.ReliabilityServiceStub(channel)

    try:
        yield stub, tenant_id, agent_id, db_manager, embedding_adapter
    finally:
        await channel.close()
        await server.stop(grace=0.05)
        # Cleanup seeded data
        async with db_manager.session() as session:
            repo = PgVectorReferenceExampleRepository(session)
            await repo.delete_all_for_agent(tenant_id, agent_id)


class TestReliabilityGateGrpcIntegration:
    """Live loopback gRPC integration test suite for ReliabilityService.Gate."""

    @pytest.mark.asyncio
    async def test_gate_pass_on_matching_precedents(
        self,
        grpc_test_environment: tuple[
            reliability_pb2_grpc.ReliabilityServiceStub,
            UUID,
            UUID,
            DatabaseManager,
            SentenceTransformerEmbeddingAdapter,
        ],
    ) -> None:
        """Verify Gate returns GATE_OUTCOME_PASS when historical evidence strongly agrees."""
        stub, tenant_id, agent_id, dm, adapter = grpc_test_environment

        # Seed 4 strongly matching APPROVE precedents
        async with dm.session() as session:
            repo = PgVectorReferenceExampleRepository(session)
            ingest_uc = IngestReferenceExamplesUseCase(
                repository=repo, embedding_provider=adapter
            )
            items = [
                IngestItemDTO(
                    example_id=f"TEST-PASS-{i}",
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    case_description=(
                        f"Customer filed dispute within 20 days. Cardholder history is clean "
                        f"with high tenure. Low chargeback ratio and no merchant dispute flags #{i}."
                    ),
                    label="APPROVE",
                    metadata={"index": i},
                )
                for i in range(1, 5)
            ]
            await ingest_uc.execute(items)

        # Send GateRequest with proposed APPROVE decision
        request = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description=(
                "Customer filed dispute within 20 days. Cardholder history is clean "
                "with high tenure. Low chargeback ratio and no merchant dispute flags."
            ),
            proposed_decision="APPROVE",
            reliability_threshold=0.50,
        )

        response = await stub.Gate(request)

        assert response.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_PASS
        assert response.confidence >= 0.50
        assert response.reason_code == "RELIABILITY_THRESHOLD_MET"
        assert len(response.neighbors) >= 3
        for n in response.neighbors:
            assert n.reference_example_id.startswith("TEST-PASS-")
            assert n.label == "APPROVE"
            assert n.similarity > 0.80

    @pytest.mark.asyncio
    async def test_gate_review_required_on_conflicting_decision(
        self,
        grpc_test_environment: tuple[
            reliability_pb2_grpc.ReliabilityServiceStub,
            UUID,
            UUID,
            DatabaseManager,
            SentenceTransformerEmbeddingAdapter,
        ],
    ) -> None:
        """Verify Gate returns GATE_OUTCOME_REVIEW_REQUIRED when proposed decision conflicts with precedents."""
        stub, tenant_id, agent_id, dm, adapter = grpc_test_environment

        # Seed 4 strongly matching APPROVE precedents
        async with dm.session() as session:
            repo = PgVectorReferenceExampleRepository(session)
            ingest_uc = IngestReferenceExamplesUseCase(
                repository=repo, embedding_provider=adapter
            )
            items = [
                IngestItemDTO(
                    example_id=f"TEST-CONF-{i}",
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    case_description=(
                        f"Customer filed dispute within 20 days. Cardholder history is clean "
                        f"with high tenure. Low chargeback ratio and no merchant dispute flags #{i}."
                    ),
                    label="APPROVE",
                    metadata={"index": i},
                )
                for i in range(1, 5)
            ]
            await ingest_uc.execute(items)

        # Propose REJECT despite APPROVE precedents
        request = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description=(
                "Customer filed dispute within 20 days. Cardholder history is clean "
                "with high tenure. Low chargeback ratio and no merchant dispute flags."
            ),
            proposed_decision="REJECT",
            reliability_threshold=0.50,
        )

        response = await stub.Gate(request)

        assert (
            response.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_REVIEW_REQUIRED
        )
        assert response.reason_code == "CONFLICTING_EVIDENCE"
        assert len(response.neighbors) >= 3

    @pytest.mark.asyncio
    async def test_gate_review_required_on_cold_start(
        self,
        grpc_test_environment: tuple[
            reliability_pb2_grpc.ReliabilityServiceStub,
            UUID,
            UUID,
            DatabaseManager,
            SentenceTransformerEmbeddingAdapter,
        ],
    ) -> None:
        """Verify cold-start returns GATE_OUTCOME_REVIEW_REQUIRED without aborting."""
        stub, _, _, _, _ = grpc_test_environment

        # Distinct empty tenant with zero reference examples
        empty_tenant_id = uuid4()
        empty_agent_id = uuid4()

        request = reliability_pb2.GateRequest(
            tenant_id=str(empty_tenant_id),
            agent_id=str(empty_agent_id),
            case_description="New dispute case with zero historical precedents in tenant repository.",
            proposed_decision="APPROVE",
        )

        response = await stub.Gate(request)

        assert (
            response.outcome == reliability_pb2.GateOutcome.GATE_OUTCOME_REVIEW_REQUIRED
        )
        assert response.confidence == 0.0
        assert response.reason_code == "COLD_START_INSUFFICIENT_EVIDENCE"
        assert len(response.neighbors) == 0

    @pytest.mark.asyncio
    async def test_gate_invalid_argument_error_on_bad_uuid(
        self,
        grpc_test_environment: tuple[
            reliability_pb2_grpc.ReliabilityServiceStub,
            UUID,
            UUID,
            DatabaseManager,
            SentenceTransformerEmbeddingAdapter,
        ],
    ) -> None:
        """Verify client receives INVALID_ARGUMENT when tenant_id is malformed."""
        stub, _, agent_id, _, _ = grpc_test_environment

        request = reliability_pb2.GateRequest(
            tenant_id="not-a-valid-uuid",
            agent_id=str(agent_id),
            case_description="Valid dispute case description.",
            proposed_decision="APPROVE",
        )

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Gate(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Invalid tenant_id" in exc_info.value.details()

    @pytest.mark.asyncio
    async def test_gate_invalid_argument_error_on_bad_decision(
        self,
        grpc_test_environment: tuple[
            reliability_pb2_grpc.ReliabilityServiceStub,
            UUID,
            UUID,
            DatabaseManager,
            SentenceTransformerEmbeddingAdapter,
        ],
    ) -> None:
        """Verify client receives INVALID_ARGUMENT when proposed_decision is unrecognized."""
        stub, tenant_id, agent_id, _, _ = grpc_test_environment

        request = reliability_pb2.GateRequest(
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
            case_description="Valid dispute case description.",
            proposed_decision="MAYBE_APPROVE",
        )

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Gate(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Invalid proposed_decision" in exc_info.value.details()
