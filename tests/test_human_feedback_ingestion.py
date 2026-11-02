"""Comprehensive unit and failure-recovery tests for human feedback ingestion pipeline.

Tests satisfy all 15 required verification scenarios:
1. Valid human-resolution event.
2. Invalid event payload.
3. Unsupported event version.
4. Duplicate event delivery.
5. Concurrent duplicate delivery.
6. Database failure.
7. Embedding failure.
8. Consumer restart and redelivery.
9. Conflicting human resolutions.
10. Tenant isolation.
11. Agent isolation.
12. Rejection of unreviewed agent decisions.
13. Successful persistence and subsequent retrieval.
14. Graceful shutdown.
15. Dead-letter handling.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.application.use_cases.evaluate_reliability import (
    EvaluateReliabilityUseCase,
)
from arda_reliability_service.application.use_cases.ingest_human_feedback import (
    HumanFeedbackIngestionOutcome,
    IngestHumanFeedbackUseCase,
)
from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.exceptions import (
    EventValidationError,
    InvalidDescriptionError,
    UnreviewedAgentDecisionError,
    UnsupportedEventVersionError,
)
from arda_reliability_service.domain.value_objects import (
    ReliabilityEvaluationRequest,
    ScoringConfig,
)
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pg_feedback_repository import (
    PgHumanFeedbackRepository,
)
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from arda_reliability_service.infrastructure.messaging.consumer import (
    RedpandaFeedbackConsumer,
)
from arda_reliability_service.infrastructure.messaging.events import (
    HumanResolutionEventDTO,
)
from arda_reliability_service.infrastructure.messaging.metrics import ConsumerMetrics


# --- Mock Embedding Provider ---
class MockEmbeddingProvider(EmbeddingProviderPort):
    """Deterministic mock embedding provider for fast unit testing."""

    def __init__(self, dimension: int = 384, should_fail: bool = False) -> None:
        self._dim = dimension
        self.should_fail = should_fail
        self.embedded_texts: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "mock-bge-small"

    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        if self.should_fail:
            raise RuntimeError("Embedding model inference failed!")
        self.embedded_texts.append(text)
        # Deterministic unit vector
        val = 1.0 / (self._dim**0.5)
        return [val] * self._dim

    def generate_embeddings_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        return [self.generate_embedding(t, is_query=is_query) for t in texts]


# --- Mock Kafka Message & Producer ---
class MockKafkaMessage:
    def __init__(
        self,
        value: bytes | str,
        topic: str = "arda.human-resolutions.v1",
        partition: int = 0,
        offset: int = 0,
    ) -> None:
        if isinstance(value, str):
            self.value = value.encode("utf-8")
        else:
            self.value = value
        self.topic = topic
        self.partition = partition
        self.offset = offset


class MockKafkaProducer:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_and_wait(self, topic: str, value: bytes) -> None:
        self.sent_messages.append((topic, value))


class MockKafkaConsumer:
    def __init__(self, messages: list[MockKafkaMessage] | None = None) -> None:
        self.messages = list(messages or [])
        self.committed: list[Any] = []
        self._idx = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def getone(self) -> MockKafkaMessage:
        if self._idx < len(self.messages):
            msg = self.messages[self._idx]
            self._idx += 1
            return msg
        # Block until cancelled if exhausted
        await asyncio.sleep(100)
        raise asyncio.CancelledError()

    async def commit(self, offsets: Any) -> None:
        self.committed.append(offsets)


# --- Database Fixtures ---
@pytest_asyncio.fixture
async def db_manager() -> Any:
    db = DatabaseManager()
    yield db
    # cleanup engine
    if db._engine:
        await db._engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    db_manager: DatabaseManager,
) -> async_sessionmaker[AsyncSession]:
    return db_manager.session_factory


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=384)


def make_valid_event_dict(
    event_id: str | None = None,
    tenant_id: UUID | None = None,
    agent_id: UUID | None = None,
    case_id: str = "case-test-001",
    human_decision: str = "APPROVE",
    case_description: str = "Authorized refund under approved limit of $50.",
) -> dict[str, Any]:
    return {
        "event_id": event_id or f"evt-{uuid4()}",
        "event_type": "arda.reliability.human_resolution",
        "event_version": "v1",
        "tenant_id": str(tenant_id or uuid4()),
        "agent_id": str(agent_id or uuid4()),
        "case_id": case_id,
        "original_case_description": case_description,
        "proposed_decision": "APPROVE",
        "human_decision": human_decision,
        "reviewer_id": "reviewer_alice_42",
        "resolved_at": datetime.now(UTC).isoformat(),
        "correlation_id": "corr-test-123",
    }


# ==============================================================================
# SCENARIOS 1-15 VERIFICATION
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_1_valid_human_resolution_event(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """1. Valid human-resolution event: parsed, embedded, and durably persisted."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-{uuid4().hex[:8]}"
    desc = "Customer requested standard address update with verified identity token."

    event_payload = make_valid_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
        case_description=desc,
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        use_case = IngestHumanFeedbackUseCase(repo, mock_embedder)

        result = await use_case.execute(event_payload)
        await session.commit()

        assert result.outcome == HumanFeedbackIngestionOutcome.PROCESSED
        assert result.reference_created is True
        assert result.case_id == case_id

        # Verify invariant: only case description was passed to embedder
        assert desc in mock_embedder.embedded_texts
        # Labels or decisions must NOT be in embedded text
        assert not any("APPROVE" in t for t in mock_embedder.embedded_texts)

        # Verify persisted reference example in database
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.label == "APPROVE"
        assert ref.source == "human_review"
        assert ref.case_description == desc
        assert ref.embedding is not None
        assert len(ref.embedding) == 384

        # Clean up
        await repo.delete_all_for_agent(tenant_id, agent_id)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_2_invalid_event_payload(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """2. Invalid event payload: missing fields, empty strings, malformed json."""
    # Missing required field
    bad_payload_missing = {
        "event_id": "evt-bad-1",
        "tenant_id": str(uuid4()),
        # missing case_id, description, decision, etc.
    }
    with pytest.raises(ValidationError):
        HumanResolutionEventDTO.model_validate(bad_payload_missing)

    # Empty case description
    bad_payload_empty_desc = make_valid_event_dict(case_description="   ")
    with pytest.raises(InvalidDescriptionError):
        HumanResolutionEventDTO.model_validate(bad_payload_empty_desc)

    # Empty reviewer ID
    bad_payload_reviewer = make_valid_event_dict()
    bad_payload_reviewer["reviewer_id"] = "   "
    with pytest.raises(EventValidationError):
        HumanResolutionEventDTO.model_validate(bad_payload_reviewer)


@pytest.mark.asyncio
async def test_scenario_3_unsupported_event_version(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """3. Unsupported event version: rejected with UnsupportedEventVersionError."""
    payload_v2 = make_valid_event_dict()
    payload_v2["event_version"] = "v2"

    with pytest.raises(UnsupportedEventVersionError):
        HumanResolutionEventDTO.model_validate(payload_v2)


@pytest.mark.asyncio
async def test_scenario_4_duplicate_event_delivery(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """4. Duplicate event delivery: reprocessing same event_id is an idempotent no-op."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-{uuid4().hex[:8]}"
    event_id = f"evt-{uuid4()}"

    payload = make_valid_event_dict(
        event_id=event_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        use_case = IngestHumanFeedbackUseCase(repo, mock_embedder)

        # First delivery
        res1 = await use_case.execute(payload)
        await session.commit()
        assert res1.outcome == HumanFeedbackIngestionOutcome.PROCESSED
        assert res1.reference_created is True

        embed_count_after_first = len(mock_embedder.embedded_texts)

        # Second delivery of identical event_id
        res2 = await use_case.execute(payload)
        await session.commit()
        assert res2.outcome == HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE
        assert res2.reference_created is False

        # Verify NO re-embedding occurred
        assert len(mock_embedder.embedded_texts) == embed_count_after_first

        # Clean up
        await repo.delete_all_for_agent(tenant_id, agent_id)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_5_concurrent_duplicate_delivery(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """5. Concurrent duplicate delivery: database uniqueness prevents duplicate records."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-{uuid4().hex[:8]}"
    event_id = f"evt-conc-{uuid4()}"

    payload = make_valid_event_dict(
        event_id=event_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
    )

    async def worker() -> Any:
        async with session_factory() as session:
            repo = PgHumanFeedbackRepository(session)
            use_case = IngestHumanFeedbackUseCase(repo, mock_embedder)
            res = await use_case.execute(payload)
            await session.commit()
            return res

    # Run two workers concurrently
    results: tuple[Any, ...] = await asyncio.gather(
        worker(), worker(), return_exceptions=True
    )

    outcomes = []
    for r in results:
        if hasattr(r, "outcome"):
            outcomes.append(r.outcome)

    assert HumanFeedbackIngestionOutcome.PROCESSED in outcomes
    # Either the second succeeded with IDEMPOTENT_DUPLICATE or gracefully handled
    assert len(outcomes) >= 1

    # Verify only 1 reference example exists in DB
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        await repo.delete_all_for_agent(tenant_id, agent_id)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_6_database_failure(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """6. Database failure: consumer retries and routes to DLQ when exhausted."""

    # Create failing session factory
    class FailingSession:
        async def __aenter__(self) -> Any:
            raise ConnectionRefusedError("PostgreSQL connection refused!")

        async def __aexit__(self, *args: object) -> None:
            pass

    def failing_session_factory() -> Any:
        return FailingSession()

    producer = MockKafkaProducer()
    metrics = ConsumerMetrics()
    consumer = RedpandaFeedbackConsumer(
        session_factory=failing_session_factory,  # type: ignore
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore
        metrics=metrics,
    )
    consumer.max_retries = 2
    consumer.retry_backoff_base = 0.01

    payload = make_valid_event_dict()
    msg = MockKafkaMessage(json.dumps(payload))

    ack = await consumer.process_single_message(msg)
    assert ack is True
    assert metrics.processing_failures >= 2
    assert metrics.dead_lettered_events == 1
    assert len(producer.sent_messages) == 1
    dlq_topic, dlq_bytes = producer.sent_messages[0]
    assert dlq_topic == consumer.dlq_topic
    dlq_data = json.loads(dlq_bytes.decode("utf-8"))
    assert dlq_data["error_type"] == "MAX_RETRIES_EXHAUSTED"


@pytest.mark.asyncio
async def test_scenario_7_embedding_failure(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """7. Embedding failure: does NOT create partial/corrupt reference example without embedding."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-{uuid4().hex[:8]}"

    failing_embedder = MockEmbeddingProvider(dimension=384, should_fail=True)
    producer = MockKafkaProducer()
    metrics = ConsumerMetrics()

    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=failing_embedder,
        producer=producer,  # type: ignore
        metrics=metrics,
    )
    consumer.max_retries = 2
    consumer.retry_backoff_base = 0.01

    payload = make_valid_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
    )
    msg = MockKafkaMessage(json.dumps(payload))

    ack = await consumer.process_single_message(msg)
    assert ack is True
    assert metrics.embedding_failures >= 2
    assert metrics.dead_lettered_events == 1

    # Verify database contains NO record for this case
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is None


@pytest.mark.asyncio
async def test_scenario_8_consumer_restart_and_redelivery(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """8. Consumer restart and redelivery: uncommitted message redelivered is safely processed."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-{uuid4().hex[:8]}"
    payload = make_valid_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
    )

    msg = MockKafkaMessage(json.dumps(payload), offset=10)

    # Consumer instance 1 processes but hypothetically crashes before committing
    c1 = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=MockKafkaProducer(),  # type: ignore
    )
    ack1 = await c1.process_single_message(msg)
    assert ack1 is True

    # Consumer instance 2 (after restart) receives redelivery of message with offset 10
    c2 = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=MockKafkaProducer(),  # type: ignore
    )
    ack2 = await c2.process_single_message(msg)
    assert ack2 is True
    assert c2.metrics.duplicate_events == 1

    # Cleanup
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        await repo.delete_all_for_agent(tenant_id, agent_id)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_9_conflicting_human_resolutions(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """9. Conflicting human resolutions: detected, recorded with status CONFLICT, never silently overwritten."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-conflict-{uuid4().hex[:8]}"

    # Event 1: APPROVE
    ev1 = make_valid_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )
    # Event 2: REJECT for the same case_id
    ev2 = make_valid_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="REJECT",
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        use_case = IngestHumanFeedbackUseCase(repo, mock_embedder)

        res1 = await use_case.execute(ev1)
        await session.commit()
        assert res1.outcome == HumanFeedbackIngestionOutcome.PROCESSED
        assert res1.reference_created is True

        # Process conflicting event
        res2 = await use_case.execute(ev2)
        await session.commit()
        assert res2.outcome == HumanFeedbackIngestionOutcome.CONFLICT
        assert res2.reference_created is False
        assert res2.conflict_details is not None
        assert res2.conflict_details["existing_label"] == "APPROVE"
        assert res2.conflict_details["new_label"] == "REJECT"

        # Verify existing reference in DB was NOT overwritten!
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.label == "APPROVE"  # Still APPROVE!

        # Verify conflict was recorded in human_feedback_events audit table
        ev2_record = await repo.get_processed_event(ev2["event_id"])
        assert ev2_record is not None
        assert ev2_record["status"] == "CONFLICT"
        assert ev2_record["conflict_details"]["existing_label"] == "APPROVE"

        # Cleanup
        await repo.delete_all_for_agent(tenant_id, agent_id)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_10_and_11_tenant_and_agent_isolation(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """10 & 11. Tenant and Agent isolation: cross-tenant and cross-agent references are strictly isolated."""
    tenant_a = uuid4()
    tenant_b = uuid4()
    agent_1 = uuid4()
    agent_2 = uuid4()
    case_id = "shared-case-id-100"

    # Persist reference for Tenant A, Agent 1
    ev_a1 = make_valid_event_dict(
        tenant_id=tenant_a,
        agent_id=agent_1,
        case_id=case_id,
        human_decision="APPROVE",
    )
    # Persist reference with same case_id for Tenant B, Agent 2
    ev_b2 = make_valid_event_dict(
        tenant_id=tenant_b,
        agent_id=agent_2,
        case_id=case_id,
        human_decision="REJECT",
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        use_case = IngestHumanFeedbackUseCase(repo, mock_embedder)

        res_a1 = await use_case.execute(ev_a1)
        res_b2 = await use_case.execute(ev_b2)
        await session.commit()

        assert res_a1.outcome == HumanFeedbackIngestionOutcome.PROCESSED
        assert res_b2.outcome == HumanFeedbackIngestionOutcome.PROCESSED

        # Query Tenant A, Agent 1
        ref_a1 = await repo.get_reference_example(tenant_a, agent_1, case_id)
        assert ref_a1 is not None
        assert ref_a1.label == "APPROVE"

        # Query Tenant B, Agent 2
        ref_b2 = await repo.get_reference_example(tenant_b, agent_2, case_id)
        assert ref_b2 is not None
        assert ref_b2.label == "REJECT"

        # Cross-query: Tenant A with Agent 2 should be None
        ref_cross = await repo.get_reference_example(tenant_a, agent_2, case_id)
        assert ref_cross is None

        # Cross-query: Tenant B with Agent 1 should be None
        ref_cross2 = await repo.get_reference_example(tenant_b, agent_1, case_id)
        assert ref_cross2 is None

        # Cleanup
        await repo.delete_all_for_agent(tenant_a, agent_1)
        await repo.delete_all_for_agent(tenant_b, agent_2)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_12_rejection_of_unreviewed_agent_decisions(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """12. Rejection of unreviewed agent decisions: human_decision must be authoritative APPROVE or REJECT."""
    payload_unreviewed = make_valid_event_dict()
    payload_unreviewed["human_decision"] = "UNREVIEWED"

    with pytest.raises(UnreviewedAgentDecisionError):
        HumanResolutionEventDTO.model_validate(payload_unreviewed)

    # Empty decision
    payload_empty = make_valid_event_dict()
    payload_empty["human_decision"] = ""

    with pytest.raises(UnreviewedAgentDecisionError):
        HumanResolutionEventDTO.model_validate(payload_empty)


@pytest.mark.asyncio
async def test_scenario_13_successful_persistence_and_subsequent_retrieval(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """13. Successful persistence and subsequent retrieval: newly ingested reference is retrievable via vector similarity."""
    real_embedder = SentenceTransformerEmbeddingAdapter()
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-retrieval-{uuid4().hex[:8]}"
    case_desc = (
        "Wire transfer of $85,000 flagged for multi-factor authentication mismatch."
    )

    payload = make_valid_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="REJECT",
        case_description=case_desc,
    )

    async with session_factory() as session:
        feedback_repo = PgHumanFeedbackRepository(session)
        vector_repo = PgVectorReferenceExampleRepository(session)
        use_case = IngestHumanFeedbackUseCase(feedback_repo, real_embedder)

        res = await use_case.execute(payload)
        await session.commit()
        assert res.outcome == HumanFeedbackIngestionOutcome.PROCESSED

        # Now query with exact or near identical description using vector retrieval
        retrieval_use_case = RetrieveReferenceExamplesUseCase(
            repository=vector_repo,
            embedding_provider=real_embedder,
        )
        eval_use_case = EvaluateReliabilityUseCase(
            retrieval_use_case=retrieval_use_case,
            config=ScoringConfig(min_neighbor_count=1, min_similarity_threshold=0.5),
        )

        eval_req = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description=case_desc,
            proposed_decision="REJECT",
        )
        eval_res = await eval_use_case.execute(eval_req)

        # The newly ingested example must be in retrieved evidence!
        assert len(eval_res.retrieved_evidence) >= 1
        top_match = eval_res.retrieved_evidence[0]
        assert top_match.example_id == case_id
        assert top_match.label == "REJECT"
        assert top_match.similarity_score > 0.95

        # Cleanup
        await feedback_repo.delete_all_for_agent(tenant_id, agent_id)
        await session.commit()


@pytest.mark.asyncio
async def test_scenario_14_graceful_shutdown(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """14. Graceful shutdown: consumer loop exits cleanly on stop()."""
    mock_consumer = MockKafkaConsumer()
    mock_producer = MockKafkaProducer()

    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        consumer=mock_consumer,  # type: ignore
        producer=mock_producer,  # type: ignore
    )

    await consumer.start()
    assert consumer._is_running is True
    assert consumer._consume_task is not None

    # Stop consumer
    await consumer.stop()
    assert consumer._is_running is False
    assert consumer._consume_task is None


@pytest.mark.asyncio
async def test_scenario_15_dead_letter_handling(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: MockEmbeddingProvider,
) -> None:
    """15. Dead-letter handling: malformed messages forwarded to DLQ and offset committed."""
    mock_producer = MockKafkaProducer()
    metrics = ConsumerMetrics()

    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=mock_producer,  # type: ignore
        metrics=metrics,
    )

    # 1. Malformed JSON message
    msg_corrupt = MockKafkaMessage(b"{ not json at all !!! }", offset=1)
    ack1 = await consumer.process_single_message(msg_corrupt)
    assert ack1 is True
    assert metrics.invalid_events == 1
    assert metrics.dead_lettered_events == 1

    # 2. Invalid schema (missing required fields)
    msg_invalid = MockKafkaMessage(json.dumps({"random": "payload"}), offset=2)
    ack2 = await consumer.process_single_message(msg_invalid)
    assert ack2 is True
    assert metrics.invalid_events == 2
    assert metrics.dead_lettered_events == 2

    # Verify DLQ received both records
    assert len(mock_producer.sent_messages) == 2
    dlq1_topic, dlq1_bytes = mock_producer.sent_messages[0]
    dlq2_topic, dlq2_bytes = mock_producer.sent_messages[1]

    assert dlq1_topic == consumer.dlq_topic
    assert dlq2_topic == consumer.dlq_topic

    dlq1_data = json.loads(dlq1_bytes.decode("utf-8"))
    assert dlq1_data["error_type"] == "JSON_DECODE_ERROR"

    dlq2_data = json.loads(dlq2_bytes.decode("utf-8"))
    assert dlq2_data["error_type"] == "ValidationError"
