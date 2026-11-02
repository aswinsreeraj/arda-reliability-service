"""Comprehensive security and conflict hardening tests for human feedback ingestion.

Tests all 15 required verification scenarios:
1. Unauthorized producer rejected and routed to DLQ.
2. Invalid reviewer authorization rejected and routed to DLQ.
3. Valid trusted human-resolution event persisted as active reference.
4. Forged reviewer identity rejected and routed to DLQ.
5. Conflicting human decisions quarantine reference example and record conflict audit.
6. Quarantined reference example excluded from vector retrieval and reliability scoring.
7. Duplicate delivery after quarantine does not reactivate reference.
8. Concurrent conflicting events result in safe quarantined state without split-brain.
9. Database uniqueness race handled gracefully via idempotent constraints.
10. DLQ acknowledgement received before original offset commit.
11. DLQ publication failure prevents offset commit.
12. Offset commit failure after DLQ publication handled safely on redelivery.
13. Sequential partition processing guarantees monotonically increasing offset commits.
14. Consumer shutdown with unfinished messages leaves uncommitted offsets.
15. Tenant and agent isolation preserved after conflict quarantine.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from aiokafka import TopicPartition
from aiokafka.structs import OffsetAndMetadata
from sqlalchemy import text
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
from arda_reliability_service.infrastructure.messaging.authorizer import (
    TrustedProducerAuthorizer,
)
from arda_reliability_service.infrastructure.messaging.consumer import (
    RedpandaFeedbackConsumer,
)


# --- Mock Embedding Provider ---
class DeterministicMockEmbedder(EmbeddingProviderPort):
    """Deterministic mock embedding provider."""

    def __init__(self, dimension: int = 384, should_fail: bool = False) -> None:
        self._dim = dimension
        self.should_fail = should_fail
        self.embedded_texts: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "BAAI/bge-small-en-v1.5"

    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        if self.should_fail:
            raise RuntimeError("Embedding model inference failed!")
        self.embedded_texts.append(text)
        val = 1.0 / (self._dim**0.5)
        return [val] * self._dim

    def generate_embeddings_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        return [self.generate_embedding(t, is_query=is_query) for t in texts]


# --- Mock Kafka Infrastructure ---
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
    def __init__(self, fail_dlq: bool = False) -> None:
        self.sent_messages: list[tuple[str, bytes]] = []
        self.call_log: list[str] = []
        self.fail_dlq = fail_dlq

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_and_wait(self, topic: str, value: bytes) -> None:
        self.call_log.append(f"send_and_wait:{topic}")
        if self.fail_dlq:
            raise RuntimeError("Kafka broker unavailable for DLQ write")
        self.sent_messages.append((topic, value))


class MockKafkaConsumer:
    def __init__(
        self,
        messages: list[MockKafkaMessage] | None = None,
        fail_commit: bool = False,
    ) -> None:
        self.messages = list(messages or [])
        self.committed: list[Any] = []
        self.call_log: list[str] = []
        self.fail_commit = fail_commit
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
        await asyncio.sleep(100)
        raise asyncio.CancelledError()

    async def commit(self, offsets: Any) -> None:
        self.call_log.append("commit")
        if self.fail_commit:
            raise RuntimeError("Kafka commit failed: group coordinator unavailable")
        self.committed.append(offsets)


# --- Database Fixtures ---
@pytest_asyncio.fixture
async def db_manager() -> Any:
    db = DatabaseManager()
    yield db
    if db._engine:
        await db._engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    db_manager: DatabaseManager,
) -> async_sessionmaker[AsyncSession]:
    return db_manager.session_factory


@pytest.fixture
def mock_embedder() -> DeterministicMockEmbedder:
    return DeterministicMockEmbedder(dimension=384)


def make_test_event_dict(
    event_id: str | None = None,
    tenant_id: UUID | None = None,
    agent_id: UUID | None = None,
    case_id: str = "case-sec-001",
    human_decision: str = "APPROVE",
    case_description: str = "Security verified customer dispute for account limit.",
    reviewer_id: str = "reviewer_alice_42",
    producer_id: str = "arda.human-review-service",
    producer_token: str = "arda-producer-secret-default",
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
        "reviewer_id": reviewer_id,
        "producer_id": producer_id,
        "producer_token": producer_token,
        "resolved_at": datetime.now(UTC).isoformat(),
        "correlation_id": "corr-sec-123",
    }


# ==============================================================================
# 15 Test Scenarios
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_1_unauthorized_producer_routed_to_dlq(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """1. Unauthorized producer: invalid producer_id or bad token routed to DLQ."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    # Unknown producer
    payload_bad_producer = make_test_event_dict(
        producer_id="malicious.rogue-actor",
        case_id="case-unauth-01",
    )
    msg1 = MockKafkaMessage(json.dumps(payload_bad_producer))
    res1 = await consumer.process_single_message(msg1)
    assert res1 is True
    assert len(producer.sent_messages) == 1
    dlq_topic1, dlq_bytes1 = producer.sent_messages[0]
    assert dlq_topic1 == "arda.human-resolutions.dlq"
    record1 = json.loads(dlq_bytes1.decode("utf-8"))
    assert record1["error_type"] == "UnauthorizedProducerError"

    # Invalid token for known producer
    payload_bad_token = make_test_event_dict(
        producer_id="arda.human-review-service",
        producer_token="wrong-secret-token",
        case_id="case-unauth-02",
    )
    msg2 = MockKafkaMessage(json.dumps(payload_bad_token))
    res2 = await consumer.process_single_message(msg2)
    assert res2 is True
    assert len(producer.sent_messages) == 2
    _dlq_topic2, dlq_bytes2 = producer.sent_messages[1]
    record2 = json.loads(dlq_bytes2.decode("utf-8"))
    assert record2["error_type"] == "UnauthorizedProducerError"

    # Verify no reference example was created
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM reference_examples WHERE example_id IN ('case-unauth-01', 'case-unauth-02')"
            )
        )
        assert result.scalar_one() == 0


@pytest.mark.asyncio
async def test_scenario_2_invalid_reviewer_authorization_routed_to_dlq(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """2. Invalid reviewer authorization: blank/missing reviewer identity routed to DLQ."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    # Empty reviewer
    payload_empty_reviewer = make_test_event_dict(
        reviewer_id="",
        case_id="case-invalid-rev-01",
    )
    msg = MockKafkaMessage(json.dumps(payload_empty_reviewer))
    res = await consumer.process_single_message(msg)
    assert res is True
    assert len(producer.sent_messages) == 1
    _dlq_topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] in (
        "UnauthorizedReviewerError",
        "EventValidationError",
    )


@pytest.mark.asyncio
async def test_scenario_3_valid_trusted_human_resolution_persisted(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """3. Valid trusted human-resolution event: persisted with active status and embedding."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-valid-sec-{uuid4().hex[:8]}"

    payload = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
        reviewer_id="reviewer_bob_101",
    )
    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)
    assert res is True
    assert len(producer.sent_messages) == 0

    # Verify database persistence
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "active"
        assert ref.label == "APPROVE"
        assert ref.source in ("human_adjudication", "human_review")
        assert ref.embedding is not None
        assert len(ref.embedding) == 384


@pytest.mark.asyncio
async def test_scenario_4_forged_reviewer_identity_routed_to_dlq(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """4. Forged reviewer identity: bot/automated identities rejected."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    forged_reviewers = [
        "anonymous",
        "system_bot",
        "automated_reviewer_v2",
        "none",
    ]
    for forged in forged_reviewers:
        payload = make_test_event_dict(
            reviewer_id=forged,
            case_id=f"case-forged-{forged}",
        )
        msg = MockKafkaMessage(json.dumps(payload))
        res = await consumer.process_single_message(msg)
        assert res is True

    assert len(producer.sent_messages) == len(forged_reviewers)
    for _, dlq_bytes in producer.sent_messages:
        record = json.loads(dlq_bytes.decode("utf-8"))
        assert record["error_type"] == "UnauthorizedReviewerError"


@pytest.mark.asyncio
async def test_scenario_5_conflicting_human_decisions_quarantine_reference(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """5. Conflicting human decisions: marks reference as quarantined and preserves both in audit."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-conflict-{uuid4().hex[:8]}"

    # 1. First event: APPROVE
    ev1 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )
    # 2. Opposing event: REJECT
    ev2 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="REJECT",
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        authorizer = TrustedProducerAuthorizer()
        use_case = IngestHumanFeedbackUseCase(
            repo, mock_embedder, authorizer=authorizer
        )

        res1 = await use_case.execute(ev1)
        await session.commit()
        assert res1.outcome == HumanFeedbackIngestionOutcome.PROCESSED
        assert res1.reference_created is True

        res2 = await use_case.execute(ev2)
        await session.commit()
        assert res2.outcome == HumanFeedbackIngestionOutcome.CONFLICT
        assert res2.reference_created is False

    # Check PostgreSQL state
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "quarantined"
        assert "conflict" in ref.metadata
        assert ref.metadata["conflict"]["existing_label"] == "APPROVE"
        assert ref.metadata["conflict"]["new_label"] == "REJECT"

        # Check audit table has both events
        result = await session.execute(
            text(
                "SELECT event_id, status FROM human_feedback_events WHERE case_id = :case_id ORDER BY created_at ASC"
            ),
            {"case_id": case_id},
        )
        rows = result.all()
        assert len(rows) == 2
        assert rows[0][1] == "PROCESSED"
        assert rows[1][1] == "CONFLICT"


@pytest.mark.asyncio
async def test_scenario_6_quarantined_example_excluded_from_retrieval(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """6. Quarantined reference example: excluded from vector retrieval and evaluation."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-excl-{uuid4().hex[:8]}"
    desc = "Quarantined description that should never be retrieved as evidence."

    ev1 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        case_description=desc,
        human_decision="APPROVE",
    )
    ev2 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        case_description=desc,
        human_decision="REJECT",
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        authorizer = TrustedProducerAuthorizer()
        use_case = IngestHumanFeedbackUseCase(
            repo, mock_embedder, authorizer=authorizer
        )
        await use_case.execute(ev1)
        await session.commit()
        await use_case.execute(ev2)
        await session.commit()

    # Now perform retrieval and evaluation using the exact description
    async with session_factory() as session:
        retrieval_repo = PgVectorReferenceExampleRepository(session)
        retrieval_use_case = RetrieveReferenceExamplesUseCase(
            retrieval_repo, mock_embedder
        )
        eval_use_case = EvaluateReliabilityUseCase(
            retrieval_use_case=retrieval_use_case,
            config=ScoringConfig(min_neighbor_count=1),
        )

        req = ReliabilityEvaluationRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description=desc,
            proposed_decision="APPROVE",
        )
        eval_res = await eval_use_case.execute(req)

        # Quarantined example is not active, so retrieved evidence must be empty
        assert len(eval_res.retrieved_evidence) == 0
        assert eval_res.decision.value == "ESCALATE"


@pytest.mark.asyncio
async def test_scenario_7_duplicate_delivery_after_quarantine_does_not_reactivate(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """7. Duplicate delivery after quarantine: stays quarantined and is not reactivated."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-dup-quar-{uuid4().hex[:8]}"

    ev1 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )
    ev2 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="REJECT",
    )
    # Duplicate of first event
    ev3_dup = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        authorizer = TrustedProducerAuthorizer()
        use_case = IngestHumanFeedbackUseCase(
            repo, mock_embedder, authorizer=authorizer
        )
        await use_case.execute(ev1)
        await session.commit()
        await use_case.execute(ev2)
        await session.commit()

        # Execute duplicate
        res3 = await use_case.execute(ev3_dup)
        await session.commit()
        assert res3.outcome == HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE

        # Confirm reference is still quarantined
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "quarantined"


@pytest.mark.asyncio
async def test_scenario_8_concurrent_conflicting_events_safe_quarantine(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """8. Concurrent conflicting events: safely resolve into quarantined state."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-concurrent-conflict-{uuid4().hex[:8]}"

    ev_app = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )
    ev_rej = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="REJECT",
    )

    async def run_event(ev: dict[str, Any]) -> HumanFeedbackIngestionOutcome:
        async with session_factory() as session:
            repo = PgHumanFeedbackRepository(session)
            authorizer = TrustedProducerAuthorizer()
            use_case = IngestHumanFeedbackUseCase(
                repo, mock_embedder, authorizer=authorizer
            )
            res = await use_case.execute(ev)
            await session.commit()
            return res.outcome

    results: tuple[Any, ...] = await asyncio.gather(
        run_event(ev_app), run_event(ev_rej)
    )
    # One is PROCESSED and the other is CONFLICT
    assert set(results) == {
        HumanFeedbackIngestionOutcome.PROCESSED,
        HumanFeedbackIngestionOutcome.CONFLICT,
    }

    # Final DB state must be quarantined
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "quarantined"


@pytest.mark.asyncio
async def test_scenario_9_database_uniqueness_race_graceful_handling(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """9. Database uniqueness race: concurrent identical events handled gracefully."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-race-{uuid4().hex[:8]}"

    ev1 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )
    ev2 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )

    async def run_event(ev: dict[str, Any]) -> HumanFeedbackIngestionOutcome:
        async with session_factory() as session:
            repo = PgHumanFeedbackRepository(session)
            authorizer = TrustedProducerAuthorizer()
            use_case = IngestHumanFeedbackUseCase(
                repo, mock_embedder, authorizer=authorizer
            )
            res = await use_case.execute(ev)
            await session.commit()
            return res.outcome

    results: tuple[Any, ...] = await asyncio.gather(run_event(ev1), run_event(ev2))
    for r in results:
        assert r in (
            HumanFeedbackIngestionOutcome.PROCESSED,
            HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE,
        )

    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM reference_examples WHERE tenant_id = :t AND agent_id = :a AND example_id = :c"
            ),
            {"t": tenant_id, "a": agent_id, "c": case_id},
        )
        assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_scenario_10_dlq_acknowledgement_before_offset_commit(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """10. DLQ acknowledgement before original offset commit."""
    producer = MockKafkaProducer()
    consumer_mock = MockKafkaConsumer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

    # Invalid JSON message
    msg = MockKafkaMessage(b"not-json", offset=42)
    should_commit = await consumer.process_single_message(msg)
    assert should_commit is True
    if should_commit:
        await consumer._commit_message_offset(msg)

    # Verify DLQ publication completed before offset commit
    assert producer.call_log == ["send_and_wait:arda.human-resolutions.dlq"]
    assert consumer_mock.call_log == ["commit"]
    tp = TopicPartition(msg.topic, msg.partition)
    assert consumer_mock.committed == [{tp: OffsetAndMetadata(43, "")}]


@pytest.mark.asyncio
async def test_scenario_11_dlq_publication_failure_prevents_offset_commit(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """11. DLQ publication failure: offset is NOT committed when broker fails."""
    # Producer configured to fail on DLQ send
    failing_producer = MockKafkaProducer(fail_dlq=True)
    consumer_mock = MockKafkaConsumer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=failing_producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

    msg = MockKafkaMessage(b"not-json", offset=50)
    should_commit = await consumer.process_single_message(msg)
    # process_single_message must return False because DLQ write failed!
    assert should_commit is False
    if should_commit:
        await consumer._commit_message_offset(msg)

    # Offset must NOT have been committed
    assert len(consumer_mock.committed) == 0


@pytest.mark.asyncio
async def test_scenario_12_offset_commit_failure_after_dlq_handled_safely(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """12. Offset commit failure after DLQ publication handled safely on redelivery."""
    producer = MockKafkaProducer()
    # Consumer configured to fail on commit
    failing_consumer = MockKafkaConsumer(fail_commit=True)
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
        consumer=failing_consumer,  # type: ignore[arg-type]
    )

    msg = MockKafkaMessage(b"bad-message", offset=99)
    should_commit = await consumer.process_single_message(msg)
    assert should_commit is True

    # Attempt offset commit
    committed_ok = await consumer._commit_message_offset(msg)
    assert committed_ok is False

    # On redelivery of same message, DLQ receives it again without crash
    redelivery_should_commit = await consumer.process_single_message(msg)
    assert redelivery_should_commit is True
    assert len(producer.sent_messages) == 2


@pytest.mark.asyncio
async def test_scenario_13_sequential_partition_processing_offset_safety(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """13. Sequential partition processing ensures monotonically increasing offset commits."""
    producer = MockKafkaProducer()
    consumer_mock = MockKafkaConsumer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

    # 3 sequential messages on partition 0 with offsets 100, 101, 102
    events = [make_test_event_dict(case_id=f"seq-case-{i}") for i in range(3)]
    msgs = [
        MockKafkaMessage(json.dumps(ev), partition=0, offset=100 + i)
        for i, ev in enumerate(events)
    ]

    for m in msgs:
        ok = await consumer.process_single_message(m)
        if ok:
            await consumer._commit_message_offset(m)

    committed_offsets = [
        commit_dict[TopicPartition(msgs[0].topic, 0)].offset
        for commit_dict in consumer_mock.committed
    ]
    assert committed_offsets == [101, 102, 103]


@pytest.mark.asyncio
async def test_scenario_14_consumer_shutdown_with_unfinished_messages(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """14. Consumer shutdown with unfinished messages leaves offset uncommitted."""
    msg = MockKafkaMessage(b"long-running-message", offset=777)
    consumer_mock = MockKafkaConsumer([msg])
    producer = MockKafkaProducer()

    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

    # Start consumer loop
    await consumer.start()
    # Immediately stop before processing finishes
    await consumer.stop()

    # Offset must NOT be committed
    assert len(consumer_mock.committed) == 0


@pytest.mark.asyncio
async def test_scenario_15_tenant_and_agent_isolation_after_conflict_quarantine(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """15. Tenant and agent isolation: quarantining in Tenant A does NOT affect Tenant B."""
    tenant_a = uuid4()
    agent_a = uuid4()
    tenant_b = uuid4()
    agent_b = uuid4()

    shared_desc = "Standard subscription cancellation dispute."
    shared_case_id = "case-shared-desc-001"

    # Tenant A receives conflicting events -> Quarantined
    ev_a1 = make_test_event_dict(
        tenant_id=tenant_a,
        agent_id=agent_a,
        case_id=shared_case_id,
        case_description=shared_desc,
        human_decision="APPROVE",
    )
    ev_a2 = make_test_event_dict(
        tenant_id=tenant_a,
        agent_id=agent_a,
        case_id=shared_case_id,
        case_description=shared_desc,
        human_decision="REJECT",
    )

    # Tenant B receives authoritative APPROVE -> Active
    ev_b = make_test_event_dict(
        tenant_id=tenant_b,
        agent_id=agent_b,
        case_id=shared_case_id,
        case_description=shared_desc,
        human_decision="APPROVE",
    )

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        authorizer = TrustedProducerAuthorizer()
        use_case = IngestHumanFeedbackUseCase(
            repo, mock_embedder, authorizer=authorizer
        )

        await use_case.execute(ev_a1)
        await session.commit()
        await use_case.execute(ev_a2)
        await session.commit()
        await use_case.execute(ev_b)
        await session.commit()

    # Verify database isolation
    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref_a = await repo.get_reference_example(tenant_a, agent_a, shared_case_id)
        ref_b = await repo.get_reference_example(tenant_b, agent_b, shared_case_id)

        assert ref_a is not None
        assert ref_a.status == "quarantined"

        assert ref_b is not None
        assert ref_b.status == "active"

        # Evaluate Tenant B: finds active reference
        retrieval_repo = PgVectorReferenceExampleRepository(session)
        retrieval_use_case = RetrieveReferenceExamplesUseCase(
            retrieval_repo, mock_embedder
        )
        eval_use_case = EvaluateReliabilityUseCase(
            retrieval_use_case=retrieval_use_case,
            config=ScoringConfig(min_neighbor_count=1),
        )

        req_b = ReliabilityEvaluationRequest(
            tenant_id=tenant_b,
            agent_id=agent_b,
            case_description=shared_desc,
            proposed_decision="APPROVE",
        )
        eval_b = await eval_use_case.execute(req_b)
        assert len(eval_b.retrieved_evidence) == 1
        assert eval_b.retrieved_evidence[0].example_id == ref_b.example_id

        # Evaluate Tenant A: quarantined example is excluded
        req_a = ReliabilityEvaluationRequest(
            tenant_id=tenant_a,
            agent_id=agent_a,
            case_description=shared_desc,
            proposed_decision="APPROVE",
        )
        eval_a = await eval_use_case.execute(req_a)
        assert len(eval_a.retrieved_evidence) == 0
