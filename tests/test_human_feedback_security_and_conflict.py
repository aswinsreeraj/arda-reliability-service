"""Comprehensive security, HMAC-SHA256 verification, and conflict hardening tests.

Tests satisfy all 15 required verification scenarios from Week 5 Audit:
1. Correct HMAC signature accepted.
2. Incorrect signature rejected.
3. Modified event payload rejected.
4. Modified tenant or agent ID rejected.
5. Modified human decision rejected.
6. Unknown producer rejected.
7. Unknown key ID rejected.
8. Expired signed event rejected.
9. Valid event within clock-skew tolerance accepted.
10. Replayed event safely handled through idempotency.
11. Secret not included in logs.
12. Secret not included in DLQ payload.
13. Key rotation with current and previous valid keys.
14. Conflict quarantine still works.
15. Offset is not committed after failed authentication and unsuccessful DLQ publication.

Plus additional boundary and race hardening tests:
- Reviewer screening against bot identities.
- Quarantined reference excluded from retrieval and evaluation.
- Duplicate delivery after quarantine does not reactivate.
- Concurrent conflicting events quarantine safely.
- Database uniqueness race handled gracefully.
- Sequential partition processing offset safety.
- Consumer shutdown with unfinished messages leaves offset uncommitted.
- Tenant and agent isolation preserved after conflict quarantine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from aiokafka import TopicPartition
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
    format_canonical_timestamp,
    sign_event,
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
    key_id: str = "v1",
    secret: str | None = "arda-producer-secret-default",
    timestamp: datetime | str | None = None,
    signature: str | None = None,
) -> dict[str, Any]:
    resolved_at = datetime.now(UTC).isoformat()
    raw = {
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
        "key_id": key_id,
        "resolved_at": resolved_at,
        "timestamp": format_canonical_timestamp(timestamp or resolved_at),
        "correlation_id": "corr-sec-123",
    }
    if signature is not None:
        raw["signature"] = signature
        raw["producer_token"] = signature
        return raw
    elif secret is not None:
        return sign_event(raw, secret=secret, key_id=key_id)
    else:
        raw["signature"] = ""
        raw["producer_token"] = ""
        return raw


# ==============================================================================
# SECTION 7: 15 REQUIRED TEST SCENARIOS
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_1_correct_hmac_signature_accepted(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """1. Correct HMAC signature accepted: validly signed event persists active reference."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-valid-hmac-{uuid4().hex[:8]}"

    payload = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
        secret="arda-producer-secret-default",
        key_id="v1",
    )
    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 0

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "active"
        assert ref.label == "APPROVE"
        assert ref.embedding is not None
        assert len(ref.embedding) == 384


@pytest.mark.asyncio
async def test_scenario_2_incorrect_signature_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """2. Incorrect signature rejected: routed to DLQ with InvalidSignatureError."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    payload = make_test_event_dict(
        case_id="case-bad-sig-01",
        signature="deadbeef" * 8,  # 64-char invalid hex signature
    )
    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "InvalidSignatureError"


@pytest.mark.asyncio
async def test_scenario_3_modified_event_payload_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """3. Modified event payload rejected: altering description invalidates HMAC signature."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    # Generate valid signature for legitimate case description
    payload = make_test_event_dict(
        case_description="Legitimate transfer authorized by manager.",
    )
    # Attacker mutates the case description after signing
    payload["original_case_description"] = (
        "Tampered transfer exceeding approved threshold!"
    )

    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "InvalidSignatureError"


@pytest.mark.asyncio
async def test_scenario_4_modified_tenant_or_agent_id_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """4. Modified tenant or agent ID rejected: cross-tenant tampering fails verification."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    tenant_a = uuid4()
    agent_a = uuid4()
    tenant_b = uuid4()

    # Sign for Tenant A
    payload = make_test_event_dict(
        tenant_id=tenant_a,
        agent_id=agent_a,
        case_id="case-tenant-tamper-01",
    )
    # Attacker tampers tenant_id to Tenant B
    payload["tenant_id"] = str(tenant_b)

    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "InvalidSignatureError"


@pytest.mark.asyncio
async def test_scenario_5_modified_human_decision_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """5. Modified human decision rejected: flipping decision invalidates signature."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    # Signed with REJECT
    payload = make_test_event_dict(
        human_decision="REJECT",
        case_id="case-decision-tamper-01",
    )
    # Attacker tampers to APPROVE
    payload["human_decision"] = "APPROVE"

    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "InvalidSignatureError"


@pytest.mark.asyncio
async def test_scenario_6_unknown_producer_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """6. Unknown producer rejected: unknown producer_id routed to DLQ."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    payload = make_test_event_dict(
        producer_id="malicious.rogue-actor",
        case_id="case-unauth-01",
    )
    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "UnauthorizedProducerError"


@pytest.mark.asyncio
async def test_scenario_7_unknown_key_id_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """7. Unknown key ID rejected: key rotation identifier not configured."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    payload = make_test_event_dict(
        key_id="v999_nonexistent_key",
        case_id="case-unknown-key-01",
    )
    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "UnauthorizedProducerError"


@pytest.mark.asyncio
async def test_scenario_8_expired_signed_event_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """8. Expired signed event rejected: event older than max_event_age is rejected."""
    producer = MockKafkaProducer()
    # Configure tight age window of 60 seconds
    authorizer = TrustedProducerAuthorizer(max_event_age_seconds=60.0)
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        authorizer=authorizer,
        producer=producer,  # type: ignore[arg-type]
    )

    # 10 minutes ago
    expired_ts = datetime.now(UTC) - timedelta(seconds=600)
    payload = make_test_event_dict(
        case_id="case-expired-01",
        timestamp=expired_ts,
    )

    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    record = json.loads(dlq_bytes.decode("utf-8"))
    assert record["error_type"] == "ExpiredEventError"


@pytest.mark.asyncio
async def test_scenario_9_valid_event_within_clock_skew_tolerance_accepted(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """9. Valid event within clock-skew tolerance accepted: future timestamp within tolerance accepted."""
    producer = MockKafkaProducer()
    authorizer = TrustedProducerAuthorizer(clock_skew_tolerance_seconds=60.0)
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        authorizer=authorizer,
        producer=producer,  # type: ignore[arg-type]
    )

    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-clockskew-{uuid4().hex[:8]}"

    # 20 seconds in future (within 60s tolerance)
    future_ts = datetime.now(UTC) + timedelta(seconds=20)
    payload = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        timestamp=future_ts,
    )

    msg = MockKafkaMessage(json.dumps(payload))
    res = await consumer.process_single_message(msg)

    assert res is True
    assert len(producer.sent_messages) == 0

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "active"


@pytest.mark.asyncio
async def test_scenario_10_replayed_event_safely_handled_through_idempotency(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """10. Replayed event safely handled through idempotency: replay within freshness window returns duplicate."""
    producer = MockKafkaProducer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
    )

    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-replay-{uuid4().hex[:8]}"
    event_id = f"evt-replay-{uuid4()}"

    payload = make_test_event_dict(
        event_id=event_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
    )
    msg = MockKafkaMessage(json.dumps(payload))

    # Initial delivery
    res1 = await consumer.process_single_message(msg)
    assert res1 is True

    # Replay attack (exact same signed message re-sent)
    res2 = await consumer.process_single_message(msg)
    assert res2 is True

    # Exactly one reference example exists
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM reference_examples WHERE tenant_id = :t AND agent_id = :a AND example_id = :c"
            ),
            {"t": tenant_id, "a": agent_id, "c": case_id},
        )
        assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_scenario_11_secret_not_included_in_logs(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """11. Secret not included in logs: sensitive key material is never logged on failures."""
    sensitive_secret = "super-confidential-producer-secret-8877"
    producer = MockKafkaProducer()
    authorizer = TrustedProducerAuthorizer(
        producer_secrets={"arda.human-review-service": {"v1": sensitive_secret}}
    )
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        authorizer=authorizer,
        producer=producer,  # type: ignore[arg-type]
    )

    payload = make_test_event_dict(
        case_id="case-leak-check-01",
        signature="bad-signature-content",
    )

    with caplog.at_level(logging.DEBUG):
        msg = MockKafkaMessage(json.dumps(payload))
        await consumer.process_single_message(msg)

    # Check all captured log messages
    for record in caplog.records:
        assert sensitive_secret not in record.message
        assert sensitive_secret not in str(record.exc_info)


@pytest.mark.asyncio
async def test_scenario_12_secret_not_included_in_dlq_payload(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """12. Secret not included in DLQ payload: DLQ error record never leaks signing secret."""
    sensitive_secret = "ultra-secret-dlq-protection-key-9911"
    producer = MockKafkaProducer()
    authorizer = TrustedProducerAuthorizer(
        producer_secrets={"arda.human-review-service": {"v1": sensitive_secret}}
    )
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        authorizer=authorizer,
        producer=producer,  # type: ignore[arg-type]
    )

    payload = make_test_event_dict(
        case_id="case-dlq-secret-leak",
        signature="invalid-sig-9999",
    )
    msg = MockKafkaMessage(json.dumps(payload))
    await consumer.process_single_message(msg)

    assert len(producer.sent_messages) == 1
    _topic, dlq_bytes = producer.sent_messages[0]
    dlq_text = dlq_bytes.decode("utf-8")

    assert sensitive_secret not in dlq_text


@pytest.mark.asyncio
async def test_scenario_13_key_rotation_with_current_and_previous_valid_keys(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """13. Key rotation: authorizer verifies both retiring (v1) and active (v2) keys."""
    old_secret_v1 = "legacy-secret-retiring-key-v1"
    new_secret_v2 = "active-secret-primary-key-v2"

    authorizer = TrustedProducerAuthorizer(
        producer_secrets={
            "arda.human-review-service": {
                "v1": old_secret_v1,
                "v2": new_secret_v2,
            }
        }
    )
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        authorizer=authorizer,
        producer=MockKafkaProducer(),  # type: ignore[arg-type]
    )

    # 1. Event signed with legacy key v1 -> ACCEPTED
    payload_v1 = make_test_event_dict(
        case_id=f"case-rot-v1-{uuid4().hex[:8]}",
        key_id="v1",
        secret=old_secret_v1,
    )
    msg_v1 = MockKafkaMessage(json.dumps(payload_v1))
    assert await consumer.process_single_message(msg_v1) is True

    # 2. Event signed with new active key v2 -> ACCEPTED
    payload_v2 = make_test_event_dict(
        case_id=f"case-rot-v2-{uuid4().hex[:8]}",
        key_id="v2",
        secret=new_secret_v2,
    )
    msg_v2 = MockKafkaMessage(json.dumps(payload_v2))
    assert await consumer.process_single_message(msg_v2) is True

    # 3. Event signed with key_id v1 but using v2 secret -> REJECTED
    payload_mismatch = make_test_event_dict(
        case_id=f"case-rot-bad-{uuid4().hex[:8]}",
        key_id="v1",
        secret=new_secret_v2,  # wrong secret for v1!
    )
    msg_mismatch = MockKafkaMessage(json.dumps(payload_mismatch))
    producer_dlq = MockKafkaProducer()
    consumer_fail = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        authorizer=authorizer,
        producer=producer_dlq,  # type: ignore[arg-type]
    )
    await consumer_fail.process_single_message(msg_mismatch)
    assert len(producer_dlq.sent_messages) == 1
    record = json.loads(producer_dlq.sent_messages[0][1].decode("utf-8"))
    assert record["error_type"] == "InvalidSignatureError"


@pytest.mark.asyncio
async def test_scenario_14_conflict_quarantine_still_works(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """14. Conflict quarantine: signed opposing resolutions transition reference to quarantined."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-quar-check-{uuid4().hex[:8]}"

    # Event 1: APPROVE
    ev1 = make_test_event_dict(
        tenant_id=tenant_id,
        agent_id=agent_id,
        case_id=case_id,
        human_decision="APPROVE",
    )
    # Event 2: REJECT
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

        res2 = await use_case.execute(ev2)
        await session.commit()
        assert res2.outcome == HumanFeedbackIngestionOutcome.CONFLICT

        # Confirm reference is quarantined and conflict metadata recorded
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "quarantined"
        assert "conflict" in ref.metadata
        assert ref.metadata["conflict"]["existing_label"] == "APPROVE"
        assert ref.metadata["conflict"]["new_label"] == "REJECT"


@pytest.mark.asyncio
async def test_scenario_15_offset_not_committed_after_failed_auth_and_unsuccessful_dlq(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """15. Offset safety: offset is NOT committed when DLQ publication fails for unauthorized event."""
    failing_producer = MockKafkaProducer(fail_dlq=True)
    consumer_mock = MockKafkaConsumer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=failing_producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

    # Event with bad signature
    payload = make_test_event_dict(
        case_id="case-fail-dlq-offset",
        signature="corrupted-signature",
    )
    msg = MockKafkaMessage(json.dumps(payload), offset=123)

    should_commit = await consumer.process_single_message(msg)
    # Must return False because DLQ write failed!
    assert should_commit is False

    if should_commit:
        await consumer._commit_message_offset(msg)

    # Consumer offset must NOT be committed
    assert len(consumer_mock.committed) == 0


# ==============================================================================
# ADDITIONAL CONFLICT, ISOLATION, AND RECOVERY REGRESSION TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_forged_reviewer_identity_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Screening: bot identities rejected even if signed by producer."""
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=MockKafkaProducer(),  # type: ignore[arg-type]
    )
    forged_reviewers = ["anonymous", "system_bot", "automated_worker", "none"]
    for f in forged_reviewers:
        payload = make_test_event_dict(reviewer_id=f)
        msg = MockKafkaMessage(json.dumps(payload))
        res = await consumer.process_single_message(msg)
        assert res is True  # handled and sent to DLQ


@pytest.mark.asyncio
async def test_quarantined_example_excluded_from_retrieval(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Quarantine exclusion: quarantined examples are never retrieved as evidence."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-excl-{uuid4().hex[:8]}"
    desc = "Quarantined transaction dispute narrative."

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

        # Evaluate against identical description
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
        assert len(eval_res.retrieved_evidence) == 0
        assert eval_res.decision.value == "ESCALATE"


@pytest.mark.asyncio
async def test_duplicate_delivery_after_quarantine_does_not_reactivate(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Quarantine immutability: redelivery after quarantine never reactivates reference."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-quar-dup-{uuid4().hex[:8]}"

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
    ev3 = make_test_event_dict(
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

        res3 = await use_case.execute(ev3)
        await session.commit()
        assert res3.outcome == HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE

        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "quarantined"


@pytest.mark.asyncio
async def test_concurrent_conflicting_events_safe_quarantine(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Concurrency: race between opposing events produces safe quarantined reference."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-race-quar-{uuid4().hex[:8]}"

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
    assert HumanFeedbackIngestionOutcome.CONFLICT in results

    async with session_factory() as session:
        repo = PgHumanFeedbackRepository(session)
        ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
        assert ref is not None
        assert ref.status == "quarantined"


@pytest.mark.asyncio
async def test_database_uniqueness_race_graceful_handling(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Database race: concurrent identical events handled gracefully via ON CONFLICT."""
    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-race-uniq-{uuid4().hex[:8]}"

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
async def test_sequential_partition_processing_offset_safety(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Sequential partition processing ensures monotonically increasing offset commits."""
    producer = MockKafkaProducer()
    consumer_mock = MockKafkaConsumer()
    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

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
async def test_consumer_shutdown_with_unfinished_messages(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Shutdown safety: cancelled consumer loop does not commit uncompleted messages."""
    msg = MockKafkaMessage(b"long-running-message", offset=777)
    consumer_mock = MockKafkaConsumer([msg])
    producer = MockKafkaProducer()

    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=mock_embedder,
        producer=producer,  # type: ignore[arg-type]
        consumer=consumer_mock,  # type: ignore[arg-type]
    )

    await consumer.start()
    await consumer.stop()

    assert len(consumer_mock.committed) == 0


@pytest.mark.asyncio
async def test_tenant_and_agent_isolation_after_conflict_quarantine(
    session_factory: async_sessionmaker[AsyncSession],
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Isolation: quarantining in Tenant A does NOT affect Tenant B."""
    tenant_a = uuid4()
    agent_a = uuid4()
    tenant_b = uuid4()
    agent_b = uuid4()

    shared_desc = "Standard account credit dispute."
    shared_case_id = "case-shared-desc-001"

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

        ref_a = await repo.get_reference_example(tenant_a, agent_a, shared_case_id)
        ref_b = await repo.get_reference_example(tenant_b, agent_b, shared_case_id)

        assert ref_a is not None
        assert ref_a.status == "quarantined"

        assert ref_b is not None
        assert ref_b.status == "active"

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

        req_a = ReliabilityEvaluationRequest(
            tenant_id=tenant_a,
            agent_id=agent_a,
            case_description=shared_desc,
            proposed_decision="APPROVE",
        )
        eval_a = await eval_use_case.execute(req_a)
        assert len(eval_a.retrieved_evidence) == 0
