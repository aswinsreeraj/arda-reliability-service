"""Integration tests running against live Redpanda and PostgreSQL containers."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from aiokafka import AIOKafkaProducer

from arda_reliability_service.config import get_settings
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pg_feedback_repository import (
    PgHumanFeedbackRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from arda_reliability_service.infrastructure.messaging.consumer import (
    RedpandaFeedbackConsumer,
)
from arda_reliability_service.infrastructure.messaging.metrics import ConsumerMetrics


async def is_redpanda_available(bootstrap_servers: str) -> bool:
    """Check if Redpanda broker is responding."""
    try:
        producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
        await asyncio.wait_for(producer.start(), timeout=3.0)
        await producer.stop()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.asyncio
async def test_live_redpanda_e2e_ingestion() -> None:
    """Test full event flow through live Redpanda broker and PostgreSQL database."""
    settings = get_settings()
    if not await is_redpanda_available(settings.redpanda_brokers):
        pytest.skip(f"Redpanda is not reachable at {settings.redpanda_brokers}")

    db = DatabaseManager()
    session_factory = db.session_factory
    embedder = SentenceTransformerEmbeddingAdapter()
    metrics = ConsumerMetrics()

    # Unique test group ID and partition scoping
    test_group = f"test-group-{uuid4().hex[:8]}"
    settings_override = settings.model_copy(
        update={
            "redpanda_consumer_group": test_group,
            "redpanda_auto_offset_reset": "latest",
        }
    )

    consumer = RedpandaFeedbackConsumer(
        session_factory=session_factory,
        embedding_provider=embedder,
        settings=settings_override,
        metrics=metrics,
    )

    # Start live consumer
    await consumer.start()

    # Set up producer
    producer = AIOKafkaProducer(bootstrap_servers=settings.redpanda_brokers)
    await producer.start()

    tenant_id = uuid4()
    agent_id = uuid4()
    case_id = f"case-live-{uuid4().hex[:8]}"
    event_id = f"evt-live-{uuid4().hex[:8]}"

    event_payload = {
        "event_id": event_id,
        "event_type": settings.redpanda_topic,
        "event_version": "v1",
        "tenant_id": str(tenant_id),
        "agent_id": str(agent_id),
        "case_id": case_id,
        "original_case_description": "Approved expedited travel reimbursement under emergency provisions.",
        "proposed_decision": "APPROVE",
        "human_decision": "APPROVE",
        "reviewer_id": "auditor_bob_99",
        "resolved_at": datetime.now(UTC).isoformat(),
        "correlation_id": f"corr-{uuid4().hex[:8]}",
    }

    try:
        # Publish event to Redpanda
        val_bytes = json.dumps(event_payload).encode("utf-8")
        await producer.send_and_wait(settings.redpanda_topic, value=val_bytes)

        # Wait for consumer to process message
        for _ in range(30):
            if metrics.events_processed >= 1:
                break
            await asyncio.sleep(0.5)

        assert metrics.events_processed >= 1

        # Verify record in PostgreSQL database
        async with session_factory() as session:
            repo = PgHumanFeedbackRepository(session)
            ref = await repo.get_reference_example(tenant_id, agent_id, case_id)
            assert ref is not None
            assert ref.label == "APPROVE"
            assert ref.source == "human_review"
            assert ref.embedding is not None
            assert len(ref.embedding) == 384

            ev_record = await repo.get_processed_event(event_id)
            assert ev_record is not None
            assert ev_record["status"] == "PROCESSED"

            # Clean up test records
            await repo.delete_all_for_agent(tenant_id, agent_id)
            await session.commit()

        # Send invalid JSON to verify DLQ routing
        await producer.send_and_wait(settings.redpanda_topic, value=b"invalid-not-json")
        for _ in range(30):
            if metrics.dead_lettered_events >= 1:
                break
            await asyncio.sleep(0.5)

        assert metrics.dead_lettered_events >= 1

    finally:
        await consumer.stop()
        await producer.stop()
        if db._engine:
            await db._engine.dispose()
