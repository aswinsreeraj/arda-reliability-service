"""Asynchronous Redpanda / Kafka consumer for authoritative human feedback events."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.structs import OffsetAndMetadata
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.application.ports.event_authorizer import (
    EventAuthorizerPort,
)
from arda_reliability_service.application.use_cases.ingest_human_feedback import (
    HumanFeedbackIngestionOutcome,
    IngestHumanFeedbackUseCase,
)
from arda_reliability_service.config import Settings, get_settings
from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    EventValidationError,
    InvalidDescriptionError,
    UnauthorizedProducerError,
    UnauthorizedReviewerError,
    UnreviewedAgentDecisionError,
    UnsupportedEventVersionError,
)
from arda_reliability_service.infrastructure.database.pg_feedback_repository import (
    PgHumanFeedbackRepository,
)
from arda_reliability_service.infrastructure.messaging.authorizer import (
    TrustedProducerAuthorizer,
)
from arda_reliability_service.infrastructure.messaging.events import (
    HumanResolutionEventDTO,
)
from arda_reliability_service.infrastructure.messaging.metrics import ConsumerMetrics

logger = logging.getLogger("arda.messaging.consumer")


class RedpandaFeedbackConsumer:
    """Consumes authoritative human resolutions, performs idempotency checks, and routes invalid events to DLQ."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_provider: EmbeddingProviderPort,
        authorizer: EventAuthorizerPort | None = None,
        settings: Settings | None = None,
        consumer: AIOKafkaConsumer | None = None,
        producer: AIOKafkaProducer | None = None,
        metrics: ConsumerMetrics | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.metrics = metrics or ConsumerMetrics()
        self.authorizer = authorizer or TrustedProducerAuthorizer(
            settings=self.settings
        )

        self.brokers = self.settings.redpanda_brokers
        self.topic = self.settings.redpanda_topic
        self.group_id = self.settings.redpanda_consumer_group
        self.dlq_topic = self.settings.redpanda_dlq_topic
        self.max_retries = self.settings.redpanda_max_retries
        self.retry_backoff_base = self.settings.redpanda_retry_backoff_base_seconds

        # Injected or lazily constructed Kafka clients
        self._consumer = consumer
        self._producer = producer
        self._is_running = False
        self._consume_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the consumer and DLQ producer background processes."""
        if self._is_running:
            return

        logger.info(
            "Starting Redpanda feedback consumer: topic=%s, group=%s, dlq=%s",
            self.topic,
            self.group_id,
            self.dlq_topic,
        )

        if self._consumer is None:
            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.brokers,
                group_id=self.group_id,
                enable_auto_commit=False,
                auto_offset_reset=self.settings.redpanda_auto_offset_reset,
            )
            await self._consumer.start()

        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.brokers,
            )
            await self._producer.start()

        self._is_running = True
        self._consume_task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Gracefully stop the consumer, flush offsets, and stop DLQ producer."""
        if not self._is_running:
            return

        logger.info("Stopping Redpanda feedback consumer gracefully...")
        self._is_running = False

        if self._consume_task is not None:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
            self._consume_task = None

        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("Error stopping Kafka consumer: %s", e)

        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("Error stopping Kafka DLQ producer: %s", e)

        logger.info("Redpanda feedback consumer stopped.")

    async def _send_to_dlq(
        self,
        raw_payload: str | bytes,
        error_type: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a permanently invalid or exhausted message to the Dead-Letter Queue (DLQ).

        Returns True if the message was acknowledged by the DLQ broker, or False if publication failed.
        """
        self.metrics.record_dead_letter()
        dlq_record = {
            "error_type": error_type,
            "error_message": error_message,
            "failed_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        if isinstance(raw_payload, bytes):
            try:
                dlq_record["raw_payload"] = raw_payload.decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                dlq_record["raw_payload"] = "<binary payload omitted>"
        else:
            dlq_record["raw_payload"] = raw_payload

        logger.warning(
            "Routing poison/invalid message to DLQ topic '%s': error_type=%s",
            self.dlq_topic,
            error_type,
        )

        if self._producer is not None:
            try:
                val_bytes = json.dumps(dlq_record).encode("utf-8")
                await self._producer.send_and_wait(self.dlq_topic, value=val_bytes)
                return True
            except Exception as ex:  # noqa: BLE001
                logger.error("Failed to forward event to DLQ: %s", ex)
                return False
        return False

    async def process_single_message(self, msg: Any) -> bool:
        """Process a single Kafka message end-to-end with validation, idempotency, and retry.

        Returns True if message was successfully processed or sent to DLQ (and offset can be committed).
        """
        self.metrics.record_received()
        start_time = time.monotonic()
        raw_value = getattr(msg, "value", b"")

        # 1. Parse JSON
        try:
            if isinstance(raw_value, bytes):
                text_val = raw_value.decode("utf-8")
            else:
                text_val = str(raw_value)
            payload_dict = json.loads(text_val)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            self.metrics.record_invalid()
            logger.error(
                "Failed to decode JSON from message on partition %s offset %s",
                getattr(msg, "partition", None),
                getattr(msg, "offset", None),
            )
            return await self._send_to_dlq(
                raw_payload=raw_value,
                error_type="JSON_DECODE_ERROR",
                error_message=str(e),
                metadata={
                    "partition": getattr(msg, "partition", None),
                    "offset": getattr(msg, "offset", None),
                },
            )

        # 2. Schema and Authorization Validation
        try:
            dto = HumanResolutionEventDTO.model_validate(payload_dict)
            if self.authorizer is not None:
                self.authorizer.authorize_event(dto.to_domain())
        except (
            ValidationError,
            EventValidationError,
            UnsupportedEventVersionError,
            UnreviewedAgentDecisionError,
            InvalidDescriptionError,
            DomainValidationError,
            UnauthorizedProducerError,
            UnauthorizedReviewerError,
        ) as val_err:
            self.metrics.record_invalid()
            logger.error(
                "Invalid event schema or authorization failure: %s (event_id=%s)",
                val_err,
                payload_dict.get("event_id"),
            )
            return await self._send_to_dlq(
                raw_payload=raw_value,
                error_type=type(val_err).__name__,
                error_message=str(val_err),
                metadata={
                    "event_id": payload_dict.get("event_id"),
                    "tenant_id": str(payload_dict.get("tenant_id", "")),
                    "partition": getattr(msg, "partition", None),
                    "offset": getattr(msg, "offset", None),
                },
            )

        # 3. Process Event with Bounded Retries
        attempts = 0
        backoff = self.retry_backoff_base

        while attempts < self.max_retries:
            attempts += 1
            try:
                async with self.session_factory() as session:
                    repo = PgHumanFeedbackRepository(session)
                    use_case = IngestHumanFeedbackUseCase(
                        repository=repo,
                        embedding_provider=self.embedding_provider,
                        authorizer=self.authorizer,
                    )
                    result = await use_case.execute(dto)
                    await session.commit()

                # Record metrics and log structured outcome
                latency_ms = (time.monotonic() - start_time) * 1000.0

                if result.outcome == HumanFeedbackIngestionOutcome.PROCESSED:
                    self.metrics.record_processed(
                        latency_ms=latency_ms,
                        reference_created=result.reference_created,
                    )
                elif (
                    result.outcome == HumanFeedbackIngestionOutcome.IDEMPOTENT_DUPLICATE
                ):
                    self.metrics.record_duplicate()
                    self.metrics.record_processed(
                        latency_ms=latency_ms,
                        reference_created=False,
                    )
                elif result.outcome == HumanFeedbackIngestionOutcome.CONFLICT:
                    self.metrics.record_conflict()
                    self.metrics.record_processed(
                        latency_ms=latency_ms,
                        reference_created=False,
                    )

                logger.info(
                    "Processed feedback event: event_id=%s, outcome=%s, latency_ms=%.2f",
                    dto.event_id,
                    result.outcome.value,
                    latency_ms,
                )
                return True

            except (UnauthorizedProducerError, UnauthorizedReviewerError) as auth_err:
                self.metrics.record_invalid()
                logger.error(
                    "Authorization failure during event processing: %s (event_id=%s)",
                    auth_err,
                    dto.event_id,
                )
                return await self._send_to_dlq(
                    raw_payload=raw_value,
                    error_type=type(auth_err).__name__,
                    error_message=str(auth_err),
                    metadata={"event_id": dto.event_id},
                )
            except Exception as proc_err:  # noqa: BLE001
                is_embedding = "embedding" in str(proc_err).lower()
                self.metrics.record_failure(is_embedding=is_embedding)

                if attempts < self.max_retries:
                    logger.warning(
                        "Transient failure processing event_id=%s (attempt %d/%d): %s. Retrying in %.2fs...",
                        dto.event_id,
                        attempts,
                        self.max_retries,
                        proc_err,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(
                        "Max retries (%d) exhausted for event_id=%s: %s. Routing to DLQ.",
                        self.max_retries,
                        dto.event_id,
                        proc_err,
                    )
                    return await self._send_to_dlq(
                        raw_payload=raw_value,
                        error_type="MAX_RETRIES_EXHAUSTED",
                        error_message=str(proc_err),
                        metadata={"event_id": dto.event_id, "attempts": attempts},
                    )

        return False

    async def _commit_message_offset(self, msg: Any) -> bool:
        """Manually commit offset for a processed message.

        Returns True if offset commit succeeded, False otherwise.
        """
        if self._consumer is not None and hasattr(self._consumer, "commit"):
            try:
                tp = TopicPartition(msg.topic, msg.partition)
                await self._consumer.commit({tp: OffsetAndMetadata(msg.offset + 1, "")})
                return True
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Failed to commit offset %s on topic %s partition %s: %s",
                    getattr(msg, "offset", None),
                    getattr(msg, "topic", None),
                    getattr(msg, "partition", None),
                    e,
                )
                return False
        return True

    async def _consume_loop(self) -> None:
        """Main polling and consumption loop."""
        logger.info("Entering consumer poll loop for topic %s...", self.topic)
        while self._is_running:
            try:
                if self._consumer is None:
                    break

                # aiokafka async iteration or getone
                msg = await self._consumer.getone()
                should_commit = await self.process_single_message(msg)
                if should_commit:
                    await self._commit_message_offset(msg)

            except asyncio.CancelledError:
                logger.info("Consumer loop received cancellation signal.")
                break
            except Exception:
                if not self._is_running:
                    break
                logger.exception("Unexpected error in consumer loop")
                await asyncio.sleep(1.0)
