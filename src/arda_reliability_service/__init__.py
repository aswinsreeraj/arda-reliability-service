from __future__ import annotations

import asyncio
import logging
import signal

from arda_reliability_service.config import get_settings
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from arda_reliability_service.infrastructure.grpc.server import create_grpc_server
from arda_reliability_service.infrastructure.messaging.consumer import (
    RedpandaFeedbackConsumer,
)

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Run the gRPC server and Redpanda feedback consumer with graceful shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    dm = DatabaseManager(settings=settings)
    embedder = SentenceTransformerEmbeddingAdapter(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        expected_dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )

    server, dm = create_grpc_server(
        settings=settings,
        db_manager=dm,
        embedding_adapter=embedder,
    )
    await server.start()

    consumer = RedpandaFeedbackConsumer(
        session_factory=dm.session_factory,
        embedding_provider=embedder,
        settings=settings,
    )
    consumer_started = False
    try:
        await consumer.start()
        consumer_started = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redpanda feedback consumer failed to start (gRPC server will continue running): %s",
            exc,
        )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Signal handling on Windows / restricted threads
            pass

    try:
        await stop_event.wait()
    finally:
        if consumer_started:
            await consumer.stop()
        await server.stop(grace=settings.grpc_grace_period_seconds)
        await dm.dispose()


def main() -> None:
    """CLI entrypoint configured in pyproject.toml."""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Service shutdown completed.")
    except SystemExit:
        logger.info("Service shutdown completed.")


if __name__ == "__main__":
    main()
