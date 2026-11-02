"""ARDA Reliability Service entrypoint."""

from __future__ import annotations

import asyncio
import logging
import signal

from arda_reliability_service.config import get_settings
from arda_reliability_service.infrastructure.grpc.server import create_grpc_server

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Run the gRPC server with signal handling for graceful shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    server, dm = create_grpc_server(settings)
    await server.start()

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
