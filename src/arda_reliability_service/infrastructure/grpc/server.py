"""Configurable async gRPC server for ARDA Reliability Service."""

from __future__ import annotations

import logging
from concurrent import futures

import grpc
from arda_reliability_service.config import Settings, get_settings
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)
from arda_reliability_service.infrastructure.grpc.generated import (
    reliability_pb2_grpc,
)
from arda_reliability_service.infrastructure.grpc.handler import ReliabilityGateHandler

logger = logging.getLogger(__name__)


class ReliabilityGrpcServer:
    """Async gRPC server wrapper managing lifecycle, port binding, and graceful shutdown."""

    def __init__(
        self,
        handler: ReliabilityGateHandler,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
        max_message_length_mb: int = 16,
        grace_period_seconds: float = 5.0,
    ) -> None:
        self.handler = handler
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.max_message_length = max_message_length_mb * 1024 * 1024
        self.grace_period_seconds = grace_period_seconds
        self._server: grpc.aio.Server | None = None
        self._bound_port: int | None = None

    @property
    def bound_port(self) -> int | None:
        """Return the actual TCP port bound by the server (useful for dynamic port :0 in tests)."""
        return self._bound_port

    async def start(self) -> int:
        """Initialize, bind, and start the async gRPC server."""
        options = [
            ("grpc.max_receive_message_length", self.max_message_length),
            ("grpc.max_send_message_length", self.max_message_length),
            ("grpc.so_reuseport", 1),
        ]

        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=options,
        )

        # Register Gate servicer
        reliability_pb2_grpc.add_ReliabilityServiceServicer_to_server(
            self.handler, self._server
        )

        bind_addr = f"{self.host}:{self.port}"
        self._bound_port = self._server.add_insecure_port(bind_addr)
        if self._bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC server on {bind_addr}")

        logger.info(
            "Starting ReliabilityService gRPC server on %s (port %d)",
            self.host,
            self._bound_port,
        )
        await self._server.start()
        logger.info("ReliabilityService gRPC server successfully running")
        return self._bound_port

    async def stop(self, grace: float | None = None) -> None:
        """Gracefully stop the server allowing in-flight requests to complete."""
        if self._server is None:
            return

        effective_grace = grace if grace is not None else self.grace_period_seconds
        logger.info(
            "Stopping ReliabilityService gRPC server (grace=%.1fs)...", effective_grace
        )
        await self._server.stop(effective_grace)
        self._server = None
        logger.info("ReliabilityService gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """Block until the gRPC server terminates."""
        if self._server is not None:
            await self._server.wait_for_termination()


def create_grpc_server(
    settings: Settings | None = None,
    db_manager: DatabaseManager | None = None,
    embedding_adapter: SentenceTransformerEmbeddingAdapter | None = None,
    port_override: int | None = None,
) -> tuple[ReliabilityGrpcServer, DatabaseManager]:
    """Factory creating fully wired production gRPC server and database manager."""
    cfg = settings or get_settings()
    dm = db_manager or DatabaseManager(settings=cfg)

    adapter = embedding_adapter or SentenceTransformerEmbeddingAdapter(
        model_name=cfg.embedding_model_name,
        device=cfg.embedding_device,
        expected_dimension=cfg.embedding_dimension,
        batch_size=cfg.embedding_batch_size,
    )

    scoring_config = cfg.get_scoring_config()

    handler = ReliabilityGateHandler(
        db_manager=dm,
        embedding_adapter=adapter,
        config=scoring_config,
    )

    server = ReliabilityGrpcServer(
        handler=handler,
        host=cfg.grpc_host,
        port=port_override if port_override is not None else cfg.grpc_port,
        max_workers=cfg.grpc_max_workers,
        max_message_length_mb=cfg.grpc_max_message_length_mb,
        grace_period_seconds=cfg.grpc_grace_period_seconds,
    )

    return server, dm
