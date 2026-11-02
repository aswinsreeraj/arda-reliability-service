"""Unit and lifecycle tests for ReliabilityGrpcServer and factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arda_reliability_service.config import Settings
from arda_reliability_service.infrastructure.grpc.handler import ReliabilityGateHandler
from arda_reliability_service.infrastructure.grpc.server import (
    ReliabilityGrpcServer,
    create_grpc_server,
)


@pytest.fixture
def mock_handler() -> ReliabilityGateHandler:
    """Fixture providing a handler with mocked use case."""
    mock_use_case = MagicMock()
    return ReliabilityGateHandler(evaluate_use_case=mock_use_case)


class TestGrpcServerLifecycle:
    """Test suite for server startup, shutdown, port binding, and lifecycle."""

    @pytest.mark.asyncio
    async def test_server_start_and_stop_dynamic_port(
        self, mock_handler: ReliabilityGateHandler
    ) -> None:
        """Verify server binds to dynamic port 0 and gracefully stops."""
        server = ReliabilityGrpcServer(
            handler=mock_handler,
            host="127.0.0.1",
            port=0,
            grace_period_seconds=0.1,
        )

        assert server.bound_port is None

        bound_port = await server.start()
        try:
            assert bound_port > 0
            assert server.bound_port == bound_port
        finally:
            await server.stop(grace=0.05)

        assert server._server is None

    @pytest.mark.asyncio
    async def test_server_stop_idempotence(
        self, mock_handler: ReliabilityGateHandler
    ) -> None:
        """Verify calling stop multiple times or before start does not raise."""
        server = ReliabilityGrpcServer(
            handler=mock_handler,
            host="127.0.0.1",
            port=0,
        )

        # Stop before start
        await server.stop(grace=0.0)

        # Start and stop twice
        await server.start()
        await server.stop(grace=0.05)
        await server.stop(grace=0.05)

        assert server._server is None

    @pytest.mark.asyncio
    async def test_server_configuration_options(
        self, mock_handler: ReliabilityGateHandler
    ) -> None:
        """Verify custom worker and message size settings."""
        server = ReliabilityGrpcServer(
            handler=mock_handler,
            host="127.0.0.1",
            port=0,
            max_workers=4,
            max_message_length_mb=8,
            grace_period_seconds=0.5,
        )

        assert server.max_workers == 4
        assert server.max_message_length == 8 * 1024 * 1024
        assert server.grace_period_seconds == 0.5

        port = await server.start()
        assert port > 0
        await server.stop(grace=0.05)

    def test_create_grpc_server_factory(self) -> None:
        """Verify create_grpc_server factory correctly wires dependencies with settings."""
        mock_dm = MagicMock()
        mock_embedding = MagicMock()
        settings = Settings(
            grpc_host="127.0.0.1",
            grpc_port=59090,
            grpc_max_workers=5,
            grpc_max_message_length_mb=32,
            grpc_grace_period_seconds=2.0,
        )

        server, dm = create_grpc_server(
            settings=settings,
            db_manager=mock_dm,
            embedding_adapter=mock_embedding,
            port_override=59091,
        )

        assert dm is mock_dm
        assert server.host == "127.0.0.1"
        assert server.port == 59091
        assert server.max_workers == 5
        assert server.max_message_length == 32 * 1024 * 1024
        assert server.grace_period_seconds == 2.0
        assert server.handler.db_manager is mock_dm
        assert server.handler.embedding_adapter is mock_embedding
