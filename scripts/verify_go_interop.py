#!/usr/bin/env python3
"""Automated verification script demonstrating Go client interoperability with Python gRPC server."""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

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
from arda_reliability_service.infrastructure.grpc.handler import ReliabilityGateHandler
from arda_reliability_service.infrastructure.grpc.server import ReliabilityGrpcServer


async def run_interop_verification() -> int:
    print("=" * 70)
    print("ARDA RELIABILITY SERVICE: GO CLIENT INTEROPERABILITY VERIFICATION")
    print("=" * 70)

    # 1. Initialize components
    print("\n[1/4] Initializing Database & SentenceTransformers embedding adapter...")
    dm = DatabaseManager()
    adapter = SentenceTransformerEmbeddingAdapter()
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
        db_manager=dm,
        embedding_adapter=adapter,
        config=scoring_config,
    )

    # 2. Start server on dynamic port
    print("\n[2/4] Starting ReliabilityService async gRPC server on 127.0.0.1:0...")
    server = ReliabilityGrpcServer(
        handler=handler,
        host="127.0.0.1",
        port=0,
        grace_period_seconds=1.0,
    )
    port = await server.start()
    addr = f"127.0.0.1:{port}"
    print(f"       -> Bound successfully to {addr}")

    # 3. Seed isolated test tenant & agent
    tenant_id = uuid4()
    agent_id = uuid4()
    print(f"\n[3/4] Seeding reference examples for Tenant {tenant_id}...")
    async with dm.session() as session:
        repo = PgVectorReferenceExampleRepository(session)
        ingest_uc = IngestReferenceExamplesUseCase(
            repository=repo, embedding_provider=adapter
        )
        items = [
            IngestItemDTO(
                example_id=f"GO-TEST-{i}",
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
        print("       -> Seeded 4 APPROVE reference examples with embeddings.")

    # 4. Run Go tests
    print(f"\n[4/4] Executing Go client tests against {addr}...")
    go_env = os.environ.copy()
    go_env["ARDA_RELIABILITY_ADDR"] = addr
    go_env["ARDA_SEEDED_TENANT_ID"] = str(tenant_id)
    go_env["ARDA_SEEDED_AGENT_ID"] = str(agent_id)

    go_dir = "/home/aswin/second-project/arda"
    cmd = ["go", "test", "-v", "./gen/go/arda/v1", "-run", "TestReliabilityGateInterop"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=go_dir,
            env=go_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out_str = stdout.decode("utf-8")
        err_str = stderr.decode("utf-8")
        print("Go test output:")
        print(out_str)
        if err_str:
            print("Go test stderr:")
            print(err_str)

        if proc.returncode != 0:
            print(f"FAILED: Go test exited with code {proc.returncode}")
            return proc.returncode if proc.returncode is not None else 1

        print("SUCCESS: Go client verified against Python gRPC server!")
        return 0

    finally:
        # Cleanup
        print("\nCleaning up seeded database records and shutting down gRPC server...")
        try:
            async with dm.session() as session:
                repo = PgVectorReferenceExampleRepository(session)
                await repo.delete_all_for_agent(tenant_id, agent_id)
        except Exception as e:  # noqa: BLE001
            print(f"Warning during DB cleanup: {e}")

        await server.stop(grace=0.1)
        await dm.dispose()
        print("Done.")


def main() -> None:
    code = asyncio.run(run_interop_verification())
    sys.exit(code)


if __name__ == "__main__":
    main()
