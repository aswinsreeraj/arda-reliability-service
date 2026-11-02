"""CLI script to ingest reference examples into PostgreSQL pgvector storage.

Reads reference examples from research/datasets/synthetic_seed_examples.json (or specified path),
generates 384-dimensional dense embeddings using BAAI/bge-small-en-v1.5 (from ratio_verbalized_description),
and idempotently persists them to the reference_examples table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arda_reliability_service.application.use_cases.ingest_reference_examples import (
    IngestReferenceExamplesUseCase,
)
from arda_reliability_service.config import get_settings
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.pgvector_repository import (
    PgVectorReferenceExampleRepository,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)

DEFAULT_DATASET = REPO_ROOT / "research" / "datasets" / "synthetic_seed_examples.json"


async def run_ingestion(
    dataset_path: Path,
    batch_size: int = 32,
) -> int:
    """Execute ingestion workflow."""
    settings = get_settings()

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:  # noqa: ASYNC230
        data = json.load(f)

    examples = (
        data["examples"] if isinstance(data, dict) and "examples" in data else data
    )
    total_count = len(examples)
    print(f"Loaded {total_count} examples from dataset.")

    # Initialize infrastructure
    print(
        f"Initializing embedding adapter: {settings.embedding_model_name} (dim={settings.embedding_dimension})..."
    )
    embedding_adapter = SentenceTransformerEmbeddingAdapter(
        model_name=settings.embedding_model_name,
        expected_dimension=settings.embedding_dimension,
        device=settings.embedding_device,
        batch_size=batch_size,
    )

    db_manager = DatabaseManager(settings)
    t0 = time.perf_counter()

    async with db_manager.session() as session:
        repo = PgVectorReferenceExampleRepository(session)
        use_case = IngestReferenceExamplesUseCase(
            repository=repo,
            embedding_provider=embedding_adapter,
        )

        print(f"Ingesting {total_count} reference examples into PostgreSQL...")
        ingested_ids = await use_case.execute(examples, on_conflict_do_update=True)

        # Verify count in database
        first_ex = examples[0]
        tenant_id = first_ex["tenant_id"]
        agent_id = first_ex["agent_id"]
        from uuid import UUID

        t_uuid = tenant_id if isinstance(tenant_id, UUID) else UUID(str(tenant_id))
        a_uuid = agent_id if isinstance(agent_id, UUID) else UUID(str(agent_id))

        stored_count = await repo.count(t_uuid, a_uuid, status="active")

    await db_manager.dispose()
    elapsed = time.perf_counter() - t0

    print(
        f"[OK] Successfully ingested {len(ingested_ids)} reference examples in {elapsed:.2f}s."
    )
    print(
        f"[OK] Total active reference examples stored for agent {agent_id}: {stored_count}"
    )
    return len(ingested_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest reference examples into pgvector"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to JSON dataset file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    args = parser.parse_args()

    asyncio.run(run_ingestion(dataset_path=args.dataset, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
