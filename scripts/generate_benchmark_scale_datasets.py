"""Generate reproducible benchmark scale datasets (1k, 10k, multi-agent) for HNSW evaluation.

Generates realistic financial dispute cases based on the audited seed dataset policy rules.
Encodes embeddings with BAAI/bge-small-en-v1.5 on CUDA/CPU.
Ingests records into PostgreSQL with dedicated benchmark tenant/agent IDs, keeping audited seed data strictly isolated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arda_reliability_service.config import get_settings
from arda_reliability_service.infrastructure.database.connection import DatabaseManager
from arda_reliability_service.infrastructure.database.models import (
    ReferenceExampleModel,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)

# Benchmark Scope Constants (Distinct from synthetic seed data)
BENCHMARK_TENANT_ID = UUID("00000000-0000-0000-0000-0000000000bb")
BENCHMARK_AGENT_1K = UUID("00000000-0000-0000-0000-000000000001")
BENCHMARK_AGENT_10K = UUID("00000000-0000-0000-0000-000000000002")
BENCHMARK_AGENT_FILTER_A = UUID(
    "00000000-0000-0000-0000-00000000000a"
)  # Sparse: 50 rows
BENCHMARK_AGENT_FILTER_B = UUID(
    "00000000-0000-0000-0000-00000000000b"
)  # Medium: 500 rows
BENCHMARK_AGENT_FILTER_C = UUID(
    "00000000-0000-0000-0000-00000000000c"
)  # Large: 5000 rows
BENCHMARK_TENANT_ALT = UUID("00000000-0000-0000-0000-0000000000cc")
BENCHMARK_AGENT_ALT = UUID("00000000-0000-0000-0000-000000000001")

DATASETS_OUT_DIR = REPO_ROOT / "research" / "datasets" / "benchmark_scales"

# Seed data templates and variations
MERCHANTS = [
    "Amazon Web Services",
    "Microsoft Cloud",
    "Apple Store",
    "Google Play",
    "Netflix",
    "Spotify Premium",
    "Uber Rides",
    "Lyft Inc",
    "DoorDash",
    "Delta Air Lines",
    "Hilton Hotels",
    "Marriott International",
    "Target Corp",
    "Walmart Supercenter",
    "Home Depot",
    "Best Buy Electronics",
    "Steam Games",
    "PlayStation Network",
    "Shell Oil",
    "Chevron Gas",
    "Starbucks Coffee",
]

CATEGORIES = [
    "unauthorized_transaction",
    "duplicate_billing",
    "non_delivery_merchandise",
    "subscription_canceled",
    "credit_not_processed",
    "incorrect_amount",
]

TEMPLATES_APPROVE = [
    (
        (
            "Cardholder reports unauthorized debit of ${amount:.2f} at {merchant} while traveling in {city}. "
            "Physical card was in cardholder possession; transaction occurred via magnetic stripe fallback in {foreign_country}. "
            "Dispute filed {filing_days} days post-transaction (within 60-day statutory window). Merchant failed to supply compelling rebuttal."
        ),
        "Dispute filing ratio is {filing_ratio:.2f} for ${amount:.2f} at {merchant} in {city}, 0 prior disputes on account, magnetic stripe fallback in {foreign_country}.",
    ),
    (
        (
            "Cardholder was billed duplicate amounts of ${amount:.2f} by {merchant} within {min_diff} minutes of initial purchase. "
            "Merchant terminal registered duplicate processing error. Cardholder attempted merchant resolution; merchant response window elapsed without refund."
        ),
        "Duplicate billing interval is {min_diff} minutes for ${amount:.2f} at {merchant}, identical authorization code, merchant refund window elapsed.",
    ),
    (
        (
            "Cardholder ordered electronic goods valued at ${amount:.2f} from {merchant} with guaranteed delivery date. "
            "Tracking number indicates shipment marked lost in transit by carrier. Cardholder filed claim {filing_days} days after estimated delivery with proof of non-receipt."
        ),
        "Carrier delivery status is lost for ${amount:.2f} shipment from {merchant}, claim delay ratio is {filing_ratio:.2f}, merchant re-shipment window of 14 days elapsed.",
    ),
    (
        (
            "Cardholder canceled recurring monthly membership with {merchant} on {cancel_date} with written confirmation. "
            "Merchant continued recurring monthly billing of ${amount:.2f} for subsequent billing cycle. Cardholder provided timestamped cancellation confirmation."
        ),
        "Subscription cancellation confirmed on {cancel_date} for {merchant}, recurring debit ratio is {filing_ratio:.2f} for ${amount:.2f}, merchant response ratio 0.0.",
    ),
]

TEMPLATES_REJECT = [
    (
        (
            "Cardholder disputes ${amount:.2f} charge at {merchant} as unauthorized. "
            "Transaction was authenticated with 3D Secure EMV chip and PIN verification matching cardholder mobile device fingerprint and known home IP address in {city}. "
            "Merchant provided complete transaction log showing biometric passkey confirmation."
        ),
        "Authentication risk ratio is 0.0 for ${amount:.2f} at {merchant} in {city}, 3D Secure biometric confirmation verified with customer mobile device.",
    ),
    (
        (
            "Cardholder filed dispute for ${amount:.2f} against {merchant} {filing_days} days after statement delivery date. "
            "Filing date exceeds mandatory 60-day contractual dispute window under cardholder agreement. No extenuating medical or military circumstances established."
        ),
        "Dispute filing ratio is {filing_ratio:.2f} (>1.0) for ${amount:.2f} at {merchant}, statutory window of 60 days exceeded by {filing_days_over} days.",
    ),
    (
        (
            "Cardholder claims ${amount:.2f} merchandise from {merchant} was not delivered to {city} address. "
            "Merchant provided carrier proof of delivery with signed delivery receipt, photo matching customer residential doorway, and matching GPS delivery coordinates."
        ),
        "Merchant compelling evidence ratio is 1.0 for ${amount:.2f} purchase from {merchant}, carrier signature and GPS coordinates match {city} address.",
    ),
    (
        (
            "Cardholder claims cancelation of service with {merchant} prior to ${amount:.2f} billing on {cancel_date}. "
            "Merchant contract stipulates annual commitment with 30-day advance notice requirement. Cardholder cancellation request submitted 2 days prior to auto-renewal date."
        ),
        "Contractual notice ratio is 0.07 (2/30 days) for ${amount:.2f} at {merchant}, auto-renewal terms accepted on {cancel_date}, merchant contractually compliant.",
    ),
]


def generate_synthetic_record(
    idx: int,
    tenant_id: UUID,
    agent_id: UUID,
    seed: int,
) -> dict:
    """Generate a single structurally consistent financial dispute reference example."""
    rng = random.Random(seed + idx)

    label = "APPROVE" if idx % 2 == 0 else "REJECT"
    merchant = rng.choice(MERCHANTS)
    amount = round(rng.uniform(15.0, 2450.0), 2)
    filing_days = rng.randint(2, 45) if label == "APPROVE" else rng.randint(65, 120)
    filing_ratio = round(filing_days / 60.0, 2)
    filing_days_over = max(0, filing_days - 60)
    min_diff = rng.randint(1, 12)
    city = rng.choice(["Chicago", "Denver", "Seattle", "Austin", "Boston"])
    foreign_country = rng.choice(["Brazil", "Romania", "Thailand", "Nigeria"])
    cancel_date = (datetime.now(UTC) - timedelta(days=rng.randint(35, 90))).strftime(
        "%Y-%m-%d"
    )

    if label == "APPROVE":
        tmpl_desc, tmpl_ratio = rng.choice(TEMPLATES_APPROVE)
    else:
        tmpl_desc, tmpl_ratio = rng.choice(TEMPLATES_REJECT)

    case_description = tmpl_desc.format(
        merchant=merchant,
        amount=amount,
        filing_days=filing_days,
        filing_days_over=filing_days_over,
        filing_ratio=filing_ratio,
        min_diff=min_diff,
        city=city,
        foreign_country=foreign_country,
        cancel_date=cancel_date,
    )

    ratio_description = tmpl_ratio.format(
        merchant=merchant,
        amount=amount,
        filing_days=filing_days,
        filing_days_over=filing_days_over,
        filing_ratio=filing_ratio,
        min_diff=min_diff,
        city=city,
        foreign_country=foreign_country,
        cancel_date=cancel_date,
    )

    return {
        "example_id": f"bench-scale-{idx:06d}",
        "tenant_id": str(tenant_id),
        "agent_id": str(agent_id),
        "case_description": case_description,
        "label": label,
        "source": "benchmark",
        "status": "active",
        "metadata": {
            "ratio_verbalized_description": ratio_description,
            "dispute_amount": amount,
            "merchant_name": merchant,
            "benchmark_index": idx,
        },
    }


def generate_records_batch(
    count: int,
    tenant_id: UUID,
    agent_id: UUID,
    seed_offset: int = 1000,
) -> list[dict]:
    """Generate a reproducible list of benchmark records."""
    records = []
    for i in range(count):
        rec = generate_synthetic_record(
            idx=i + 1,
            tenant_id=tenant_id,
            agent_id=agent_id,
            seed=seed_offset + i,
        )
        records.append(rec)
    return records


async def ingest_benchmark_records(
    db_manager: DatabaseManager,
    records: list[dict],
    embeddings: list[list[float]],
    batch_size: int = 1000,
) -> int:
    """Batch-ingest benchmark records into PostgreSQL."""
    assert len(records) == len(embeddings), "Record and embedding count mismatch!"

    db_records = []
    for r, emb in zip(records, embeddings):
        db_records.append(
            {
                "example_id": r["example_id"],
                "tenant_id": UUID(r["tenant_id"]),
                "agent_id": UUID(r["agent_id"]),
                "case_description": r["case_description"],
                "label": r["label"],
                "source": r["source"],
                "status": r["status"],
                "metadata": r["metadata"],
                "embedding": emb,
                "model_name": "BAAI/bge-small-en-v1.5",
                "model_version": "v1.5",
            }
        )

    total_inserted = 0
    t0 = time.perf_counter()

    async with db_manager.session() as s:
        for offset in range(0, len(db_records), batch_size):
            chunk = db_records[offset : offset + batch_size]
            stmt = insert(ReferenceExampleModel).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["tenant_id", "agent_id", "example_id"],
                set_={
                    "case_description": stmt.excluded.case_description,
                    "label": stmt.excluded.label,
                    "source": stmt.excluded.source,
                    "status": stmt.excluded.status,
                    "metadata": stmt.excluded.metadata,
                    "embedding": stmt.excluded.embedding,
                    "updated_at": func_now_utc(),
                },
            )
            await s.execute(stmt)
            total_inserted += len(chunk)
            print(f"  Inserted {total_inserted}/{len(db_records)} records...")

    elapsed = time.perf_counter() - t0
    print(
        f"Database ingestion of {total_inserted} records completed in {elapsed:.2f}s."
    )
    return total_inserted


def func_now_utc():
    return datetime.now(UTC)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate and ingest benchmark scale datasets."
    )
    parser.add_argument(
        "--scale",
        choices=["1k", "10k", "all"],
        default="all",
        help="Dataset scale to generate",
    )
    args = parser.parse_args()

    settings = get_settings()
    dm = DatabaseManager(settings)

    DATASETS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing embedding adapter (CUDA preferred)...")
    adapter = SentenceTransformerEmbeddingAdapter(
        model_name="BAAI/bge-small-en-v1.5",
        expected_dimension=384,
        device="cuda"
        if settings.embedding_device == "auto"
        else settings.embedding_device,
        batch_size=256,
    )

    # 1. 1K Benchmark Dataset
    if args.scale in ["1k", "all"]:
        print("\n--- Generating 1,000 Reference Benchmark Dataset ---")
        records_1k = generate_records_batch(
            1000, BENCHMARK_TENANT_ID, BENCHMARK_AGENT_1K, seed_offset=1000
        )
        json_path_1k = DATASETS_OUT_DIR / "benchmark_examples_1k.json"
        with open(json_path_1k, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump({"examples": records_1k}, f, indent=2)
        print(f"Saved {len(records_1k)} records to {json_path_1k}")

        texts_1k = [r["metadata"]["ratio_verbalized_description"] for r in records_1k]
        t0 = time.perf_counter()
        embs_1k = adapter.generate_embeddings_batch(texts_1k, is_query=False)
        print(f"Encoded 1,000 vectors in {time.perf_counter() - t0:.2f}s.")

        await ingest_benchmark_records(dm, records_1k, embs_1k)

    # 2. 10K Benchmark Dataset
    if args.scale in ["10k", "all"]:
        print("\n--- Generating 10,000 Reference Benchmark Dataset ---")
        records_10k = generate_records_batch(
            10000, BENCHMARK_TENANT_ID, BENCHMARK_AGENT_10K, seed_offset=20000
        )
        json_path_10k = DATASETS_OUT_DIR / "benchmark_examples_10k.json"
        with open(json_path_10k, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump({"examples": records_10k}, f, indent=2)
        print(f"Saved {len(records_10k)} records to {json_path_10k}")

        texts_10k = [r["metadata"]["ratio_verbalized_description"] for r in records_10k]
        t0 = time.perf_counter()
        embs_10k = adapter.generate_embeddings_batch(texts_10k, is_query=False)
        print(f"Encoded 10,000 vectors in {time.perf_counter() - t0:.2f}s.")

        await ingest_benchmark_records(dm, records_10k, embs_10k)

    # 3. Multi-Agent & Multi-Tenant Partitioning Distribution
    if args.scale == "all":
        print("\n--- Generating Multi-Agent & Multi-Tenant Partition Distribution ---")
        # Agent Filter A (50 records - sparse)
        rec_fa = generate_records_batch(
            50, BENCHMARK_TENANT_ID, BENCHMARK_AGENT_FILTER_A, seed_offset=50000
        )
        embs_fa = adapter.generate_embeddings_batch(
            [r["metadata"]["ratio_verbalized_description"] for r in rec_fa],
            is_query=False,
        )
        await ingest_benchmark_records(dm, rec_fa, embs_fa)

        # Agent Filter B (500 records - medium)
        rec_fb = generate_records_batch(
            500, BENCHMARK_TENANT_ID, BENCHMARK_AGENT_FILTER_B, seed_offset=60000
        )
        embs_fb = adapter.generate_embeddings_batch(
            [r["metadata"]["ratio_verbalized_description"] for r in rec_fb],
            is_query=False,
        )
        await ingest_benchmark_records(dm, rec_fb, embs_fb)

        # Multi-Tenant: Tenant Alt (1,000 records)
        rec_alt = generate_records_batch(
            1000, BENCHMARK_TENANT_ALT, BENCHMARK_AGENT_ALT, seed_offset=70000
        )
        embs_alt = adapter.generate_embeddings_batch(
            [r["metadata"]["ratio_verbalized_description"] for r in rec_alt],
            is_query=False,
        )
        await ingest_benchmark_records(dm, rec_alt, embs_alt)

    # Verify counts in database
    async with dm.session() as s:
        counts = await s.execute(
            text(
                """
            SELECT tenant_id, agent_id, count(*), source
            FROM reference_examples
            GROUP BY tenant_id, agent_id, source
            ORDER BY count(*) ASC;
            """
            )
        )
        print("\nDatabase Scope Verification:")
        for r in counts.all():
            print(f"  Tenant: {r[0]} | Agent: {r[1]} | Count: {r[2]} | Source: {r[3]}")

    await dm.dispose()
    print("\nBenchmark scale generation and ingestion completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
