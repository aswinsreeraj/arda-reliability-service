"""Thread-safe observability metrics for human feedback ingestion consumer."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass


@dataclass
class MetricsSnapshot:
    """Read-only snapshot of current consumer metrics."""

    events_received: int = 0
    events_processed: int = 0
    duplicate_events: int = 0
    invalid_events: int = 0
    processing_failures: int = 0
    embedding_failures: int = 0
    reference_examples_created: int = 0
    conflicting_resolutions: int = 0
    dead_lettered_events: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


class ConsumerMetrics:
    """Observability metrics tracker for the Redpanda human feedback consumer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self.events_received = 0
        self.events_processed = 0
        self.duplicate_events = 0
        self.invalid_events = 0
        self.processing_failures = 0
        self.embedding_failures = 0
        self.reference_examples_created = 0
        self.conflicting_resolutions = 0
        self.dead_lettered_events = 0
        self._total_latency_ms = 0.0
        self._latency_samples = 0

    def record_received(self) -> None:
        with self._lock:
            self.events_received += 1

    def record_processed(
        self, latency_ms: float, reference_created: bool = False
    ) -> None:
        with self._lock:
            self.events_processed += 1
            if reference_created:
                self.reference_examples_created += 1
            self._total_latency_ms += latency_ms
            self._latency_samples += 1

    def record_duplicate(self) -> None:
        with self._lock:
            self.duplicate_events += 1

    def record_conflict(self) -> None:
        with self._lock:
            self.conflicting_resolutions += 1

    def record_invalid(self) -> None:
        with self._lock:
            self.invalid_events += 1

    def record_failure(self, is_embedding: bool = False) -> None:
        with self._lock:
            self.processing_failures += 1
            if is_embedding:
                self.embedding_failures += 1

    def record_dead_letter(self) -> None:
        with self._lock:
            self.dead_lettered_events += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            avg_lat = (
                (self._total_latency_ms / self._latency_samples)
                if self._latency_samples > 0
                else 0.0
            )
            uptime = time.monotonic() - self._start_time
            return MetricsSnapshot(
                events_received=self.events_received,
                events_processed=self.events_processed,
                duplicate_events=self.duplicate_events,
                invalid_events=self.invalid_events,
                processing_failures=self.processing_failures,
                embedding_failures=self.embedding_failures,
                reference_examples_created=self.reference_examples_created,
                conflicting_resolutions=self.conflicting_resolutions,
                dead_lettered_events=self.dead_lettered_events,
                avg_latency_ms=round(avg_lat, 2),
                uptime_seconds=round(uptime, 2),
            )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self.snapshot())

    def reset(self) -> None:
        with self._lock:
            self._start_time = time.monotonic()
            self.events_received = 0
            self.events_processed = 0
            self.duplicate_events = 0
            self.invalid_events = 0
            self.processing_failures = 0
            self.embedding_failures = 0
            self.reference_examples_created = 0
            self.conflicting_resolutions = 0
            self.dead_lettered_events = 0
            self._total_latency_ms = 0.0
            self._latency_samples = 0
