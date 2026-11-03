"""Trusted producer HMAC-SHA256 authenticator and reviewer authorization implementation."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arda_reliability_service.config import Settings
    from arda_reliability_service.infrastructure.messaging.events import (
        HumanResolutionEventDTO,
    )

from arda_reliability_service.application.ports.event_authorizer import (
    EventAuthorizerPort,
)
from arda_reliability_service.domain.entities import HumanResolutionEvent
from arda_reliability_service.domain.exceptions import (
    ExpiredEventError,
    InvalidEventTimestampError,
    InvalidSignatureError,
    UnauthorizedProducerError,
    UnauthorizedReviewerError,
)

logger = logging.getLogger(__name__)

FORBIDDEN_REVIEWER_PATTERNS = re.compile(
    r"(?:anonymous|unauthorized|bot|system|automated|unknown|none|^$)",
    re.IGNORECASE,
)


def format_canonical_timestamp(ts: datetime | str) -> str:
    """Format timestamp into deterministic canonical RFC3339 UTC string with second precision."""
    if isinstance(ts, str):
        clean_str = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
    elif isinstance(ts, datetime):
        dt = ts
    else:
        raise TypeError(f"Invalid timestamp type: {type(ts)}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_canonical_payload(
    event: HumanResolutionEvent | HumanResolutionEventDTO | dict[str, Any],
) -> bytes:
    """Build deterministic UTF-8 JSON canonical payload for HMAC-SHA256 signing.

    Included Fields (strictly sorted lexicographically):
    1. agent_id: UUID string (lowercase)
    2. case_id: string (trimmed)
    3. event_id: string (trimmed)
    4. event_type: string (trimmed)
    5. event_version: string (trimmed)
    6. human_decision: string (trimmed)
    7. key_id: string (trimmed)
    8. original_case_description: string
    9. producer_id: string (trimmed)
    10. proposed_decision: string (trimmed)
    11. reviewer_id: string (trimmed)
    12. tenant_id: UUID string (lowercase)
    13. timestamp: canonical RFC3339 UTC string (YYYY-MM-DDTHH:MM:SSZ)

    Excluded Fields:
    - signature / producer_signature (cannot sign itself)
    - correlation_id (tracing metadata that can be altered by proxies/sidecars)
    - transport-level metadata (Kafka topic, partition, offset, broker timestamps)
    """
    if isinstance(event, dict):
        agent_id = str(event["agent_id"]).lower()
        case_id = str(event["case_id"]).strip()
        event_id = str(event["event_id"]).strip()
        event_type = str(
            event.get("event_type", "arda.reliability.human_resolution")
        ).strip()
        event_version = str(event.get("event_version", "v1")).strip()
        human_decision = str(event["human_decision"]).strip()
        key_id = str(event.get("key_id", "v1")).strip()
        original_case_description = str(event["original_case_description"])
        producer_id = str(event.get("producer_id", "arda.human-review-service")).strip()
        proposed_decision = str(event.get("proposed_decision", "APPROVE")).strip()
        reviewer_id = str(event["reviewer_id"]).strip()
        tenant_id = str(event["tenant_id"]).lower()
        ts_val = event.get("timestamp") or event.get("resolved_at")
        if not ts_val:
            raise ValueError(
                "Event must include 'timestamp' or 'resolved_at' for canonical signing"
            )
        timestamp = format_canonical_timestamp(ts_val)
    elif isinstance(event, HumanResolutionEvent):
        agent_id = str(event.agent_id).lower()
        case_id = event.case_id.strip()
        event_id = event.event_id.strip()
        event_type = event.event_type.strip()
        event_version = event.event_version.strip()
        human_decision = event.human_decision.strip()
        key_id = event.key_id.strip()
        original_case_description = event.original_case_description
        producer_id = event.producer_id.strip()
        proposed_decision = event.proposed_decision.strip()
        reviewer_id = event.reviewer_id.strip()
        tenant_id = str(event.tenant_id).lower()
        timestamp = format_canonical_timestamp(event.get_event_timestamp())
    elif hasattr(event, "to_domain"):
        dom = event.to_domain()
        return build_canonical_payload(dom)
    else:
        raise TypeError(f"Unsupported event type for canonical payload: {type(event)}")

    canonical_dict = {
        "agent_id": agent_id,
        "case_id": case_id,
        "event_id": event_id,
        "event_type": event_type,
        "event_version": event_version,
        "human_decision": human_decision,
        "key_id": key_id,
        "original_case_description": original_case_description,
        "producer_id": producer_id,
        "proposed_decision": proposed_decision,
        "reviewer_id": reviewer_id,
        "tenant_id": tenant_id,
        "timestamp": timestamp,
    }
    return json.dumps(
        canonical_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_hmac_signature(secret: str, canonical_payload: bytes) -> str:
    """Compute HMAC-SHA256 lowercase hex signature over canonical payload."""
    secret_bytes = secret.encode("utf-8")
    return hmac.new(secret_bytes, canonical_payload, hashlib.sha256).hexdigest()


def sign_event(
    event_dict: dict[str, Any],
    secret: str,
    key_id: str = "v1",
) -> dict[str, Any]:
    """Helper for producers and test fixtures to sign an event payload with HMAC-SHA256."""
    ev = dict(event_dict)
    ev["key_id"] = key_id
    if "timestamp" not in ev:
        ev["timestamp"] = ev.get("resolved_at") or datetime.now(UTC).isoformat()
    canonical_bytes = build_canonical_payload(ev)
    signature = compute_hmac_signature(secret, canonical_bytes)
    ev["signature"] = signature
    return ev


class TrustedProducerAuthorizer(EventAuthorizerPort):
    """Authenticates event producer identity via HMAC-SHA256 signature verification,

    enforces timestamp freshness for replay defense, and screens reviewer legitimacy.
    """

    def __init__(
        self,
        producer_secrets: dict[str, dict[str, str]] | None = None,
        trusted_producers: AbstractSet[str] | list[str] | None = None,
        producer_secret: str | None = None,
        max_event_age_seconds: float | None = None,
        clock_skew_tolerance_seconds: float | None = None,
        enforce_producer_auth: bool | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is None:
            from arda_reliability_service.config import get_settings

            settings = get_settings()

        if trusted_producers is None:
            trusted_producers = set(settings.redpanda_trusted_producers)
        if producer_secret is None:
            producer_secret = settings.redpanda_producer_secret
        if producer_secrets is None:
            producer_secrets = settings.redpanda_producer_secrets
        if max_event_age_seconds is None:
            max_event_age_seconds = settings.redpanda_max_event_age_seconds
        if clock_skew_tolerance_seconds is None:
            clock_skew_tolerance_seconds = (
                settings.redpanda_clock_skew_tolerance_seconds
            )
        if enforce_producer_auth is None:
            enforce_producer_auth = settings.redpanda_enforce_producer_auth

        self.trusted_producers: AbstractSet[str] = (
            set(trusted_producers)
            if trusted_producers is not None
            else {
                "arda.human-review-service",
                "arda.gateway.review-worker",
                "arda.workflow-engine",
            }
        )
        self.enforce_producer_auth: bool = (
            enforce_producer_auth if enforce_producer_auth is not None else True
        )
        self.max_event_age_seconds: float = (
            max_event_age_seconds if max_event_age_seconds is not None else 300.0
        )
        self.clock_skew_tolerance_seconds: float = (
            clock_skew_tolerance_seconds
            if clock_skew_tolerance_seconds is not None
            else 60.0
        )

        # Build secrets map: producer_id -> {key_id: secret}
        # In multi-key rotation environments, only explicitly configured producer keys are active.
        # Fallback to shared producer_secret only applies when producer_secrets map is empty.
        self.producer_secrets: dict[str, dict[str, str]] = {}
        if producer_secrets:
            for p, keys in producer_secrets.items():
                self.producer_secrets[p] = {k: v for k, v in keys.items() if v}
        elif producer_secret:
            for p in self.trusted_producers:
                self.producer_secrets[p] = {"v1": producer_secret}

    def authorize_event(
        self,
        event: HumanResolutionEvent,
        now: datetime | None = None,
    ) -> None:
        """Validate producer authenticity and reviewer legitimacy.

        Boundary Note:
        - Producer authentication guarantees that the event originated from a trusted internal service.
        - Reviewer identity screening is a defensive sanitization filter against automated/bot
          identities; it is NOT an RBAC mechanism.
        - Reviewer authentication, tenant authorization, and RBAC are the strict responsibility
          of the upstream review service boundary, which must authenticate human reviewers before signing.
        """
        if self.enforce_producer_auth:
            # Verify producer is in trusted whitelist
            if event.producer_id not in self.trusted_producers:
                raise UnauthorizedProducerError(
                    f"Untrusted event producer '{event.producer_id}'. Authorized producers: {sorted(self.trusted_producers)}"
                )

            # Look up signing key
            prod_keys = self.producer_secrets.get(event.producer_id, {})
            if not prod_keys:
                raise UnauthorizedProducerError(
                    f"No active signing keys configured for producer '{event.producer_id}'."
                )

            key_id = (event.key_id or "").strip()
            if not key_id:
                raise UnauthorizedProducerError(
                    f"Missing key_id for event '{event.event_id}' from producer '{event.producer_id}'."
                )

            secret = prod_keys.get(key_id)
            if not secret:
                raise UnauthorizedProducerError(
                    f"Unknown key_id '{key_id}' for producer '{event.producer_id}'."
                )

            # Verify signature presence
            supplied_sig = (event.signature or "").strip()
            if not supplied_sig:
                raise UnauthorizedProducerError(
                    f"Missing HMAC signature for event '{event.event_id}' from producer '{event.producer_id}'."
                )

            # Verify timestamp freshness to mitigate replay attacks
            current_time = now or datetime.now(UTC)
            event_ts = event.get_event_timestamp()
            if event_ts.tzinfo is None:
                event_ts = event_ts.replace(tzinfo=UTC)
            else:
                event_ts = event_ts.astimezone(UTC)

            age = (current_time - event_ts).total_seconds()
            if age > self.max_event_age_seconds:
                raise ExpiredEventError(
                    f"Event '{event.event_id}' expired. Timestamp {event_ts.isoformat()} is {age:.1f}s old (max allowed: {self.max_event_age_seconds}s)."
                )
            if age < -self.clock_skew_tolerance_seconds:
                raise InvalidEventTimestampError(
                    f"Event '{event.event_id}' timestamp {event_ts.isoformat()} is {abs(age):.1f}s in future (exceeds clock skew tolerance of {self.clock_skew_tolerance_seconds}s)."
                )

            # Verify HMAC-SHA256 signature against canonical payload
            canonical_payload = build_canonical_payload(event)
            computed_sig = compute_hmac_signature(secret, canonical_payload)

            clean_supplied = supplied_sig.lower().removeprefix("sha256=")

            if not hmac.compare_digest(computed_sig, clean_supplied):
                raise InvalidSignatureError(
                    f"HMAC-SHA256 signature verification failed for event '{event.event_id}' from producer '{event.producer_id}'."
                )

        # Defensive filter: screen reviewer identity against bot/unauthorized patterns
        if not event.reviewer_id or not event.reviewer_id.strip():
            raise UnauthorizedReviewerError(
                "Missing or empty reviewer_id in human resolution event."
            )

        clean_reviewer = event.reviewer_id.strip()
        if FORBIDDEN_REVIEWER_PATTERNS.search(clean_reviewer):
            raise UnauthorizedReviewerError(
                f"Reviewer identity '{event.reviewer_id}' indicates forged or unauthorized non-human resolution."
            )
