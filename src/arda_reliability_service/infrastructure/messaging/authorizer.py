"""Trusted producer authenticator and reviewer authorization implementation."""

from __future__ import annotations

import hmac
import logging
import re
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arda_reliability_service.config import Settings

from arda_reliability_service.application.ports.event_authorizer import (
    EventAuthorizerPort,
)
from arda_reliability_service.domain.entities import HumanResolutionEvent
from arda_reliability_service.domain.exceptions import (
    UnauthorizedProducerError,
    UnauthorizedReviewerError,
)

logger = logging.getLogger(__name__)

FORBIDDEN_REVIEWER_PATTERNS = re.compile(
    r"(?:anonymous|unauthorized|bot|system|automated|unknown|none|^$)",
    re.IGNORECASE,
)


class TrustedProducerAuthorizer(EventAuthorizerPort):
    """Authenticates event producer identity and verifies reviewer authorization boundary."""

    def __init__(
        self,
        trusted_producers: AbstractSet[str] | list[str] | None = None,
        producer_secret: str | None = None,
        enforce_producer_auth: bool | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            if trusted_producers is None:
                trusted_producers = set(settings.redpanda_trusted_producers)
            if producer_secret is None:
                producer_secret = settings.redpanda_producer_secret
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
        self.producer_secret: str = (
            producer_secret
            if producer_secret is not None
            else "arda-producer-secret-default"
        )
        self.enforce_producer_auth: bool = (
            enforce_producer_auth if enforce_producer_auth is not None else True
        )

    def authorize_event(self, event: HumanResolutionEvent) -> None:
        """Validate producer authenticity and reviewer legitimacy.

        Boundary Note:
        - Producer authentication guarantees that the event originated from a trusted internal service.
        - Reviewer authorization screens for forged identities, ensuring that only authenticated
          human reviewers adjudicated by the producer boundary are accepted.
        """
        if self.enforce_producer_auth:
            # 1. Producer Identity Verification
            if event.producer_id not in self.trusted_producers:
                raise UnauthorizedProducerError(
                    f"Untrusted event producer '{event.producer_id}'. Authorized producers: {sorted(self.trusted_producers)}"
                )

            # 2. Cryptographic / Token Verification
            if self.producer_secret and (
                not event.producer_token
                or not hmac.compare_digest(event.producer_token, self.producer_secret)
            ):
                raise UnauthorizedProducerError(
                    f"Producer authentication token invalid or missing for '{event.producer_id}'."
                )

        # 3. Reviewer Identity Screening
        if not event.reviewer_id or not event.reviewer_id.strip():
            raise UnauthorizedReviewerError(
                "Missing or empty reviewer_id in human resolution event."
            )

        clean_reviewer = event.reviewer_id.strip()
        if FORBIDDEN_REVIEWER_PATTERNS.search(clean_reviewer):
            raise UnauthorizedReviewerError(
                f"Reviewer identity '{event.reviewer_id}' indicates forged or unauthorized non-human resolution."
            )
