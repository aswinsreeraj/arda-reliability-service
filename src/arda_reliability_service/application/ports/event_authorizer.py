"""Port defining event producer authentication and reviewer authorization interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from arda_reliability_service.domain.entities import HumanResolutionEvent


class EventAuthorizerPort(ABC):
    """Abstract port for verifying that human resolution events originate from trusted producers."""

    @abstractmethod
    def authorize_event(self, event: HumanResolutionEvent) -> None:
        """Validate that the event originates from a trusted producer and carries authorized reviewer credentials.

        Args:
            event: The normalized domain resolution event.

        Raises:
            UnauthorizedProducerError: If the producer is untrusted or unauthenticated.
            UnauthorizedReviewerError: If the reviewer identity is forged or unauthorized.
        """
