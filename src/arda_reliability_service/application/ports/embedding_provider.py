"""Port defining embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProviderPort(ABC):
    """Abstract port for generating dense normalized vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension produced by this model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model name or identifier."""

    @abstractmethod
    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        """Generate a single unit-normalized embedding vector.

        Args:
            text: Input natural language description.
            is_query: Whether this text is an asymmetric search query (requiring query prefix/instruction).

        Returns:
            List of floats representing the embedding vector.
        """

    @abstractmethod
    def generate_embeddings_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Generate a batch of unit-normalized embedding vectors.

        Args:
            texts: List of natural language descriptions.
            is_query: Whether these texts are search queries.

        Returns:
            List of embedding vectors.
        """
