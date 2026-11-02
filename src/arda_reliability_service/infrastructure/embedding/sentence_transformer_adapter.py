"""SentenceTransformer adapter implementing the EmbeddingProviderPort."""

from __future__ import annotations

import threading
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.domain.exceptions import (
    InvalidDescriptionError,
    InvalidEmbeddingDimensionError,
)

# Known model prefix rules
MODEL_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
}

MODEL_PASSAGE_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "",
    "intfloat/e5-base-v2": "passage: ",
}


class SentenceTransformerEmbeddingAdapter(EmbeddingProviderPort):
    """Adapter for generating normalized dense embeddings using SentenceTransformers."""

    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        expected_dimension: int = 384,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> SentenceTransformer:
        """Thread-safe lazy loader ensuring the model is loaded once per process/model."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model_name not in self._model_cache:
                dev = None if self._device == "auto" else self._device
                self._model_cache[self._model_name] = SentenceTransformer(
                    self._model_name, device=dev
                )
            self._model = self._model_cache[self._model_name]
            return self._model

    def _format_text(self, text: str, is_query: bool) -> str:
        """Format input text with recommended model prefix while rejecting empty input."""
        if not text or not text.strip():
            raise InvalidDescriptionError(
                "Text to embed must not be empty or whitespace."
            )

        cleaned = text.strip()
        if is_query:
            prefix = MODEL_QUERY_PREFIXES.get(self._model_name, "")
        else:
            prefix = MODEL_PASSAGE_PREFIXES.get(self._model_name, "")

        return f"{prefix}{cleaned}"

    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        """Generate a single unit-normalized embedding vector."""
        formatted = self._format_text(text, is_query=is_query)
        model = self._get_model()

        emb = model.encode(
            formatted,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        dim = len(emb)
        if dim != self._expected_dimension:
            raise InvalidEmbeddingDimensionError(
                f"Generated embedding dimension {dim} does not match expected {self._expected_dimension}"
            )

        return [float(x) for x in emb]

    def generate_embeddings_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Generate a batch of unit-normalized embedding vectors."""
        if not texts:
            return []

        formatted_texts = [self._format_text(t, is_query=is_query) for t in texts]
        model = self._get_model()

        embs: np.ndarray = model.encode(
            formatted_texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        results: list[list[float]] = []
        for emb in embs:
            dim = len(emb)
            if dim != self._expected_dimension:
                raise InvalidEmbeddingDimensionError(
                    f"Generated embedding dimension {dim} does not match expected {self._expected_dimension}"
                )
            results.append([float(x) for x in emb])

        return results
