"""Application use case for executing exact or approximate vector retrieval."""

from __future__ import annotations

import time

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.application.ports.reference_repository import (
    ReferenceExampleRepositoryPort,
)
from arda_reliability_service.domain.exceptions import DomainValidationError
from arda_reliability_service.domain.value_objects import (
    RetrievalQuery,
    RetrievalResultSet,
)


class RetrieveReferenceExamplesUseCase:
    """Orchestrates query embedding generation and filtered vector retrieval."""

    def __init__(
        self,
        repository: ReferenceExampleRepositoryPort,
        embedding_provider: EmbeddingProviderPort,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider

    async def execute(
        self,
        query: RetrievalQuery,
        mode: str = "exact",
    ) -> RetrievalResultSet:
        """Execute vector retrieval for the given query.

        Args:
            query: Domain query containing tenant_id, agent_id, case_description, and top_k.
            mode: 'exact' (flat cosine scan) or 'approximate' (HNSW index scan).

        Returns:
            RetrievalResultSet with retrieved neighbors and execution latency.
        """
        if mode not in {"exact", "approximate", "approximate_hnsw"}:
            raise DomainValidationError(
                f"Invalid retrieval mode '{mode}'. Must be 'exact' or 'approximate'."
            )

        t0 = time.perf_counter()

        # Generate query embedding (is_query=True applies query prefix/instruction)
        query_embedding = self.embedding_provider.generate_embedding(
            query.case_description, is_query=True
        )

        # Execute retrieval via repository port
        if mode == "exact":
            neighbors = await self.repository.retrieve_exact(
                query_embedding=query_embedding,
                tenant_id=query.tenant_id,
                agent_id=query.agent_id,
                top_k=query.top_k,
            )
        else:
            neighbors = await self.repository.retrieve_approximate(
                query_embedding=query_embedding,
                tenant_id=query.tenant_id,
                agent_id=query.agent_id,
                top_k=query.top_k,
                ef_search=query.ef_search,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResultSet(
            neighbors=neighbors,
            query_description=query.case_description,
            top_k=query.top_k,
            retrieval_mode="exact" if mode == "exact" else "approximate_hnsw",
            execution_latency_ms=round(latency_ms, 3),
        )
