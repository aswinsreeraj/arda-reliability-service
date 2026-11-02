"""Application use case for ingesting reference examples and generating vector embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.application.ports.reference_repository import (
    ReferenceExampleRepositoryPort,
)
from arda_reliability_service.domain.entities import ReferenceExample
from arda_reliability_service.domain.exceptions import DomainValidationError


class IngestItemDTO(BaseModel):
    """Data transfer object for a single reference example to ingest."""

    example_id: str = Field(
        ..., description="Unique business identifier (e.g. 'ref-example-001')"
    )
    tenant_id: UUID = Field(..., description="Tenant scope identifier")
    agent_id: UUID = Field(..., description="Agent scope identifier")
    case_description: str = Field(
        ..., min_length=1, description="Natural language case narrative"
    )
    label: str = Field(..., description="Adjudication outcome ('APPROVE' or 'REJECT')")
    source: str = Field(
        default="synthetic", description="Data origin (e.g. 'synthetic', 'production')"
    )
    status: str = Field(
        default="active", description="Lifecycle status ('active', 'archived')"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary case metadata"
    )
    model_name: str | None = Field(
        default=None, description="Optional override for embedding model name"
    )
    model_version: str | None = Field(
        default=None, description="Optional override for model version"
    )


class IngestReferenceExamplesUseCase:
    """Orchestrates validation, embedding generation, and idempotent persistence."""

    def __init__(
        self,
        repository: ReferenceExampleRepositoryPort,
        embedding_provider: EmbeddingProviderPort,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider

    async def execute(
        self,
        items: Sequence[IngestItemDTO | dict[str, Any] | ReferenceExample],
        on_conflict_do_update: bool = True,
    ) -> list[str]:
        """Ingest a batch of reference examples.

        Args:
            items: List of raw dictionaries, IngestItemDTOs, or domain entities.
            on_conflict_do_update: Whether to update existing records if duplicate IDs exist.

        Returns:
            List of successfully ingested example_ids.
        """
        if not items:
            return []

        # Convert to standardized domain entities
        entities: list[ReferenceExample] = []
        texts_to_embed: list[str] = []

        for item in items:
            if isinstance(item, ReferenceExample):
                entity = item
            elif isinstance(item, IngestItemDTO):
                entity = ReferenceExample(
                    example_id=item.example_id,
                    tenant_id=item.tenant_id,
                    agent_id=item.agent_id,
                    case_description=item.case_description,
                    label=item.label,
                    source=item.source,
                    status=item.status,
                    metadata=item.metadata,
                    model_name=item.model_name or self.embedding_provider.model_name,
                    model_version=item.model_version or "v1.5",
                )
            elif isinstance(item, dict):
                tenant_raw = item["tenant_id"]
                agent_raw = item["agent_id"]
                tenant_id = (
                    tenant_raw
                    if isinstance(tenant_raw, UUID)
                    else UUID(str(tenant_raw))
                )
                agent_id = (
                    agent_raw if isinstance(agent_raw, UUID) else UUID(str(agent_raw))
                )

                entity = ReferenceExample(
                    example_id=str(item["example_id"]),
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    case_description=str(item["case_description"]),
                    label=str(item["label"]),
                    source=str(item.get("source", "synthetic")),
                    status=str(item.get("status", "active")),
                    metadata=dict(item.get("metadata", {})),
                    model_name=str(
                        item.get("model_name", self.embedding_provider.model_name)
                    ),
                    model_version=str(item.get("model_version", "v1.5")),
                )
            else:
                raise DomainValidationError(
                    f"Unsupported ingestion item type: {type(item)}"
                )

            # Canonical embedding source:
            # Per operational configuration, use ratio_verbalized_description if present in metadata;
            # otherwise fall back to case_description.
            text = (
                entity.metadata.get("ratio_verbalized_description")
                or entity.case_description
            )
            texts_to_embed.append(text)
            entities.append(entity)

        # Batch embedding generation (is_query=False for passages/documents)
        embeddings = self.embedding_provider.generate_embeddings_batch(
            texts_to_embed, is_query=False
        )

        # Attach embeddings to domain entities
        expected_dim = self.embedding_provider.dimension
        for entity, emb in zip(entities, embeddings):
            entity.attach_embedding(emb, expected_dim=expected_dim)

        # Persist through repository port
        return await self.repository.save_batch(
            entities, on_conflict_do_update=on_conflict_do_update
        )
