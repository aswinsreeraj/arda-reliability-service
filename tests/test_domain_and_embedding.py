"""Unit tests for domain entities, value objects, exceptions, and embedding adapter."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from arda_reliability_service.application.ports.embedding_provider import (
    EmbeddingProviderPort,
)
from arda_reliability_service.application.ports.reference_repository import (
    ReferenceExampleRepositoryPort,
)
from arda_reliability_service.application.use_cases.ingest_reference_examples import (
    IngestItemDTO,
    IngestReferenceExamplesUseCase,
)
from arda_reliability_service.application.use_cases.retrieve_reference_examples import (
    RetrieveReferenceExamplesUseCase,
)
from arda_reliability_service.domain.entities import ReferenceExample
from arda_reliability_service.domain.exceptions import (
    DomainValidationError,
    InvalidDescriptionError,
    InvalidEmbeddingDimensionError,
)
from arda_reliability_service.domain.value_objects import (
    ReferenceNeighbor,
    RetrievalQuery,
    RetrievalResultSet,
)
from arda_reliability_service.infrastructure.embedding.sentence_transformer_adapter import (
    SentenceTransformerEmbeddingAdapter,
)


class TestDomainEntitiesAndValueObjects:
    """Tests domain invariants for ReferenceExample, RetrievalQuery, and value objects."""

    def test_reference_example_valid_instantiation(self) -> None:
        tenant_id = uuid4()
        agent_id = uuid4()
        entity = ReferenceExample(
            example_id="ref-test-001",
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Customer filed an unauthorized debit claim.",
            label="APPROVE",
        )
        assert entity.example_id == "ref-test-001"
        assert entity.tenant_id == tenant_id
        assert entity.agent_id == agent_id
        assert entity.label == "APPROVE"
        assert entity.status == "active"
        assert entity.source == "synthetic"
        assert entity.embedding is None
        assert isinstance(entity.id, UUID)

    def test_reference_example_rejects_empty_id(self) -> None:
        with pytest.raises(DomainValidationError, match="example_id must not be empty"):
            ReferenceExample(
                example_id="   ",
                tenant_id=uuid4(),
                agent_id=uuid4(),
                case_description="Valid text",
                label="APPROVE",
            )

    def test_reference_example_rejects_empty_description(self) -> None:
        with pytest.raises(
            InvalidDescriptionError, match="case_description must not be empty"
        ):
            ReferenceExample(
                example_id="ref-001",
                tenant_id=uuid4(),
                agent_id=uuid4(),
                case_description="",
                label="APPROVE",
            )

    def test_reference_example_rejects_invalid_label(self) -> None:
        with pytest.raises(DomainValidationError, match="Invalid label 'MAYBE'"):
            ReferenceExample(
                example_id="ref-001",
                tenant_id=uuid4(),
                agent_id=uuid4(),
                case_description="Valid text",
                label="MAYBE",
            )

    def test_reference_example_rejects_invalid_status(self) -> None:
        with pytest.raises(DomainValidationError, match="Invalid status 'deleted'"):
            ReferenceExample(
                example_id="ref-001",
                tenant_id=uuid4(),
                agent_id=uuid4(),
                case_description="Valid text",
                label="APPROVE",
                status="deleted",
            )

    def test_attach_embedding_success_and_dimension_validation(self) -> None:
        entity = ReferenceExample(
            example_id="ref-001",
            tenant_id=uuid4(),
            agent_id=uuid4(),
            case_description="Valid text",
            label="APPROVE",
        )
        dummy_vec = [0.1] * 384
        entity.attach_embedding(dummy_vec, expected_dim=384)
        assert entity.embedding == dummy_vec

        # Dimension mismatch
        with pytest.raises(
            InvalidEmbeddingDimensionError, match="expected 384, got 128"
        ):
            entity.attach_embedding([0.1] * 128, expected_dim=384)

    def test_retrieval_query_invariants(self) -> None:
        tenant_id = uuid4()
        agent_id = uuid4()
        query = RetrievalQuery(
            tenant_id=tenant_id,
            agent_id=agent_id,
            case_description="Test dispute description",
            top_k=5,
            ef_search=64,
        )
        assert query.top_k == 5
        assert query.ef_search == 64

        # Invalid descriptions
        with pytest.raises(InvalidDescriptionError):
            RetrievalQuery(tenant_id=tenant_id, agent_id=agent_id, case_description="")

        with pytest.raises(InvalidDescriptionError):
            RetrievalQuery(
                tenant_id=tenant_id, agent_id=agent_id, case_description="   "
            )

        # Invalid top_k
        with pytest.raises(
            DomainValidationError, match="top_k must be a positive integer"
        ):
            RetrievalQuery(
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description="Test",
                top_k=0,
            )

        # Invalid ef_search
        with pytest.raises(DomainValidationError, match="ef_search must be positive"):
            RetrievalQuery(
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description="Test",
                ef_search=-1,
            )

    def test_reference_neighbor_and_result_set(self) -> None:
        neighbor = ReferenceNeighbor(
            example_id="ref-001",
            label="APPROVE",
            source="synthetic",
            case_description="Description",
            similarity_score=0.9234,
            metadata={"category": "fraud"},
        )
        assert neighbor.example_id == "ref-001"
        assert neighbor.similarity_score == 0.9234

        result_set = RetrievalResultSet(
            neighbors=[neighbor],
            query_description="Query",
            top_k=1,
            retrieval_mode="exact",
            execution_latency_ms=12.5,
        )
        assert len(result_set.neighbors) == 1
        assert result_set.retrieval_mode == "exact"
        assert result_set.execution_latency_ms == 12.5


class TestSentenceTransformerAdapter:
    """Tests the SentenceTransformer adapter text formatting, embedding output, and caching."""

    def test_formatting_with_prefixes(self) -> None:
        adapter = SentenceTransformerEmbeddingAdapter(
            model_name="BAAI/bge-small-en-v1.5"
        )
        query_text = adapter._format_text("Dispute text", is_query=True)
        passage_text = adapter._format_text("Dispute text", is_query=False)

        assert (
            query_text
            == "Represent this sentence for searching relevant passages: Dispute text"
        )
        assert passage_text == "Dispute text"

        e5_adapter = SentenceTransformerEmbeddingAdapter(
            model_name="intfloat/e5-base-v2", expected_dimension=768
        )
        assert (
            e5_adapter._format_text("Dispute text", is_query=True)
            == "query: Dispute text"
        )
        assert (
            e5_adapter._format_text("Dispute text", is_query=False)
            == "passage: Dispute text"
        )

    def test_formatting_rejects_empty_text(self) -> None:
        adapter = SentenceTransformerEmbeddingAdapter()
        with pytest.raises(InvalidDescriptionError):
            adapter._format_text("", is_query=False)
        with pytest.raises(InvalidDescriptionError):
            adapter._format_text("   ", is_query=True)

    def test_embedding_generation_dimension_and_normalization(self) -> None:
        adapter = SentenceTransformerEmbeddingAdapter(
            model_name="BAAI/bge-small-en-v1.5", expected_dimension=384
        )
        emb = adapter.generate_embedding(
            "Cardholder reported unauthorized billing after card stolen.",
            is_query=False,
        )

        assert len(emb) == 384
        # L2 norm check
        l2_norm = math.sqrt(sum(x * x for x in emb))
        assert abs(l2_norm - 1.0) < 1e-4

    def test_batch_embedding_generation(self) -> None:
        adapter = SentenceTransformerEmbeddingAdapter(
            model_name="BAAI/bge-small-en-v1.5", expected_dimension=384
        )
        texts = [
            "Customer claims duplicate charge.",
            "Merchant provided signed delivery slip.",
        ]
        embeddings = adapter.generate_embeddings_batch(texts, is_query=False)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        for emb in embeddings:
            l2_norm = math.sqrt(sum(x * x for x in emb))
            assert abs(l2_norm - 1.0) < 1e-4

    def test_dimension_mismatch_raises_exception(self) -> None:
        # If adapter is configured with wrong expected dimension, it should raise InvalidEmbeddingDimensionError
        adapter = SentenceTransformerEmbeddingAdapter(
            model_name="BAAI/bge-small-en-v1.5",
            expected_dimension=512,  # Actual is 384
        )
        with pytest.raises(
            InvalidEmbeddingDimensionError, match="does not match expected 512"
        ):
            adapter.generate_embedding("Test text", is_query=False)


class TestApplicationUseCases:
    """Tests orchestration in IngestReferenceExamplesUseCase and RetrieveReferenceExamplesUseCase."""

    @pytest.mark.asyncio
    async def test_ingest_use_case_from_dtos(self) -> None:
        mock_repo = AsyncMock(spec=ReferenceExampleRepositoryPort)
        mock_repo.save_batch.return_value = ["ref-001", "ref-002"]

        mock_embedding = MagicMock(spec=EmbeddingProviderPort)
        mock_embedding.model_name = "BAAI/bge-small-en-v1.5"
        mock_embedding.dimension = 384
        mock_embedding.generate_embeddings_batch.return_value = [
            [0.1] * 384,
            [0.2] * 384,
        ]

        use_case = IngestReferenceExamplesUseCase(
            repository=mock_repo, embedding_provider=mock_embedding
        )

        tenant_id = uuid4()
        agent_id = uuid4()
        items = [
            IngestItemDTO(
                example_id="ref-001",
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description="Raw description 1",
                label="APPROVE",
                metadata={
                    "ratio_verbalized_description": "Ratio verbalized description 1"
                },
            ),
            IngestItemDTO(
                example_id="ref-002",
                tenant_id=tenant_id,
                agent_id=agent_id,
                case_description="Raw description 2",
                label="REJECT",
                metadata={},  # Falls back to case_description
            ),
        ]

        result = await use_case.execute(items)
        assert result == ["ref-001", "ref-002"]

        # Verify embedding generator called with ratio verbalized description for item 1 and raw for item 2
        mock_embedding.generate_embeddings_batch.assert_called_once_with(
            ["Ratio verbalized description 1", "Raw description 2"],
            is_query=False,
        )

        # Verify repo.save_batch called
        assert mock_repo.save_batch.call_count == 1
        saved_entities = mock_repo.save_batch.call_args[0][0]
        assert len(saved_entities) == 2
        assert saved_entities[0].example_id == "ref-001"
        assert saved_entities[0].embedding == [0.1] * 384
        assert saved_entities[1].example_id == "ref-002"
        assert saved_entities[1].embedding == [0.2] * 384

    @pytest.mark.asyncio
    async def test_ingest_use_case_empty_items(self) -> None:
        mock_repo = AsyncMock(spec=ReferenceExampleRepositoryPort)
        mock_embedding = MagicMock(spec=EmbeddingProviderPort)
        use_case = IngestReferenceExamplesUseCase(
            repository=mock_repo, embedding_provider=mock_embedding
        )

        result = await use_case.execute([])
        assert result == []
        mock_repo.save_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_use_case_rejects_unsupported_type(self) -> None:
        mock_repo = AsyncMock(spec=ReferenceExampleRepositoryPort)
        mock_embedding = MagicMock(spec=EmbeddingProviderPort)
        use_case = IngestReferenceExamplesUseCase(
            repository=mock_repo, embedding_provider=mock_embedding
        )

        with pytest.raises(
            DomainValidationError, match="Unsupported ingestion item type"
        ):
            await use_case.execute([12345])  # type: ignore

    @pytest.mark.asyncio
    async def test_retrieve_use_case_exact_and_approximate(self) -> None:
        mock_repo = AsyncMock(spec=ReferenceExampleRepositoryPort)
        mock_repo.retrieve_exact.return_value = [
            ReferenceNeighbor(
                example_id="ref-001",
                label="APPROVE",
                source="synthetic",
                case_description="Desc",
                similarity_score=0.95,
            )
        ]
        mock_repo.retrieve_approximate.return_value = [
            ReferenceNeighbor(
                example_id="ref-002",
                label="REJECT",
                source="synthetic",
                case_description="Desc 2",
                similarity_score=0.88,
            )
        ]

        mock_embedding = MagicMock(spec=EmbeddingProviderPort)
        mock_embedding.generate_embedding.return_value = [0.05] * 384

        use_case = RetrieveReferenceExamplesUseCase(
            repository=mock_repo, embedding_provider=mock_embedding
        )

        tenant_id = uuid4()
        agent_id = uuid4()
        query = RetrievalQuery(
            tenant_id=tenant_id, agent_id=agent_id, case_description="Dispute claim"
        )

        # Test exact retrieval
        exact_res = await use_case.execute(query, mode="exact")
        assert exact_res.retrieval_mode == "exact"
        assert len(exact_res.neighbors) == 1
        assert exact_res.neighbors[0].example_id == "ref-001"
        mock_repo.retrieve_exact.assert_called_once()

        # Test approximate retrieval
        approx_res = await use_case.execute(query, mode="approximate")
        assert approx_res.retrieval_mode == "approximate_hnsw"
        assert len(approx_res.neighbors) == 1
        assert approx_res.neighbors[0].example_id == "ref-002"
        mock_repo.retrieve_approximate.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_use_case_invalid_mode(self) -> None:
        mock_repo = AsyncMock(spec=ReferenceExampleRepositoryPort)
        mock_embedding = MagicMock(spec=EmbeddingProviderPort)
        use_case = RetrieveReferenceExamplesUseCase(
            repository=mock_repo, embedding_provider=mock_embedding
        )

        query = RetrievalQuery(
            tenant_id=uuid4(), agent_id=uuid4(), case_description="Dispute claim"
        )
        with pytest.raises(
            DomainValidationError, match="Invalid retrieval mode 'invalid'"
        ):
            await use_case.execute(query, mode="invalid")
