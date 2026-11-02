"""Embedding generation, caching, and Leave-One-Out (LOO) retrieval engine.

Implements:
- Model initialization with timing and peak memory tracking
- SentenceTransformers encoding with query/passage prefix handling
- Caching of normalized embeddings to disk (.npz)
- Strict Leave-One-Out (LOO) retrieval evaluation where queries never retrieve themselves
- Deterministic majority-vote aggregation with similarity-based tie-breaking
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from research.experiments.config import (
    CACHE_DIR,
    ExperimentConfig,
)
from research.experiments.metrics import (
    ExperimentResult,
    LOOResult,
    PerformanceMetrics,
    RetrievedNeighbor,
    calculate_category_relevance,
    calculate_classification_metrics,
    calculate_cohort_metrics,
    calculate_global_similarity_metrics,
    calculate_retrieval_baselines,
    calculate_similarity_metrics,
    evaluate_contrastive_pairs,
)


class EmbeddingEvaluator:
    """Manages embedding generation, caching, and LOO retrieval for an experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        device: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.config = config
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: SentenceTransformer | None = None
        self.model_load_time_seconds: float = 0.0
        self.peak_memory_mb: float = 0.0

    def load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model with telemetry and resource tracking."""
        if self.model is not None:
            return self.model

        tracemalloc.start()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            mem_before = 0.0

        t0 = time.perf_counter()
        try:
            model = SentenceTransformer(self.config.model.name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.config.model.name}': {exc}"
            ) from exc

        t_load = time.perf_counter() - t0
        self.model_load_time_seconds = round(t_load, 4)

        if self.device == "cuda":
            peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.peak_memory_mb = round(peak_gpu_mem - mem_before, 2)
        else:
            _, peak_cpu = tracemalloc.get_traced_memory()
            self.peak_memory_mb = round(peak_cpu / (1024 * 1024), 2)
        tracemalloc.stop()

        self.model = model
        return self.model

    def _get_cache_path(self) -> Path:
        """Derive reproducible cache filename for model and template configuration."""
        safe_model = self.config.model.short_name
        safe_template = self.config.template.name
        return self.cache_dir / f"{safe_model}_{safe_template}_embeddings.npz"

    def encode_dataset(
        self, examples: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Encode all examples into query embeddings and passage embeddings.

        Returns:
            Tuple of (query_embeddings, passage_embeddings, generation_time_seconds).
        """
        cache_path = self._get_cache_path()
        if self.use_cache and cache_path.exists():
            data = np.load(cache_path)
            queries = data["queries"]
            passages = data["passages"]
            return queries, passages, 0.0

        model = self.load_model()
        template_field = self.config.template.metadata_field

        # Extract strictly the template text from metadata
        raw_texts = [ex["metadata"][template_field] for ex in examples]

        # Apply recommended prefixes
        query_texts = [f"{self.config.model.query_prefix}{t}" for t in raw_texts]
        passage_texts = [f"{self.config.model.passage_prefix}{t}" for t in raw_texts]

        t0 = time.perf_counter()
        query_embeddings = model.encode(
            query_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.model.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        passage_embeddings = model.encode(
            passage_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.model.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        t_gen = time.perf_counter() - t0

        if self.use_cache:
            np.savez_compressed(
                cache_path,
                queries=query_embeddings,
                passages=passage_embeddings,
                example_ids=[ex["example_id"] for ex in examples],
            )

        return query_embeddings, passage_embeddings, round(t_gen, 4)

    def run_loo_evaluation(self, dataset_path: Path) -> ExperimentResult:
        """Execute full Leave-One-Out (LOO) retrieval evaluation on the dataset."""
        t_start = time.perf_counter()

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)

        examples = dataset_data["examples"]
        n_examples = len(examples)
        id_to_idx = {ex["example_id"]: i for i, ex in enumerate(examples)}

        # Generate / load normalized embeddings
        query_embs, passage_embs, t_gen = self.encode_dataset(examples)

        # Full similarity matrix: Query_i dot Passage_j
        # Since embeddings are L2 normalized, dot product == cosine similarity
        sim_matrix = np.matmul(query_embs, passage_embs.T)

        t_retrieval_start = time.perf_counter()
        loo_results: list[LOOResult] = []

        max_k = max(self.config.k_values)

        for i, query_ex in enumerate(examples):
            q_id = query_ex["example_id"]
            true_label = query_ex["label"]

            # Vector of similarities to all passage embeddings
            sims = sim_matrix[i].copy()

            # STRICT LOO RULE: The query example must never retrieve itself
            sims[i] = -np.inf

            # Sort indices by similarity descending
            sorted_indices = np.argsort(-sims)
            top_k_indices = sorted_indices[:max_k]

            retrieved_neighbors: list[RetrievedNeighbor] = []
            for rank, idx in enumerate(top_k_indices, start=1):
                neighbor_ex = examples[idx]
                retrieved_neighbors.append(
                    RetrievedNeighbor(
                        example_id=neighbor_ex["example_id"],
                        label=neighbor_ex["label"],
                        similarity=round(float(sims[idx]), 4),
                        rank=rank,
                    )
                )

            # Calculate predictions for each k in k_values
            predicted_labels: dict[int, str] = {}
            is_correct: dict[int, bool] = {}

            for k in self.config.k_values:
                k_neighbors = retrieved_neighbors[:k]
                approve_votes = [n for n in k_neighbors if n.label == "APPROVE"]
                reject_votes = [n for n in k_neighbors if n.label == "REJECT"]

                if len(approve_votes) > len(reject_votes):
                    pred = "APPROVE"
                elif len(reject_votes) > len(approve_votes):
                    pred = "REJECT"
                else:
                    # Deterministic Tie-Breaking:
                    # Choose label of the candidate with higher maximum similarity in the top-k
                    max_app_sim = max(
                        (n.similarity for n in approve_votes), default=-1.0
                    )
                    max_rej_sim = max(
                        (n.similarity for n in reject_votes), default=-1.0
                    )
                    if max_app_sim > max_rej_sim:
                        pred = "APPROVE"
                    elif max_rej_sim > max_app_sim:
                        pred = "REJECT"
                    else:
                        # Secondary tie-breaker: sum of similarities
                        sum_app = sum(n.similarity for n in approve_votes)
                        sum_rej = sum(n.similarity for n in reject_votes)
                        pred = "APPROVE" if sum_app >= sum_rej else "REJECT"

                predicted_labels[k] = pred
                is_correct[k] = pred == true_label

            loo_results.append(
                LOOResult(
                    example_id=q_id,
                    true_label=true_label,
                    retrieved_neighbors=retrieved_neighbors,
                    predicted_labels=predicted_labels,
                    is_correct=is_correct,
                )
            )

        t_retrieval_total = time.perf_counter() - t_retrieval_start
        per_query_ms = (t_retrieval_total / n_examples) * 1000.0

        # Calculate Classification Metrics for each k
        k_metrics = {}
        y_true = [ex["label"] for ex in examples]
        for k in self.config.k_values:
            y_pred = [res.predicted_labels[k] for res in loo_results]
            k_metrics[k] = calculate_classification_metrics(y_true, y_pred)

        # Calculate Similarity Separation across top retrieved neighbors
        sim_metrics = calculate_similarity_metrics(loo_results)

        # Calculate Global Similarity Metrics across all candidate pairs
        global_sim_metrics = calculate_global_similarity_metrics(sim_matrix, y_true)

        # Break down metrics by cohort: contrastive (N=20) vs non-contrastive (N=80)
        cohort_metrics = calculate_cohort_metrics(loo_results, examples)

        # Calculate semantic topical relevance (Dispute category Precision@K)
        category_relevance = calculate_category_relevance(loo_results, examples)

        # Finite-sample retrieval baselines
        baselines = calculate_retrieval_baselines()

        # Contrastive Pairs Analysis
        # Note: Symmetric passage-to-passage similarity matrix for intrinsic pair similarity
        passage_sim_matrix = np.matmul(passage_embs, passage_embs.T)
        contrastive_metrics = evaluate_contrastive_pairs(
            examples, passage_sim_matrix, id_to_idx
        )

        t_total = time.perf_counter() - t_start

        # Detailed serializable results
        detailed_records = []
        for res in loo_results:
            detailed_records.append(
                {
                    "example_id": res.example_id,
                    "true_label": res.true_label,
                    "predicted_labels": res.predicted_labels,
                    "is_correct": res.is_correct,
                    "top_neighbors": [
                        {
                            "example_id": n.example_id,
                            "label": n.label,
                            "similarity": n.similarity,
                            "rank": n.rank,
                        }
                        for n in res.retrieved_neighbors
                    ],
                }
            )

        perf = PerformanceMetrics(
            model_load_time_seconds=self.model_load_time_seconds,
            embedding_generation_seconds=t_gen,
            per_example_embedding_ms=round((t_gen / max(1, n_examples)) * 1000.0, 2),
            per_query_retrieval_ms=round(per_query_ms, 3),
            total_pipeline_seconds=round(t_total, 3),
            peak_memory_mb=self.peak_memory_mb,
            inference_device=self.device,
        )

        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            model_name=self.config.model.name,
            template_name=self.config.template.name,
            total_examples=n_examples,
            k_metrics=k_metrics,
            similarity_metrics=sim_metrics,
            contrastive_pairs=contrastive_metrics,
            performance=perf,
            detailed_loo_results=detailed_records,
            global_similarity_metrics=global_sim_metrics,
            cohort_metrics=cohort_metrics,
            category_relevance=category_relevance,
            baselines=baselines,
        )
