"""
RAG (Retrieval-Augmented Generation) poisoning detection.

Implements a cosine-similarity anomaly detector for the RAG retrieval mechanism.
When a retrieved document deviates significantly from the statistical norm of
the agent's core alignment data, it is flagged and optionally discarded.

Reference: AgentPoison (2024) - backdoor triggers in RAG knowledge bases.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
import numpy as np


@dataclass
class RAGRetrievalResult:
    """A single retrieved document with its anomaly analysis."""
    document_id: str
    content: str
    embedding: np.ndarray
    similarity_to_query: float
    anomaly_score: float
    is_anomalous: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGDetectionReport:
    """Aggregate report from RAG poisoning detection."""
    total_documents: int
    anomalous_documents: int
    filtered_documents: int
    anomaly_rate: float
    results: List[RAGRetrievalResult] = field(default_factory=list)
    safe_documents: List[RAGRetrievalResult] = field(default_factory=list)


class RAGPoisoningDetector:
    """Cosine-similarity anomaly detector for RAG retrieval pipelines.

    Maintains a statistical profile of the agent's core alignment data
    (trusted document embeddings). When a retrieved document's embedding
    deviates beyond the configured threshold, it is flagged as potentially
    poisoned.

    Detection strategies:
    1. **Centroid distance**: Compare document embedding to the centroid
       of trusted embeddings.
    2. **Distribution outlier**: Flag embeddings outside N standard
       deviations of the trusted distribution.
    3. **Trigger pattern detection**: Identify known backdoor trigger
       tokens/phrases in document content.
    """

    KNOWN_TRIGGER_PATTERNS = [
        "EXECUTE_OVERRIDE_",
        "PRIORITY_DIRECTIVE_",
        "SYS_UPDATE_",
        "ignore previous instructions",
        "system override",
        "admin override",
        "developer mode activated",
    ]

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        outlier_std_factor: float = 2.5,
        min_trusted_samples: int = 10,
        enable_trigger_detection: bool = True,
        custom_trigger_patterns: Optional[List[str]] = None,
    ):
        self.logger = logging.getLogger("auto_art.rag_detector")
        self.similarity_threshold = similarity_threshold
        self.outlier_std_factor = outlier_std_factor
        self.min_trusted_samples = min_trusted_samples
        self.enable_trigger_detection = enable_trigger_detection

        self.trigger_patterns = self.KNOWN_TRIGGER_PATTERNS.copy()
        if custom_trigger_patterns:
            self.trigger_patterns.extend(custom_trigger_patterns)

        self._trusted_embeddings: Optional[np.ndarray] = None
        self._centroid: Optional[np.ndarray] = None
        self._distances_mean: float = 0.0
        self._distances_std: float = 1.0

    def fit(self, trusted_embeddings: np.ndarray) -> None:
        """Fit the detector on trusted/aligned document embeddings.

        Args:
            trusted_embeddings: Array of shape (n_docs, embedding_dim)
                                containing embeddings of known-good documents.
        """
        if trusted_embeddings.shape[0] < self.min_trusted_samples:
            self.logger.warning(
                f"Only {trusted_embeddings.shape[0]} trusted samples provided "
                f"(minimum recommended: {self.min_trusted_samples}). "
                "Detection accuracy may be reduced."
            )

        self._trusted_embeddings = trusted_embeddings.copy()
        self._centroid = np.mean(trusted_embeddings, axis=0)

        distances = self._compute_distances_to_centroid(trusted_embeddings)
        self._distances_mean = float(np.mean(distances))
        self._distances_std = float(np.std(distances)) if np.std(distances) > 1e-8 else 1e-8

        self.logger.info(
            f"RAG detector fitted on {trusted_embeddings.shape[0]} trusted documents. "
            f"Mean distance: {self._distances_mean:.4f}, Std: {self._distances_std:.4f}"
        )

    def _compute_distances_to_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine distances from embeddings to the centroid."""
        if self._centroid is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return 1.0 - self._cosine_similarity_batch(embeddings, self._centroid)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _cosine_similarity_batch(embeddings: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between each row in embeddings and reference."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        ref_norm = np.linalg.norm(reference)
        norms = np.where(norms < 1e-10, 1.0, norms)
        ref_norm = max(ref_norm, 1e-10)
        return (embeddings @ reference) / (norms.squeeze() * ref_norm)

    def detect(
        self,
        documents: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None,
    ) -> RAGDetectionReport:
        """Analyze retrieved documents for poisoning indicators.

        Args:
            documents: List of dicts, each with at minimum 'id', 'content',
                       and 'embedding' (np.ndarray) keys.
            query_embedding: Optional query embedding for relevance scoring.

        Returns:
            RAGDetectionReport with per-document analysis.
        """
        results: List[RAGRetrievalResult] = []
        safe_docs: List[RAGRetrievalResult] = []
        anomalous_count = 0

        for doc in documents:
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", "")
            embedding = doc.get("embedding")

            if embedding is None:
                self.logger.warning(f"Document {doc_id} has no embedding; skipping.")
                continue

            embedding = np.asarray(embedding, dtype=np.float32)
            anomaly_score = self._compute_anomaly_score(embedding, content)

            query_sim = 0.0
            if query_embedding is not None:
                query_sim = self._cosine_similarity(embedding, query_embedding)

            is_anomalous = anomaly_score > self.similarity_threshold

            result = RAGRetrievalResult(
                document_id=doc_id,
                content=content[:500],
                embedding=embedding,
                similarity_to_query=query_sim,
                anomaly_score=anomaly_score,
                is_anomalous=is_anomalous,
                metadata=doc.get("metadata", {}),
            )
            results.append(result)

            if is_anomalous:
                anomalous_count += 1
                self.logger.warning(
                    f"Anomalous document detected: {doc_id} "
                    f"(anomaly_score={anomaly_score:.4f})"
                )
            else:
                safe_docs.append(result)

        total = len(results)
        return RAGDetectionReport(
            total_documents=total,
            anomalous_documents=anomalous_count,
            filtered_documents=total - anomalous_count,
            anomaly_rate=anomalous_count / total if total > 0 else 0.0,
            results=results,
            safe_documents=safe_docs,
        )

    def _compute_anomaly_score(self, embedding: np.ndarray, content: str) -> float:
        """Compute a composite anomaly score for a single document."""
        scores: List[float] = []

        if self._centroid is not None:
            cos_dist = 1.0 - self._cosine_similarity(embedding, self._centroid)
            z_score = abs(cos_dist - self._distances_mean) / self._distances_std
            dist_anomaly = min(1.0, z_score / (self.outlier_std_factor * 2))
            scores.append(dist_anomaly)

        if self.enable_trigger_detection:
            trigger_score = self._detect_triggers(content)
            scores.append(trigger_score)

        if not scores:
            return 0.0
        return float(np.max(scores))

    def _detect_triggers(self, content: str) -> float:
        """Detect known backdoor trigger patterns in content."""
        content_lower = content.lower()
        matches = sum(1 for p in self.trigger_patterns if p.lower() in content_lower)
        return min(1.0, matches / max(len(self.trigger_patterns) * 0.1, 1))

    def filter_safe(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Convenience method: filter and return only safe documents."""
        report = self.detect(documents, **kwargs)
        return [
            {"id": r.document_id, "content": r.content, "embedding": r.embedding}
            for r in report.safe_documents
        ]
