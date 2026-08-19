"""
Retriever interface and single-shot implementation.

The ABC defines the contract that every retriever (single-shot, multi-hop,
QP-optimized) must satisfy.  Phase 2 will add a MultiHopRetriever that
subclasses this, and Phase 3-4 will inject the QP/SNN solver into it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from snn_rag.config import RetrieverConfig, RerankerConfig
from snn_rag.core.vector_store import FAISSVectorStore, RetrievedDoc


@dataclass
class RetrievalResult:
    """Output of a retrieval pass (one or more hops)."""
    docs: list[RetrievedDoc]
    num_hops: int = 1
    # Phase 3+: will carry QP solution vector, cost, etc.
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """Contract for all retriever variants."""

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        ...


class SingleShotRetriever(BaseRetriever):
    """Phase 1 retriever: embed query → FAISS top-k → optional rerank."""

    def __init__(
        self,
        store: FAISSVectorStore,
        cfg: RetrieverConfig | None = None,
        reranker_cfg: RerankerConfig | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg or RetrieverConfig()
        self.reranker_cfg = reranker_cfg or RerankerConfig()
        self._reranker = None
        if self.reranker_cfg.enabled:
            self._init_reranker()

    # ------------------------------------------------------------------
    def _init_reranker(self) -> None:
        from sentence_transformers import CrossEncoder  # lazy import
        self._reranker = CrossEncoder(
            self.reranker_cfg.model_name, max_length=512
        )

    def _rerank(
        self, query: str, docs: list[RetrievedDoc]
    ) -> list[RetrievedDoc]:
        if self._reranker is None or not docs:
            return docs
        pairs = [(query, d.text) for d in docs]
        scores = self._reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.score = float(score)   # overwrite with cross-encoder score
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[: self.reranker_cfg.top_k]

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> RetrievalResult:
        docs = self.store.search(
            query,
            top_k=self.cfg.top_k,
            score_threshold=self.cfg.score_threshold,
        )
        if self.reranker_cfg.enabled:
            docs = self._rerank(query, docs)
        return RetrievalResult(docs=docs, num_hops=1)
