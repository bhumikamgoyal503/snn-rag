"""
RAG pipeline — the top-level orchestrator.

Phase 1: query → SingleShotRetriever → Generator → answer
Phase 2+: query → MultiHopRetriever (with planner) → Generator → answer

The pipeline doesn't know *how* retrieval works; it just calls
retriever.retrieve() and hands the docs to the generator.
This is the seam where the QP/SNN planner will plug in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from snn_rag.config import PipelineConfig
from snn_rag.core.retriever import BaseRetriever, RetrievalResult
from snn_rag.core.generator import Generator


@dataclass
class PipelineOutput:
    """Everything we need for evaluation and analysis."""
    question: str
    answer: str
    retrieval: RetrievalResult
    latency_s: float
    # Convenience for evaluation
    retrieved_titles: list[str] = field(default_factory=list)


class RAGPipeline:
    """Orchestrates retrieval → generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: Generator,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.cfg = cfg or PipelineConfig()

    def run(self, question: str) -> PipelineOutput:
        t0 = perf_counter()
        retrieval = self.retriever.retrieve(question)
        answer = self.generator.generate(
            question=question,
            docs=retrieval.docs,
            max_context_docs=self.cfg.max_context_docs,
        )
        latency = perf_counter() - t0
        return PipelineOutput(
            question=question,
            answer=answer,
            retrieval=retrieval,
            latency_s=latency,
        )
