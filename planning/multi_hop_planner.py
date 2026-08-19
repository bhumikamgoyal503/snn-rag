"""
Agentic multi-hop retrieval planner.

Loop, per hop: retrieve -> ask the LLM to detect a knowledge gap ->
formulate a sub-query -> retrieve again. Stops early once the LLM
reports no gap, or after cfg.max_hops.

This is the seam Phase 3-4 plug into: after each hop's retrieval, an
optional `doc_selector` (a BaseQPSolver — ClassicalQPSolver or
SNNQPSolver) decides which of the accumulated docs to keep, formulated
as the QP  min (1/2) x^T Q x + c^T x  s.t. 0 <= x <= 1.  Passing None
falls back to plain multi-hop retrieval with no QP-based pruning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from snn_rag.config import QPConfig, RerankerConfig, RetrieverConfig
from snn_rag.core.generator import Generator
from snn_rag.core.retriever import BaseRetriever, RetrievalResult, SingleShotRetriever
from snn_rag.core.vector_store import FAISSVectorStore, RetrievedDoc
from snn_rag.solvers.base import BaseQPSolver
from snn_rag.solvers.qp_utils import build_qp_matrices


@dataclass
class HopRecord:
    """Diagnostics for a single hop, kept for evaluation/ablation."""
    hop: int
    query: str
    num_docs_retrieved: int
    gap_detected: bool = False
    missing_info: str = ""
    qp: dict | None = None


class MultiHopRetriever(BaseRetriever):
    """Retrieve -> detect gap -> sub-query, for up to `max_hops` hops."""

    def __init__(
        self,
        store: FAISSVectorStore,
        generator: Generator,
        retriever_cfg: RetrieverConfig | None = None,
        reranker_cfg: RerankerConfig | None = None,
        max_hops: int = 2,
        doc_selector: BaseQPSolver | None = None,
        qp_cfg: QPConfig | None = None,
    ) -> None:
        self.store = store
        self.generator = generator
        self.max_hops = max_hops
        self.doc_selector = doc_selector
        self.qp_cfg = qp_cfg or QPConfig()
        self._base = SingleShotRetriever(store, retriever_cfg, reranker_cfg)

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> RetrievalResult:
        seen: dict[int, RetrievedDoc] = {}
        hop_records: list[HopRecord] = []
        current_query = query
        hop = 0

        for hop in range(1, self.max_hops + 1):
            hop_docs = self._base.retrieve(current_query).docs
            for doc in hop_docs:
                seen[doc.doc_id] = doc
            retained = list(seen.values())

            qp_record = None
            if self.doc_selector is not None and len(retained) > 1:
                retained, qp_record = self._apply_doc_selection(retained)
                seen = {d.doc_id: d for d in retained}

            record = HopRecord(
                hop=hop, query=current_query,
                num_docs_retrieved=len(hop_docs), qp=qp_record,
            )

            if hop >= self.max_hops:
                hop_records.append(record)
                break

            gap = self.generator.detect_gap(query, retained)
            record.gap_detected = gap.has_gap
            record.missing_info = gap.missing_info
            hop_records.append(record)

            if not gap.has_gap:
                break
            current_query = self.generator.generate_subquery(
                query, retained, gap.missing_info
            )

        return RetrievalResult(
            docs=list(seen.values()),
            num_hops=hop,
            metadata={"hops": hop_records},
        )

    # ------------------------------------------------------------------
    def _apply_doc_selection(
        self, docs: list[RetrievedDoc]
    ) -> tuple[list[RetrievedDoc], dict]:
        """Run the QP doc-selection hook and filter `docs` to the retained set."""
        Q, c = build_qp_matrices(docs, self.qp_cfg)
        solution = self.doc_selector.solve(Q, c, threshold=self.qp_cfg.retain_threshold)

        retained = [d for d, keep in zip(docs, solution.selected) if keep]
        if not retained:
            # Never let the QP zero out every candidate doc.
            retained = [max(docs, key=lambda d: d.score)]

        record = {
            "num_docs_in": len(docs),
            "num_docs_retained": len(retained),
            "objective": solution.objective,
            "solve_time_s": solution.solve_time_s,
            "solver_name": solution.solver_name,
            "comparison": solution.comparison,
        }
        return retained, record
