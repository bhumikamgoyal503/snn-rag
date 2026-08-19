"""Unit tests for the document-selection QP formulation (Phase 3)."""
from __future__ import annotations

import numpy as np

from snn_rag.config import QPConfig
from snn_rag.core.vector_store import RetrievedDoc
from snn_rag.solvers.classical import ClassicalQPSolver
from snn_rag.solvers.qp_utils import build_qp_matrices, ensure_psd


def _doc(doc_id: int, score: float, embedding: list[float]) -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc_id, text=f"doc{doc_id}", score=score,
        embedding=np.array(embedding, dtype=np.float32),
    )


class TestEnsurePSD:
    def test_already_psd_matrix_is_unchanged_up_to_symmetry(self):
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = ensure_psd(Q)
        assert np.allclose(out, Q)

    def test_indefinite_matrix_is_projected_to_psd(self):
        # eigenvalues of [[0.01, 1], [1, 0.01]] are 1.01 and -0.99 -> indefinite
        Q = np.array([[0.01, 1.0], [1.0, 0.01]])
        out = ensure_psd(Q)
        eigvals = np.linalg.eigvalsh(out)
        assert eigvals.min() >= -1e-8
        assert np.allclose(out, out.T)


class TestClassicalQPSolver:
    def test_two_relevant_non_redundant_docs_are_both_retained(self):
        # No cross-redundancy; both docs highly relevant -> keep both.
        Q = np.diag([0.01, 0.01])
        c = np.array([-1.0, -1.0])
        solution = ClassicalQPSolver().solve(Q, c)
        assert solution.selected.tolist() == [1, 1]
        assert np.all(solution.x > 0.9)

    def test_irrelevant_doc_is_dropped(self):
        # doc 2's unconstrained optimum (c_i / Q_ii) sits well under threshold.
        Q = np.diag([0.01, 0.01])
        c = np.array([-1.0, -0.001])
        solution = ClassicalQPSolver().solve(Q, c)
        assert solution.selected.tolist() == [1, 0]
        assert solution.x[1] < 0.5

    def test_redundant_pair_retains_less_mass_than_independent_pair(self):
        # Same relevance in both cases; only redundancy differs.
        c = np.array([-1.0, -1.0])
        Q_independent = ensure_psd(np.diag([0.01, 0.01]))
        Q_redundant = ensure_psd(np.array([[1.0, 0.9], [0.9, 1.0]]))

        sol_independent = ClassicalQPSolver().solve(Q_independent, c)
        sol_redundant = ClassicalQPSolver().solve(Q_redundant, c)

        assert sol_redundant.x.sum() < sol_independent.x.sum()


class TestBuildQPMatrices:
    def test_identical_embeddings_flagged_more_redundant_than_orthogonal(self):
        cfg = QPConfig(redundancy_weight=1.0)
        identical_docs = [_doc(0, 0.9, [1.0, 0.0]), _doc(1, 0.9, [1.0, 0.0])]
        orthogonal_docs = [_doc(0, 0.9, [1.0, 0.0]), _doc(1, 0.9, [0.0, 1.0])]

        Q_identical, c = build_qp_matrices(identical_docs, cfg)
        Q_orthogonal, _ = build_qp_matrices(orthogonal_docs, cfg)

        # ensure_psd may rescale magnitudes, but identical embeddings must
        # always carry a strictly larger redundancy penalty than orthogonal.
        assert Q_identical[0, 1] > Q_orthogonal[0, 1]
        # Equal raw scores -> degenerate min-max normalization -> both fully relevant.
        assert np.allclose(c, -np.array([1.0, 1.0]))

    def test_output_Q_is_psd(self):
        docs = [
            _doc(0, 0.9, [1.0, 0.0]),
            _doc(1, 0.8, [0.0, 1.0]),
            _doc(2, 0.7, [0.9, 0.1]),
        ]
        Q, _ = build_qp_matrices(docs, QPConfig())
        eigvals = np.linalg.eigvalsh(Q)
        assert eigvals.min() >= -1e-8

    def test_orthogonal_relevant_docs_are_both_retained_through_the_pipeline(self):
        # Regression test for the bug flagged in review: two non-redundant,
        # highly relevant docs must survive QP selection. Before the fix, a
        # uniform diagonal self-cost (either the raw Gram diagonal of 1, or
        # an arbitrary constant needing large PSD repair) suppressed recall
        # even with zero real redundancy between candidates.
        docs = [
            _doc(0, 0.95, [1.0, 0.0]),
            _doc(1, 0.90, [0.0, 1.0]),
        ]
        Q, c = build_qp_matrices(docs, QPConfig())
        solution = ClassicalQPSolver().solve(Q, c)
        assert solution.selected.tolist() == [1, 1]

    def test_relevance_scores_are_normalized_into_a_known_range(self):
        # Raw scores on wildly different scales (e.g. cosine sim vs a
        # cross-encoder logit) must not change the *relative* c values.
        docs_small_scale = [_doc(0, 0.9, [1.0, 0.0]), _doc(1, 0.1, [0.0, 1.0])]
        docs_large_scale = [_doc(0, 9.0, [1.0, 0.0]), _doc(1, 1.0, [0.0, 1.0])]

        _, c_small = build_qp_matrices(docs_small_scale, QPConfig())
        _, c_large = build_qp_matrices(docs_large_scale, QPConfig())

        assert np.allclose(c_small, c_large)
        assert c_small.min() >= -1.0 and c_small.max() <= 0.0

    def test_moderately_similar_docs_pay_no_redundancy_penalty(self):
        # cos_sim ~ 0.5 (topically adjacent, e.g. two people in the same
        # film) is well below the default 0.8 threshold -> not redundant.
        docs = [
            _doc(0, 0.9, [1.0, 0.0]),
            _doc(1, 0.9, [0.5, np.sqrt(1 - 0.5 ** 2)]),
        ]
        Q, _ = build_qp_matrices(docs, QPConfig())
        assert Q[0, 1] == 0.0

    def test_near_duplicate_docs_still_pay_redundancy_penalty(self):
        # cos_sim ~ 0.95 is above the default 0.8 threshold -> genuinely
        # near-duplicate, still penalized.
        docs = [
            _doc(0, 0.9, [1.0, 0.0]),
            _doc(1, 0.9, [0.95, np.sqrt(1 - 0.95 ** 2)]),
        ]
        Q, _ = build_qp_matrices(docs, QPConfig())
        assert Q[0, 1] > 0.0

    def test_topically_adjacent_but_distinct_relevant_docs_both_survive(self):
        # Regression test mirroring the real HotpotQA failure: two relevant,
        # moderately-similar (bridge-question-like) docs must both be kept,
        # not just the orthogonal-embedding case already covered above.
        docs = [
            _doc(0, 0.95, [1.0, 0.0]),
            _doc(1, 0.90, [0.5, np.sqrt(1 - 0.5 ** 2)]),
        ]
        Q, c = build_qp_matrices(docs, QPConfig())
        solution = ClassicalQPSolver().solve(Q, c)
        assert solution.selected.tolist() == [1, 1]
