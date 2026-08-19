"""
Utilities for building and validating the document-selection QP.

QP formulation (minimize):
    (1/2) x^T Q x + c^T x   s.t.  0 <= x_i <= 1   for all i

Where, per doc i:
    x_i   in [0,1]  — probability of keeping document i
    c_i = -relevance_weight * relevance_norm(doc_i)   [relevance_norm in [0,1]]
    Q_ij = redundancy_weight * cosine_sim_norm(emb_i, emb_j)   for i != j,
           but only if the raw cosine_sim >= redundancy_similarity_threshold;
           otherwise Q_ij = 0
    Q_ii = 0    (no diagonal self-cost — see note below)

Scaling note: c and Q must live on comparable scales, or one term silently
dominates the other regardless of the *_weight knobs. Relevance is
normalized to [0,1] per retrieval batch; the redundancy matrix is built with
a zero diagonal and normalized by its own largest entry before
redundancy_weight is applied, so redundancy_weight is a single clean scale
knob rather than fighting an uncontrolled matrix magnitude.

Similarity-threshold note: cosine similarity between paragraph embeddings
conflates "same topic" with "same information." On bridge questions in
particular, the two gold documents are often topically adjacent but
complementary (that's what makes it a bridge question) — e.g. two people
who appear in the same film. A naive redundancy penalty punishes that
adjacency as if it were duplication, while an irrelevant outlier document
(dissimilar to everything, because it's off-topic) pays almost no penalty
and gets kept by default. Thresholding at redundancy_similarity_threshold
restricts the penalty to genuinely near-duplicate pairs and leaves merely
topic-adjacent pairs alone.

The diagonal is intentionally zero going in: a document's self-similarity
is 1 for every document, so if that landed on the diagonal it would act as
a flat "cost to keep any document at all" that suppresses recall regardless
of relevance or actual redundancy with other candidates. ensure_psd() still
has to repair the diagonal to make the matrix convex (a zero-diagonal matrix
with nonzero off-diagonal is never PSD), but because both c and Q are now
scale-bounded first, that repair stays small relative to the relevance
signal instead of swamping it.
"""
from __future__ import annotations

import numpy as np

from snn_rag.config import QPConfig
from snn_rag.core.vector_store import RetrievedDoc


def ensure_psd(Q: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Symmetrize Q and project onto the PSD cone if needed.

    cvxpy/OSQP require Q to be PSD for the QP to be convex. A zero-diagonal
    redundancy matrix with nonzero off-diagonal entries is never PSD, so
    this repair always runs — every Q must pass through here before being
    handed to a solver.
    """
    Q_sym = (Q + Q.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(Q_sym)
    if eigvals.min() >= -epsilon:
        return Q_sym
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def _normalize_relevance(scores: np.ndarray) -> np.ndarray:
    """Normalize relevance scores to [0,1] by shifting to a non-negative
    floor and dividing by the batch max — NOT batch min-max range.

    Min-max normalization was tried and rejected: it forces whichever doc
    is locally weakest in a batch down to exactly 0 reward, even when it's
    still highly relevant in absolute terms (e.g. two docs scoring 0.95 and
    0.90 both indicate strong relevance, but min-max would zero out the
    second one purely for being the batch's minimum). Anchoring the floor
    at 0 instead of the batch minimum only suppresses genuinely low/negative
    scores (relevant for unbounded cross-encoder logits) while leaving
    already-non-negative scores (the common cosine-similarity case) untouched
    before scaling.
    """
    floor = min(scores.min(), 0.0)
    shifted = scores - floor
    max_val = shifted.max()
    if max_val < 1e-12:
        return np.ones_like(scores)  # nothing to differentiate; treat as equally relevant
    return shifted / max_val


def build_qp_matrices(
    docs: list[RetrievedDoc],
    cfg: QPConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (Q, c) pair for the document-selection QP from retrieval state."""
    cfg = cfg or QPConfig()
    embs = np.stack([d.embedding for d in docs]).astype(np.float64)
    scores = np.array([d.score for d in docs], dtype=np.float64)

    sim = embs @ embs.T  # cosine sim, since embeddings are L2-normalized
    np.fill_diagonal(sim, 0.0)
    sim = np.where(sim >= cfg.redundancy_similarity_threshold, sim, 0.0)
    max_abs = np.abs(sim).max()
    if max_abs > 1e-12:
        sim = sim / max_abs  # single comparable scale, so redundancy_weight is the only knob
    Q = cfg.redundancy_weight * sim
    Q = ensure_psd(Q)

    relevance_norm = _normalize_relevance(scores)
    c = -cfg.relevance_weight * relevance_norm
    return Q, c
