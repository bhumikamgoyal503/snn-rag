"""
Solver interface for the document-selection QP.

Phase 1-2: not used (retriever handles doc selection via top-k + rerank).
Phase 3:   ClassicalQPSolver fills in (cvxpy/OSQP).
Phase 4:   SNNQPSolver fills in (snnTorch-based).

Both solvers will implement the same interface so we can compare
solution quality (SNN vs classical) as a first-class result.

QP formulation (minimize):
    (1/2) x^T Q x + c^T x
    subject to  0 <= x_i <= 1  for all i

Where:
    x ∈ [0,1]^n  — probability of keeping each doc
    c ∈ R^n      — negative relevance (so minimization → max relevance)
    Q ∈ R^{n×n}  — redundancy penalty (PSD)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class QPSolution:
    """Output of a QP solve."""
    x: np.ndarray           # continuous solution in [0,1]^n
    selected: np.ndarray    # binary mask after thresholding
    objective: float        # optimal objective value
    solve_time_s: float     # wall-clock time for the solve
    solver_name: str
    # Populated by SNNQPSolver: classical-solver comparison (objective gap,
    # ||x_snn - x_classical||, timing). None for the classical solver itself.
    comparison: dict | None = None


class BaseQPSolver(ABC):
    """Contract for all QP solvers."""

    @abstractmethod
    def solve(
        self,
        Q: np.ndarray,
        c: np.ndarray,
        threshold: float = 0.5,
    ) -> QPSolution:
        """
        Solve  min (1/2) x^T Q x + c^T x  s.t. 0 <= x <= 1.

        Parameters
        ----------
        Q : (n, n) PSD matrix — redundancy + cost
        c : (n,) vector — negative relevance scores
        threshold : x_i > threshold → retain doc i
        """
        ...
