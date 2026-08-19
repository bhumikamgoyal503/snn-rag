"""
Classical QP solver: cvxpy + OSQP.

This is the ground truth Phase 4's SNN solver is measured against — it
runs on every SNN solve by default (see SNNQPSolver.compare_with_classical).
"""
from __future__ import annotations

from time import perf_counter

import cvxpy as cp
import numpy as np

from snn_rag.solvers.base import BaseQPSolver, QPSolution
from snn_rag.solvers.qp_utils import ensure_psd


class ClassicalQPSolver(BaseQPSolver):
    """Solves  min (1/2) x^T Q x + c^T x  s.t. 0 <= x <= 1  with OSQP."""

    def solve(self, Q: np.ndarray, c: np.ndarray, threshold: float = 0.5) -> QPSolution:
        Q = ensure_psd(Q)
        n = len(c)
        x = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Q)) + c @ x)
        constraints = [x >= 0, x <= 1]
        problem = cp.Problem(objective, constraints)

        t0 = perf_counter()
        problem.solve(solver=cp.OSQP)
        solve_time = perf_counter() - t0

        if x.value is None:
            raise RuntimeError(f"OSQP failed to solve QP (status={problem.status})")

        x_val = np.clip(x.value, 0.0, 1.0)
        selected = (x_val > threshold).astype(int)
        return QPSolution(
            x=x_val,
            selected=selected,
            objective=float(problem.value),
            solve_time_s=solve_time,
            solver_name="cvxpy-OSQP",
        )
