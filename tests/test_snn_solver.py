"""
Tests for the Phase 4 SNN QP solver.

These do NOT assert the SNN matches the classical solver exactly — it's
an approximate spiking dynamical system, not a numerical QP solver, and
CLAUDE.md is explicit that a solution gap is an honest, expected result.
What we do assert: the solver runs, returns a well-formed solution, and
always attaches a classical-solver comparison.
"""
from __future__ import annotations

import numpy as np

from snn_rag.solvers.classical import ClassicalQPSolver
from snn_rag.solvers.qp_utils import ensure_psd
from snn_rag.solvers.snn_solver import SNNQPSolver


class TestSNNQPSolver:
    def test_returns_well_formed_solution(self):
        Q = ensure_psd(np.diag([0.01, 0.01]))
        c = np.array([-1.0, -0.001])
        solution = SNNQPSolver(num_steps=50).solve(Q, c)

        assert solution.x.shape == (2,)
        assert np.all((solution.x >= 0.0) & (solution.x <= 1.0))
        assert set(solution.selected.tolist()) <= {0, 1}
        assert solution.solver_name == "SNN-LIF"
        assert solution.solve_time_s >= 0.0

    def test_comparison_with_classical_is_attached_by_default(self):
        Q = ensure_psd(np.diag([0.01, 0.01]))
        c = np.array([-1.0, -0.001])
        solution = SNNQPSolver(num_steps=50).solve(Q, c)

        assert solution.comparison is not None
        for key in (
            "classical_objective", "snn_objective", "objective_gap",
            "x_l2_gap", "selected_agreement",
            "classical_solve_time_s", "snn_solve_time_s",
        ):
            assert key in solution.comparison

    def test_comparison_can_be_disabled(self):
        Q = ensure_psd(np.diag([0.01, 0.01]))
        c = np.array([-1.0, -0.001])
        solution = SNNQPSolver(num_steps=20, compare_with_classical=False).solve(Q, c)
        assert solution.comparison is None

    def test_converges_roughly_toward_classical_on_an_easy_problem(self):
        # A well-separated, low-dimensional problem should be easy enough
        # for the spiking dynamics to land in the same basin as OSQP.
        Q = ensure_psd(np.diag([0.01, 0.01]))
        c = np.array([-1.0, -0.001])

        classical = ClassicalQPSolver().solve(Q, c)
        snn_solution = SNNQPSolver(num_steps=200).solve(Q, c)

        assert snn_solution.comparison["x_l2_gap"] < 0.5
        assert snn_solution.selected.tolist() == classical.selected.tolist()
