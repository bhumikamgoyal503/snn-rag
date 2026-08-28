"""
Spiking Neural Network QP solver (Phase 4).

Approach: energy-based / Hopfield-style spiking dynamics. One Leaky
Integrate-and-Fire (snnTorch) neuron per candidate document. At each
timestep, every neuron is driven by the negative gradient of the QP
objective evaluated at the current rate estimate:

    grad = Q x + c
    current_i(t) = -step_size * grad_i

This is the classical continuous Hopfield-network construction for
minimizing a quadratic energy function under box constraints, run here
on a spiking substrate: the decision variable x_i is read out as neuron
i's spike rate (spike count / t) over T timesteps, clamped to [0, 1].

Step-size calibration: step_size is NOT a fixed constant. It is derived
per-problem as k0_scale / L, where L is the Lipschitz constant of the QP
gradient (grad(x) = Qx + c is L-Lipschitz with L = largest eigenvalue of
Q). This follows the projected-gradient calibration used by the SNN_opt
solver (github.com/ahkhan03/SNN_opt): a fixed step_size tuned on one
Q's scale silently mis-converges once Q's magnitude changes (e.g. after
reformulating the redundancy penalty), whereas k0_scale is a small,
scale-invariant knob - lower k0_scale trades convergence speed for
accuracy, matching the reference solver's guidance to "tune the pair,
never k0_scale alone" (i.e. alongside num_steps).

Honesty constraint (see CLAUDE.md): the SNN's contribution is only
meaningful in comparison to the classical solver, so by default every
solve() also runs ClassicalQPSolver on the same (Q, c) and attaches the
gap (objective, ||x_snn - x_classical||, timing, selection agreement) to
QPSolution.comparison. Never report SNN results without this attached.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import snntorch as snn
import torch

from snn_rag.solvers.base import BaseQPSolver, QPSolution
from snn_rag.solvers.classical import ClassicalQPSolver
from snn_rag.solvers.qp_utils import ensure_psd


class SNNQPSolver(BaseQPSolver):
    """Hopfield-style LIF spiking network that solves the document-selection QP."""

    def __init__(
        self,
        num_steps: int = 200,
        beta: float = 0.9,
        k0_scale: float = 20.0,
        compare_with_classical: bool = True,
    ) -> None:
        self.num_steps = num_steps
        self.beta = beta
        self.k0_scale = k0_scale
        self.compare_with_classical = compare_with_classical

    def solve(self, Q: np.ndarray, c: np.ndarray, threshold: float = 0.5) -> QPSolution:
        Q = ensure_psd(Q)
        n = len(c)
        Q_t = torch.tensor(Q, dtype=torch.float32)
        c_t = torch.tensor(c, dtype=torch.float32)

        # Lipschitz constant of grad(x) = Qx + c is Q's largest eigenvalue;
        # floor it so a near-zero Q (e.g. no redundancy at all) can't blow
        # the step size up.
        L = max(float(np.linalg.eigvalsh(Q).max()), 1e-6)
        step_size = self.k0_scale / L

        lif = snn.Leaky(beta=self.beta, threshold=1.0)
        mem = torch.zeros(n)
        x_rate = torch.full((n,), 0.5)
        spike_counts = torch.zeros(n)

        t0 = perf_counter()
        for t in range(self.num_steps):
            grad = Q_t @ x_rate + c_t
            current = -step_size * grad
            spk, mem = lif(current, mem)
            spike_counts = spike_counts + spk
            x_rate = torch.clamp(spike_counts / (t + 1), 0.0, 1.0)
        solve_time = perf_counter() - t0

        x = x_rate.detach().numpy()
        selected = (x > threshold).astype(int)
        objective = float(0.5 * x @ Q @ x + c @ x)

        comparison = None
        if self.compare_with_classical:
            comparison = self._compare(Q, c, threshold, x, selected, objective, solve_time)

        return QPSolution(
            x=x,
            selected=selected,
            objective=objective,
            solve_time_s=solve_time,
            solver_name="SNN-LIF",
            comparison=comparison,
        )

    # ------------------------------------------------------------------
    def _compare(
        self,
        Q: np.ndarray,
        c: np.ndarray,
        threshold: float,
        x: np.ndarray,
        selected: np.ndarray,
        objective: float,
        solve_time: float,
    ) -> dict:
        classical = ClassicalQPSolver().solve(Q, c, threshold=threshold)
        return {
            "classical_objective": classical.objective,
            "snn_objective": objective,
            "objective_gap": objective - classical.objective,
            "x_l2_gap": float(np.linalg.norm(x - classical.x)),
            "selected_agreement": float(np.mean(selected == classical.selected)),
            "classical_solve_time_s": classical.solve_time_s,
            "snn_solve_time_s": solve_time,
        }
