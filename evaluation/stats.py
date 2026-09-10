"""
Bootstrap confidence intervals for the ablation metrics.

Two kinds of interval, and the distinction matters for the paper:

  * `bootstrap_ci` - a marginal CI on one arm's mean. Answers "how precisely
    do we know this arm's precision?"
  * `paired_bootstrap_ci` - a CI on the per-example *difference* between two
    arms. Every arm is evaluated on the identical example set, so the two
    means are correlated; resampling them independently throws that pairing
    away and inflates the interval. The paired interval is the one that
    supports a claim like "the QP improves precision", and it excludes zero
    exactly when the improvement is significant at the given level.
"""
from __future__ import annotations

import numpy as np

DEFAULT_N_BOOT = 10_000


def bootstrap_ci(
    values: list[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": n}


def paired_bootstrap_ci(
    treatment: list[float],
    baseline: list[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean paired difference (treatment - baseline).

    Resamples *examples*, keeping each example's pair of scores together, so the
    interval reflects the within-example correlation between the two arms.
    `significant` is True when the interval excludes zero.
    """
    t = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if len(t) != len(b):
        raise ValueError(f"paired arms must be the same length, got {len(t)} and {len(b)}")
    diff = t - b
    n = len(diff)
    if n == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "significant": False, "n": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "n": n,
    }


def format_ci(ci: dict[str, float], places: int = 3) -> str:
    """Render a CI as 'mean [lo, hi]' for console tables."""
    key = "mean" if "mean" in ci else "mean_diff"
    return f"{ci[key]:.{places}f} [{ci['lo']:.{places}f}, {ci['hi']:.{places}f}]"
