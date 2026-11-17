"""Distribution-free bounds and resampling.

The point of this module: a simulation gives you a number, and a number is not
an answer. These functions turn a sample into an interval, or into a hard bound
that holds without assuming anything about the shape of the distribution.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Concentration inequalities (no distributional assumptions)
# --------------------------------------------------------------------------


def markov_upper(mean: float, t: float) -> float:
    """P(X >= t) <= E[X] / t for non-negative X."""
    if t <= 0:
        return 1.0
    return float(min(1.0, mean / t))


def cantelli_upper(mean: float, var: float, t: float) -> float:
    """One-sided Cantelli (Chebyshev) bound: P(X - mu >= a) <= s^2 / (s^2 + a^2).

    Holds for any distribution with finite mean and variance. Tight for
    heavy tails in the sense that it degrades gracefully instead of lying.
    """
    a = t - mean
    if a <= 0:
        return 1.0
    if var <= 0:
        return 0.0
    return float(var / (var + a * a))


def cantelli_lower(mean: float, var: float, t: float) -> float:
    """Mirror bound: P(X < t) <= s^2 / (s^2 + a^2) with a = mu - t."""
    a = mean - t
    if a <= 0:
        return 1.0
    if var <= 0:
        return 0.0
    return float(var / (var + a * a))


def hoeffding_upper(mean: float, t: float, n: int, lo: float, hi: float) -> float:
    """Bound on the tail of the *sample mean*, for bounded X in [lo, hi]."""
    if t <= mean or n <= 0:
        return 1.0
    rng = hi - lo
    if rng <= 0:
        return 0.0
    return float(min(1.0, math.exp(-2.0 * n * (t - mean) ** 2 / rng ** 2)))


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def bootstrap_ci(
    samples: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for any statistic."""
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "width": float("nan"), "n": int(x.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    point = float(stat(x))
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "n": int(x.size),
        "method": "bootstrap",
    }


def p95_ci(samples: np.ndarray, **kw) -> Dict[str, float]:
    return bootstrap_ci(samples, stat=lambda a: float(np.quantile(a, 0.95)), **kw)


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation. Above ~1.96/sqrt(n) the samples are not iid."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return float("nan")
    xc = x - np.mean(x)
    denom = float(np.sum(xc * xc))
    if denom <= 0:
        return 0.0
    return float(np.sum(xc[:-1] * xc[1:]) / denom)


_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def batch_means_ci(
    samples: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_batches: int = 20,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Confidence interval that survives autocorrelated samples.

    Splits the run into k contiguous batches, computes the statistic per batch
    and uses a t-interval on the batch values. The bootstrap assumes iid and
    understates the width when it is violated; this does not.
    """
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    # keep at least ~30 observations per batch, else batch statistics are noise
    n_batches = max(4, min(n_batches, x.size // 30))
    if x.size < 40 or n_batches < 4:
        return bootstrap_ci(x, stat=stat, alpha=alpha)
    b = x.size // n_batches
    batches = np.array([stat(x[i * b:(i + 1) * b]) for i in range(n_batches)])
    batches = batches[np.isfinite(batches)]
    k = batches.size
    if k < 2:
        return bootstrap_ci(x, stat=stat, alpha=alpha)
    se = float(np.std(batches, ddof=1) / np.sqrt(k))
    t = _T975.get(k - 1, 1.96)
    # The batch values estimate the standard error; the interval is centred on
    # the full-sample statistic so the point estimate always lies inside it.
    point = float(stat(x))
    return {
        "point": point,
        "lo": point - t * se,
        "hi": point + t * se,
        "width": 2.0 * t * se,
        "n": int(x.size),
        "method": f"batch means ({k} batches)",
    }


def required_n_for_mean(
    sd: float, half_width: float, z: float = 1.96, batch: int = 1
) -> int:
    """Sample size so the 95% CI half-width on a mean is <= half_width.

    For a batch-means estimator, pass the batch sd and multiply by batch.
    """
    if half_width <= 0 or sd <= 0:
        return 0
    n = (z * sd / half_width) ** 2
    return int(math.ceil(n / batch) * batch)


def relative_precision(ci: Dict[str, float]) -> float:
    """CI half-width as a fraction of the point estimate."""
    p = ci.get("point", float("nan"))
    if not p or np.isnan(p) or p == 0:
        return float("nan")
    return float((ci["hi"] - ci["lo"]) / 2.0 / abs(p))


# --------------------------------------------------------------------------
# Steady-state detection
# --------------------------------------------------------------------------


def warmup_index(
    samples: np.ndarray,
    frac: float = 0.05,
    tolerance: float = 0.05,
) -> int:
    """Welch-style crude warm-up: first index where a moving average settles.

    Reports where the cumulative mean stops drifting beyond `tolerance` of the
    full-sample mean. A large index means the run is dominated by the transient,
    and any steady-state formula applied to it is wrong.
    """
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 20:
        return 0
    target = float(np.mean(x))
    cum = np.cumsum(x) / np.arange(1, x.size + 1)
    for i in range(int(x.size * frac), x.size):
        window = cum[i:]
        if np.all(np.abs(window - target) <= tolerance * abs(target) + 1e-12):
            return int(i)
    return int(x.size * frac)
