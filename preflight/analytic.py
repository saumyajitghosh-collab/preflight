"""Closed-form results: what the maths says before you simulate anything.

Two jobs. First, give a cheap analytic answer where the assumptions hold.
Second — the important one — supply the *checks*: Little's Law and the serial
composition of moments must be satisfied by any correct simulation, so they
are free invariants the simulator can be audited against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .queue import QueueStage, QueueWorkflow


# --------------------------------------------------------------------------
# M/M/c
# --------------------------------------------------------------------------


@dataclass
class MMCMetrics:
    lam: float
    mu: float
    c: int
    rho: float
    p_wait: float          # Erlang C: probability an arrival must queue
    wq: float              # mean time in queue
    w: float               # mean time in system
    lq: float              # mean number in queue
    l: float               # mean number in system
    stable: bool


def _erlang_c(lam: float, mu: float, c: int) -> float:
    """Probability of queueing in an M/M/c system (Erlang's C formula)."""
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return 1.0
    # Erlang B by the numerically stable recursion
    b = 1.0
    for k in range(1, c + 1):
        b = (a * b) / (k + a * b)
    return b / (1.0 - rho + rho * b)


def mmc(lam: float, mu: float, c: int = 1) -> MMCMetrics:
    """Exact M/M/c metrics. Assumes Poisson arrivals and exponential service."""
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be positive")
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return MMCMetrics(lam, mu, c, rho, 1.0, math.inf, math.inf, math.inf, math.inf, False)
    pw = _erlang_c(lam, mu, c)
    wq = pw / (c * mu - lam)
    w = wq + 1.0 / mu
    return MMCMetrics(lam, mu, c, rho, pw, wq, w, lam * wq, lam * w, True)


def mm1(lam: float, mu: float) -> MMCMetrics:
    return mmc(lam, mu, 1)


# --------------------------------------------------------------------------
# G/G/c approximations
# --------------------------------------------------------------------------


def kingman_wq(lam: float, service_mean: float, ca2: float, cs2: float) -> float:
    """Kingman's formula for the mean wait in a G/G/1 queue.

    Wq ~= (rho / (1 - rho)) * ((ca^2 + cs^2) / 2) * E[S]
    The (ca^2 + cs^2)/2 factor is what makes variability, not just load, the
    thing that hurts. It is the single most useful approximation here.
    """
    rho = lam * service_mean
    if rho >= 1.0:
        return math.inf
    return (rho / (1.0 - rho)) * ((ca2 + cs2) / 2.0) * service_mean


def allen_cunneen_wq(
    lam: float,
    service_mean: float,
    ca2: float,
    cs2: float,
    c: int = 1,
) -> float:
    """Allen-Cunneen: scale the M/M/c wait by the variability correction."""
    base = mmc(lam, 1.0 / service_mean, c)
    if not base.stable:
        return math.inf
    return base.wq * ((ca2 + cs2) / 2.0)


# --------------------------------------------------------------------------
# Little's Law
# --------------------------------------------------------------------------


def little_check(l_observed: float, lam: float, w_observed: float) -> Dict[str, float]:
    """L = lambda * W. A free audit on any simulation output."""
    predicted = lam * w_observed
    rel = (l_observed - predicted) / predicted if predicted > 0 else float("nan")
    return {
        "l_observed": l_observed,
        "lam_times_w": predicted,
        "relative_error": rel,
        "holds": bool(abs(rel) < 0.10),
    }


# --------------------------------------------------------------------------
# Serial composition (no queueing)
# --------------------------------------------------------------------------


def serial_moments(stages: List[QueueStage]) -> Dict[str, float]:
    """Mean and variance of a serial chain with independent, uncontended stages.

    Mean composes linearly; variance composes only because of independence.
    Break that assumption (common-cause failures) and this stops being true —
    which is exactly the kind of thing the ledger flags.
    """
    mean = sum(s.service_mean for s in stages)
    var = sum(s.service.var() for s in stages)
    return {
        "mean": float(mean),
        "sd": float(np.sqrt(var)),
        "cv": float(np.sqrt(var) / mean) if mean > 0 else float("nan"),
    }


def serial_tail_lognormal_normal_approx(moments: Dict[str, float], t: float) -> float:
    """Normal approximation to P(T > t) from composed moments.

    Deliberately included as a straw man: for heavy-tailed stages this
    understates the tail badly, and the lab shows by how much.
    """
    m, sd = moments["mean"], moments["sd"]
    if sd <= 0:
        return 0.0 if t <= m else 1.0
    z = (t - m) / sd
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def stage_load(wf: QueueWorkflow) -> Dict[str, Dict[str, float]]:
    """Offered load per stage: rho = lambda / (c * mu)."""
    out: Dict[str, Dict[str, float]] = {}
    lam = wf.arrival_rate
    if lam is None:
        return out
    for s in wf.stages:
        rho = lam / (s.capacity * s.service_rate)
        out[s.name] = {
            "rho": rho,
            "stable": rho < 1.0,
            "service_mean": s.service_mean,
            "capacity": float(s.capacity),
        }
    return out


def max_stable_arrival(wf: QueueWorkflow) -> float:
    """The arrival rate beyond which the bottleneck saturates."""
    return min(s.capacity * s.service_rate for s in wf.stages)
