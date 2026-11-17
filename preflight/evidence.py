"""
preflight.evidence
==================
The input side of the problem. Per-step transition probabilities are not known;
they are estimated from counts. This module carries that uncertainty through
the exact solver and reports what it does to the answer.

Evidence per step is a count vector (n_ok, n_retry, n_back, n_esc, n_fail)
observed over n visits. With a Dirichlet(prior, ..., prior) prior the posterior
over the step's transition vector is Dirichlet(counts + prior); each marginal is
Beta, so a single category's credible interval is available in closed form.

propagate()        sample the inputs, solve exactly for every draw (batched),
                   report credible intervals on every output
sobol()            first- and total-order Sobol indices (Jansen estimators) of
                   P(success) with respect to each step's uncertainty
value_of_info()    preposterior: expected narrowing of the success interval if
                   step i were observed `more` further times
littles_law()      conservation check: L = lambda * W must hold for any
                   stationary system; a violation means the model is wrong,
                   not the parameters

The batched solver is the reason all of this is fast enough to be interactive:
propagate() and value_of_info() invert a whole stack of matrices at once
instead of looping over draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import Step, Workflow

CATS = ("ok", "retry", "back", "esc", "fail")


@dataclass
class Evidence:
    """Observed counts for one step, with a Dirichlet prior."""

    name: str
    n_ok: int
    n_retry: int = 0
    n_back: int = 0
    n_esc: int = 0
    n_fail: int = 0
    back_to: int = 0
    cost: float = 0.0
    latency: float = 1.0
    sigma: float = 0.5
    prior: float = 1.0          # Dirichlet concentration; 1 = flat

    @property
    def counts(self) -> np.ndarray:
        return np.array([self.n_ok, self.n_retry, self.n_back, self.n_esc,
                         self.n_fail], float)

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    def alpha(self) -> np.ndarray:
        return self.counts + self.prior

    def mean_step(self) -> Step:
        a = self.alpha(); p = a / a.sum()
        return Step(self.name, p[0], p[1], p[2], self.back_to, p[3],
                    self.cost, self.latency, self.sigma)

    def posterior_mean(self) -> dict:
        p = self.alpha() / self.alpha().sum()
        return {c: float(p[i]) for i, c in enumerate(CATS)}

    def beta_interval(self, cat: str = "ok", level: float = 0.9) -> tuple[float, float]:
        """Closed-form credible interval for one category, via the Beta marginal.

        Uses the Beta quantile through the regularised incomplete beta function,
        implemented by bisection on the continued fraction — no scipy needed.
        """
        i = CATS.index(cat)
        a = self.alpha()[i]
        b = self.alpha().sum() - a
        lo = (1 - level) / 2
        return (_beta_ppf(lo, a, b), _beta_ppf(1 - lo, a, b))

    def sample(self, rng: np.random.Generator, m: int) -> np.ndarray:
        """m draws of (p_ok, p_retry, p_back, p_esc) from the posterior."""
        return rng.dirichlet(self.alpha(), size=m)[:, :4]

    def with_more(self, more: int, rng: np.random.Generator) -> "Evidence":
        """Simulate observing this step `more` further times (preposterior)."""
        p = rng.dirichlet(self.alpha())
        new = rng.multinomial(more, p)
        c = (self.counts + new).astype(int).tolist()
        return Evidence(self.name, *c, back_to=self.back_to, cost=self.cost,
                        latency=self.latency, sigma=self.sigma, prior=self.prior)

    @staticmethod
    def from_steps_and_visits(steps: Sequence[Step], visits: Sequence[int]) -> list["Evidence"]:
        """Turn typed point probabilities plus 'how many times was it observed'
        into pseudo-counts. Use when all you have is a belief and a sample size."""
        out = []
        for s, n in zip(steps, visits):
            c = np.array([s.p_ok, s.p_retry, s.p_back, s.p_esc, max(s.p_fail, 0)]) * n
            out.append(Evidence(s.name, *c.round().astype(int).tolist(),
                                back_to=s.back_to, cost=s.cost,
                                latency=s.latency, sigma=s.sigma))
        return out


# --------------------------------------------------------------------------
# Beta quantile, no scipy
# --------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    from math import lgamma, exp, log
    lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x)
    bt = exp(lbt)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def _beta_ppf(q: float, a: float, b: float, iters: int = 90) -> float:
    """Beta quantile by bisection on the CDF. Plenty accurate for reporting."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Posterior propagation
# --------------------------------------------------------------------------


def _carrier(ev: Sequence[Evidence]) -> Workflow:
    """A workflow holding the structure (back_to, cost, latency) of the steps."""
    return Workflow([e.mean_step() for e in ev])


def _draws(ev: Sequence[Evidence], rng: np.random.Generator, m: int) -> np.ndarray:
    return np.stack([e.sample(rng, m) for e in ev], axis=1)      # (m, k, 4)


def propagate(ev: Sequence[Evidence], budget: int | None = None, draws: int = 4000,
              seed: int = 0, level: float = 0.9) -> dict:
    """Posterior draws through the exact solver; credible intervals on outputs."""
    rng = np.random.default_rng(seed)
    wf = _carrier(ev)
    out = wf.solve_batch(_draws(ev, rng, draws), budget)

    plug = wf.solve(budget)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    res = {"level": level, "draws": draws, "n_obs": [e.n for e in ev]}
    for key, point in (("p_success", plug.p_success),
                       ("p_escalate", plug.p_escalate),
                       ("p_fail", plug.p_fail),
                       ("expected_steps", plug.expected_steps),
                       ("expected_cost", plug.expected_cost)):
        x = np.asarray(out[key], dtype=float)
        res[key] = {"point": float(point), "mean": float(x.mean()),
                    "lo": float(np.percentile(x, lo)), "hi": float(np.percentile(x, hi)),
                    "sd": float(x.std())}
    res["samples_success"] = np.asarray(out["p_success"], dtype=float)
    return res


# --------------------------------------------------------------------------
# Sobol
# --------------------------------------------------------------------------


def sobol(ev: Sequence[Evidence], budget: int | None = None, base: int = 1024,
          seed: int = 0) -> list[dict]:
    """Saltelli scheme with Jansen estimators.

    A, B and the k hybrid matrices A_B^(i), each of size `base`. Jansen's
    first-order estimator is less noisy here than the Saltelli-2010 form, but at
    this sample size first order is still the noisier of the two indices; read
    the total order for ranking and treat S1 as indicative.
    """
    rng = np.random.default_rng(seed)
    wf = _carrier(ev)
    k = len(ev)

    A = _draws(ev, rng, base)
    B = _draws(ev, rng, base)

    fA = wf.solve_batch(A, budget)["p_success"]
    fB = wf.solve_batch(B, budget)["p_success"]
    var = float(np.var(np.concatenate([fA, fB])))
    if var <= 0:
        raise ValueError("output has zero variance across draws; nothing to attribute")

    out = []
    for i in range(k):
        AB = A.copy()
        AB[:, i, :] = B[:, i, :]
        fAB = wf.solve_batch(AB, budget)["p_success"]
        ST = 0.5 * float(np.mean((fA - fAB) ** 2)) / var
        S1 = (var - 0.5 * float(np.mean((fB - fAB) ** 2))) / var
        ST = min(max(ST, 0.0), 1.0)
        S1 = min(max(S1, 0.0), ST)
        out.append({"step": ev[i].name, "n_obs": ev[i].n, "S1": S1, "ST": ST})
    return out


# --------------------------------------------------------------------------
# Value of information
# --------------------------------------------------------------------------


def value_of_info(ev: Sequence[Evidence], more: int = 200, budget: int | None = None,
                  outer: int = 40, draws: int = 600, seed: int = 0,
                  level: float = 0.9) -> list[dict]:
    """Preposterior analysis: for each step, imagine observing it `more` further
    times (outcomes drawn from its current posterior), recompute the success
    interval, average its width. Report the expected shrinkage.

    This is the quantitative form of 'what would settle it', and it routinely
    says the useful thing: the least-observed step is the only one worth
    measuring. Batched solves plus common random numbers keep it fast and
    low-variance.
    """
    rng = np.random.default_rng(seed)
    k = len(ev)
    wf = _carrier(ev)

    now = wf.solve_batch(_draws(ev, rng, draws), budget)["p_success"]
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    width_now = float(np.percentile(now, hi_q) - np.percentile(now, lo_q))

    out = []
    for i, e in enumerate(ev):
        widths = []
        for _ in range(outer):
            e2 = e.with_more(more, rng)
            trial = list(ev)
            trial[i] = e2
            y = wf.solve_batch(_draws(trial, rng, draws), budget)["p_success"]
            widths.append(float(np.percentile(y, hi_q) - np.percentile(y, lo_q)))
        w = float(np.mean(widths))
        out.append({"step": e.name, "n_obs": e.n, "more": more,
                    "width_now": width_now, "width_after": w,
                    "expected_shrink_pts": max(width_now - w, 0.0) * 100})
    return sorted(out, key=lambda d: -d["expected_shrink_pts"])


# --------------------------------------------------------------------------
# Model check and verdict
# --------------------------------------------------------------------------


def littles_law(arrival_rate: float, mean_in_system: float, mean_time_in_system: float,
                tol: float = 0.1) -> dict:
    """L = lambda W. Parameter-free, so a violation indicts the model rather
    than the numbers."""
    implied = arrival_rate * mean_time_in_system
    rel = abs(mean_in_system - implied) / max(implied, 1e-12)
    return {"L_observed": mean_in_system, "L_implied": implied, "relative_error": rel,
            "consistent": rel <= tol,
            "note": "" if rel <= tol else
            "L != lambda*W: the system is not stationary over the window, or "
            "arrivals and completions are being counted differently. Fix the "
            "model before trusting parameters."}


SUPPORTED, CONTRADICTED, UNDERDETERMINED = "supported", "contradicted", "underdetermined"


def verdict(prop: dict, claim: float | None = None, threshold: float = 0.1) -> dict:
    """Classify a claim about P(success) against the posterior interval.

    Three outcomes, and the third is the honest one: if the interval is wide
    enough that the claim could go either way, say so and say what would
    settle it.
    """
    s = prop["p_success"]
    width = s["hi"] - s["lo"]
    if claim is None:
        status = UNDERDETERMINED if width > threshold else SUPPORTED
        return {"status": status, "claim": None, "interval": [s["lo"], s["hi"]],
                "width": width, "point": s["point"]}
    if claim < s["lo"] or claim > s["hi"]:
        status = CONTRADICTED
    elif width > threshold:
        status = UNDERDETERMINED
    else:
        status = SUPPORTED
    return {"status": status, "claim": claim, "interval": [s["lo"], s["hi"]],
            "width": width, "point": s["point"]}
