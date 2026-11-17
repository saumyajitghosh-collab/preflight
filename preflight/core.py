"""
preflight.core
==============
An agent workflow is modelled as an absorbing Markov chain.

Transient states  : the steps of the workflow (plan, retrieve, call tool, verify ...)
Absorbing states  : SUCCESS, FAIL, ESCALATE (handed to a human)

Per visit, step i does exactly one of:
    advance  -> step i+1 (or SUCCESS from the last step)   with probability p_ok
    retry    -> step i again                               with probability p_retry
    go back  -> step back_to                               with probability p_back
    escalate -> ESCALATE                                   with probability p_esc
    fail     -> FAIL                                       with the remainder

Everything except latency percentiles is computed exactly, never sampled:

    Q  = transient-to-transient matrix,  R = transient-to-absorbing matrix
    N  = (I - Q)^-1                      fundamental matrix: expected visits
    B  = N R                             absorption probabilities
    t  = N 1                             expected number of steps
    Var(steps) = (2N - I) t - t*t
    E[cost]    = N c

With a step budget K the chain is propagated K times instead (exact transient
analysis); whatever mass is still transient after K steps is FAIL (budget).

solve_batch() does all of the above for a whole stack of parameter draws at
once. That is what makes posterior propagation and value-of-information cheap:
the linear algebra is batched rather than looped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, log
from typing import Sequence

import numpy as np

SUCCESS, FAIL, ESCALATE = 0, 1, 2
OUTCOMES = ("success", "fail", "escalate")


@dataclass
class Step:
    name: str
    p_ok: float
    p_retry: float = 0.0
    p_back: float = 0.0
    back_to: int = 0
    p_esc: float = 0.0
    cost: float = 0.0            # currency units per visit
    latency: float = 1.0         # median seconds per visit
    sigma: float = 0.5           # lognormal shape of latency

    @property
    def p_fail(self) -> float:
        return 1.0 - self.p_ok - self.p_retry - self.p_back - self.p_esc

    def validate(self, n: int) -> None:
        for k in ("p_ok", "p_retry", "p_back", "p_esc"):
            if getattr(self, k) < 0:
                raise ValueError(f"{self.name}: {k} is negative")
        if self.p_fail < -1e-12:
            raise ValueError(f"{self.name}: probabilities sum to more than 1")
        if not 0 <= self.back_to < n:
            raise ValueError(f"{self.name}: back_to out of range")


@dataclass
class Result:
    p_success: float
    p_fail: float
    p_escalate: float
    expected_steps: float
    sd_steps: float
    expected_cost: float
    expected_visits: np.ndarray           # per step
    p_budget_exhausted: float = 0.0       # part of p_fail
    cost_per_success: float = field(init=False)

    def __post_init__(self) -> None:
        self.cost_per_success = (
            self.expected_cost / self.p_success if self.p_success > 0 else float("inf")
        )


class Workflow:
    def __init__(self, steps: Sequence[Step]):
        self.steps = list(steps)
        n = len(self.steps)
        if n == 0:
            raise ValueError("a workflow needs at least one step")
        for s in self.steps:
            s.validate(n)

    def __len__(self) -> int:
        return len(self.steps)

    # ---- matrices ---------------------------------------------------------
    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.steps)
        Q = np.zeros((n, n))
        R = np.zeros((n, 3))
        for i, s in enumerate(self.steps):
            Q[i, i] += s.p_retry
            Q[i, s.back_to] += s.p_back
            if i + 1 < n:
                Q[i, i + 1] += s.p_ok
            else:
                R[i, SUCCESS] += s.p_ok
            R[i, ESCALATE] += s.p_esc
            R[i, FAIL] += max(s.p_fail, 0.0)
        return Q, R

    # ---- exact analysis ---------------------------------------------------
    def solve(self, budget: int | None = None) -> Result:
        """Exact outcome probabilities, step count moments and expected cost."""
        Q, R = self.matrices()
        n = len(self.steps)
        cost = np.array([s.cost for s in self.steps])

        if budget is None:
            N = np.linalg.inv(np.eye(n) - Q)
            B = N @ R
            t = N @ np.ones(n)
            var = (2 * N - np.eye(n)) @ t - t * t
            return Result(
                p_success=B[0, SUCCESS], p_fail=B[0, FAIL], p_escalate=B[0, ESCALATE],
                expected_steps=t[0], sd_steps=float(np.sqrt(max(var[0], 0.0))),
                expected_cost=float(N[0] @ cost), expected_visits=N[0].copy(),
            )

        pi = np.zeros(n); pi[0] = 1.0
        absorbed = np.zeros(3)
        visits = np.zeros(n)
        m1 = m2 = 0.0
        for k in range(1, budget + 1):
            visits += pi
            newly = pi @ R
            absorbed += newly
            pi = pi @ Q
            mass = newly.sum() + (pi.sum() if k == budget else 0.0)
            m1 += k * mass
            m2 += k * k * mass
        left = pi.sum()
        return Result(
            p_success=absorbed[SUCCESS], p_fail=absorbed[FAIL] + left,
            p_escalate=absorbed[ESCALATE], expected_steps=m1,
            sd_steps=float(np.sqrt(max(m2 - m1 * m1, 0.0))),
            expected_cost=float(visits @ cost), expected_visits=visits,
            p_budget_exhausted=left,
        )

    # ---- batched exact analysis ------------------------------------------
    def solve_batch(self, P: np.ndarray, budget: int | None = None) -> dict:
        """Exact analysis for a stack of draws.

        P has shape (m, k, 4) holding (p_ok, p_retry, p_back, p_esc) per draw
        and per step; p_fail is the remainder. Returns arrays of length m.
        """
        P = np.asarray(P, dtype=float)
        m, k, _ = P.shape
        if k != len(self.steps):
            raise ValueError(f"P has {k} steps, workflow has {len(self.steps)}")

        Q = np.zeros((m, k, k))
        R = np.zeros((m, k, 3))
        for i, s in enumerate(self.steps):
            Q[:, i, i] += P[:, i, 1]
            Q[:, i, s.back_to] += P[:, i, 2]
            if i + 1 < k:
                Q[:, i, i + 1] += P[:, i, 0]
            else:
                R[:, i, SUCCESS] += P[:, i, 0]
            R[:, i, ESCALATE] += P[:, i, 3]
            R[:, i, FAIL] += np.clip(1.0 - P[:, i, :4].sum(axis=1), 0.0, 1.0)

        cost = np.array([s.cost for s in self.steps])

        if budget is None:
            eye = np.eye(k)[None, :, :]
            N = np.linalg.inv(eye - Q)
            B = N @ R
            t = N @ np.ones(k)
            return {
                "p_success": B[:, 0, SUCCESS],
                "p_fail": B[:, 0, FAIL],
                "p_escalate": B[:, 0, ESCALATE],
                "expected_steps": t[:, 0],
                "expected_cost": N[:, 0, :] @ cost,
                "expected_visits": N[:, 0, :],
            }

        pi = np.zeros((m, k)); pi[:, 0] = 1.0
        absorbed = np.zeros((m, 3))
        visits = np.zeros((m, k))
        m1 = np.zeros(m); m2 = np.zeros(m)
        for step in range(1, budget + 1):
            visits += pi
            newly = np.einsum("mk,mkj->mj", pi, R)
            absorbed += newly
            pi = np.einsum("mk,mkj->mj", pi, Q)
            mass = newly.sum(axis=1) + (pi.sum(axis=1) if step == budget else 0.0)
            m1 += step * mass
            m2 += step * step * mass
        left = pi.sum(axis=1)
        return {
            "p_success": absorbed[:, SUCCESS],
            "p_fail": absorbed[:, FAIL] + left,
            "p_escalate": absorbed[:, ESCALATE],
            "expected_steps": m1,
            "expected_cost": visits @ cost,
            "expected_visits": visits,
            "p_budget_exhausted": left,
        }

    def sensitivity(self, budget: int | None = None, eps: float = 0.01) -> list[dict]:
        """Change in end-to-end success when one step converts `eps` of its
        failure probability into success. Answers: which step is worth fixing?"""
        base = self.solve(budget).p_success
        out = []
        for i, s in enumerate(self.steps):
            d = min(eps, max(s.p_fail, 0.0))
            if d <= 0:
                out.append({"step": s.name, "d_success_per_point": 0.0})
                continue
            s.p_ok += d
            try:
                lifted = self.solve(budget).p_success
            finally:
                s.p_ok -= d
            out.append({"step": s.name, "d_success_per_point": (lifted - base) / d * 0.01})
        return out

    # ---- Monte Carlo (latency only) --------------------------------------
    def simulate(self, runs: int = 20000, budget: int | None = None,
                 seed: int | None = 0) -> dict:
        """Sample whole trajectories. Returns outcome frequencies plus latency
        and cost samples — the quantities with no closed form."""
        rng = np.random.default_rng(seed)
        n = len(self.steps)
        Q, R = self.matrices()
        P = np.hstack([Q, R])
        P = P / P.sum(axis=1, keepdims=True)
        cum = np.cumsum(P, axis=1)
        mu = np.array([log(s.latency) for s in self.steps])
        sg = np.array([s.sigma for s in self.steps])
        cost = np.array([s.cost for s in self.steps])

        state = np.zeros(runs, dtype=int)
        alive = np.ones(runs, dtype=bool)
        outcome = np.full(runs, FAIL)
        lat = np.zeros(runs); cst = np.zeros(runs); stp = np.zeros(runs, dtype=int)
        k = 0
        while alive.any() and (budget is None or k < budget):
            idx = np.flatnonzero(alive)
            s = state[idx]
            lat[idx] += rng.lognormal(mu[s], sg[s])
            cst[idx] += cost[s]
            stp[idx] += 1
            u = rng.random(idx.size)
            nxt = (u[:, None] > cum[s]).sum(axis=1).clip(max=n + 2)
            done = nxt >= n
            outcome[idx[done]] = nxt[done] - n
            alive[idx[done]] = False
            state[idx[~done]] = nxt[~done]
            k += 1
        freq = np.bincount(outcome, minlength=3) / runs
        q = lambda a, p: float(np.percentile(a, p))
        return {
            "p_success": freq[SUCCESS], "p_fail": freq[FAIL], "p_escalate": freq[ESCALATE],
            "mean_steps": float(stp.mean()), "mean_cost": float(cst.mean()),
            "latency_p50": q(lat, 50), "latency_p95": q(lat, 95), "latency_p99": q(lat, 99),
            "latency": lat, "cost": cst, "outcome": outcome,
            "stderr_success": float(np.sqrt(freq[SUCCESS] * (1 - freq[SUCCESS]) / runs)),
        }


# ---- companions ------------------------------------------------------------
def erlang_c(arrival_rate: float, service_time: float, servers: int) -> dict:
    """M/M/c queue: how many concurrent agent workers (or rate-limit slots)
    does a given load need? Returns utilisation, P(wait) and mean wait."""
    a = arrival_rate * service_time
    rho = a / servers
    if rho >= 1:
        return {"utilisation": rho, "p_wait": 1.0, "mean_wait": float("inf")}
    head = sum(a**k / factorial(k) for k in range(servers))
    tail = a**servers / (factorial(servers) * (1 - rho))
    pw = tail / (head + tail)
    return {"utilisation": rho, "p_wait": pw,
            "mean_wait": pw * service_time / (servers * (1 - rho))}


def verifier_lift(p: float, sensitivity: float, specificity: float,
                  max_attempts: int = 3) -> dict:
    """A step is right with probability p. A checker catches wrong outputs with
    `sensitivity` and passes right ones with `specificity`; rejected outputs
    are regenerated, up to `max_attempts`. Returns what reaches the next step."""
    accept_right = p * specificity
    accept_wrong = (1 - p) * (1 - sensitivity)
    reject = 1 - accept_right - accept_wrong
    geo = (1 - reject**max_attempts) / (1 - reject) if reject < 1 else max_attempts
    return {"p_right_delivered": accept_right * geo,
            "p_wrong_delivered": accept_wrong * geo,
            "p_gave_up": reject**max_attempts,
            "expected_attempts": geo}
