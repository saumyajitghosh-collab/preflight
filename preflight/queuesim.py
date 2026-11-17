"""Discrete-event simulation of an agent workflow.

A hand-rolled event loop (heapq) rather than SimPy, so the library has no
dependency beyond numpy and the scheduling semantics are explicit and testable.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .queue import QueueWorkflow

ARRIVE = 0
COMPLETE = 1


@dataclass
class QueueStageStats:
    name: str
    waits: List[float] = field(default_factory=list)
    services: List[float] = field(default_factory=list)
    attempts: int = 0
    failures: int = 0
    timeouts: int = 0
    busy_time: float = 0.0
    utilisation: float = 0.0

    @property
    def mean_wait(self) -> float:
        return float(np.mean(self.waits)) if self.waits else 0.0

    @property
    def p95_wait(self) -> float:
        return float(np.quantile(self.waits, 0.95)) if self.waits else 0.0


@dataclass
class SimResult:
    e2e: np.ndarray
    failed_e2e: np.ndarray
    cost: np.ndarray
    n_tasks: int
    horizon: float
    arrival_rate: float
    stage_stats: Dict[str, QueueStageStats]
    mean_queue_length: Dict[str, float]

    # -- basic accessors ---------------------------------------------------

    @property
    def n_failed(self) -> int:
        return int(self.failed_e2e.size)

    @property
    def failure_prob(self) -> float:
        return float(self.n_failed / max(self.n_tasks, 1))

    def latency(self, include_failures: bool = False) -> np.ndarray:
        if include_failures and self.n_failed:
            return np.concatenate([self.e2e, self.failed_e2e])
        return self.e2e

    def p(self, q: float, include_failures: bool = False) -> float:
        x = self.latency(include_failures)
        return float(np.quantile(x, q)) if x.size else float("nan")

    def tail_prob(self, t: float, include_failures: bool = False) -> float:
        x = self.latency(include_failures)
        return float(np.mean(x > t)) if x.size else float("nan")

    def summary(self) -> dict:
        x = self.latency()
        return {
            "mean": float(np.mean(x)) if x.size else float("nan"),
            "sd": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
            "p50": self.p(0.50),
            "p95": self.p(0.95),
            "p99": self.p(0.99),
            "max": float(np.max(x)) if x.size else float("nan"),
            "failure_prob": self.failure_prob,
            "cost_mean": float(np.mean(self.cost)) if self.cost.size else float("nan"),
            "horizon": self.horizon,
        }


def simulate(
    wf: QueueWorkflow,
    n_tasks: Optional[int] = None,
    seed: Optional[int] = None,
) -> SimResult:
    """Run one replication of the workflow and return the sample paths."""
    rng = np.random.default_rng(wf.seed if seed is None else seed)
    n = int(n_tasks if n_tasks is not None else wf.n_tasks)

    stages = wf.by_name()
    succ = wf.successors()
    roots = wf.roots()

    busy = {k: 0 for k in stages}
    queues: Dict[str, List[tuple]] = {k: [] for k in stages}
    stats = {k: QueueStageStats(k) for k in stages}
    q_integral = {k: 0.0 for k in stages}
    q_last = {k: 0.0 for k in stages}

    done: List[set] = [set() for _ in range(n)]
    t_start = np.zeros(n)
    e2e = np.full(n, np.nan)
    e2e_failed = np.full(n, np.nan)
    cost = np.zeros(n)
    dead = np.zeros(n, dtype=bool)

    heap: List[tuple] = []
    now = 0.0
    if wf.arrival_rate is None:
        for i in range(n):
            heapq.heappush(heap, (0.0, ARRIVE, i, None, 0, 0.0, 0.0, False))
    else:
        for i in range(n):
            now += float(rng.exponential(1.0 / wf.arrival_rate))
            heapq.heappush(heap, (now, ARRIVE, i, None, 0, 0.0, 0.0, False))

    def flush_q(name: str, t: float) -> None:
        q_integral[name] += (t - q_last[name]) * len(queues[name])
        q_last[name] = t

    def start_service(name, task, attempt, enq, t):
        st = stages[name]
        busy[name] += 1
        stats[name].waits.append(t - enq)
        svc = float(st.service.sample(rng))
        if attempt > 0:
            svc += st.retry_backoff * attempt
        timed_out = st.timeout is not None and svc > st.timeout
        eff = min(svc, st.timeout) if st.timeout is not None else svc
        stats[name].busy_time += eff
        heapq.heappush(heap, (t + eff, COMPLETE, task, name, attempt, svc, 0.0, timed_out))

    def enqueue(name, task, attempt, t):
        flush_q(name, t)
        if busy[name] < stages[name].capacity:
            start_service(name, task, attempt, t, t)
        else:
            queues[name].append((task, attempt, t))

    def dispatch(name, t):
        flush_q(name, t)
        if queues[name] and busy[name] < stages[name].capacity:
            task, attempt, enq = queues[name].pop(0)
            start_service(name, task, attempt, enq, t)

    horizon = 0.0
    while heap:
        t, kind, task, name, attempt, svc, _enq, timed_out = heapq.heappop(heap)
        horizon = max(horizon, t)

        if kind == ARRIVE:
            t_start[task] = t
            for r in roots:
                enqueue(r, task, 0, t)
            continue

        # ---- completion of one service at `name` ----
        st = stages[name]
        busy[name] -= 1
        stats[name].attempts += 1
        stats[name].services.append(svc)
        cost[task] += st.cost_per_attempt

        if timed_out:
            stats[name].timeouts += 1
            ok = False
        else:
            ok = rng.random() >= st.p_fail
            if not ok:
                stats[name].failures += 1

        if ok:
            done[task].add(name)
            for nxt in succ[name]:
                if not dead[task] and all(d in done[task] for d in stages[nxt].deps):
                    enqueue(nxt, task, 0, t)
            if not dead[task] and len(done[task]) == len(stages):
                e2e[task] = t - t_start[task]
        else:
            if attempt < st.max_retries and not dead[task]:
                enqueue(name, task, attempt + 1, t)
            else:
                dead[task] = True
                e2e_failed[task] = t - t_start[task]

        dispatch(name, t)

    for name, spec in stages.items():
        flush_q(name, horizon)
        st = stats[name]
        denom = spec.capacity * horizon
        st.utilisation = float(st.busy_time / denom) if denom > 0 else float("nan")

    mean_q = {k: (q_integral[k] / horizon if horizon > 0 else 0.0) for k in stages}

    ok_mask = ~np.isnan(e2e)
    fail_mask = ~np.isnan(e2e_failed)

    lam = wf.arrival_rate
    if lam is None:
        lam = n / horizon if horizon > 0 else float("nan")

    return SimResult(
        e2e=e2e[ok_mask],
        failed_e2e=e2e_failed[fail_mask],
        cost=cost,
        n_tasks=n,
        horizon=float(horizon),
        arrival_rate=float(lam),
        stage_stats=stats,
        mean_queue_length=mean_q,
    )


def replications(
    wf: QueueWorkflow,
    reps: int = 20,
    n_tasks: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> List[SimResult]:
    """Independent replications with a common base seed for reproducibility."""
    base = wf.seed if base_seed is None else base_seed
    return [simulate(wf, n_tasks=n_tasks, seed=base + 7919 * k) for k in range(reps)]
