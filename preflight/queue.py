"""Queueing model: service-time distributions, stages, and the DAG.

An agent workflow is modelled as a queueing network. Every stage is a service
station with a finite number of servers (concurrency limit), a service-time
distribution, and an optional failure/retry policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Service-time distributions
# --------------------------------------------------------------------------


class Dist:
    """Base class for a service-time distribution."""

    name = "dist"

    def sample(self, rng: np.random.Generator, size: Optional[int] = None):
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def var(self) -> float:
        raise NotImplementedError

    def cv(self) -> float:
        m = self.mean()
        if m <= 0:
            return float("inf")
        return float(np.sqrt(self.var()) / m)

    def describe(self) -> str:
        return self.name


class Lognormal(Dist):
    """Heavy right tail. The realistic default for LLM and network latency."""

    name = "lognormal"

    def __init__(self, median: float, sigma: float):
        if median <= 0:
            raise ValueError("median must be positive")
        self.median = float(median)
        self.sigma = float(sigma)

    def sample(self, rng, size=None):
        out = rng.lognormal(mean=np.log(self.median), sigma=self.sigma, size=size)
        return float(out) if size is None else out

    def mean(self) -> float:
        return self.median * float(np.exp(self.sigma ** 2 / 2))

    def var(self) -> float:
        s2 = self.sigma ** 2
        return (self.median ** 2) * (np.exp(s2) - 1.0) * float(np.exp(s2))

    def describe(self) -> str:
        return f"lognormal(median={self.median:g}, sigma={self.sigma:g})"


class Exponential(Dist):
    """Markovian. Kept so the analytic layer's assumptions can be violated on purpose."""

    name = "exponential"

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)

    def sample(self, rng, size=None):
        out = rng.exponential(1.0 / self.rate, size=size)
        return float(out) if size is None else out

    def mean(self) -> float:
        return 1.0 / self.rate

    def var(self) -> float:
        return 1.0 / (self.rate ** 2)

    def describe(self) -> str:
        return f"exponential(mean={self.mean():g})"


class Gamma(Dist):
    """Shape < 1 gives CV > 1; shape > 1 gives CV < 1."""

    name = "gamma"

    def __init__(self, shape: float, scale: float):
        self.shape = float(shape)
        self.scale = float(scale)

    def sample(self, rng, size=None):
        out = rng.gamma(self.shape, self.scale, size=size)
        return float(out) if size is None else out

    def mean(self) -> float:
        return self.shape * self.scale

    def var(self) -> float:
        return self.shape * self.scale ** 2

    def describe(self) -> str:
        return f"gamma(shape={self.shape:g}, scale={self.scale:g})"


class Deterministic(Dist):
    """Zero variance. Useful as a limit test."""

    name = "deterministic"

    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, rng, size=None):
        return self.value if size is None else np.full(size, self.value)

    def mean(self) -> float:
        return self.value

    def var(self) -> float:
        return 0.0

    def describe(self) -> str:
        return f"deterministic({self.value:g})"


# --------------------------------------------------------------------------
# Stages and workflows
# --------------------------------------------------------------------------


@dataclass
class QueueStage:
    """One service station in the workflow."""

    name: str
    service: Dist
    capacity: int = 1
    p_fail: float = 0.0
    max_retries: int = 0
    retry_backoff: float = 0.0
    timeout: Optional[float] = None
    deps: Sequence[str] = field(default_factory=tuple)
    cost_per_attempt: float = 0.0

    def __post_init__(self):
        self.deps = tuple(self.deps)
        if self.capacity < 1:
            raise ValueError(f"stage {self.name!r}: capacity must be >= 1")
        if not 0.0 <= self.p_fail <= 1.0:
            raise ValueError(f"stage {self.name!r}: p_fail must be in [0, 1]")

    @property
    def service_mean(self) -> float:
        return self.service.mean()

    @property
    def service_rate(self) -> float:
        return 1.0 / self.service_mean


@dataclass
class QueueWorkflow:
    """A DAG of stages plus an arrival process."""

    stages: List[QueueStage]
    arrival_rate: Optional[float] = None
    n_tasks: int = 2000
    seed: int = 0
    label: str = "workflow"

    def __post_init__(self):
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        known = set(names)
        for s in self.stages:
            for d in s.deps:
                if d not in known:
                    raise ValueError(f"stage {s.name!r} depends on unknown stage {d!r}")
        self.topo_order()  # raises on cycles

    def by_name(self) -> Dict[str, QueueStage]:
        return {s.name: s for s in self.stages}

    def successors(self) -> Dict[str, List[str]]:
        succ: Dict[str, List[str]] = {s.name: [] for s in self.stages}
        for s in self.stages:
            for d in s.deps:
                succ[d].append(s.name)
        return succ

    def roots(self) -> List[str]:
        return [s.name for s in self.stages if not s.deps]

    def topo_order(self) -> List[str]:
        by = self.by_name()
        indeg = {n: len(by[n].deps) for n in by}
        ready = [n for n, d in indeg.items() if d == 0]
        order: List[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in self.successors()[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        if len(order) != len(by):
            raise ValueError("workflow contains a cycle")
        return order

    def bottleneck(self) -> QueueStage:
        """The stage with the highest offered load, assuming serial flow."""
        if self.arrival_rate is None:
            return max(self.stages, key=lambda s: s.service_mean / s.capacity)
        return max(
            self.stages,
            key=lambda s: (self.arrival_rate / (s.capacity * s.service_rate)),
        )
