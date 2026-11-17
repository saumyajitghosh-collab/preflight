"""The queueing half: contention, bounds and the simulation ledger."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import preflight as ad
from preflight.queue import QueueStage as Stage, QueueWorkflow as Workflow


# --------------------------------------------------------------------------
# Analytic layer
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def test_mm1_matches_closed_form():
    """M/M/1 with lam=4, mu=5: Wq = rho/(mu-lam) = 0.8, W = 1.0, L = 4.0."""
    m = ad.mm1(lam=4.0, mu=5.0)
    assert m.stable
    assert m.rho == pytest.approx(0.8)
    assert m.wq == pytest.approx(0.8, rel=1e-9)
    assert m.w == pytest.approx(1.0, rel=1e-9)
    assert m.lq == pytest.approx(3.2, rel=1e-9)
    assert m.l == pytest.approx(4.0, rel=1e-9)


def test_mmc_p_wait_matches_erlang_c():
    """M/M/2 with a=1.6: P(wait) = 6.4/(2.6+6.4) = 0.7111, from two routes."""
    a, c, rho = 1.6, 2, 0.8
    num = a ** c / (2 * (1 - rho))
    den = sum(a ** n / float(math.factorial(n)) for n in range(c)) + num
    assert ad.mmc(1.6, 1.0, 2).p_wait == pytest.approx(num / den, rel=1e-9)
    # core.erlang_c is the agent-facing form: (arrival rate, service time, servers)
    assert ad.erlang_c(1.6, 1.0, 2)["p_wait"] == pytest.approx(0.7111, abs=2e-4)


def test_mmc_reduces_to_mm1():
    m1 = ad.mm1(0.6, 1.0)
    mc = ad.mmc(0.6, 1.0, 1)
    assert mc.wq == pytest.approx(m1.wq, rel=1e-9)


def test_unstable_load_reports_infinite():
    m = ad.mm1(lam=6.0, mu=5.0)
    assert not m.stable
    assert np.isinf(m.wq)


def test_kingman_reduces_to_mm1_when_exponential():
    """With ca2 = cs2 = 1, Kingman gives the M/M/1 wait exactly."""
    lam, svc = 0.6, 1.0
    mm1_wq = ad.mm1(lam, 1.0 / svc).wq
    assert ad.kingman_wq(lam, svc, 1.0, 1.0) == pytest.approx(mm1_wq, rel=1e-9)


def test_kingman_grows_with_variability():
    low = ad.kingman_wq(0.6, 1.0, 0.25, 0.25)
    high = ad.kingman_wq(0.6, 1.0, 4.0, 4.0)
    assert high > low


def test_serial_moments():
    wf = Workflow(stages=[
        Stage("a", ad.Deterministic(1.0)),
        Stage("b", ad.Deterministic(2.0), deps=("a",)),
    ])
    sm = ad.serial_moments(wf.stages)
    assert sm["mean"] == pytest.approx(3.0)
    assert sm["sd"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


def test_cantelli_is_valid_and_nontrivial():
    rng = np.random.default_rng(0)
    x = rng.lognormal(0.0, 1.0, 200_000)
    t = 8.0
    emp = float(np.mean(x > t))
    bnd = ad.cantelli_upper(float(np.mean(x)), float(np.var(x, ddof=1)), t)
    assert bnd >= emp * 0.95           # a bound must not be violated
    assert bnd < 1.0                    # and must say something


def test_markov_bound():
    assert ad.markov_upper(1.0, 2.0) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(1)
    x = rng.exponential(2.0, 5000)
    ci = ad.bootstrap_ci(x, np.mean)
    assert ci["lo"] < 2.0 < ci["hi"]
    assert ci["width"] < 0.3


# --------------------------------------------------------------------------
# Simulator: verify against theory
# --------------------------------------------------------------------------


def test_simulation_reproduces_mm1_wait():
    """The event loop must land on the analytic M/M/1 answer."""
    lam, mu = 0.7, 1.0
    wf = Workflow(
        arrival_rate=lam, n_tasks=40_000, seed=5,
        stages=[Stage("s", ad.Exponential(mu), capacity=1)],
    )
    res = ad.simulate(wf)
    theory = ad.mm1(lam, mu)
    # total time in system = queue + service
    observed = float(np.mean(res.e2e))
    assert observed == pytest.approx(theory.w, rel=0.12)


def test_little_law_holds_in_simulation():
    wf = Workflow(
        arrival_rate=0.8, n_tasks=30_000, seed=9,
        stages=[Stage("s", ad.Exponential(1.0), capacity=1)],
    )
    res = ad.simulate(wf)
    led = ad.build_ledger(wf, res)
    assert led.get("little").status == ad.HOLDS


def test_conservation_all_tasks_accounted_for():
    wf = Workflow(
        arrival_rate=1.0, n_tasks=2000, seed=2,
        stages=[Stage("a", ad.Lognormal(0.5, 0.4), capacity=4, p_fail=0.3, max_retries=1),
                Stage("b", ad.Lognormal(0.5, 0.4), capacity=4, deps=("a",))],
    )
    res = ad.simulate(wf)
    assert res.e2e.size + res.failed_e2e.size == 2000


def test_capacity_reduces_wait():
    def run(c):
        wf = Workflow(
            arrival_rate=1.5, n_tasks=6000, seed=4,
            stages=[Stage("s", ad.Lognormal(0.5, 0.5), capacity=c)],
        )
        return ad.simulate(wf).p(0.95)

    assert run(4) < run(2)


def test_retries_increase_load():
    def run(retries):
        wf = Workflow(
            arrival_rate=0.9, n_tasks=6000, seed=6,
            stages=[Stage("s", ad.Exponential(1.0), capacity=1,
                             p_fail=0.25, max_retries=retries, retry_backoff=0.1)],
        )
        return ad.simulate(wf).p(0.95)

    assert run(3) > run(0)


