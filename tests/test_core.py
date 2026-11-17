"""The exact solver must be exact. These tests check the maths, not the plumbing."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import preflight as pf
from preflight import Step, Workflow


def test_single_step_success_is_p_ok():
    wf = Workflow([Step("only", p_ok=0.9, p_esc=0.05)])
    r = wf.solve()
    assert r.p_success == pytest.approx(0.9)
    assert r.p_escalate == pytest.approx(0.05)
    assert r.p_fail == pytest.approx(0.05)


def test_serial_success_is_the_product():
    steps = [Step("a", 0.9), Step("b", 0.8), Step("c", 0.7)]
    r = Workflow(steps).solve()
    assert r.p_success == pytest.approx(0.9 * 0.8 * 0.7)
    # a run can die early, so expected steps is below 3
    assert r.expected_steps == pytest.approx(1 + 0.9 + 0.9 * 0.8)


def test_retry_loop_has_closed_form():
    """One step, p_ok = p, p_retry = r, no other exits.

    Success per attempt is p, retries are geometric, so P(success) = 1 when
    p + r = 1 (retries are unlimited) and the expected visits are 1/p.
    """
    p, r_ = 0.5, 0.5
    wf = Workflow([Step("s", p_ok=p, p_retry=r_)])
    res = wf.solve()
    assert res.p_success == pytest.approx(1.0)
    assert res.expected_steps == pytest.approx(1.0 / p)
    assert res.expected_visits[0] == pytest.approx(1.0 / p)


def test_retry_with_failure_has_closed_form():
    """p_ok + p_retry + p_fail = 1: success = p/(p+f), visits = 1/(p+f)."""
    p, r_, f = 0.4, 0.4, 0.2
    wf = Workflow([Step("s", p_ok=p, p_retry=r_, p_esc=0.0)])
    res = wf.solve()
    assert res.p_success == pytest.approx(p / (p + f))
    assert res.expected_visits[0] == pytest.approx(1.0 / (p + f))


def test_back_edge_matches_hand_computation():
    """a -> b, and b goes back to a with prob 0.5, else succeeds.

    Each a-visit reaches b with certainty; each b-visit succeeds w.p. 0.5 and
    otherwise restarts. P(success) = 1 by eventual absorption; expected visits
    of a = 1/(1-0.5) = 2.
    """
    wf = Workflow([
        Step("a", p_ok=1.0),
        Step("b", p_ok=0.5, p_back=0.5, back_to=0),
    ])
    res = wf.solve()
    assert res.p_success == pytest.approx(1.0)
    assert res.expected_visits[0] == pytest.approx(2.0)
    assert res.expected_steps == pytest.approx(4.0)


def test_budget_truncation_reduces_success():
    wf = Workflow([Step("a", 0.9, p_retry=0.1)])
    full = wf.solve().p_success
    tight = wf.solve(budget=3)
    assert tight.p_success < full
    assert tight.p_budget_exhausted > 0
    assert tight.p_success + tight.p_fail + tight.p_escalate == pytest.approx(1.0)


def test_expected_cost_accounts_for_retries():
    wf = Workflow([Step("s", p_ok=0.5, p_retry=0.5, cost=2.0)])
    # expected visits = 2, so expected cost = 4
    assert wf.solve().expected_cost == pytest.approx(4.0)


def test_solve_batch_matches_solve():
    """The batched solver is the fast path; it must equal the scalar one."""
    steps = [Step("a", 0.95, p_retry=0.03), Step("b", 0.9, p_back=0.05, back_to=0, cost=1.0),
             Step("c", 0.85, p_esc=0.02)]
    wf = Workflow(steps)
    P = np.array([[[s.p_ok, s.p_retry, s.p_back, s.p_esc] for s in steps]])
    batched = wf.solve_batch(P)
    scalar = wf.solve()
    assert batched["p_success"][0] == pytest.approx(scalar.p_success, rel=1e-12)
    assert batched["p_fail"][0] == pytest.approx(scalar.p_fail, rel=1e-12)
    assert batched["p_escalate"][0] == pytest.approx(scalar.p_escalate, rel=1e-12)
    assert batched["expected_steps"][0] == pytest.approx(scalar.expected_steps, rel=1e-12)
    assert batched["expected_cost"][0] == pytest.approx(scalar.expected_cost, rel=1e-12)


def test_solve_batch_matches_solve_with_budget():
    steps = [Step("a", 0.9, p_retry=0.1), Step("b", 0.8, p_esc=0.05)]
    wf = Workflow(steps)
    P = np.array([[[s.p_ok, s.p_retry, s.p_back, s.p_esc] for s in steps]])
    batched = wf.solve_batch(P, budget=6)
    scalar = wf.solve(budget=6)
    assert batched["p_success"][0] == pytest.approx(scalar.p_success, rel=1e-12)
    assert batched["p_fail"][0] == pytest.approx(scalar.p_fail, rel=1e-12)


def test_monte_carlo_agrees_with_exact():
    wf = Workflow([Step("a", 0.95, p_retry=0.03), Step("b", 0.9, p_back=0.05, back_to=0),
                   Step("c", 0.85, p_esc=0.02)])
    exact = wf.solve().p_success
    mc = wf.simulate(runs=40000, seed=1)["p_success"]
    assert mc == pytest.approx(exact, abs=0.01)


def test_erlang_c():
    q = pf.erlang_c(arrival_rate=1.6, service_time=1.0, servers=2)
    assert q["utilisation"] == pytest.approx(0.8)
    assert q["p_wait"] == pytest.approx(0.7111, abs=2e-4)
    assert pf.erlang_c(2.0, 1.0, 2)["mean_wait"] == float("inf")


def test_verifier_lift_monotone():
    weak = pf.verifier_lift(0.6, sensitivity=0.5, specificity=0.9)
    strong = pf.verifier_lift(0.6, sensitivity=0.95, specificity=0.95)
    assert strong["p_right_delivered"] > weak["p_right_delivered"]
    assert strong["p_wrong_delivered"] < weak["p_wrong_delivered"]


def test_invalid_step_rejected():
    with pytest.raises(ValueError):
        Workflow([Step("bad", p_ok=0.9, p_esc=0.2)])
