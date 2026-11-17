"""Evidence mode: counts in, intervals and verdicts out."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import preflight as pf
from preflight import Evidence, Step, Workflow
from preflight.evidence import _beta_ppf, _betainc


def ev(n=2000):
    return [
        Evidence("Plan", int(0.95 * n), int(0.03 * n), 0, 0, int(0.02 * n)),
        Evidence("Retrieve", int(0.99 * n), int(0.01 * n), 0, 0, 0),
        Evidence("Call tool", int(0.90 * n), int(0.05 * n), 0, 0, int(0.05 * n), cost=1.0),
        Evidence("Verify", int(0.85 * n), 0, int(0.10 * n), int(0.03 * n), int(0.02 * n),
                 back_to=2),
    ]


# --------------------------------------------------------------------------
# Beta marginal
# --------------------------------------------------------------------------


def test_betainc_known_values():
    # Beta(1,1) is uniform: I_x = x
    for x in (0.1, 0.5, 0.9):
        assert _betainc(1, 1, x) == pytest.approx(x, abs=1e-6)
    # Beta(2,1) CDF is x^2
    assert _betainc(2, 1, 0.5) == pytest.approx(0.25, abs=1e-6)


def test_beta_interval_brackets_the_mean():
    e = Evidence("s", 90, 0, 0, 0, 10)
    lo, hi = e.beta_interval("ok", 0.9)
    mean = e.posterior_mean()["ok"]
    assert lo < mean < hi
    assert 0 <= lo and hi <= 1


def test_beta_interval_narrows_with_n():
    small = Evidence("s", 9, 0, 0, 0, 1).beta_interval("ok")
    big = Evidence("s", 900, 0, 0, 0, 100).beta_interval("ok")
    assert (big[1] - big[0]) < (small[1] - small[0]) / 5


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------


def test_interval_contains_point_and_shrinks_with_data():
    small = pf.propagate(ev(20), draws=1500)
    big = pf.propagate(ev(2000), draws=1500)
    for p in (small, big):
        s = p["p_success"]
        assert s["lo"] <= s["point"] <= s["hi"]
    assert (big["p_success"]["hi"] - big["p_success"]["lo"]) < \
           (small["p_success"]["hi"] - small["p_success"]["lo"])


def test_propagate_point_matches_exact_solve_of_posterior_means():
    e = ev(1000)
    wf = Workflow([x.mean_step() for x in e])
    p = pf.propagate(e, draws=50)
    assert p["p_success"]["point"] == pytest.approx(wf.solve().p_success, rel=1e-12)


def test_posterior_interval_collapses_to_point_with_huge_n():
    e = [Evidence("s", 990000, 0, 0, 0, 10000)]
    p = pf.propagate(e, draws=2000)
    s = p["p_success"]
    assert (s["hi"] - s["lo"]) < 0.005
    assert s["point"] == pytest.approx(0.99, abs=0.002)


def test_budget_is_respected_in_propagation():
    e = ev(2000)
    full = pf.propagate(e, budget=None, draws=800)
    tight = pf.propagate(e, budget=4, draws=800)
    assert tight["p_success"]["point"] < full["p_success"]["point"]


# --------------------------------------------------------------------------
# Sobol
# --------------------------------------------------------------------------


def test_sobol_finds_the_uncertain_step():
    e = ev(5000)
    e[1] = Evidence("Retrieve", 9, 0, 0, 0, 1)      # only 10 observations
    s = pf.sobol(e, base=512)
    worst = max(s, key=lambda d: d["ST"])
    assert worst["step"] == "Retrieve"
    assert worst["ST"] > 0.8
    for r in s:
        assert 0 <= r["S1"] <= r["ST"] <= 1.0


def test_sobol_indices_sum_to_about_one_plus_interaction():
    e = ev(2000)
    s = pf.sobol(e, base=512)
    total = sum(r["ST"] for r in s)
    assert 0.9 <= total <= 1.4          # interactions push it slightly above 1


# --------------------------------------------------------------------------
# Value of information
# --------------------------------------------------------------------------


def test_voi_prefers_the_least_observed_step():
    e = ev(5000)
    e[2] = Evidence("Call tool", 9, 1, 0, 0, 0, cost=1.0)   # 10 observations
    v = pf.value_of_info(e, more=200, outer=12, draws=200)
    assert v[0]["step"] == "Call tool"
    assert v[0]["expected_shrink_pts"] > 0
    assert v[0]["expected_shrink_pts"] > 5 * max(
        sum(r["expected_shrink_pts"] for r in v[1:]), 1e-9)


def test_voi_reports_no_gain_when_everything_is_observed():
    e = [Evidence("s", 100000, 0, 0, 0, 100)]
    v = pf.value_of_info(e, more=200, outer=6, draws=200)
    assert v[0]["expected_shrink_pts"] < 0.5


# --------------------------------------------------------------------------
# Model check and verdict
# --------------------------------------------------------------------------


def test_littles_law():
    assert pf.littles_law(2.0, 22.0, 11.0)["consistent"]
    assert not pf.littles_law(2.0, 40.0, 11.0)["consistent"]


def test_verdict_three_way():
    p = {"p_success": {"lo": 0.70, "hi": 0.80, "point": 0.75}}
    assert pf.verdict(p, 0.90)["status"] == pf.CONTRADICTED
    assert pf.verdict(p, 0.60)["status"] == pf.CONTRADICTED
    assert pf.verdict(p, 0.75, threshold=0.05)["status"] == pf.UNDERDETERMINED
    narrow = {"p_success": {"lo": 0.745, "hi": 0.755, "point": 0.75}}
    assert pf.verdict(narrow, 0.75, threshold=0.05)["status"] == pf.SUPPORTED


def test_verdict_without_claim_flags_wide_intervals():
    wide = {"p_success": {"lo": 0.60, "hi": 0.85, "point": 0.72}}
    assert pf.verdict(wide)["status"] == pf.UNDERDETERMINED
    tight = {"p_success": {"lo": 0.79, "hi": 0.81, "point": 0.80}}
    assert pf.verdict(tight)["status"] == pf.SUPPORTED


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------


def test_evidence_ledger_flags_thin_data():
    e = ev(2000)
    e[0] = Evidence("Plan", 3, 1, 0, 0, 1)          # 5 observations
    led = pf.build_evidence_ledger(e, pf.propagate(e, draws=200))
    assert led.get("counts").status == pf.VIOLATED
    assert led.get("prior").status == pf.VIOLATED


def test_evidence_ledger_passes_with_enough_data():
    led = pf.build_evidence_ledger(ev(20000), pf.propagate(ev(20000), draws=200))
    assert led.get("counts").status == pf.HOLDS
    assert led.get("sampler").status == pf.HOLDS


# --------------------------------------------------------------------------
# Round trip from typed probabilities
# --------------------------------------------------------------------------


def test_from_steps_and_visits_round_trips():
    steps = [Step("a", 0.9, p_retry=0.05), Step("b", 0.8, p_esc=0.05)]
    e = Evidence.from_steps_and_visits(steps, [10000, 10000])
    wf = Workflow([x.mean_step() for x in e])
    assert wf.solve().p_success == pytest.approx(Workflow(steps).solve().p_success, abs=0.01)
