"""Text reports: counts in, verdict out."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .analytic import mmc, serial_moments, stage_load
from .bounds import cantelli_upper
from .core import Workflow
from .evidence import Evidence, propagate, sobol, value_of_info, verdict
from .ledger import Ledger, build_evidence_ledger

BAR = "=" * 74
THIN = "-" * 74


def _f(x: float, nd: int = 3) -> str:
    return "inf" if not np.isfinite(x) else f"{x:.{nd}f}"


def report_evidence(
    ev: Sequence[Evidence],
    budget: Optional[int] = None,
    claim: Optional[float] = None,
    level: float = 0.9,
    draws: int = 4000,
    base: int = 1024,
    more: int = 200,
    ledger: Optional[Ledger] = None,
    with_voi: bool = True,
) -> str:
    """The full counts-based report: intervals, verdict, Sobol, value of information."""
    out: List[str] = [BAR, "PREFLIGHT — evidence-based reliability", BAR]

    out.append("EVIDENCE (counts observed per step)")
    for e in ev:
        pm = e.posterior_mean()
        lo, hi = e.beta_interval("ok", level)
        out.append(
            f"  {e.name:<14} n={e.n:<6} p_ok {pm['ok']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]   fail {pm['fail']:.4f}  "
            f"retry {pm['retry']:.3f}  back {pm['back']:.3f}  esc {pm['esc']:.4f}"
        )
    out.append("")

    plug = Workflow([e.mean_step() for e in ev]).solve(budget)
    out.append("EXACT SOLVE (plug-in posterior means, no sampling)")
    out.append(
        f"  P(success) {_f(plug.p_success, 4)}   P(fail) {_f(plug.p_fail, 4)}   "
        f"P(escalate) {_f(plug.p_escalate, 4)}"
    )
    out.append(
        f"  expected steps {_f(plug.expected_steps, 2)} (sd {_f(plug.sd_steps, 2)})   "
        f"expected cost {_f(plug.expected_cost, 2)}   "
        f"cost per success {_f(plug.cost_per_success, 2)}"
    )
    if budget is not None:
        out.append(f"  budget {budget} steps: {_f(plug.p_budget_exhausted, 4)} of runs run out")
    out.append("")

    p = propagate(ev, budget, draws=draws, level=level)
    s = p["p_success"]
    out.append(f"POSTERIOR INTERVALS ({int(level * 100)}%, {draws} draws)")
    out.append(
        f"  P(success) {s['point']:.4f}   interval [{s['lo']:.4f}, {s['hi']:.4f}]   "
        f"width {(s['hi'] - s['lo']) * 100:.1f} pts"
    )
    for k in ("p_fail", "p_escalate", "expected_steps", "expected_cost"):
        d = p[k]
        out.append(f"  {k:<15} {d['point']:>9.4f}   [{d['lo']:.4f}, {d['hi']:.4f}]")
    out.append("")

    v = verdict(p, claim)
    head = v["status"].upper()
    out.append("VERDICT")
    if claim is not None:
        out.append(f"  claim: P(success) = {claim:.3f}")
    out.append(f"  {head}")
    out.append(f"  evidence: {v['point']:.4f} with interval [{v['interval'][0]:.4f}, "
               f"{v['interval'][1]:.4f}]")
    if v["status"] == "contradicted":
        side = "above" if claim > v["interval"][1] else "below"
        out.append(f"  why: the claimed rate lies {side} the entire interval")
    elif v["status"] == "underdetermined":
        out.append(f"  why: the interval is {(v['width']) * 100:.1f} pts wide — "
                   f"wide enough that the claim could go either way")
    out.append("")

    try:
        so = sobol(ev, budget, base=base)
        out.append("SOBOL — which step's uncertainty owns the spread")
        for r in sorted(so, key=lambda d: -d["ST"]):
            bar = "#" * max(0, int(round(r["ST"] * 34)))
            out.append(f"  {r['step']:<14} n={r['n_obs']:<6} S_T {r['ST']:.3f}  "
                       f"S_1 {r['S1']:.3f}  {bar}")
        out.append(f"  ({base * (len(ev) + 2)} exact solves; read S_T, S_1 is noisier)")
        out.append("")
    except ValueError as exc:
        out.append(f"SOBOL — skipped: {exc}")
        out.append("")

    if with_voi:
        voi = value_of_info(ev, more=more, budget=budget)
        out.append(f"WHAT WOULD SETTLE IT — observing each step {more} more times")
        for r in voi:
            out.append(
                f"  {r['step']:<14} width {r['width_now'] * 100:5.1f} -> "
                f"{r['width_after'] * 100:5.1f} pts   "
                f"expected shrink {r['expected_shrink_pts']:5.2f} pts"
            )
        top = voi[0]
        rest = sum(r["expected_shrink_pts"] for r in voi[1:])
        if top["expected_shrink_pts"] > 5 * max(rest, 1e-9):
            out.append(f"  -> observe {top['step']} only; every other step is a rounding error")
        out.append("")

    led = ledger if ledger is not None else build_evidence_ledger(ev, p)
    out.append(led.render())
    out.append(BAR)
    return "\n".join(out)


def report(
    wf,
    res,
    ledger=None,
    claims=None,
    unit: str = "s",
    sobol_result=None,
) -> str:
    """The queueing report: contention, tail bounds and the simulation ledger."""
    from .verdict import interval as _interval  # noqa: F401  (kept optional)

    out: List[str] = [BAR, f"QUEUEING — {wf.label}", BAR]

    out.append("STAGES")
    loads = stage_load(wf)
    for s in wf.stages:
        rho = loads.get(s.name, {}).get("rho")
        load_txt = f"rho={rho:.3f}" if rho is not None else "rho=n/a"
        out.append(f"  {s.name:<18} {s.service.describe():<38} c={s.capacity}  {load_txt}")
    out.append("")

    sm = serial_moments(wf.stages)
    out.append("ANALYTIC BASELINE (no contention, independent stages)")
    out.append(f"  serial moments: mean {_f(sm['mean'])}  sd {_f(sm['sd'])}  "
               f"CV {_f(sm['cv'], 2)}")
    if wf.arrival_rate is not None:
        for s in wf.stages:
            m = mmc(wf.arrival_rate, s.service_rate, s.capacity)
            if m.stable:
                out.append(f"  M/M/c {s.name:<18} Wq {_f(m.wq)}  W {_f(m.w)}  "
                           f"Lq {_f(m.lq, 2)}  P(wait) {_f(m.p_wait)}")
            else:
                out.append(f"  M/M/c {s.name:<18} UNSTABLE at this arrival rate")
    out.append("")

    s = res.summary()
    out.append(f"SIMULATION  {res.n_tasks} tasks, {res.n_failed} failed, "
               f"horizon {_f(res.horizon, 1)}{unit}")
    out.append(f"  latency [{unit}]  mean {_f(s['mean'])}  p50 {_f(s['p50'])}  "
               f"p95 {_f(s['p95'])}  p99 {_f(s['p99'])}  max {_f(s['max'])}")
    out.append(f"  failure probability {_f(s['failure_prob'], 4)}")
    out.append("")

    out.append("TAIL CHECK")
    for mult in (2.0, 3.0):
        t = s["mean"] * mult
        emp = res.tail_prob(t)
        bnd = cantelli_upper(s["mean"], s["sd"] ** 2, t)
        out.append(f"  P(T > {_f(t)}{unit}) empirical {_f(emp, 4)}  Cantelli {_f(bnd, 4)}")
    out.append("")

    out.append("PER-STAGE QUEUEING")
    for name, st in res.stage_stats.items():
        out.append(f"  {name:<18} util {_f(st.utilisation, 3)}  "
                   f"wait mean {_f(st.mean_wait)} p95 {_f(st.p95_wait)}  "
                   f"attempts {st.attempts}  fails {st.failures}")
    out.append("")

    if sobol_result is not None:
        out.append(sobol_result.render())
        out.append("")
    if ledger is not None:
        out.append(ledger.render())
        out.append("")
    out.append(BAR)
    return "\n".join(out)
