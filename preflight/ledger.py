"""The assumption ledger.

Every quantitative claim about an agent workflow rests on assumptions. This
module makes them explicit and machine-checkable, and records what each one
breaks if it fails. A number without its ledger is not a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .analytic import little_check
from .bounds import lag1_autocorr
from .queuesim import SimResult
from .queue import QueueWorkflow


HOLDS = "holds"
VIOLATED = "violated"
UNTESTED = "untested"


@dataclass
class Assumption:
    key: str
    statement: str
    status: str
    evidence: str
    breaks: str = ""
    severity: str = "info"      # info | warning | critical

    def __str__(self) -> str:
        mark = {HOLDS: "PASS", VIOLATED: "FAIL", UNTESTED: "N/A "}[self.status]
        return f"[{mark}] {self.statement} — {self.evidence}"


@dataclass
class Ledger:
    assumptions: List[Assumption] = field(default_factory=list)

    def add(self, a: Assumption) -> None:
        self.assumptions.append(a)

    def get(self, key: str) -> Optional[Assumption]:
        for a in self.assumptions:
            if a.key == key:
                return a
        return None

    @property
    def failures(self) -> List[Assumption]:
        return [a for a in self.assumptions if a.status == VIOLATED]

    @property
    def critical_failures(self) -> List[Assumption]:
        return [a for a in self.assumptions if a.status == VIOLATED and a.severity == "critical"]

    def score(self) -> float:
        tested = [a for a in self.assumptions if a.status != UNTESTED]
        if not tested:
            return float("nan")
        return sum(1 for a in tested if a.status == HOLDS) / len(tested)

    def render(self) -> str:
        lines = ["ASSUMPTION LEDGER", "-" * 72]
        for a in self.assumptions:
            lines.append(str(a))
            if a.status == VIOLATED and a.breaks:
                lines.append(f"        -> breaks: {a.breaks}")
        lines.append("-" * 72)
        n_bad = len(self.failures)
        lines.append(
            f"{len(self.assumptions) - n_bad}/{len(self.assumptions)} assumptions survive; "
            f"{n_bad} violated."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_stability(wf: QueueWorkflow) -> Assumption:
    if wf.arrival_rate is None:
        return Assumption(
            "stability", "Offered load rho < 1 at every stage",
            UNTESTED, "no external arrival process (batch mode)", severity="warning",
        )
    worst = None
    for s in wf.stages:
        rho = wf.arrival_rate / (s.capacity * s.service_rate)
        if worst is None or rho > worst[1]:
            worst = (s.name, rho)
    name, rho = worst
    ok = rho < 1.0
    return Assumption(
        "stability",
        "Offered load rho < 1 at every stage",
        HOLDS if ok else VIOLATED,
        f"worst stage {name!r} at rho = {rho:.3f}",
        breaks="queue length diverges; mean and p95 latency are infinite, "
               "not merely large",
        severity="critical",
    )


def check_stationarity(res: SimResult) -> Assumption:
    x = res.e2e
    if x.size < 50:
        return Assumption("stationarity", "Latency distribution is stationary",
                          UNTESTED, "too few completed tasks", severity="warning")
    h = x.size // 2
    a, b = float(np.mean(x[:h])), float(np.mean(x[h:]))
    drift = abs(b - a) / max(a, 1e-12)
    ok = drift < 0.10
    return Assumption(
        "stationarity",
        "Latency distribution is stationary over the run",
        HOLDS if ok else VIOLATED,
        f"first-half mean {a:.3f} vs second-half {b:.3f} ({drift * 100:.1f}% drift)",
        breaks="a percentile taken over a drifting run is a mixture, not a "
               "property of the system; add warm-up or shorten the horizon",
        severity="warning",
    )


def check_warmup(res: SimResult) -> Assumption:
    """Compare the first and last fifth of the run on batch means.

    A cumulative-mean test is unreliable here: with autocorrelated latencies
    the cumulative mean wanders, so it looks unsettled long after the transient
    is over. Batched windows are stable under dependence.
    """
    x = res.e2e
    if x.size < 200:
        return Assumption("warmup", "Transient is negligible", UNTESTED, "too few tasks",
                          severity="warning")
    k = 10
    head, tail = x[: x.size // 5], x[-(x.size // 5):]

    def batch_means(a: np.ndarray) -> np.ndarray:
        b = a.size // k
        return np.array([float(np.mean(a[i * b:(i + 1) * b])) for i in range(k)])

    bh, bt = batch_means(head), batch_means(tail)
    se = float(np.sqrt(np.var(bh, ddof=1) / k + np.var(bt, ddof=1) / k))
    diff = abs(float(np.mean(bt) - np.mean(bh)))
    ref = float(np.mean(tail))
    rel = diff / ref if ref > 0 else 0.0
    ok = (diff < 2.0 * se) or (rel < 0.05)
    return Assumption(
        "warmup",
        "Transient is negligible: start and end of the run agree",
        HOLDS if ok else VIOLATED,
        f"first fifth {float(np.mean(head)):.3f} vs last fifth {ref:.3f} "
        f"({rel * 100:.1f}% apart, {diff / se if se > 0 else 0:.1f} standard errors)",
        breaks="the run measures start-up, not steady state; add warm-up or "
               "extend the horizon",
        severity="warning",
    )


def check_iid_latency(res: SimResult) -> Assumption:
    """Lag-1 autocorrelation of end-to-end latency."""
    x = res.e2e
    if x.size < 50:
        return Assumption("iid", "Task latencies are independent", UNTESTED,
                          "too few tasks", severity="warning")
    r1 = lag1_autocorr(x)
    # 95% band for white noise is about +-1.96/sqrt(n)
    band = 1.96 / math.sqrt(x.size)
    ok = abs(r1) < band
    return Assumption(
        "iid",
        "Task latencies are independent across tasks",
        HOLDS if ok else VIOLATED,
        f"lag-1 autocorrelation {r1:+.3f} (white-noise band +/-{band:.3f})"
        + ("" if ok else "; confidence intervals widened to batch means"),
        breaks="bootstrap intervals would be too narrow and confidence "
               "overstated; mitigated by switching to a batch-means estimator",
        severity="warning",
    )


def check_exponential_service(wf: QueueWorkflow) -> Assumption:
    """Only relevant to the M/M/c closed form."""
    worst = None
    for s in wf.stages:
        cv = s.service.cv()
        err = abs(cv - 1.0)
        if worst is None or err > worst[1]:
            worst = (s.name, err, cv)
    name, err, cv = worst
    ok = err < 0.25
    return Assumption(
        "markovian",
        "Service times are exponential (CV = 1), as M/M/c requires",
        HOLDS if ok else VIOLATED,
        f"stage {name!r} has CV = {cv:.2f}",
        breaks="M/M/c formulas no longer apply; use the Allen-Cunneen "
               "correction or trust the simulation instead",
        severity="warning",
    )


def check_retry_independence(wf: QueueWorkflow, res: SimResult) -> Assumption:
    """Compare observed failure rate to the independent-failure product."""
    stages_with_fail = [s for s in wf.stages if s.p_fail > 0]
    if not stages_with_fail:
        return Assumption("retry_independence", "Failures are independent across stages",
                          UNTESTED, "no failing stages configured")
    # A stage fails permanently only when every attempt fails, so the
    # independent model must account for the retry budget.
    predicted = 1.0 - float(
        np.prod([1.0 - s.p_fail ** (s.max_retries + 1) for s in stages_with_fail])
    )
    observed = res.failure_prob
    if not np.isfinite(observed) or res.n_tasks < 100:
        return Assumption("retry_independence", "Failures are independent across stages",
                          UNTESTED, "too few tasks", severity="warning")
    # Rare events: compare on the scale of the binomial standard error, not on
    # a relative-error threshold, which is meaningless when p is near zero.
    se = math.sqrt(max(predicted * (1.0 - predicted), 1e-18) / res.n_tasks)
    z = abs(observed - predicted) / se if se > 0 else 0.0
    ok = z < 2.5
    return Assumption(
        "retry_independence",
        "Failures are independent across stages",
        HOLDS if ok else VIOLATED,
        f"observed task failure {observed:.5f} vs independent model "
        f"{predicted:.5f} ({z:.1f} standard errors apart)",
        breaks="retry and failure analysis is optimistic; correlated failures "
               "(shared provider outage, common dependency) are not additive",
        severity="critical",
    )


def check_little(wf: QueueWorkflow, res: SimResult) -> Assumption:
    """L = lambda W, per stage, using independently measured quantities."""
    worst = None
    for name, st in res.stage_stats.items():
        lam_stage = st.attempts / res.horizon if res.horizon > 0 else 0.0
        if lam_stage <= 0 or not st.waits:
            continue
        wq = float(np.mean(st.waits))
        lq = res.mean_queue_length[name]
        # a stage with no measurable queue carries no information: L = 0 = 0
        if lam_stage * wq < 1e-6 or lq < 1e-6:
            continue
        chk = little_check(lq, lam_stage, wq)
        err = abs(chk["relative_error"]) if np.isfinite(chk["relative_error"]) else 1.0
        if worst is None or err > worst[1]:
            worst = (name, err, chk)
    if worst is None:
        return Assumption("little", "Little's Law holds per stage", UNTESTED,
                          "no stage queued appreciably; nothing to audit",
                          severity="warning")
    name, err, chk = worst
    ok = err < 0.10
    return Assumption(
        "little",
        "Little's Law L_q = lambda W_q holds per stage",
        HOLDS if ok else VIOLATED,
        f"worst stage {name!r}: L_q {chk['l_observed']:.2f} vs lambda*W_q "
        f"{chk['lam_times_w']:.2f} ({err * 100:.1f}% error)",
        breaks="the simulator itself is suspect — this is an identity, so a "
               "violation means the event loop is losing or double-counting time",
        severity="critical",
    )


def check_variability(wf: QueueWorkflow) -> Assumption:
    """High service CV makes mean-based planning wrong."""
    worst = max(wf.stages, key=lambda s: s.service.cv())
    cv = worst.service.cv()
    ok = cv < 1.5
    return Assumption(
        "variability",
        "Service-time variability is moderate (CV < 1.5)",
        HOLDS if ok else VIOLATED,
        f"stage {worst.name!r} has CV = {cv:.2f}",
        breaks="mean-based capacity planning understates p95 by a wide margin; "
               "the tail, not the mean, sets the user experience",
        severity="warning",
    )


# --------------------------------------------------------------------------
# Build the whole ledger
# --------------------------------------------------------------------------


def build_ledger(wf: QueueWorkflow, res: SimResult) -> Ledger:
    led = Ledger()
    led.add(check_stability(wf))
    led.add(check_little(wf, res))
    led.add(check_iid_latency(res))
    led.add(check_stationarity(res))
    led.add(check_warmup(res))
    led.add(check_exponential_service(wf))
    led.add(check_retry_independence(wf, res))
    led.add(check_variability(wf))
    return led


# --------------------------------------------------------------------------
# Evidence-mode checks: is the input data good enough to answer at all?
# --------------------------------------------------------------------------


def check_counts_adequate(ev) -> Assumption:
    """The least-observed step sets the precision of the whole answer."""
    if not ev:
        return Assumption("counts", "Enough observations per step", UNTESTED,
                          "no evidence supplied", severity="warning")
    worst = min(ev, key=lambda e: e.n)
    lo, hi = worst.beta_interval("ok", level=0.90)
    width_pts = (hi - lo) * 100
    ok = width_pts <= 10.0
    return Assumption(
        "counts",
        "Every step is observed enough to pin its success rate to ±10 pts",
        HOLDS if ok else VIOLATED,
        f"least-observed step {worst.name!r} has n = {worst.n}, "
        f"90% interval on p_ok is {width_pts:.1f} pts wide",
        breaks="the reported interval is dominated by that step's sampling "
               "error, not by the system's behaviour; the verdict cannot be "
               "trusted until it is observed more",
        severity="critical",
    )


def check_prior_dominance(ev) -> Assumption:
    """If the prior carries a large share of the posterior, the data is thin."""
    if not ev:
        return Assumption("prior", "The prior does not dominate the posterior",
                          UNTESTED, "no evidence supplied", severity="warning")
    worst = min(ev, key=lambda e: e.n / max(e.alpha().sum(), 1e-9))
    share = (worst.prior * 5) / worst.alpha().sum()
    ok = share < 0.10
    return Assumption(
        "prior",
        "The Dirichlet prior contributes under 10% of the posterior",
        HOLDS if ok else VIOLATED,
        f"step {worst.name!r}: prior is {share * 100:.0f}% of the posterior mass",
        breaks="the answer is partly a statement about the prior, not about "
               "the workflow; report it as a prior-dominated estimate or add data",
        severity="warning",
    )


def build_evidence_ledger(ev, prop: Optional[dict] = None) -> Ledger:
    """Ledger for a counts-based analysis."""
    led = Ledger()
    led.add(check_counts_adequate(ev))
    led.add(check_prior_dominance(ev))
    if prop is not None and "samples_success" in prop:
        x = np.asarray(prop["samples_success"], dtype=float)
        r1 = lag1_autocorr(x)
        band = 1.96 / math.sqrt(max(x.size, 1))
        # posterior draws are iid by construction; this catches a broken sampler
        led.add(Assumption(
            "sampler",
            "Posterior draws are independent (sampler sanity check)",
            HOLDS if abs(r1) < band or not np.isfinite(r1) else VIOLATED,
            f"lag-1 autocorrelation of the draw stream {r1:+.3f}",
            breaks="the propagation loop is reusing random numbers; intervals "
                   "are too narrow",
            severity="warning",
        ))
    return led
