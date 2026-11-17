"""
preflight.mcp_server
====================
Expose preflight to any MCP client.  Run:  python -m preflight.mcp_server

Design rule: the LLM supplies counts it has read from traces; the maths is
deterministic and the filled-in spec is returned for inspection. The model is
never asked to do arithmetic it can get wrong.

Tools
  analyze               credible intervals, exact outcomes, Sobol, verdict
  what_would_settle_it  expected interval shrink per step for m more observations
  check_littles_law     parameter-free conservation check on the model
  queue_latency         contention: p95 latency with a distribution-free bound
  verifier_lift         is a checker step worth its cost?
"""
from __future__ import annotations

from typing import Optional, Sequence

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None

from .core import Step, Workflow, erlang_c, verifier_lift as _lift
from .evidence import Evidence, littles_law, propagate, sobol, value_of_info, verdict


def _ev(steps: Sequence[dict]) -> list[Evidence]:
    return [Evidence(**s) for s in steps]


def analyze(steps: list[dict], budget: Optional[int] = None,
            claim: Optional[float] = None, level: float = 0.9) -> dict:
    """steps: [{name, n_ok, n_retry, n_back, n_esc, n_fail, back_to, cost, latency}]

    Counts observed per step. Returns the exact solve on posterior means, the
    posterior interval on P(success), Sobol indices over input uncertainty, and
    a verdict on `claim` if given.
    """
    ev = _ev(steps)
    p = propagate(ev, budget, level=level)
    p.pop("samples_success", None)
    wf = Workflow([e.mean_step() for e in ev])
    exact = wf.solve(budget)
    return {
        "intervals": p,
        "exact": {"p_success": exact.p_success, "p_fail": exact.p_fail,
                  "p_escalate": exact.p_escalate,
                  "expected_steps": exact.expected_steps,
                  "expected_cost": exact.expected_cost},
        "sobol": sobol(ev, budget, base=512),
        "verdict": verdict(p, claim),
        "spec_used": [{"name": e.name, "posterior": e.posterior_mean(),
                       "n_obs": e.n, "back_to": e.back_to} for e in ev],
    }


def what_would_settle_it(steps: list[dict], more: int = 200,
                         budget: Optional[int] = None) -> list[dict]:
    """Expected narrowing of the success interval per step, if that step were
    observed `more` further times. Sorted best-first; the tail is usually zero."""
    return value_of_info(_ev(steps), more=more, budget=budget, outer=30, draws=500)


def check_littles_law(arrival_rate: float, mean_in_system: float,
                      mean_time_in_system: float) -> dict:
    """L = lambda * W. Parameter-free, so a violation indicts the model."""
    return littles_law(arrival_rate, mean_in_system, mean_time_in_system)


def queue_latency(stages: list[dict], arrival_rate: float, n_tasks: int = 4000,
                  seed: int = 0) -> dict:
    """stages: [{name, median, sigma, capacity, p_fail, max_retries, deps}]

    Runs the workflow under contention and returns latency percentiles with a
    distribution-free (Cantelli) bound on the tail, plus per-stage utilisation.
    Use this when several tasks share capacity; use `analyze` for a single run.
    """
    from .bounds import cantelli_upper
    from .queue import Lognormal, QueueStage, QueueWorkflow
    from .queuesim import simulate

    qs = [QueueStage(name=s["name"], service=Lognormal(s["median"], s.get("sigma", 0.5)),
                     capacity=s.get("capacity", 1), p_fail=s.get("p_fail", 0.0),
                     max_retries=s.get("max_retries", 0), deps=tuple(s.get("deps", ())))
          for s in stages]
    wf = QueueWorkflow(stages=qs, arrival_rate=arrival_rate, n_tasks=n_tasks, seed=seed)
    res = simulate(wf)
    mean, sd = res.summary()["mean"], res.summary()["sd"]
    return {
        "p50": res.p(0.50), "p95": res.p(0.95), "p99": res.p(0.99),
        "mean": mean, "failure_prob": res.failure_prob,
        "bottleneck": wf.bottleneck().name,
        "max_stable_arrival": min(s.capacity / s.service_mean for s in qs),
        "utilisation": {k: v.utilisation for k, v in res.stage_stats.items()},
        "cantelli_bound_p95": cantelli_upper(mean, sd * sd, res.p(0.95)),
        "note": "a single number is not an answer: report the interval and the "
                "assumptions, or use analyze() for a verdict",
    }


def verifier_lift(p: float, sensitivity: float, specificity: float,
                  max_attempts: int = 3) -> dict:
    """Is a checker step worth it? Returns what reaches the next step."""
    return _lift(p, sensitivity, specificity, max_attempts)


if FastMCP:
    mcp = FastMCP("preflight")
    for _fn in (analyze, what_would_settle_it, check_littles_law, queue_latency, verifier_lift):
        mcp.tool()(_fn)

if __name__ == "__main__":
    if FastMCP is None:
        raise SystemExit("pip install mcp")
    mcp.run()
