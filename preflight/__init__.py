"""
preflight — reliability mathematics for agent workflows.

An agent workflow is an absorbing Markov chain whose transition probabilities
are not known, only counted. `preflight` solves the chain exactly, propagates
the uncertainty in those counts through that solve, and reports intervals,
verdicts and what would settle them.

    from preflight import Evidence, propagate, verdict, value_of_info
    ev = [Evidence("Plan", n_ok=1920, n_retry=40, n_fail=40),
          Evidence("Verify", n_ok=34, n_back=4, back_to=0, n_esc=1, n_fail=1)]
    p = propagate(ev)
    verdict(p, claim=0.85)        # -> contradicted / supported / underdetermined
    value_of_info(ev, more=200)   # -> observe Verify, not the others

Two halves:

  core + evidence   exact Markov chain, counts in, credible intervals out
  queue + analytic  the same workflow under contention: M/M/c, a discrete-event
                    engine, distribution-free tail bounds, and an assumption
                    ledger that checks the maths rather than asserting it
"""
from .analytic import (
    MMCMetrics,
    allen_cunneen_wq,
    kingman_wq,
    little_check,
    max_stable_arrival,
    mm1,
    mmc,
    serial_moments,
    stage_load,
)
from .bounds import (
    batch_means_ci,
    bootstrap_ci,
    cantelli_lower,
    cantelli_upper,
    hoeffding_upper,
    lag1_autocorr,
    markov_upper,
    p95_ci,
    relative_precision,
    required_n_for_mean,
    warmup_index,
)
from .core import (
    ESCALATE,
    FAIL,
    OUTCOMES,
    SUCCESS,
    Result,
    Step,
    Workflow,
    erlang_c,
    verifier_lift,
)
from .evidence import (
    CATS,
    CONTRADICTED,
    SUPPORTED,
    UNDERDETERMINED,
    Evidence,
    littles_law,
    propagate,
    sobol,
    value_of_info,
    verdict,
)
from .ledger import (
    HOLDS,
    UNTESTED,
    VIOLATED,
    Assumption,
    Ledger,
    build_evidence_ledger,
    build_ledger,
)
from .queue import (
    Deterministic,
    Dist,
    Exponential,
    Gamma,
    Lognormal,
    QueueStage,
    QueueWorkflow,
)
from .queuesim import SimResult, replications, simulate
from .report import report, report_evidence

__version__ = "0.2.0"

__all__ = [
    # exact Markov chain
    "Step", "Workflow", "Result", "erlang_c", "verifier_lift",
    "SUCCESS", "FAIL", "ESCALATE", "OUTCOMES",
    # evidence / uncertainty
    "Evidence", "CATS", "propagate", "sobol", "value_of_info", "verdict",
    "littles_law", "SUPPORTED", "CONTRADICTED", "UNDERDETERMINED",
    # queueing
    "QueueWorkflow", "QueueStage", "Dist", "Lognormal", "Exponential", "Gamma",
    "Deterministic", "simulate", "replications", "SimResult",
    "mmc", "mm1", "kingman_wq", "allen_cunneen_wq", "MMCMetrics",
    "serial_moments", "little_check", "stage_load", "max_stable_arrival",
    # bounds
    "bootstrap_ci", "batch_means_ci", "p95_ci", "lag1_autocorr",
    "cantelli_upper", "cantelli_lower", "markov_upper", "hoeffding_upper",
    "required_n_for_mean", "relative_precision", "warmup_index",
    # ledger
    "Ledger", "Assumption", "build_ledger", "build_evidence_ledger",
    "HOLDS", "VIOLATED", "UNTESTED",
    # reports
    "report", "report_evidence",
]
