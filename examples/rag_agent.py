"""A RAG agent with a verify step, analysed from observation counts.

The point of the example: four of the five steps have been observed thousands of
times. One — Verify — has been observed 40 times. Everything you want to know
about this workflow is therefore a statement about 40 observations, and the
honest answer is an interval plus a shopping list.

Run:  python examples/rag_agent.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import preflight as pf

# counts read off traces: (n_ok, n_retry, n_back, n_esc, n_fail)
EVIDENCE = [
    pf.Evidence("Plan",       n_ok=1920, n_retry=40, n_fail=40, cost=0.4),
    pf.Evidence("Retrieve",   n_ok=1980, n_retry=20, n_fail=0,  cost=0.1),
    pf.Evidence("Call tool",  n_ok=1900, n_retry=60, n_fail=40, cost=0.2),
    pf.Evidence("Verify",     n_ok=34, n_back=4, n_esc=1, n_fail=1, back_to=2, cost=0.3),
    pf.Evidence("Answer",     n_ok=1960, n_retry=20, n_fail=20, cost=0.5),
]


def main() -> None:
    print(pf.report_evidence(
        EVIDENCE,
        claim=0.85,
        level=0.90,
        draws=4000,
        base=1024,
        more=200,
    ))

    # the queueing view: the same workflow under contention
    print()
    print("=" * 74)
    print("CONTENTION — what happens when 2.2 tasks/second share the workers")
    print("=" * 74)
    wf = pf.QueueWorkflow(
        label="rag agent under load",
        arrival_rate=2.2,
        n_tasks=6000,
        seed=7,
        stages=[
            pf.QueueStage("plan",      pf.Lognormal(0.40, 0.50), capacity=6),
            pf.QueueStage("retrieve",  pf.Lognormal(0.10, 0.60), capacity=8, deps=("plan",)),
            pf.QueueStage("call_tool", pf.Lognormal(0.30, 0.70), capacity=3,
                          p_fail=0.02, max_retries=2, deps=("retrieve",)),
            pf.QueueStage("verify",    pf.Lognormal(0.25, 0.55), capacity=3, deps=("call_tool",)),
            pf.QueueStage("answer",    pf.Lognormal(0.90, 0.60), capacity=4, deps=("verify",)),
        ],
    )
    res = pf.simulate(wf)
    print(pf.report(wf, res, ledger=pf.build_ledger(wf, res), unit="s"))


if __name__ == "__main__":
    main()
