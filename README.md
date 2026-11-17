# preflight

**Reliability mathematics for agent workflows. Counts in, credible intervals out.**

You do not know your agent's success rate. You know counts: this step succeeded 1,920
times, retried 40, failed 40. `preflight` treats each step's transition vector as a
Dirichlet posterior, solves the workflow as an absorbing Markov chain **exactly**, and
carries that input uncertainty through the solve — so the answer arrives as an interval
with a verdict attached, not a number with false precision.

**[Live lab →](https://saumyajitghosh-collab.github.io/preflight/)** — edit the counts,
watch the interval, the Sobol indices and the verdict recompute in the browser.

```python
from preflight import Evidence, propagate, verdict, value_of_info

ev = [Evidence("Plan",      n_ok=1920, n_retry=40, n_fail=40),
      Evidence("Retrieve",  n_ok=1980, n_retry=20),
      Evidence("Call tool", n_ok=1900, n_retry=60, n_fail=40),
      Evidence("Verify",    n_ok=34, n_back=4, n_esc=1, n_fail=1, back_to=2),
      Evidence("Answer",    n_ok=1960, n_retry=20, n_fail=20)]

p = propagate(ev)
p["p_success"]          # point 0.846, 90% interval [0.761, 0.908] — 14.7 pts wide
verdict(p, claim=0.85)  # -> underdetermined
value_of_info(ev, more=200)
# Verify: 14.8 -> 6.7 pts.  Every other step: zero.
```

---

## Why

Every tool in the AI stack runs an agent, guards an agent, or watches an agent. Almost
none of them do the arithmetic *before* deployment. Ask a typical stack "will this agent
succeed 85% of the time?" and you get a benchmark on yesterday's traffic, or a guess.

Two things make that question answerable:

1. **An agent workflow is an absorbing Markov chain.** Each step advances, retries, loops
   back, escalates, or fails. That gives closed forms for everything except latency.
2. **The transition probabilities are not known, only counted.** Typed probabilities are a
   belief. Traces give counts. The uncertainty in those counts is the dominant term in
   the answer, so it belongs inside the calculation.

## What it computes

| Question | Method |
|---|---|
| How often does the workflow succeed? | absorption probabilities `B = N R`, `N = (I-Q)⁻¹` — exact |
| How many steps, what cost, with what variance? | `t = N 1`, `E[cost] = N c`, `Var = (2N-I)t - tt` — exact |
| What does a step budget cost in success? | K-step transient propagation — exact |
| **What interval, not what number?** | `propagate` — Dirichlet draws through the batched exact solver |
| **Which step's *not knowing* owns the spread?** | `sobol` — Saltelli scheme, Jansen estimators |
| **What would settle it?** | `value_of_info` — preposterior interval shrink per step |
| Is the model itself wrong? | `littles_law` — `L = λW`, parameter-free |
| Under load, what is the p95? | `simulate` — discrete-event queueing network |
| How loose is the tail really? | Cantelli bound, distribution-free |

## Install

```bash
pip install numpy            # the only dependency
git clone https://github.com/saumyajitghosh-collab/preflight
cd preflight && python -m pytest tests/ -q      # 45 passed
```

## MCP server

```bash
pip install mcp
python -m preflight.mcp_server
```

Five tools, so an agent can ask before it ships:

| Tool | Purpose |
|---|---|
| `analyze` | intervals, exact outcomes, Sobol, verdict — from counts |
| `what_would_settle_it` | expected interval shrink per step for *m* more observations |
| `check_littles_law` | parameter-free conservation check on the model |
| `queue_latency` | p95 under contention, with a distribution-free bound |
| `verifier_lift` | is a checker step worth its cost? |

The design rule: **the LLM supplies counts it read from traces; the maths is
deterministic.** The filled-in spec comes back for inspection. The model is never asked
to do arithmetic it can get wrong.

## The maths

**Absorbing chain.** `N = (I-Q)⁻¹` is the fundamental matrix: `N[i,j]` is the expected
number of visits to `j` starting from `i`. Absorption probabilities are `B = NR`. Exact,
no sampling, no convergence worry.

**Dirichlet posterior.** With counts `(n_ok, n_retry, n_back, n_esc, n_fail)` and a flat
prior, the step's transition vector is `Dirichlet(counts + 1)`; each marginal is Beta, so
a single category's interval is available in closed form.

**Batched solve.** `solve_batch` inverts a whole stack of matrices at once. That is what
makes value-of-information interactive: the naive loop took 9.5 s, the batched version
takes 0.6 s for the same analysis.

**Sobol.** `S_Ti − S_1i` is the part explained only through interactions — the part
one-at-a-time analysis cannot see. First order is the noisier index at this sample size;
read the total order for ranking.

**Little's Law.** `L = λW` is an identity, not an approximation, so a violation indicts
the *model* rather than the parameters. The queueing run measures `L_q = 3.35` against
`λW_q = 3.35`.

## The headline result

The reference workflow has five steps. Four have been observed ~2,000 times. Verify has
been observed **40** times.

- P(success) = 0.846, 90% interval **[0.761, 0.908]** — 14.7 points wide.
- "Succeeds 85% of the time" is **underdetermined**: 85% sits inside a 14.7-point interval.
- Verify's uncertainty owns **95%** of the output variance. The other four steps, with
  2,000 observations each, own about 1%.
- 200 more Verify observations shrink the interval from 14.8 to 6.7 points. 200 more of
  anything else: zero.

That last line is the whole product. It converts an unanswerable question into one
afternoon of instrumentation on one step.

## Limitations

- **First-order Sobol is noisy.** At 256–1,024 Saltelli draws it can come out negative;
  it is clipped to `[0, S_T]` and should be read as indicative. Total order is the one to
  rank with. Coupling Sobol to a stochastic simulator needs a surrogate to do properly.
- **Latency is Monte Carlo.** Outcome probabilities are exact; end-to-end latency has no
  closed form and is sampled.
- **The queueing model is a separate view.** The chain assumes steps run alone; the
  discrete-event engine adds contention. They share a workflow shape but not a parameter
  set, and the tool does not pretend otherwise.
- **Independence across steps.** The Dirichlet posteriors are per-step; correlated
  failures (a shared provider outage) are not representable.
- **A model is not the system.** Every number is conditional on the step graph and the
  counts you wrote down. The ledger says which assumptions the answer leans on; it cannot
  tell you whether your trace data is representative.

## Layout

```
preflight/
  core.py          absorbing Markov chain: exact solve, batched solve, trajectory MC
  evidence.py      Dirichlet posteriors, propagate, Sobol, value of information
  queue.py         service-time distributions, stages, the workflow DAG
  queuesim.py      discrete-event engine
  analytic.py      M/M/c, Erlang C, Kingman, Allen-Cunneen, Little's Law
  bounds.py        Cantelli, Markov, Hoeffding, bootstrap, batch means
  ledger.py        assumption checks with blast radius
  mcp_server.py    five tools for any MCP client
index.html         the live lab (GitHub Pages)
tests/             45 tests, including solver-vs-closed-form
```

## Licence

MIT.
