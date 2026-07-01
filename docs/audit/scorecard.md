# Verification Scorecard (Campaign Phase 5)

`tools/audit/scorecard.py` is the ongoing per-module verification scorecard.
Per operator decision D3 (docs/audit/operator-decisions.md) the hermes study is
terminated, so the scorecard is a **standalone runner**, not a hermes-pipeline
report: it re-executes the campaign's read-only invariant sweeps against a
corpus root and rolls their results up into one verdict per module.

## How to run it

```bash
# full scorecard against the frozen corpus (default root)
python3 tools/audit/scorecard.py

# machine-readable copy as well
python3 tools/audit/scorecard.py --json /tmp/scorecard.json

# incremental scorecard: only snapshot days >= 2026-06-20
python3 tools/audit/scorecard.py --since 20260620
```

Exit codes: `0` all PASS/KNOWN/INCONCLUSIVE, `1` any WARN, `2` any ERROR
(or unusable `--root`). Runtime is a few minutes (the sweeps run as
subprocesses, four at a time). The tool is strictly read-only over the corpus.

The header always prints the **corpus window actually covered** (first/last
snapshot day, distinct days, per-node snapshot counts). Check it: a stale or
sparse corpus makes every verdict weaker, and a zero-snapshot window makes the
entire card INCONCLUSIVE by construction.

## Statuses

| Status | Meaning |
|---|---|
| **PASS** | Every mapped check is clean and at least one check was non-vacuous (actually fed by corpus data). |
| **KNOWN** | Only allowlisted findings fired — documented, operator-accepted anomalies (each carries its doc reference in the notes column). Not new signal. |
| **WARN** | At least one non-allowlisted violation. New signal; the violating check names and example rows are listed. Investigate before anything else. |
| **INCONCLUSIVE** | Every mapped check was vacuous (zero checkable instances in the window) or the module has no corpus-observable checks at all. **This is data starvation, not a health verdict** — the module may be perfectly healthy (or broken); this window simply contains nothing that exercises it. Operationally: do not read INCONCLUSIVE as regression or as clearance; if you need a verdict, generate data (e.g. boltz activity for `boltz_manager`) or fall back to the Phase 2 test/code verdicts in docs/audit/verification/. |
| **ERROR** | The sweep crashed or its output did not parse. Never treated as absence — fix the sweep or the parser before trusting the rest of the row's module. |

Additional labels in the notes column:

- `vacuous: <check>` — that check had zero checkable instances (or is
  force-labeled vacuous, e.g. BM-H2 passing over an all-zero boltz surface).
- `lossy-echo checks: …` — these checks read the `recent_fee_changes` RPC
  echo, a rolling 10-row window that lost **61% of nexus-01's change records**
  in the frozen corpus (docs/audit/decision-loops.md, defect 5). Their pass
  counts are lower bounds; a future collector must capture the `fee_changes`
  DB table instead of the RPC echo.

## What it runs and how modules map

The scorecard shells out to the campaign sweeps in `tools/audit/`
(`sweep_fee_stack`, `sweep_rebalancer`, `sweep_profitability`,
`sweep_planner_boltz_hints`, `sweep_data_budget`, `sweep_routing_stack`,
`loop_sweep_fee`, `loop_sweep_rebalance`, `loop_sweep_planner`,
`check_hermes_forwards_chain`) and parses their per-check results. Checks map
to modules by invariant prefix:

| Prefixes / source | Scorecard row |
|---|---|
| FC-*, FA-*, PM-* | fee_controller / flow_analysis / policy_manager |
| RB-*, RE-* | rebalancer / rebalance_engine_v2 |
| PA-* | profitability_analyzer |
| CP-* (+ D1-flag, pool, ledger-continuity), BM-*, HH-* | capacity_planner / boltz_manager / hive_hints |
| SL/TCB/DB* , CB*, ML-*, SO*, DF-* | database / capex_budget / capital_efficiency / segment_observations / demand_flow |
| sweep_routing_stack C/S/O/L groups | routing_stack (the RX/R2/R3/HR/RHR/RCO/NX/RP2 surface) |
| LF-* | fee_loop (Phase 3 fee decision loop) |
| LP-* | rebalance_loop |
| loop_sweep_planner L1–L5, inv-* | planner_loop |
| listforwards chain contiguity | forwards_chain |

`policy_manager` and `demand_flow` have no corpus-observable checks (their
Phase 2 verdicts are code/test based), so they are permanently INCONCLUSIVE
here — by design, not by accident.

## Pointing it at a future data source

The scorecard and every sweep resolve the corpus root from
`CL_MYCELIUM_HERMES_ROOT` (the scorecard sets it for its children from
`--root`). Any future collector only has to reproduce the corpus layout:

```
<root>/<node>/<YYYYMMDD>/<YYYYMMDDTHHMMSSZ>/commands/*.json[.gz]
```

with the same artifact names the plugin RPCs emit (`revenue-status.json`,
`revenue-rebalance-debug.json`, `listpeerchannels.json`,
`listforwards-window.json.gz`, `revenue-planner-history.json`, …). Node names
are currently pinned to `hive-nexus-01`/`hive-nexus-02` inside the sweeps; a
fleet change means updating each sweep's `NODES` tuple.

`--since YYYYMMDD` is implemented as a temporary symlink shadow of the root
(only day-directories `>= since` are linked), so the sweeps themselves never
grow window logic. Use it for "what changed since the last capture"
scorecards once a new data source produces fresh days.

Two data-quality lessons to carry into any new collector (from
docs/audit/decision-loops.md and phase2-summary.md): capture the
`fee_changes` DB table (the RPC echo is lossy — see `[lossy-echo]` above),
and capture planner history frequently enough that the 7-day RPC window
cannot drop action ids.

## Extending the allowlist

The allowlist lives in `tools/audit/scorecard.py` as the `ALLOWLIST` constant.
Each entry is:

```python
{
    "sweep": "loop_sweep_fee",          # sweep whose check may fire
    "match": "LF-7",                    # prefix of check id OR printed name
    "reason": "why this is accepted",   # one sentence, specific
    "doc":    "docs/audit/...",         # REQUIRED: where it is documented
}
```

Rules for adding one:

1. A finding goes on the allowlist only after it is **documented** (a
   verification doc, a decision-loop report, or an operator decision) — the
   `doc` field is not optional. Undocumented findings stay WARN.
2. Match as narrowly as possible (full check id, or the exact printed name
   prefix). A too-broad match can hide a *new* failure inside a known one —
   e.g. `ML-` is acceptable because every ML-* check is the same
   cl-hive-owned seam, but do not allowlist all of `LF-`.
3. When the underlying defect is fixed, **remove the entry**; the next run
   flips the module from KNOWN back to PASS and confirms the fix (or WARNs
   if the fix regressed something else).

Current allowlisted findings (frozen-corpus state): ML-* metabolism-ledger
anomalies (cl-hive owned), DB7/TCB covered_hours drift (commit 2247370),
LF-7 external-fee-writer episode, LP-I6 pre-coverage observation, planner
L2 delegation counter-cases + L3 defibrillation placebo, D1 hive-member
defibrillation/fee-reduce exposure, and planner-ledger id-continuity gaps
(bounded RPC window artifact).

## Validation (2026-07-01, frozen corpus)

Full-corpus run: 10 PASS / 6 KNOWN / 3 INCONCLUSIVE / 0 WARN / 0 ERROR,
matching the campaign ground truth in docs/audit/verification/ and
docs/audit/decision-loops.md. Spot checks per the Phase 5 done-criterion:
`fee_controller` = PASS (4/4 non-vacuous checks clean, FC-I1a/I9/I10 labeled
lossy-echo), `capital_efficiency` = KNOWN (all five ML-* checks allowlisted,
zero non-allowlisted violations).
