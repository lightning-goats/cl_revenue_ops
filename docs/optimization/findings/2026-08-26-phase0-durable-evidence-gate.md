# Phase 0 durable-evidence gate — final disposition

**Adjudicated:** 2026-08-26

**Gate window:** `[2026-08-22 00:00:00, 2026-08-25 00:00:00) UTC`

**Production node:** `0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3`

**Production plugin SHA:** `39fe455dab8112ad8934ba068c5508fefc25dde8`

**Core Lightning:** `v26.06.6`
**Runtime config version:** `103`

## Disposition

**PHASE 0 PRODUCTION GATE: PASS.**

The aligned three-day window contains 72 of 72 durable hourly reconciliation
slots. Every run completed cleanly, ledger projection was aligned, fee-intent
completeness was `ok`, and no run was missing, duplicate, failed, skipped,
incomplete, truncated, or unexplained. Each closed UTC day also has complete
budget coverage, a complete required-evidence manifest, and complete,
untruncated canonical forward-archive coverage.

This disposition removes the Phase 0 measurement prerequisite from later
reviewed shadow proposals. It does **not** activate the successor production
evaluation, deploy or enable rebalance replay capture, or authorize any live
economic optimization. The configured identity remains `preflight` with
`formal_window_active=false`.

## Boundary selection

The earliest eligible boundary was `2026-08-20 18:00:00 UTC`. The first
sequence reached 25 clean hours, but a full daemon/plugin restart at
`2026-08-21 19:32 UTC` invalidated that sequence as the final gate proof. The
adjudicated interval starts at the next complete UTC-day boundary and is wholly
after that restart.

At `2026-08-25 00:10 UTC`, the preflight monitor already reported 72 clean
hours for the exact aligned interval, while the just-closed August 24 archive
was correctly still pending. At `01:10 UTC`, archive coverage for August
22–24 was complete and the trailing reconciliation window remained 72 hours
clean. The subsequent immutable closed-day collection independently confirmed
the exact aligned UTC days below.

## Daily completeness

| UTC day | Plugin coverage | Reconciliation | Fee intent | Budget | Manifest/watch | Forward archive | Result |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 2026-08-22 | 86,096.754 s | 24/24 clean; 0 missing/duplicate/failed/skipped/incomplete/unexplained | `ok`, complete | 24/24 h complete | complete / green | complete; not truncated; 29 settled | countable |
| 2026-08-23 | 86,378.650 s | 24/24 clean; 0 missing/duplicate/failed/skipped/incomplete/unexplained | `ok`, complete | 24/24 h complete | complete / green | complete; not truncated; 29 settled | countable |
| 2026-08-24 | 86,400.000 s | 24/24 clean; 0 missing/duplicate/failed/skipped/incomplete/unexplained | `ok`, complete | 24/24 h complete | complete / green | complete; not truncated; 62 settled | countable |

The required plugin-coverage floor is 79,200 seconds per UTC day. The August
22 CLN crash lasted approximately 303.246 seconds, from the fatal signal at
`02:48:45.522 UTC` to plugin initialization at `02:53:48.768 UTC`. The failure
originated in CLN's Bitcoin backend after a `getrawblockbyheight` timeout, not
in `cl_revenue_ops`. The August 23 JSON-RPC shutdown interrupted the plugin for
approximately 21.350 seconds, from `14:27:03.778` to `14:27:25.128 UTC`.
Systemd reports the daemon continuously active from that restart through the
end of August 24. Both days remain well above the frozen 22-hour minimum.

## Economic evidence retained

The canonical archive totals for the gate days are:

| UTC day | Settled forwards | Forwarded out | Routing fees |
| --- | ---: | ---: | ---: |
| 2026-08-22 | 29 | 1,780,019.975 sats | 118.644 sats |
| 2026-08-23 | 29 | 1,617,802.025 sats | 307.677 sats |
| 2026-08-24 | 62 | 5,415,182.936 sats | 496.538 sats |

Budget evidence reported `coverage_status=complete`, `covered_hours=24`, and
`coverage_hours=24` on all three days. Recorded spend was 0 sats on August 22
and 214 sats on August 23 and 24; active reservations were zero in all three
collections. These values are evidence observations, not an economic verdict
or activation recommendation.

## Reproducibility

The read-only evidence sources are:

- `results/revenue-validation/preflight/lnnode.jsonl`;
- `results/revenue-validation/manifests/2026-08-{22,23,24}.json`;
- `results/revenue-validation/watch/2026-08-{22,23,24}.json`;
- `results/revenue-validation/2026-08-{22,23,24}/lnnode/revenue-econ-reconcile.json`;
- `results/revenue-validation/2026-08-{22,23,24}/lnnode/revenue-budget.json`;
- `results/revenue-validation/2026-08-{22,23,24}/lnnode/revenue-forward-history.json`;
- production `cln.log` around `2026-08-22 02:48–02:54 UTC` and
  `2026-08-23 14:27 UTC`; and
- read-only `getinfo`, plugin Git, and systemd service-state queries performed
  during adjudication.

No action RPC, configuration mutation, fee change, rebalance, deployment,
plugin reload, or daemon restart was performed to adjudicate this gate.

## Remaining activation gates

The successor evaluation still requires a separate committed activation
record, an explicit stable evaluation identity, a frozen production/config and
capital boundary, and 30 complete baseline UTC days under the successor
specification. Only data after that activation record may count toward the
formal successor window.

Rebalance replay capture remains default-off and not deployed on production.
Any shadow rollout requires its own reviewed, operator-approved proposal and
must remain observational.
