# cl_revenue_ops Deep Audit — Final Report

Date: 2026-07-02. Scope: the whole plugin (~73,800 tracked source lines across 68 files),
every dimension the June–July 2026 verification campaign did not cover. Ledger:
`findings-ledger.md`. Coverage proof: `coverage-manifest.md` + `attestations.md`
(`tools/audit/deep_manifest.py --coverage` = 100.00%, `--check` CLEAN at HEAD).
Standing gate: `tools/audit/scorecard.py --deep-only`.

## Outcome

- **114 findings** logged; **89 FIXED**, 1 profiling CLOSED-CLEAN, 5 WONTFIX/accepted,
  **20 OPEN** — all OPEN are Low/Info: accepted-as-design, documented test-coverage gaps,
  minor enhancements, or conservative-direction (never overspend). **Zero Critical/High/
  Medium left open.**
- **100.00% accountable coverage**: every one of 184 blob-pinned source chunks is COVERED
  by a finding (58) or a structured attestation (126) at its current blob; the coverage
  math (every source line maps to exactly one chunk) is independently verified.
- Full test suite: **3199 passed, 5 skipped** (env-gated). ~169 commits (5449c53→HEAD).

## What the audit found and fixed, by dimension

**Control plane + 58-RPC security (Phase 1)** — the 8,517-line main file was never a
verified module. 30 findings: input-range validation across the RPC surface, `LIMIT -1`
DoS, type-confusion crashes, socket/shutdown resilience, hive-datastore hardening, and a
permanent 1,051-case `test_rpc_param_validation.py` matrix. No Critical; the boltz
subprocess is shell-safe.

**Concurrency + daemon survival (Phase 2)** — C-1 (DB corruption) and C-3 (config torn
read) both proved CLOSED-BY-MECHANISM; no deadlock. Fixed 7 unprotected-shared-state
races, a "database is locked" batch-hold, and demonstrated all 6 daemon loops die silently
on a tail exception → (with Phase 8) a per-iteration guard + heartbeat. A stress harness
(`stress_concurrency.py`) that reproduces budget overspend is now a standing check.

**Migrations, resource, supply chain (Phase 3)** — **MIG-1 (broken upgrade)**: a
pre-2025-12-19 DB would lose all forward-revenue recording until a 2nd restart on upgrade —
fixed order-independent. Resource: no table on a 1GB path (fee_changes is bounded at 90d).
Supply chain: `requirements.txt` pinned+hashed, CycloneDX SBOM, non-fatal CLN/boltzcli
version probes, `check_pins.py` gate.

**Tier-1 money core (Phase 4)** — ~26k lines, 100% double-read + 6 adversarial refutation
passes. The headline: the operator-approved **DD1 unified cross-category budget** had **six**
spending paths and originally covered **three**; each refutation pass surfaced the next
overspend gap (boltz, capex opens/closes, defibrillation) plus a sixth (chainswap), and
twice proved the enumeration guard itself incomplete. **All six spenders now reserve
atomically inside `BEGIN IMMEDIATE`; the guarantee is backstopped by `test_all_spenders_
atomic.py`** — a non-evadable enumeration guard (any `create*` swap, the full money-RPC set,
computed-method and attribute-style dispatch, with a forbid-pattern lint for aliasing) that
fails on any new/moved/uncovered spender. None of these were caught by six line-by-line
auditors — the failure mode was architectural completeness, not local correctness.

**Tier-2/3 + audit tooling (Phase 5)** — decision/routing/accounting logic clean (no
Medium+); fixes: an EV-sign inversion, an open-cost optimism, the June-campaign min→max
route-selector tautology finally killed. **Audited the auditors**: fixed a laundered
scorecard verdict and a coverage over-count, and — critically — confirmed the load-bearing
tools (stress-harness sampler, `utils` rounding) are correct, validating the Phase-4 verdicts.

**Docs conformance + deferred adjudication (Phase 6)** — 12 doc gaps fixed (the
HIVE_REBALANCE_REPORTING contract rewritten to the real payload; conf.full/README/header
corrections; undocumented hive caps now documented). Re-adjudicated all 94 prior-era
deferred findings: **all three recovered Jan-2026 Red Team CRITICALs are provably CLOSED**
(budget race → DD1/P4; 1-ppm floor → composed floor; unbounded `_wait_for_job` → removed
with Sling). 23 were still-open; the real ones (SCID-breaker splice-reset, batch rate-limit
ordering, a drift-guard hole) are fixed.

**Performance (Phase 7)** — CLEAN: adequately indexed, no O(n²)/full-scan; baseline +
regression guard committed.

**Consolidation + closure (Phase 8)** — DD6–DD9 operator rulings applied (force overrides
hive-zero; boltz journal cap; fee_changes left at 90d; no schema gating, documented). The
coverage reconciliation caught one more Medium the audits missed — **P8-001, a partial-retry
double-pay** on `payment_pending` — now fixed. A fresh independent adversary re-derived the
money core and found nothing ≥ Low.

## Residual OPEN (20, all Low/Info — none block deploy)

Accepted-as-design (fee-cycle `_state_lock` hold), conservative-direction concurrency Lows
(P2-009/010 — over-count/self-correct, never overspend), test-coverage gaps
(DEF-066/085/086), enhancements (predictive rebalance DEF-064, fleet-path threshold
DEF-067-S14), dead-code (DEF-088 demand_flow, cleanup optional), the Kalman-window
correlation (DEF-028, accept-or-subsample), and a lab-only tournament script bug (P8-004).
Each is documented in the ledger with rationale.

## Production deploy checklist (two HARD gates)

1. **Verify production dependency versions before deploying `requirements.txt`.** Pins are
   from the dev runtime (`pyln-client==25.12.1`, `numpy==1.26.4`, `PyYAML==6.0.1`). A node
   on an older pyln-client will FAIL to start on the pin. Run `tools/audit/check_pins.py`
   against each node's plugin runtime (or relax the pin to the node's version). Regenerate
   `requirements.lock` + `sbom.cyclonedx.json` from the production interpreter.
2. **Ship the MIG-1 migration fix before upgrading any old-schema node.** A DB created
   before 2025-12-19 (node 1 may qualify) needs the fixed `_migrate_forwards_schema` or the
   first post-upgrade start loses forward-revenue recording until a second restart. The fix
   makes upgrade safe; deploying it is strictly safer than not.

Then: standard restart. After deploy, `revenue-health` exposes the new per-thread
heartbeat (DD5) and `scorecard.py --deep-only` verifies the guarantees live.

## Standing regression armor (permanent, from this campaign)

`test_all_spenders_atomic.py` (6-spender budget enumeration + alias lint),
`test_rpc_param_validation.py` (RPC fuzz matrix), `test_migrations.py`,
`test_daemon_survival.py`, `stress_concurrency.py` (budget-overspend soak),
`test_perf_regression_guard.py`, `check_pins.py`, and the extended
`scorecard.py --deep-only` one-command health gate.
