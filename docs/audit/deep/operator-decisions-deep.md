# Deep-Audit Operator Decisions

Rulings from Sat on the Phase 1–2 behavioral-HOLD findings, 2026-07-02. All approved
as recommended. These authorize the behavioral fixes that the deep-audit campaign held
pending operator input.

## DD1 — Unified budget enforcement (P1-003 + P2-011 + P2-008) [priority]

**Ruling:** Enforce a **live cross-category budget inside the reservation transaction**;
remove the 30-second memoization from the reservation gating path; make the
`spend_event` write failure **loud/retried** rather than swallowed.

- P2-011: `_total_cost_budget_status()` is memoized 30s (cl-revenue-ops.py:6833); the
  `revenue-spend-reserve` gate (6388-6415) reads a stale `remaining_sats` so N
  reservations in one window pass against the same snapshot → overspend at low
  concurrency (soak: 842x). Gating must read a live total, not the memo. (Non-gating
  telemetry reads may keep a cache.)
- P1-003: the reserve must account for ALL categories (generic ledger +
  rebalance + boltz/capex) atomically — a shared reservation lock or a single
  cross-category reserve inside `BEGIN IMMEDIATE`.
- P2-008: `record_spend_event`/`reserve_spend` swallow `OperationalError` → silently
  lost write, never retried; under-counts in the overspend-permitting direction. The
  write must fail loud (propagate) or retry so the reservation is not reported spent
  on a lost event.

## DD2 — Force semantics (P1-001, P1-004)

**Ruling:** Hard ceilings bind even under `force`. `force` overrides soft gates
(deadband, cooldown, per-cycle limits) but NEVER the absolute rails: max rebalance
amount, the `[min_fee_ppm, max_fee_ppm]` ceiling, and a sane fee upper bound. Also
rate-limit `force=false` the same as `force=true` (remove the asymmetry where
force=false is *less* gated).

## DD3 — Boltz withdraw cap (P1-006 remainder)

**Ruling:** Add a **configurable max-withdraw-sats** bound (sane default) so a typo or
automation bug cannot sweep the wallet in one call. No budget-category gate (withdraw
returns owned funds, not a spend). Address-format validation already landed (9637c54).

## DD4 — Auto-cycle force (P1-018)

**Ruling:** Expose a `dry_run` option on `revenue-boltz-auto-cycle-run-now` (defaulting
safe) so the operator can preview; keep `force`'s ability to run one live cycle while
the `boltz_auto_cycle_enabled` toggle is off. Document that force overrides the toggle.

## DD5 — Daemon restart + heartbeat (P1-010)

**Ruling:** Wrap the **entire per-iteration loop body including the tail**
(config.snapshot / interval / jitter / sleep) in one `try/except Exception` with logging
+ bounded backoff, so a tail error can no longer kill the thread. Add a **per-thread
heartbeat** (last-iteration timestamp) surfaced on `revenue-health`, and make it a
standing scorecard check so a stalled/dead loop becomes operator-detectable. Apply the
canonical guard to all 7 daemon loops (incl. the one-shot startup-snapshot). The
`tests/test_daemon_survival.py` tail-death assertions flip to assert survival.
