# Intent Contract: modules/capital_efficiency.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`CapitalEfficiencyAnalyzer` (modules/capital_efficiency.py:44) computes a fleet-wide
capital-efficiency snapshot from the profitability and flow analyzers: per-channel RPSD
(lifetime gross revenue per sat deployed, ppm), a 30-day NET windowed RPSD (audit F7),
a blended percentile efficiency rank, forward velocity, and a conservative dead-capital
classification staged via `dead_capital_stage` DB rows. Its `FleetEfficiency` output is a
pure in-memory snapshot consumed by the capex engine (budget multipliers) and the
capacity planner (close/recycle decisions). It performs no RPC and writes nothing.

## Inputs / Outputs

- **Constructed at** cl-revenue-ops.py:2066–2073 (profitability_analyzer, flow_analyzer,
  database, hive_hints, config).
- **Reads**: `profitability.analyze_all_channels()` and `flow.analyze_all_channels()`
  (:63–64), `database.get_dead_capital_stages()` (:66, table at
  modules/database.py:1271/6671), config `capex_grace_days` / `flow_window_days`
  (:105–110), hive membership (:220–225).
- **Consumers**: `CapexBudgetEngine` (`_capital_efficiency.analyze()`,
  modules/capex_budget.py:159–163 → `_get_efficiency_multiplier` :558–586);
  `capacity_planner` (modules/capacity_planner.py:830, :1622).
- **No RPC surface and no datastore key of its own.**

## Invariants

- **CE-1** RPSD is msat-precise gross revenue per deployed sat:
  `fees_earned_msat * 1000 / capacity_sats`, 0.0 for zero capacity (:151–164).
- **CE-2** The windowed-net blend activates only when EVERY channel exposes a numeric
  `marginal_profit_30d_sats`; one missing field disables the blend fleet-wide rather than
  synthesizing values (:91–103, :166–182). When active, rank = 0.5·lifetime + 0.5·windowed.
- **CE-3** Percentile ranks are tie-averaged in [0,1]; a single channel ranks 1.0
  (`_calculate_percentile_ranks`, :184–207).
- **CE-4** Dead capital requires all of: flow metrics present, zero forwards in window,
  `days_open > capex_grace_days`, and peer not a hive member (`_is_dead_capital`,
  :209–227). Absence of flow data is never treated as death (:211–212).
- **CE-5** Fleet totals are non-negative sums of per-channel capacities; dead-capital
  sats only accumulate for channels classified dead (:112–136).
- **CE-6** SCID-format drift cannot orphan flow or stage data: flow metrics and stages
  are looked up via `normalize_scid` aliases (:69–78, :122–127, :145).

## Observable surface — metabolism ledger (known anomaly documented)

The hermes artifact `hive-organism-status.json` carries a `metabolism` profile with a
canonical ledger labeled `canonical_source: "cl_revenue_ops"`. **That ledger is not built
by this module** — it is assembled by cl-hive
(`_build_canonical_metabolism_ledger`, cl-hive/modules/organism/runtime.py:2265–2372)
from this plugin's RPC surfaces (`revenue-profitability`, `revenue-dashboard`,
`revenue-total-cost-budget`, `revenue-capex-status`, `revenue-spend-ledger`).

What the code SAYS the windows should contain
(`_METABOLISM_LEDGER_WINDOWS`, runtime.py:1141–1147: 1h/6h/24h/7d/30d): per-window
`energy_intake`, `metabolic_burn`, `developmental_expenditure`, `net_usable_energy`,
reserves, stranded liquidity, plus a `coverage` block
(`status`/`covered_hours`/`required_hours`) and per-window confidence/freshness. Burn and
development are supposed to come from per-window `revenue-spend-ledger` calls
(runtime.py:1364–1372), with development = the category set at runtime.py:1149–1160.

Why production shows identical values across all five windows:

1. `energy_intake_msat`, `energy_reserves`, and `stranded_liquidity` are computed ONCE
   outside the window loop from point-in-time/lifetime payloads (runtime.py:2280–2301)
   and copied into every window (:2335–2348). They are identical by construction —
   intake is never windowed at all.
2. Burn/development are per-window only when the spend ledger has nonzero data;
   `_ledger_spend_totals_msat` (runtime.py:2061–2080) returns `has_spend_ledger=False`
   for empty/zero windows, triggering fallback to the same lifetime profitability/
   dashboard burn for every window (:2320–2324). With no `spend_events` rows in 30d,
   all five windows are byte-identical.

Why coverage is always `unknown`: `_ledger_window_coverage` (runtime.py:2240–2262)
expects `covered_hours`/`coverage_hours` in the spend payload, but
`Database.get_spend_ledger_summary` (modules/database.py:3912–4001) never emits either
field — so status is `"unknown"` with `covered_hours: null` whenever the RPC responds,
and `insufficient_coverage` only when it doesn't respond at all.

## Revenue role

Indirect. It re-weights capex budgets toward channels with proven revenue-per-sat and
flags stranded capital for the planner's close/recycle path; it moves no sats itself.

## Uncertainties

- Should intake be windowed on the cl_revenue_ops side (a windowed
  `revenue-spend-ledger`-style income endpoint) or fixed in cl-hive? Cross-repo design
  question; this module is currently uninvolved in the ledger despite the naming.
- CE-2's all-or-nothing blend means one mocked/partial prof object in production would
  silently revert ranking to lifetime-gross; no telemetry exposes which mode ran.
- `forward_velocity` divides by `flow_window_days` resolved from two config fallbacks
  (:106–110); mismatch with the flow analyzer's actual window would skew velocity.
- Dead-capital stages come from the DB but stage progression logic lives elsewhere
  (capacity planner); staleness of `dead_capital_stage` rows is not validated here.
