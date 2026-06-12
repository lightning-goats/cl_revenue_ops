# Intent Contract: modules/database.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`Database` (modules/database.py:246) is the single SQLite persistence layer for the whole
plugin: channel flow state, fee-controller P&O state, fee/rebalance audit history, raw
forwards, cost accounting (opens, closes, rebalances), generic spend events and
reservations, peer reputation/uptime, planner candidates/actions, and operator config
overrides. It owns connection lifecycle (one connection per thread via `threading.local`,
modules/database.py:278), schema creation/migration (`initialize`, modules/database.py:560),
input sanitization, and data retention (aggressive 8-day forwards pruning with lifetime
rollup, modules/database.py:6130). Every budget ceiling, profitability number, and
historical claim the plugin makes ultimately rests on rows this module wrote.

## Inputs / Outputs

- **Instantiation**: `Database(config.db_path, safe_plugin)` at cl-revenue-ops.py:1870
  (production file `advisor.db` in the plugin directory).
- **Callers**: essentially every module — fee_controller, rebalancer, rebalance_engine_v2,
  profitability_analyzer, capex_budget (`get_total_capex_by_channel`,
  `get_spend_ledger_summary`, `get_channel_rebalance_success_rate`), capacity_planner
  (planner_candidates/actions, recycle ops), capital_efficiency
  (`get_dead_capital_stages`, modules/database.py:6671), boltz_manager
  (`record_spend_event`).
- **RPC surfaces backed directly by this module** (cl-revenue-ops.py): `revenue-spend-ledger`
  (:5935 → `get_spend_ledger_summary` :3912), `revenue-spend-reserve` (:5961, serialized by
  `_spend_reserve_lock` :5958), plus dashboard/budget RPCs that aggregate via
  `get_total_rebalance_fees` (:2663), `get_total_routing_revenue` (:2770), etc.
- **Schema** (all `CREATE TABLE IF NOT EXISTS`, modules/database.py:560–1297 + migrations):
  `schema_version`, `channel_states` (+flow-v2/kalman/temporal columns via migrations
  :1436–:1537), `fee_strategy_state`, `fee_changes`, `rebalance_history` (+`payment_hash`
  for pending settlement :648), `forwards`, `channel_costs`, `rebalance_costs`,
  `channel_failures`, `pair_rebalance_failures`, `peer_reputation`,
  `peer_connection_history`, `lifetime_aggregates` (singleton row :747),
  `channel_probes`, `ignored_peers` (legacy, migrated to) `peer_policies`,
  `hot_channel_protection_overrides`, `config_overrides`, `mempool_fee_history`,
  `daily_forwarding_stats`(+`_inbound`), `budget_reservations`, `spend_reservations`,
  `spend_events`, `financial_snapshots`, `channel_closure_costs`, `closed_channels`,
  `planner_candidates`, `planner_actions`, `dead_capital_stage`, `planner_recycle_ops`,
  `kalman_state` (:1495), `plugin_flags` (:1955).

## Invariants

- **DB-1** Each thread gets its own SQLite connection; connections are created with
  WAL journal mode, `busy_timeout=5000`, `synchronous=NORMAL`, `foreign_keys=ON`
  (`_get_connection`, modules/database.py:278–340). No connection is *used* (execute etc.)
  by more than one thread; connections are however created with
  `check_same_thread=False` (:307) specifically so the shutdown handler can `close()`
  them cross-thread — that close is the one sanctioned cross-thread touch.
- **DB-2** Rebalance budget reservations are atomic and ceiling-enforced: `_reserve_budget_atomic`
  (modules/database.py:94–180) runs `BEGIN IMMEDIATE`, sums `rebalance_costs` +
  active `budget_reservations` against daily and optional weekly limits, and rolls back
  rather than over-reserving (daily rollback :138–140, weekly rollback :160–162).
- **DB-3** Generic spend events are idempotent by `event_id`: `record_spend_event`
  uses `INSERT OR REPLACE` keyed on `event_id` (modules/database.py:3816–3820) and
  lowercases categories (:3809), so replays (e.g. repeated Boltz journal updates)
  never double-count.
- **DB-4** Pruning never loses lifetime revenue: `cleanup_old_data`
  (modules/database.py:6130+) aggregates to-be-deleted forwards into
  `daily_forwarding_stats` and `daily_forwarding_stats_inbound` (per-channel, per-day)
  inside one `BEGIN IMMEDIATE` transaction before deleting. The singleton
  `lifetime_aggregates` row is **no longer written** by pruning — it is frozen legacy
  history from the pre-daily-stats era (:6225–6226) that revenue rollups still add in
  (:5254–5256).
- **DB-5** Read paths tolerate both SCID spellings: `_scid_aliases`
  (modules/database.py:40–51) expands canonical `AxBxC` and legacy `A:B:C` forms, used by
  revenue/P&L lookups so renamed channels keep their history.
- **DB-6** Monetary inputs are clamped before write: `_sanitize_fee` (:488) zeroes
  non-numeric/NaN/negative fees and caps at `MAX_FEE_SATS`; `_sanitize_amount` (:526)
  zeroes non-numeric/NaN but **permits negative amounts** (clamping magnitude to
  `MAX_AMOUNT_SATS`, :550–557) — negative spend events are storable by design;
  `_validate_channel_id`/`_validate_peer_id` (:458/:473) gate identifiers.
- **DB-7** `get_spend_ledger_summary(window_hours)` (modules/database.py:3912) windows all
  sums by `now - window_hours*3600`, but names the fields `spent_24h_sats` /
  `reserved_24h_sats` regardless of the requested window (:3964–3965), and emits **no**
  `covered_hours`/`coverage_hours` field — downstream consumers (cl-hive metabolism
  ledger) cannot tell how much history backs the number.

## Revenue role

Plumbing, indirect. It earns nothing itself, but budget ceilings (DB-2), spend
idempotency (DB-3), and revenue-history preservation (DB-4) are the substrate for every
fee, rebalance, and capex decision; corruption here silently mis-prices the whole fleet.

## Observable surface

`revenue-spend-ledger.json` in the hermes corpus is a direct dump of
`get_spend_ledger_summary`; `revenue-capex-status.json` and the revenue dashboard reflect
its cost/aggregate queries; cl-hive's `hive-organism-status.json` metabolism-ledger
sources name `revenue_spend_ledger` and `revenue_total_cost_budget`. The DB file itself
(`advisor.db`) is the ground truth but is not collected.

## Uncertainties

- `budget_reservations` (rebalance) and `spend_reservations` (generic) are parallel
  systems; whether any flow can hold both simultaneously for one action (double
  reservation) has not been traced end-to-end.
- The `spent_24h_sats` field-name misnomer (DB-7) is consumed by external code; renaming
  it is a cross-repo compatibility question.
- `schema_version` table exists but migrations are ad-hoc `ALTER TABLE ... except
  OperationalError`; actual version row appears never bumped past 1.
- Retention interplay: `_revenue_by_size_bucket_sql` (:58) documents ~8-day accuracy, but
  callers requesting `window_days=30` on forwards-backed queries silently undercount.
