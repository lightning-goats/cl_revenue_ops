# Persistence map (baseline 5e8f747)

Pin test: `tests/test_persistence_inventory.py`.
Prior art (authoritative for retention/growth): `docs/audit/deep/resource-growth.md`
(per-table BOUNDED/unbounded classification with retention line cites),
`docs/audit/deep/prod-baseline-T0.md` (production row counts, 53 MiB DB).

## SQLite database

Path: CLN lightning-dir `revenue_ops.db`; owner `modules/database.py`
(single `Database` class, 7,778 lines). All DDL is
`CREATE TABLE/INDEX IF NOT EXISTS` + guarded `ALTER TABLE` in
`Database.__init__` — additive-only, no migration framework.

`schema_version` is WRITE-ONLY by operator ruling DD9/MIG-3 (2026-07-02,
`modules/database.py:606`): the plugin stamps it but never refuses to run
on version mismatch. Any refactor migration tooling must not assume a
version gate exists.

### Tables (40) and writers

| Table | Written by (domain) |
|---|---|
| acquisition_experiments | restart-safe single-lane acquisition episodes, baseline restoration, and bounded-loss evidence (fee controller) |
| budget_reservations | rebalance budget reservations (database.py `reserve_budget`) |
| channel_closure_costs | close-cost accounting (planner/profitability) |
| channel_costs | per-channel open/rebalance cost ledger (profitability) |
| channel_failures | failure evidence (rebalance/fee decisioning) |
| channel_probes | probe results (fee controller) |
| channel_states | flow-analysis channel state (kalman/flow) |
| closed_channels | closed-channel P&L archive (lifecycle events) |
| config_overrides | `revenue-config set` persisted overrides |
| daily_forwarding_stats | per-day forwarding aggregates (outbound) |
| daily_forwarding_stats_inbound | per-day forwarding aggregates (inbound) |
| dead_capital_stage | staged dead-capital closes (capacity planner) |
| fee_changes | fee broadcast history (fee controller; 90d retention) |
| fee_strategy_state | DTS/PID strategy state (fee controller) |
| financial_snapshots | 24h financial snapshots (snapshot thread) |
| forward_archive_v1 | canonical CLN `listforwards` evidence (observational archive; 400d raw retention after verified aggregation) |
| forward_archive_sync_state_v1 | independent created/updated cursor watermarks (observational archive) |
| forward_daily_channel_v1 | replacement daily/channel aggregates derived from the canonical archive (retained indefinitely) |
| forward_archive_coverage_v1 | per-UTC-day archive completeness and reconciliation evidence (retained indefinitely) |
| forwards | forward event ledger (flow/profitability source) |
| hot_channel_protection_overrides | hot-channel protection (fee/rebalance) |
| ignored_peers | `revenue-ignore` operator list |
| kalman_state | kalman filter state (flow analysis) |
| lifetime_aggregates | lifetime per-channel aggregates (profitability) |
| lnplus_peers | LN+ peer history incl. defections |
| lnplus_swaps | LN+ swap obligations (external contracts) |
| mempool_fee_history | chain-fee history (fee floor / planner) |
| pair_rebalance_failures | per-pair rebalance failure evidence |
| peer_connection_history | peer uptime evidence |
| peer_policies | operator policy tags (no_close, bans, strategies) |
| peer_reputation | peer reputation scores (planner) |
| planner_actions | planner action history (opens/closes) |
| planner_candidates | planner candidate history |
| planner_recycle_ops | capital-recycle operations |
| plugin_flags | misc persistent flags |
| rebalance_costs | per-channel rebalance cost ledger |
| rebalance_history | rebalance attempt/result history |
| schema_version | write-only version stamp (see above) |
| spend_events | generic spend ledger — events |
| spend_reservations | generic spend ledger — reservations |

Retention/growth classification per table: see
`docs/audit/deep/resource-growth.md` (23 bounded with retention
mechanisms; corrects the earlier "fee_changes unbounded" claim — it is
90d-bounded).

The four `forward_*_v1` archive tables are additive Phase 0.6 measurement
infrastructure owned by `modules/forward_archive.py`. They are deliberately
not read by fee, flow, profitability, planner, or rebalance decision paths.
The raw archive has a 400-day retention floor and can be pruned only after the
affected day has a complete reconciled aggregate; daily aggregates and
coverage are indefinite evidence. Older plugin code ignores the tables, so
code rollback does not require a destructive schema rollback.

## Restart-recovery state (Workstream D/E input)

State that must survive restart and its current recovery path:

- `spend_reservations` / `spend_events` — generic spend ledger;
  stale-reservation cleanup `cleanup_stale_spend_reservations`
  (`modules/database.py:4168`) and RPC `revenue-spend-release-stale`
- `budget_reservations` — rebalance reservations (:3693 lineage)
- `lnplus_swaps` — external obligations (must be honored across restart;
  refactor invariant 6)
- Boltz swap journal (in `boltz_manager`; boltzcli owns swap state
  externally, journal reconciles)
- `dead_capital_stage` — staged closes pending execution
- In-flight sendpay: recovered via waitsendpay/listpays on next cycle

## Econ ledger (Phase 1 wiring, 2026-07-12)

`econ_ledger.db` — a SEPARATE sqlite file beside `revenue_ops.db`, owned
by `modules/econ_ledger.py` (table `econ_ledger_events`, append-only).
Created lazily by `modules/econ_shadow.py` only when
`econ_shadow_enabled` is set; records `intent_proposed` events from live
fee cycles AND (Phase 2 pilot A) the full generic spend lifecycle
(budget_reserved / cost_recorded / execution_succeeded /
reservation_released) via `Database.spend_journal` hooks. Not part of the production DB schema (the table pin test
scans `modules/database.py` only). Becomes authoritative in Phase 2.

## CLN datastore keys (telemetry projections, read-only contracts)

Writers: `data_service.datastore_push` (`modules/data_service.py:461`) +
one raw-RPC fallback (`modules/rebalance_engine_v2.py:2869`).

Complete key inventory at baseline (from
`grep -rn "datastore_push" modules/ cl-revenue-ops.py`):

| Key | Writer | Contract |
|---|---|---|
| `["revenue","profitability-summary"]` | `modules/profitability_analyzer.py:758` | `docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md` |
| `["revenue","capex-summary"]` | `cl-revenue-ops.py:7242` | `docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md` (TTL 1800s) |
| `["revenue","segment-observations"]` | `modules/rebalance_engine_v2.py:2862` (key from `modules/segment_observations.py:13`) | `docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md` (schema_version 1) |
| `["revenue","status"]` | `cl-revenue-ops.py:3510` | undocumented (status mirror) |
| `["revenue","fee-bounds"]` | `cl-revenue-ops.py:3515` | undocumented (fee bounds mirror) |
| `["revenue","dashboard"]` | `cl-revenue-ops.py:3557` | undocumented (dashboard mirror) |

The three documented keys are contract-tested by
`tests/test_cross_plugin_contracts.py` — the refactor's projection layer
must keep that test green (refactor invariant 3). The three undocumented
mirrors need either a contract doc or an explicit "internal, no
compatibility promise" marker during Workstream I.
