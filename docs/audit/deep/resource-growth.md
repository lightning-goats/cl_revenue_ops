# Resource Exhaustion / Unbounded Growth — Deep Audit Phase 3B

**Scope:** DB tables (retention), in-memory caches/dicts/lists, and persisted state
files, audited for unbounded growth over daemon uptime. Never audited before this phase.

**Baseline:** `prod-baseline-T0.md` (node hive-nexus-01 LIVE: 53 MiB DB, 33 tables,
92,410-row `fee_changes` @ ~1,027 rows/day, oldest row 2026-04-03).

**Headline correction to T0:** T0 flagged `fee_changes` as the "primary exhaustion
risk … no observed pruning." **This is wrong.** `cleanup_old_data()` (database.py:6551)
DELETEs `fee_changes` older than 90 days, and it runs every flow-analysis cycle
(cl-revenue-ops.py:2587). The oldest row (2026-04-03) sits **exactly at the 90-day
cutoff** from the T0 capture (2026-07-02) — that is the retention working, not the
absence of it. `fee_changes` is at **steady state**: 1,027 rows/day × 90 days ≈ 92,430
rows ≈ the observed 92,410. It is BOUNDED, and it is the dominant table only in the
sense that 90 days of a high-frequency audit log is legitimately large.

---

## 1. 33-table classification

`cleanup_old_data(days_to_keep)` (database.py:6418) is the central retention routine,
invoked every flow cycle. `days_to_keep = max(8, flow_window_days + 1)` (default 8).

### BOUNDED (23 tables) — retention/trigger/close-eviction proven

| table | mechanism | retention |
|---|---|---|
| forwards | cleanup_old_data:6458-6524, **rolled into daily_forwarding_stats first** | 8 days |
| peer_connection_history | cleanup_old_data:6546 | 8 days (`days_to_keep`) |
| fee_changes | cleanup_old_data:6551 | 90 days |
| rebalance_history | cleanup_old_data:6552 | 90 days |
| financial_snapshots | cleanup_old_data:6556 | 365 days |
| budget_reservations | cleanup_old_data:6567 (terminal rows only) | 90 days |
| spend_reservations | cleanup_old_data:6573 + cleanup_stale_spend_reservations:4011 | 90 days |
| pair_rebalance_failures | cleanup_old_data:6578 + record-path prune:2632 | 90 days (expired) |
| mempool_fee_history | record_mempool_fee prune:7030 | 48 hours |
| peer_policies | cleanup_expired_policies:7169 (expires_at) | on expiry |
| peer_reputation | decay-to-(0,0) delete:6403 | decays out |
| channel_states | remove_closed_channel_data:6073 + flow reconcile (flow_analysis.py:1538) | on channel close |
| fee_strategy_state | close delete:6115 + reset_fee_strategy_state:2252 | on channel close |
| kalman_state | close delete:6094 | on channel close |
| channel_failures | close delete:6080 / :6216 | on channel close |
| channel_probes | close delete:6087 + :2045/:2052 | on channel close |
| dead_capital_stage | per-channel delete:7010 | on stage clear |
| hot_channel_protection_overrides | per-peer delete:6760 | on clear |
| config_overrides | per-key delete:6838 | bounded by config-key set |
| planner_candidates | per-peer replace/delete:6905 | replaced per plan |
| ignored_peers | operator blacklist add/remove | bounded by operator |
| lifetime_aggregates | `CHECK (id = 1)` | exactly 1 row |
| schema_version | single version marker | 1 row |

`plugin_flags` (bounded by the fixed flag-key set, effectively ≤ a handful of rows) is
the 24th effectively-bounded table.

### UNBOUNDED-IN-TIME (9 tables) — never pruned, but see projections

These are accounting sources of truth or lifetime records; `cleanup_old_data` explicitly
excludes them (database.py:6564: "Do NOT prune spend_events, rebalance_costs, or
planner_actions — they are accounting sources of truth"). None is deleted on channel
close.

| table | T0 rows | growth driver | ~rows/day (node-1 scale) | ~bytes/row | days-to-1GB* |
|---|---|---|---|---|---|
| daily_forwarding_stats | 661 | 1 row per (out-channel, day) | ≤ N_channels (~36) | ~80 | ~350,000 (≈950 yr) |
| daily_forwarding_stats_inbound | 673 | 1 row per (in-channel, day) | ≤ N_channels (~36) | ~72 | ~390,000 (≈1000 yr) |
| planner_actions | 981 | 1 row per planner action | ~5-15 (periodic planner) | ~250 | ~530,000 (≈1450 yr) |
| rebalance_costs | (not in T0**) | 1 row per rebalance | ~5-6 (tracks rebalance_history) | ~80 | ~2.6M (millennia) |
| spend_events | 63 | 1 row per capex spend | ~0.9 (currently idle since 2026-05-27) | ~200 | idle |
| closed_channels | 72 | 1 row per lifetime channel close | ≪1 | ~300 | lifetime-bounded |
| channel_closure_costs | 72 | 1 row per lifetime channel close | ≪1 | ~150 | lifetime-bounded |
| channel_costs | 105 | 1 row per channel ever opened (PK, upsert; NOT deleted on close) | ≪1 | ~120 | lifetime-bounded |
| planner_recycle_ops | 0 | 1 row per recycle op | ~0 (idle) | ~250 | idle |

*days-to-1GB = 1e9 bytes ÷ (rows/day × bytes/row), ignoring the existing ~53 MiB.
**`rebalance_costs` (database.py:717, written at :5332) is a real table but is absent
from T0's 33-count — T0 named only 32 distinct tables + 1 gap; this is the gap. It is
low-churn (rebalance cadence) and effectively negligible.

### Bottom line on DB growth

**No table is on a path to 1 GB within any operationally-relevant horizon at node-1
scale.** The dominant table (`fee_changes`) is bounded at 90 days and already at steady
state. Every genuinely-unbounded-in-time table is a low- or idle-churn accounting/lifetime
table whose days-to-1GB is measured in centuries.

**The only real lever is node-activity scaling.** All growth is proportional to node
throughput. `fee_changes` retention (90 days × fee-update rate) is the single largest
contributor and scales linearly: a 100×-busier node would hold ~9M `fee_changes` rows
(~1–2 GB from that table alone). That is a **retention-policy question, not a bug** — see
§4 for the recommended operator ruling.

---

## 2. In-memory cache / dict / list inventory

Swept `cl-revenue-ops.py` and all `modules/*.py` for runtime-growing structures and
checked each for an eviction bound. **Result: the codebase is already well-defended.**
Every per-channel/per-peer structure has close-eviction, per-cycle rebuild, TTL, a
size-cap, or a finite key domain — **except one** (fixed in §3).

| structure | file:line | keyed by | eviction | verdict |
|---|---|---|---|---|
| `_scid_to_peer_cache` | cl-revenue-ops.py:871 | SCID | hourly full clear (TTL 3600) + 512 neg-cap | BOUNDED |
| `ForceRateLimiter._timestamps` | cl-revenue-ops.py:121 | RPC command name (finite: revenue-set-fee, revenue-rebalance) | per-call window prune | BOUNDED |
| `_total_cost_budget_memo` | cl-revenue-ops.py:7039 | window-hours clamped [1,168] | bounded key domain (≤168) | BOUNDED |
| `_loop_heartbeats` | cl-revenue-ops.py:188 | loop name (finite) | finite keys | BOUNDED |
| `data_service._cache` | data_service.py:48 | RPC key | oldest-first evict, cap 256 + per-read TTL evict | BOUNDED |
| `data_service._forever` | data_service.py:51 | fixed keys (getinfo/listconfigs) | 2 entries | BOUNDED |
| fee_controller `_cycle_states`, `_channel_fee_states`, `_persisted_shared_fields`, `_last_dts_summaries` | fee_controller.py:2584-2644 | channel_id | `_prune_stale_states(active)` close-evict:4159 (called :4554) | BOUNDED |
| `_neighbor_fee_cache` | fee_controller.py:2616 | peer_id | >500 stale sweep + TTL:3252 | BOUNDED |
| `_cycle_peer_latency_memo` | fee_controller.py:2639 | peer_id | `.clear()` each cycle:4534 | BOUNDED |
| `_hive_member_set_at` | fee_controller.py:2604 | peer_id | `.pop` on release:2763; bounded by hive membership | BOUNDED |
| `_pending_fee_strategy_rows` | fee_controller.py:2633 | channel_id | reset each cycle:3954 | BOUNDED |
| `_dynamic_htlcmin_baselines` | fee_controller.py:2594 | — | **never written (dead)** | BOUNDED (empty) |
| observation windows | fee_controller.py:150 (MAX_OBSERVATIONS=200), :216 (MAX_BIAS_NUDGES=50) | — | hard cap | BOUNDED |
| `contextual_posteriors` | fee_controller.py:927 | context | top-104 slice | BOUNDED |
| rebalancer `_fee_cache` | rebalancer.py:305 | (chan,amt) | reset each cycle:1265/1398 | BOUNDED |
| rebalancer `_peer_inbound_fees` | rebalancer.py:314 | peer_id | wholesale rebuild:1846 | BOUNDED |
| rebalancer `_pending` | rebalancer.py:302 | to_channel | `.pop` on completion:1940/1982/2089 | BOUNDED |
| rebalancer `source_failure_counts` | rebalancer.py:200 | channel_id | **never written (dead)**; orphan pruner `prune_stale_source_failures`:233 never called | BOUNDED (empty) |
| `_pair_failures` | rebalance_engine_v2.py:163 | (src,dst) scid | pop-empty:2021 + pop-success:2055 | BOUNDED |
| `_dest_success_rate_memo` | rebalance_engine_v2.py:171 | dest | `.clear()` each cycle:1325 | BOUNDED |
| `_kalman_filters` | flow_analysis.py:782 | channel_id | close-evict `.pop`:1541 | BOUNDED |
| `_flow_cache` | flow_analysis.py:794 | channel_id | wholesale replace:1361 + TTL 300 | BOUNDED |
| `_profitability_cache` | profitability_analyzer.py:589 | scid | replace:669 + pop-closed:1168/1241 | BOUNDED |
| `policy_manager._cache` | policy_manager.py:206 | peer_id | write-through, clear on invalidate; bounded by policy count | BOUNDED |
| `policy_manager._change_timestamps` | policy_manager.py:215 | peer_id | >500 sweep:272 + per-window prune | BOUNDED |
| `segment _observations` | segment_observations.py:35 | (list) | max 200 + TTL 900 | BOUNDED |
| `hive_router._route_cache` / `_fleet_balances` | hive_router.py:56-58 | peer_id | clear:284/395 / replace:425 | BOUNDED |
| capacity_planner per-cycle gossip caches | capacity_planner.py:144-145 | — | `.clear()` each cycle:260 | BOUNDED |
| **`_boltz_balance_last_action`** | **cl-revenue-ops.py:850** | **channel_id** | **NONE (before fix)** | **UNBOUNDED → FIXED §3** |

**Notes on two structures the T0 prompt hypothesized as risks:**
- **`ForceRateLimiter`** — hypothesized "bound the deque." No unbounded growth exists:
  `_timestamps` is keyed by the *finite* set of RPC command strings (only
  `"revenue-set-fee"` and `"revenue-rebalance"` are ever passed), and each per-command
  list is pruned to the sliding window on every call. No change needed.
- **`source_failure_counts`** (rebalancer.py:200) — its close-eviction pruner
  (`prune_stale_source_failures`:233) has **no caller**, but the dict also has **no
  writer**, so it stays empty. Dead code, not a leak.

---

## 3. What was FIXED (non-behavioral)

### `_boltz_balance_last_action` unbounded growth — FIXED

- **File:** cl-revenue-ops.py:850 (`Dict[str, int]`, channel_id → last-action unix ts).
- **Defect:** purely a per-channel Boltz-balance cooldown gate (read :8639/:8998, written
  :8650/:9003). **Nothing ever evicted entries** — no `pop`/`del`/`clear`/TTL. A
  long-running node accumulated one entry per channel ever swapped, including closed
  channels, monotonically for the process lifetime.
- **Fix:** added `_BOLTZ_BALANCE_ACTION_TTL_SECONDS = 30 * 86400` and
  `_prune_boltz_balance_actions(now)` (acquires `_boltz_balance_lock`), called once at the
  start of both balance-cycle executors (`_execute_boltz_balance_cycle` and the treasury
  executor) before the recommendation loop.
- **Why non-behavioral:** the value is only ever compared as
  `(now - last_ts) < cooldown_seconds`. The default cooldown is 4h and hint overrides are
  hours; the 30-day TTL dwarfs any realistic cooldown by ≥180×. Any entry the prune
  removes is already `> 30 days` old, i.e. `(now - last_ts)` already ≫ any cooldown, so
  the cooldown check already treats it as expired. Removing it changes no decision. The
  dict is now bounded to "channels touched within 30 days" (≈ active-channel count).
- **Tests (TDD):** `tests/test_boltz_balance_action_eviction.py` — prune removes
  stale/keeps recent, TTL-dwarfs-cooldown safety invariant, empty-dict no-op.

**No other in-memory structure required a fix** — the sweep found every other cache
already bounded.

---

## 4. What needs an OPERATOR RETENTION RULING (behavioral — data loss, NOT applied)

Adding a DELETE-based retention to a currently-unbounded table is behavioral (drops rows
an operator may want). These are surfaced for ruling, not applied:

1. **`fee_changes` — the one worth acting on (dominant table, node-activity-scaled).**
   Currently 90-day retention, ~92k rows at node-1 scale, and the single largest DB
   contributor. Recommendation: rule on whether 90 days of per-fee-change audit rows is
   needed, or shorten to **~30 days** and/or **roll aggregate counts into an aggregate
   table before deletion** — mirroring the low-risk `forwards → daily_forwarding_stats`
   pattern (cleanup_old_data:6497). Because it is an audit log (not an accounting source
   of truth), a rolled-up retention is **low-risk**. This is the only DB lever that
   matters if the node scales up materially.

2. **`daily_forwarding_stats` / `daily_forwarding_stats_inbound`** — unbounded in time but
   grow ≤ N_channels/day (centuries to 1 GB). These are already the compact roll-up
   *target* of `forwards`. Recommendation: OPTIONAL very-long retention (e.g. 2–3 years)
   only if an operator wants a hard cap; otherwise leave as-is (negligible).

3. **`planner_actions`, `rebalance_costs`, `spend_events`, `planner_recycle_ops`,
   `closed_channels`, `channel_closure_costs`, `channel_costs`** — accounting/lifetime
   sources of truth, explicitly excluded from pruning (database.py:6564). Centuries-to-1GB
   or lifetime-bounded. Recommendation: **no retention** (deleting these loses accounting
   history); leave as sources of truth. `channel_costs` is the only one that could
   optionally be close-evicted (it is keyed by channel_id and not deleted on close), but
   it is used for lifetime cost attribution, so that too is a behavioral ruling, not a
   bug.

4. **Boltz swap journal file (disk, not DB, not RAM)** — `boltz_manager.py:1425`
   `_save_swap_journal(merged)` dedups by swap id but never caps the entry count, so
   `cl_revenue_ops_swap_journal.json` grows one JSON record per swap ever recorded.
   Low-churn (swaps are infrequent) but strictly monotonic on disk. Recommendation: rule
   on a cap (e.g. keep last N=500 swaps or 180 days) — behavioral because the journal
   backs swap-history listing and capex attribution, so it is surfaced, not applied.

---

## 5. Findings

```
Low     | resource/DB      | modules/database.py:6551 | T0 mislabel: fee_changes is BOUNDED (90d retention via cleanup_old_data, runs every flow cycle cl-revenue-ops.py:2587); oldest row sits exactly at 90d cutoff = retention working, not absent | steady-state ~92k rows @ node-1 scale; scales linearly with fee-update rate (100x node ≈ 1-2 GB from this table alone → retention ruling §4.1)
Medium  | resource/memory  | cl-revenue-ops.py:850    | _boltz_balance_last_action (channel_id->cooldown ts) had NO eviction — grew one entry per channel ever swapped, incl. closed, for process lifetime | ~1 dict entry per distinct swapped channel forever; unbounded in channel-churn → FIXED: 30d TTL prune (non-behavioral), bounded to ~active-channel count
Low     | resource/disk    | modules/boltz_manager.py:1425 | swap journal JSON dedups by swap id but never caps count; cl_revenue_ops_swap_journal.json grows per swap ever recorded | ~1 JSON record (~0.3-1 KB) per swap forever; low churn (swaps infrequent) → retention ruling §4.4
Info    | resource/memory  | modules/rebalancer.py:233 | prune_stale_source_failures (close-eviction pruner) has no caller AND source_failure_counts (rebalancer.py:200) has no writer — dead code pair | 0 growth (empty dict); cosmetic dead-code, not a leak
Info    | resource/memory  | modules/fee_controller.py:2594 | _dynamic_htlcmin_baselines declared but never written | 0 growth; dead field
Info    | resource/DB      | modules/database.py:717   | rebalance_costs table exists + written (:5332) but is absent from T0's 33-table row-count capture (T0 gap) | low churn (rebalance cadence); unbounded-in-time accounting source, centuries-to-1GB
```

---

## 6. Verification

- `python3 -m pytest tests/ -q` → **3007 passed, 5 skipped** (skips are env-only:
  pyln.testing absent, CLN_INTEGRATION unset). Full green.
- New test file: `tests/test_boltz_balance_action_eviction.py` (3 tests, TDD red→green).
