# Performance Baseline — Deep Audit Phase 7

Reproducible cProfile + wall-clock profile of the fee-adjustment cycle, the
rebalance planning cycle, the profitability cycle, and the heaviest DB read
queries, exercised against a synthetic SQLite DB seeded to the **production T0
row counts** (docs/audit/deep/prod-baseline-T0.md): ~92,410 fee_changes,
~600 forwards (8-day window), 477 rebalance_history, ~500 rebalance_costs,
~1080 daily rollup rows, 63 spend_events, 360 financial_snapshots, 2,706
peer_connection_history — across a 36-channel topology.

- **Generator:** `tools/perf/profile_cycles.py` (+ `tools/perf/cycle_driver.py`)
- **Reproduce:** `python3 tools/perf/profile_cycles.py --repeats 30 --markdown OUT`
- **Regression guard:** `tests/test_perf_regression_guard.py` (fast; structural
  query-plan check + generous latency ceiling on the top-3 hot paths)
- **Corpus note:** profiled against the **synthetic T0-scale DB only**. The live
  production DB (`revenue_ops.db`, ~53 MiB on hive-nexus-01) is reachable only
  via read-only ssh on the node and is **not** present in this worktree, so no
  read-only copy was profilable here. The synthetic DB reproduces the T0 row
  counts and index set exactly; pass `--db PATH` to profile a real copy when one
  is obtainable. The repo-root `advisor.db` is a 0-byte placeholder.

## Headline

**The codebase is adequately indexed and performance-engineered at production
scale. No egregious performance findings.** All cycle and query timings are
single-digit milliseconds or less at T0 scale; every scale-sensitive read is
index-backed (verified by `EXPLAIN QUERY PLAN`), and the per-cycle work is
O(channels) (~36), not O(history). This is a clean outcome — no fixes required.

### Top cumulative-time functions (mean over 30 runs, T0 scale)

**Fee-adjustment cycle — `FeeController.adjust_all_fees` ≈ 1.2 ms**
- `_adjust_all_fees_inner` → `_adjust_all_fees_channel_loop` → `_adjust_channel_fee` (×36)
- per-channel `get_volume_since` / `get_forward_count_since` (×36) — indexed on
  `forwards(out_channel, timestamp)`; `forwards` is pruned to ~8 days (~600 rows)
- `_get_channels_info` (one `listpeerchannels`), `_prefetch_neighbor_gossip`

**Rebalance planning cycle — `RebalanceEngine.run_cycle` ≈ 0.8 ms**
- `_run_cycle_locked` → `find_candidates` → `_build_snapshot` → `build_state_snapshot`
- `_normalize_channel_input` (×36), `_historical_fee_rate_ppm` (×72, pure arithmetic — no DB)

**Profitability cycle — `ChannelProfitabilityAnalyzer.analyze_all_channels` ≈ 4.6 ms**
- batched `get_all_channels_full_pnl(30)` + `_get_all_revenue_data` +
  `_get_all_fee_states` (one query each), then `analyze_channel` (×36)
- per-channel `get_channel_cost` / `get_channel_rebalance_costs` /
  `get_channel_rebalance_success_rate` / `get_last_forward_time_any_direction`
  (×36) — all index-backed; a benign N+1 over the small 36-channel set

**Heaviest DB read — `get_all_channels_full_pnl` ≈ 1.4–1.6 ms** (30d/90d window).
Three GROUP-BY passes over `forwards` (indexed by timestamp) + daily rollups +
`rebalance_costs`; a `USE TEMP B-TREE FOR GROUP BY` appears because the group key
is `REPLACE(out_channel,':','x')` (a scid-alias normalization) rather than a raw
column, but the input is already reduced to the windowed row set by the timestamp
index, so the sort is over a tiny set. Not a concern at T0 scale.

## Egregious-findings sweep (the four risk classes from the plan)

| risk class | result | evidence |
|---|---|---|
| O(n^2)-or-worse over `forwards`/`fee_changes` history | **none** | cycles are O(channels)=36; no nested loop over history rows |
| query with no supporting index doing a full scan of a large table | **none** | every scale-sensitive read shows `USING INDEX`/`USING COVERING INDEX` in `EXPLAIN QUERY PLAN` (see below). The only unbounded reads of `fee_changes` (92k) are `get_recent_fee_changes` (`ORDER BY timestamp DESC LIMIT` → index scan, stops at LIMIT) and the retention `DELETE ... WHERE timestamp < ?` (indexed). All `forwards` scans are on a table pruned to ~8 days (~600 rows). |
| per-cycle recompute that rescans all history each time | **none** | fee/profitability/rebalance cycles read windowed, indexed slices or batched aggregates; profitability caches for its TTL. No full-history rescan. |
| unbounded in-memory build proportional to a large table | **none** | in-memory dicts are keyed per channel (~36 entries); `get_all_channels_full_pnl` builds one dict per active channel, not per history row |

### Index coverage confirmed (existing schema, database.py)

`fee_changes(channel_id, timestamp)` + `fee_changes(timestamp)`;
`forwards(timestamp)`, `forwards(out_channel, timestamp)`, `forwards(in_channel, timestamp)`;
`daily_forwarding_stats(date)` (+ inbound); `rebalance_costs(timestamp, ...)` covering
indexes; `rebalance_history(to_channel, timestamp) WHERE status='success'` (partial);
`spend_events(timestamp, channel_id, amount_sats)` covering. No index was missing.

## Fixes applied

None. The profile revealed no egregious issue meeting the fix bar (missing index
or obviously-linear-rewritable O(n^2)). Per the Phase 7 mandate ("fix only
egregious findings; do NOT refactor hot paths for micro-optimizations"), no hot
path was touched. A clean, adequately-indexed codebase is a valid outcome.

## Non-egregious observations (informational — NOT fixed, no ledger action)

- **Profitability N+1 over channels:** `analyze_channel` issues ~4 tiny indexed
  per-channel queries (`get_channel_cost`, `get_channel_rebalance_costs`,
  `get_channel_rebalance_success_rate`, `get_last_forward_time_any_direction`).
  This is O(channels)=36, ~1–2 ms total. Batchable in principle (like the
  already-batched P&L path), but well below the micro-optimization bar and out
  of Phase 7 scope. Informational only.
- **`USE TEMP B-TREE FOR GROUP BY`** in `get_all_channels_full_pnl` /
  `get_recent_fee_changes` variants stems from grouping on
  `REPLACE(col,':','x')` (scid-alias normalization). The temp sort is over the
  already index-reduced windowed row set (tiny), so it does not materialize a
  large intermediate. No action.

---

_The machine-generated tables and full cProfile cumtime dumps follow._

<!-- generated by tools/perf/profile_cycles.py; source: synthetic DB seeded to T0 row counts -->

## DB read-query timings

| query / cycle | best ms | mean ms | worst ms |
|---|---:|---:|---:|
| get_all_channels_full_pnl(30) | 1.392 | 1.408 | 1.449 |
| get_all_channels_full_pnl(90) | 1.540 | 1.560 | 1.610 |
| get_total_routing_revenue(30d) | 0.124 | 0.127 | 0.147 |
| get_node_realized_fee_ppm_30d | 0.242 | 0.245 | 0.253 |
| get_recent_fee_changes(limit=50) | 0.098 | 0.101 | 0.137 |
| get_recent_fee_changes(per-channel) | 0.103 | 0.106 | 0.135 |
| get_daily_rebalance_spend(24h) | 0.014 | 0.015 | 0.049 |
| get_budget_status(30d) | 0.011 | 0.011 | 0.021 |
| get_spend_ledger_summary(30d) | 0.037 | 0.040 | 0.065 |
| get_category_spend_sats(rebalance,30d) | 0.003 | 0.003 | 0.011 |

## Cycle timings

| query / cycle | best ms | mean ms | worst ms |
|---|---:|---:|---:|
| profitability.analyze_all_channels(force=True) | 4.558 | 4.585 | 4.637 |
| fee_controller.adjust_all_fees | 1.167 | 1.211 | 1.302 |
| rebalance_engine.run_cycle | 0.789 | 0.811 | 0.869 |

## EXPLAIN QUERY PLAN (scale-sensitive reads)

**forwards revenue-sum (get_total_routing_revenue)**
  - `SEARCH forwards USING INDEX idx_forwards_time (timestamp>?)`

**daily_forwarding_stats date-range sum**
  - `SEARCH daily_forwarding_stats USING INDEX idx_daily_fwd_stats_date (date>? AND date<?)`

**fee_changes ORDER BY timestamp DESC LIMIT**
  - `SCAN fee_changes USING INDEX idx_fee_changes_time`

**fee_changes per-channel recent**
  - `SEARCH fee_changes USING INDEX idx_fee_changes_channel (channel_id=?)`

**forwards GROUP BY REPLACE(out_channel) (full_pnl exit)**
  - `SEARCH forwards USING INDEX idx_forwards_time (timestamp>?)`
  - `USE TEMP B-TREE FOR GROUP BY`

**rebalance_costs SUM by channel (full_pnl)**
  - `SEARCH rebalance_costs USING INDEX idx_rebalance_costs_time (timestamp>?)`
  - `USE TEMP B-TREE FOR GROUP BY`

**rebalance_costs timestamp SUM (budget)**
  - `SEARCH rebalance_costs USING COVERING INDEX idx_rebalance_costs_time (timestamp>?)`

**spend_events timestamp SUM (budget)**
  - `SEARCH spend_events USING COVERING INDEX idx_spend_events_time_channel (timestamp>?)`


## cProfile cumtime (top 20) — DB reads

### get_all_channels_full_pnl(30)

```
         5129 function calls in 0.011 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.011    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:302(<lambda>)
        7    0.001    0.000    0.011    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3202(get_all_channels_full_pnl)
       21    0.009    0.000    0.009    0.000 {method 'execute' of 'sqlite3.Connection' objects}
       21    0.001    0.000    0.001    0.000 {method 'fetchall' of 'sqlite3.Cursor' objects}
      756    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3228(_entry)
      756    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:13(normalize_scid)
     1260    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:73(base_to_sats_ceil)
      756    0.000    0.000    0.000    0.000 {method 'setdefault' of 'dict' objects}
      756    0.000    0.000    0.000    0.000 {method 'replace' of 'str' objects}
      253    0.000    0.000    0.000    0.000 {built-in method builtins.max}
      252    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:81(base_to_sats_floor)
      252    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:91(base_delta_to_sats_toward_zero)
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        7    0.000    0.000    0.000    0.000 {built-in method time.time}
        7    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}



```

### get_all_channels_full_pnl(90)

```
         5129 function calls in 0.012 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.012    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:304(<lambda>)
        7    0.001    0.000    0.012    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3202(get_all_channels_full_pnl)
       21    0.009    0.000    0.009    0.000 {method 'execute' of 'sqlite3.Connection' objects}
       21    0.001    0.000    0.001    0.000 {method 'fetchall' of 'sqlite3.Cursor' objects}
      756    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3228(_entry)
      756    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:13(normalize_scid)
     1260    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:73(base_to_sats_ceil)
      756    0.000    0.000    0.000    0.000 {method 'setdefault' of 'dict' objects}
      756    0.000    0.000    0.000    0.000 {method 'replace' of 'str' objects}
      253    0.000    0.000    0.000    0.000 {built-in method builtins.max}
      252    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:81(base_to_sats_floor)
      252    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:91(base_delta_to_sats_toward_zero)
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        7    0.000    0.000    0.000    0.000 {built-in method time.time}
        7    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}



```

### get_total_routing_revenue(30d)

```
         54 function calls in 0.001 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:306(<lambda>)
        7    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:2829(get_total_routing_revenue)
        7    0.001    0.000    0.001    0.000 {method 'execute' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        7    0.000    0.000    0.000    0.000 {built-in method time.time}
        1    0.000    0.000    0.000    0.000 {built-in method builtins.max}



```

### get_node_realized_fee_ppm_30d

```
         54 function calls in 0.002 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.002    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:308(<lambda>)
        7    0.000    0.000    0.002    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:2784(get_node_realized_fee_ppm_30d)
        7    0.002    0.000    0.002    0.000 {method 'execute' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        7    0.000    0.000    0.000    0.000 {built-in method time.time}
        1    0.000    0.000    0.000    0.000 {built-in method builtins.max}



```

### get_recent_fee_changes(limit=50)

```
         61 function calls in 0.001 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:310(<lambda>)
        7    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:2314(get_recent_fee_changes)
        7    0.000    0.000    0.000    0.000 {method 'fetchall' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.min}
        8    0.000    0.000    0.000    0.000 {built-in method builtins.max}
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}



```

### get_recent_fee_changes(per-channel)

```
         68 function calls in 0.001 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:312(<lambda>)
        7    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:2314(get_recent_fee_changes)
        7    0.000    0.000    0.000    0.000 {method 'fetchall' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:77(_scids)
        7    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.min}
        8    0.000    0.000    0.000    0.000 {built-in method builtins.max}
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}



```

### get_daily_rebalance_spend(24h)

```
         131 function calls in 0.000 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:314(<lambda>)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:4342(get_daily_rebalance_spend)
       28    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
       28    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3727(count_stale_reservations)
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
       14    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.round}
       14    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
       14    0.000    0.000    0.000    0.000 {built-in method time.time}
        1    0.000    0.000    0.000    0.000 {built-in method builtins.max}



```

### get_budget_status(30d)

```
         75 function calls in 0.000 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:316(<lambda>)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:4416(get_budget_status)
       28    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
       14    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        1    0.000    0.000    0.000    0.000 {built-in method builtins.max}



```

### get_spend_ledger_summary(30d)

```
         180 function calls in 0.000 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:318(<lambda>)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:4235(get_spend_ledger_summary)
       56    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
       28    0.000    0.000    0.000    0.000 {method 'fetchall' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:4174(_earliest_evidence_timestamp)
       28    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:4198(_coverage_from_earliest)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {built-in method builtins.round}
        8    0.000    0.000    0.000    0.000 {built-in method builtins.max}
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        7    0.000    0.000    0.000    0.000 {built-in method time.time}



```

### get_category_spend_sats(rebalance,30d)

```
         61 function calls in 0.000 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:320(<lambda>)
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:4036(get_category_spend_sats)
        7    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 /usr/lib/python3.12/contextlib.py:141(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:275(profiled)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        7    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        7    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:316(_get_connection)
        7    0.000    0.000    0.000    0.000 {method 'strip' of 'str' objects}
        7    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        7    0.000    0.000    0.000    0.000 {method 'lower' of 'str' objects}
        1    0.000    0.000    0.000    0.000 {built-in method builtins.max}



```

## cProfile cumtime (top 20) — cycles

### profitability.analyze_all_channels(force=True)

```
         8212 function calls in 0.007 seconds

   Ordered by: cumulative time
   List reduced from 113 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.007    0.007 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:412(<lambda>)
        1    0.000    0.000    0.007    0.007 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:610(analyze_all_channels)
       36    0.000    0.000    0.003    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:757(analyze_channel)
      151    0.003    0.000    0.003    0.000 {method 'execute' of 'sqlite3.Connection' objects}
       36    0.000    0.000    0.002    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:1999(_get_channel_costs)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:2557(_get_all_full_pnl_batch)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3202(get_all_channels_full_pnl)
        1    0.000    0.000    0.001    0.001 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:2459(_get_all_revenue_data)
       36    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:2542(_get_last_routing_time)
       36    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3361(get_last_forward_time_any_direction)
        1    0.000    0.000    0.001    0.001 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:3496(get_all_channels_revenue_totals)
        1    0.000    0.000    0.001    0.001 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:705(_push_profitability_summary)
       36    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:5463(get_channel_rebalance_success_rate)
       36    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:5413(get_channel_rebalance_costs)
       36    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:5364(get_channel_cost)
       36    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:5512(_get_rebalance_success_rate_for_where)
      144    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:40(_scid_aliases)
       42    0.000    0.000    0.000    0.000 {method 'fetchall' of 'sqlite3.Cursor' objects}
      109    0.000    0.000    0.000    0.000 {method 'fetchone' of 'sqlite3.Cursor' objects}
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/profitability_analyzer.py:1920(_get_all_channels)



```

### fee_controller.adjust_all_fees

```
         6070 function calls in 0.003 seconds

   Ordered by: cumulative time
   List reduced from 94 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.003    0.003 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:415(<lambda>)
        1    0.000    0.000    0.003    0.003 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:4350(adjust_all_fees)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:4464(_adjust_all_fees_inner)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:4614(_adjust_all_fees_channel_loop)
       36    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:5326(_adjust_channel_fee)
       38    0.000    0.000    0.001    0.000 /usr/lib/python3.12/unittest/mock.py:1129(__call__)
       38    0.000    0.000    0.001    0.000 /usr/lib/python3.12/unittest/mock.py:1140(_increment_mock_call)
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:7909(_get_channels_info)
      113    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
       36    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:5100(get_volume_since)
       36    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:5126(get_forward_count_since)
       72    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/policy_manager.py:443(get_policy)
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:1711(get_all_channel_states)
       36    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/database.py:2052(get_channel_probe)
      152    0.000    0.000    0.000    0.000 /usr/lib/python3.12/unittest/mock.py:2508(__new__)
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:4423(_prefetch_neighbor_gossip)
      180    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/utils.py:22(parse_msat)
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/config.py:656(snapshot)
      942    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
       36    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/fee_controller.py:6945(_extract_local_htlc_bounds)



```

### rebalance_engine.run_cycle

```
         6552 function calls (6480 primitive calls) in 0.002 seconds

   Ordered by: cumulative time
   List reduced from 118 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/tools/perf/profile_cycles.py:418(<lambda>)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_engine_v2.py:3361(run_cycle)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_engine_v2.py:3393(_run_cycle_locked)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_engine_v2.py:1362(find_candidates)
        1    0.000    0.000    0.002    0.002 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_engine_v2.py:971(_build_snapshot)
        1    0.000    0.000    0.001    0.001 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_state_v2.py:266(build_state_snapshot)
       36    0.000    0.000    0.001    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_state_v2.py:125(_normalize_channel_input)
        1    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_audit_v2.py:188(log_skips)
        2    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/config.py:656(snapshot)
        2    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/config.py:1060(from_config)
       72    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_engine_v2.py:96(_historical_fee_rate_ppm)
      797    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
       10    0.000    0.000    0.000    0.000 /usr/lib/python3.12/unittest/mock.py:1129(__call__)
        8    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_audit_v2.py:165(log_skip)
       10    0.000    0.000    0.000    0.000 /usr/lib/python3.12/unittest/mock.py:1140(_increment_mock_call)
  572/500    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        7    0.000    0.000    0.000    0.000 {method 'execute' of 'sqlite3.Connection' objects}
     1038    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
      544    0.000    0.000    0.000    0.000 {built-in method builtins.max}
      144    0.000    0.000    0.000    0.000 /home/sat/bin/cl_revenue_ops/.claude/worktrees/agent-a5f9234b13d8d6bcc/modules/rebalance_engine_v2.py:73(_nonnegative_msat_int)



```

