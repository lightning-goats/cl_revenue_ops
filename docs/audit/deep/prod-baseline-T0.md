# Production Baseline T₀ — Deep Audit (Phase 0C)

Read-only production baseline for the cl_revenue_ops deep audit. This is the **T₀
reference** for Phase 3's DB-exhaustion / growth-delta analysis. All figures are either
**LIVE** (captured this session), **CORPUS** (frozen hermes snapshot), or **UNAVAILABLE**
(access denied this session), labelled per row.

- **Capture timestamp (UTC):** 2026-07-02T03:09:32Z (wrapper wall clock)
  - Corroborating RPC `revenue-health.generated_at`: `1782961777` = 2026-07-02T03:09:37Z
- **Audit repo commit at capture (local HEAD):** `07bd400c6c314f1444d2a0d845b9462e4cea5480`
  ("Add deep-audit coverage-accounting infrastructure (Phase 0A)")
- **Access channels used:** sanctioned `lg-readonly-rpc` wrapper (RPC) + read-only `ssh lnnode`
  (filesystem stat / `sqlite3 -readonly`). Direct `docker exec` into hive-nexus-02 is **denied**
  by the permission classifier, so node 2 filesystem stats are UNAVAILABLE this session.

---

## 1. Deployed version / commit per node

| Field | hive-nexus-01 | hive-nexus-02 |
|---|---|---|
| Plugin version (RPC `revenue-status.version`) | 2.10.0 — **LIVE** | 2.10.0 — **LIVE** |
| Deployed git commit | `5449c53ba65b2117b4b1ed262cb89cb0993312e2` — **LIVE** (read-only ssh `git rev-parse` in plugin dir) | UNAVAILABLE (docker exec denied; not exposed by any read-only RPC) |
| `git describe` of deployed tree | `v2.8.0-61-g5449c53` — **LIVE** | UNAVAILABLE |
| CLN version (`getinfo.version`) | v26.06.1 — **LIVE** | v26.06.1 — **LIVE** |
| Node id | `0382d5583...b03afdc3` — **LIVE** | `03992b9a1...e07bfb00` — **LIVE** |
| Plugin path (RPC `plugin list`) | `/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py` — **LIVE** | `/opt/cl-revenue-ops/cl-revenue-ops.py` — **LIVE** |

**Notes / discrepancies:**
- The RPC-reported plugin version string (`2.10.0`) does **not** match the deployed tree's
  `git describe` (`v2.8.0-61-g5449c53`). The version string is a hardcoded constant, so this is
  expected drift, not corruption — but Phase 6 (docs/version conformance) should confirm the
  `2.10.0` constant is intended for commit `5449c53`.
- Deployed commit `5449c53` is only **2 commits behind** local audit HEAD (`07bd400`); those two
  are audit-infra/doc commits, so node 1 is effectively running current `main` code.
- No read-only `revenue-*` RPC echoes a git commit/build hash; only the semantic `version`
  string. Commit identity had to come from the filesystem (ssh) on node 1.

---

## 2. SQLite DB — path, size, per-table row counts

### hive-nexus-01 — **LIVE** (read-only ssh + `sqlite3 -readonly`)

- **Active DB path:** `/data/lightningd/.lightning/revenue_ops.db` (matches configured
  `revenue-ops-db-path`)
- **Main DB file size:** 55,603,200 bytes (**~53.0 MiB / 55.6 MB**) as of 2026-07-01 20:52 MDT
- **WAL:** `revenue_ops.db-wal` = 4,144,752 bytes (~4.0 MiB); `-shm` = 32,768 bytes
- **Total on-disk (db+wal+shm):** ~59,780,720 bytes (~57.0 MiB)
- Three stale/empty 0-byte `revenue_ops.db` / `cl_revenue_ops.db` files exist under other
  lightning dirs (`/data/lightningd/`, `/data/lightningd/bitcoin/`,
  `/data/lightningd/bitcoin/bitcoin/`) — **not** the active DB; ignore for growth math.

**Per-table row counts (33/33 tables), hive-nexus-01, LIVE:**

| # | table | rows | | # | table | rows |
|--|--|--|--|--|--|--|
| 1 | fee_changes | **92,410** | | 18 | closed_channels | 72 |
| 2 | peer_connection_history | **2,706** | | 19 | spend_reservations | 72 |
| 3 | planner_actions | 981 | | 20 | dead_capital_stage | 68 |
| 4 | daily_forwarding_stats_inbound | 673 | | 21 | spend_events | 63 |
| 5 | daily_forwarding_stats | 661 | | 22 | channel_states | 36 |
| 6 | forwards | 500 | | 23 | fee_strategy_state | 36 |
| 7 | rebalance_history | 477 | | 24 | kalman_state | 36 |
| 8 | financial_snapshots | 360 | | 25 | config_overrides | 11 |
| 9 | budget_reservations | 271 | | 26 | peer_reputation | 10 |
| 10 | channel_costs | 105 | | 27 | channel_failures | 5 |
| 11 | mempool_fee_history | 95 | | 28 | pair_rebalance_failures | 5 |
| 12 | channel_closure_costs | 72 | | 29 | channel_probes | 1 |
| 13 | planner_candidates | 32 | | 30 | lifetime_aggregates | 1 |
| — | — | — | | 31 | peer_policies | 1 |
| — | — | — | | 32 | plugin_flags | 1 |
| — | — | — | | 33 | schema_version | 1 |
| — | — | — | | — | hot_channel_protection_overrides | 0 |
| — | — | — | | — | ignored_peers | 0 |
| — | — | — | | — | planner_recycle_ops | 0 |

(Two tables above with count 0 not numbered; full 33-table set confirmed present. `schema_version`
= 1 row — single current-version marker.)

### hive-nexus-02 — DB stats **UNAVAILABLE** this session

- **Configured DB path (CORPUS, listconfigs 2026-07-01T20:35:41Z):**
  `/data/lightning/bitcoin/bitcoin/revenue_ops.db`
- DB **file size, per-table row counts, and oldest-row timestamps could not be captured**:
  node 2 is a docker container (`cl-hive-node-hive-nexus-02`) and `docker exec` is denied by
  the permission classifier; no read-only ssh path to the container filesystem was sanctioned.
  Per audit rules, this is recorded as "not accessible this session" — **not** worked around.
- The frozen hermes corpus (`/home/sat/cl-mycelium-hermes/hive-nexus-02/…`) contains **no**
  DB-internal snapshots: every archived command is an RPC/CLN output (revenue-status,
  revenue-dashboard, revenue-health, listforwards, etc.), and — confirmed against the live node 1
  outputs — **none of those RPCs report DB file size or table row counts**. So the corpus cannot
  fill this gap.
- Scale context (so Phase 3 can weight node 2): node 2 is a **tiny 2-channel node**
  (`revenue-health`: channels.total = 2, managed_channels = 0, 0 forwards today/week, 0 spend).
  Its DB is expected to be far smaller / lower-churn than node 1, but this must be **re-captured
  live before Phase 3** for any real growth delta.

---

## 3. Plugin process — RSS + thread count

| Field | hive-nexus-01 | hive-nexus-02 |
|---|---|---|
| Source | **LIVE** (`ps` via read-only ssh) | UNAVAILABLE (docker exec denied; no RSS/thread RPC) |
| PID | 1943651 | — |
| RSS | 119,060 KB (**~116 MiB**) | — |
| Threads (`nlwp`) | **11** | — |
| Process uptime (`etimes`) | 1699 s (~28 min at capture — recently restarted) | — |

Node 2 RSS/thread count are not exposed by any read-only `revenue-*` RPC and the filesystem/ps
channel is denied, so these are UNAVAILABLE this session.

---

## 4. Oldest-row timestamps + rows/day for high-churn tables (hive-nexus-01, LIVE)

Timestamp column is `timestamp` (INTEGER, unix epoch) on all five. rows/day = rows ÷ span.

| table | rows | oldest row (UTC) | newest row (UTC) | span (days) | ~rows/day | growth character |
|---|---|---|---|---|---|---|
| **fee_changes** | 92,410 | 2026-04-03T03:15:48Z | 2026-07-02T02:35:40Z | ~89.97 | **~1,027 /day** | **UNBOUNDED — largest table, primary Phase 3 exhaustion risk** |
| forwards | 500 | 2026-06-24T02:50:52Z | 2026-07-02T03:09:57Z | ~8.01 | ~62 /day | **BOUNDED** — `cleanup_old_data()` prunes to a ~8-day window; rolled into `lifetime_aggregates` (database.py:61-62, :1338) |
| rebalance_history | 477 | 2026-04-03T16:55:12Z | 2026-07-01T10:51:18Z | ~88.75 | ~5.4 /day | low churn; not pruned |
| mempool_fee_history | 95 | 2026-06-30T03:10:55Z | 2026-07-02T02:45:45Z | ~1.98 | ~48 /day | short window on disk (~2 days) — appears pruned/short-retention; confirm retention in Phase 3 |
| spend_events | 63 | 2026-03-20T18:07:44Z | 2026-05-27T16:12:15Z | ~67.9 | ~0.9 /day | effectively idle since 2026-05-27 (0 spend activity currently) |

Also relevant for Phase 3 (no oldest-row captured, but large & likely append-only):
`peer_connection_history` (2,706 rows) and `planner_actions` (981 rows).

---

## 5. Summary for Phase 3

- **Headline DB size:** node 1 = **~53 MiB main file** (~57 MiB incl. WAL). Node 2 = **unknown
  (re-capture needed).**
- **Biggest / fastest-growing table:** `fee_changes` — 92,410 rows, ~1,027 rows/day, ~3 months
  deep, **no observed pruning**. This dominates DB size and is the headline exhaustion candidate.
- **Bounded tables confirmed:** `forwards` (~8-day retention window, ~500 rows steady state).
- **Idle/stale:** `spend_events` (last row 2026-05-27), several 0-row tables.
- **Node 2 is a live re-capture blocker:** its DB size / row counts / oldest-rows / RSS / threads
  are all UNAVAILABLE this session and are **not** recoverable from the frozen corpus (corpus
  holds only RPC outputs, which carry no DB internals). **A live re-capture of node 2 — via an
  operator-sanctioned read-only filesystem channel into `cl-hive-node-hive-nexus-02`
  (path `/data/lightning/bitcoin/bitcoin/revenue_ops.db`) — is required before Phase 3's
  growth delta on node 2 is meaningful.**

### Figure provenance at a glance

- **LIVE (node 1, this session):** version string, deployed commit + describe, CLN version, node
  id, plugin path, DB path + size + WAL, all 33 table row counts, high-churn oldest/newest
  timestamps, RSS, thread count, uptime.
- **LIVE (node 2, RPC only):** version string, CLN version, node id, plugin path, channel scale.
- **CORPUS (node 2):** configured `revenue-ops-db-path` (2026-07-01 listconfigs).
- **UNAVAILABLE (node 2, docker exec denied; absent from corpus):** DB file size, all 33 table
  row counts, oldest-row timestamps, plugin RSS, thread count, deployed git commit.
