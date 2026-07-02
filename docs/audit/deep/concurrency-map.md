# Concurrency Map — Deep Audit Phase 2A

Authoritative lock-graph, thread inventory, and shared-mutable-state adjudication for
`cl_revenue_ops`. Consumed by Phase 2B (stress harness) and 2C (daemon fault-injection),
and by the fix stage. **Read-only artifact — no behavioral fixes proposed here; where a
fix is implied it is marked `REC:` (recommendation only).**

Line citations are against `git HEAD 4d08f45` (worktree blob). Prior-ledger line numbers
(blob `acbdb42`) have drifted; every citation below was re-derived from current code.

---

## 0. Headline — the threading reality (corrects the "single-threaded timer" assumption)

The deferred ledger repeatedly mitigated races with "single-threaded timer" / "single-threaded
write pattern". **That assumption is false, but in a specific shape that matters:**

- **pyln-client dispatches every RPC handler AND every notification handler SERIALLY on ONE
  thread** — the stdin read loop in `pyln/client/plugin.py:run()` (`_multi_dispatch` →
  `_dispatch_request` → `_exec_func`), with **no per-request thread**. So the 58 RPC handlers
  and the 4 subscriptions (`forward_event`, `connect`, `disconnect`, `channel_state_changed`)
  **never overlap each other**. This is confirmed in-code by the comment at
  `cl-revenue-ops.py:5606-5609` ("pyln dispatches all notifications on one thread").
- **But that single dispatch thread runs concurrently with 7 daemon threads.** So the true
  concurrency is **dispatch-thread vs each daemon**, and **daemon vs daemon** — up to **8
  concurrent writers** to shared module state and to SQLite. Every "RPC vs timer" race in the
  ledger is therefore a **real 2-thread race** (not N-way among RPCs, but genuinely concurrent
  with the owning daemon).

Net: most historical "single-threaded" mitigations are **invalid as stated**; safety today
rests on **explicit locks** (where they exist) and on **SQLite WAL single-writer semantics**,
not on serialization of the work.

---

## 1. Thread inventory

| # | Thread (name) | Kind | Start | Entry points it executes |
|---|---|---|---|---|
| T0 | **main / pyln dispatch** | 1, long-lived | `plugin.run()` | ALL 58 `@plugin.method` RPC handlers + 4 `@plugin.subscribe` handlers (`forward_event` 5612, `connect` 5751, `disconnect` 5794, `channel_state_changed` 5837). Serial — one at a time. Also runs `init` (2223) at startup. |
| T1 | **flow-analysis** | daemon | 2799 | `run_flow_analysis`→`FlowAnalyzer.analyze_all_channels` (2830), `database.cleanup_old_data` (2521), `policy_manager.cleanup_expired_policies` (2526) |
| T2 | **fee-adjustment** | daemon | 2800 | `_refresh_fee_cycle_hive_inputs` (2555), `run_fee_adjustment`→`FeeController.adjust_all_fees` (2559) |
| T3 | **rebalance-check** | daemon | 2801 | `refresh_hive_runtime` (2586), `run_rebalance_check`→`EVRebalancer`/`RebalanceEngine._run_cycle` (2591) |
| T4 | **startup-snapshot** | daemon, **one-shot** | 2802 | `_snapshot_peers_once` (2724) then exits |
| T5 | **financial-snapshot** | daemon | 2803 | `_take_financial_snapshot`→`database.record_financial_snapshot` (2742/2758) |
| T6 | **boltz-auto-cycle** | daemon | 2804 | `_refresh_dynamic_config` (2632), `_run_boltz_auto_cycle_once` (2633) → boltz swaps, spend-ledger reserves |
| T7 | **capacity-planner** | daemon | 2805 | `_refresh_dynamic_config` (2687), `capacity_planner.execute_cycle` (2689) → opens/closes/recycle |
| P-main | **rpc worker pool** | `ThreadPoolExecutor(16)` | 623 | ONLY outbound `plugin.rpc.<method>` socket calls submitted by `_submit_main` (699). Touches no plugin business state; releases `_main_submit_slots` in done-callback (703). Sets per-thread `_rpc_socket_timeout` (692). |
| P-async | **rpc async pool** | `ThreadPoolExecutor(4)` | 624 | ONLY `fire_and_forget` outbound RPC (717). Touches `_async_fail_count` (locked, 630/757/760). |
| X | **rpc-shutdown** | daemon, transient | 381 | atexit drain of the executors, joined with timeout (P1-029 fix) |

**Boltz single-flight:** `_boltz_auto_cycle_run_lock` (805) — **non-blocking** acquire
(`acquire(blocking=False)`, 1676) so a manual `revenue-boltz-auto-cycle-run-now` on T0 and the
scheduled cycle on T6 cannot both run a cycle; the loser returns "already running". Released in
`finally` (1893). Swap creation is separately single-flighted by
`BoltzCliManager._swap_creation_lock` (boltz_manager.py:248; used 1575/1656/2053).

**The rpc worker pools are NOT a source of shared-state races** — they execute only the pyln
socket call `fn(*args)`; the plugin's own logic stays on the submitting thread, which blocks on
`future.result()`. They matter only for saturation (P1-007) and lock-hold-duration (a daemon
holding a lock while blocked on `future.result()` extends the hold — see §2 fee `_state_lock`).

---

## 2. Lock graph

Every `threading.Lock/RLock/Semaphore/Event` in `cl-revenue-ops.py` + `modules/`. "Contended
by" lists the threads that can acquire it. "Acquire mode" flags non-blocking / timeout acquires
(the property that makes the graph deadlock-resistant).

| Lock | file:line | Protects | Contended by | Acquire mode | Held across RPC/DB I/O? |
|---|---|---|---|---|---|
| `shutdown_event` (Event) | cro:479 | shutdown signal | all daemons + signal handler | n/a | n/a |
| `_main_submit_slots` (Sem 48) | cro:627 | main pool backpressure | any thread issuing RPC | `acquire(timeout≤1.0s)` (676) | released in done-cb (703) |
| `_async_submit_slots` (Sem 68) | cro:628 | async pool backpressure | any thread `fire_and_forget` | `acquire(blocking=False)` (707) | released in done-cb (721) |
| `_async_lock` | cro:630 | `_async_fail_count` int | RPC callers + async workers | blocking, trivial | no |
| `_boltz_balance_lock` | cro:804 | `_boltz_balance_last_action` dict | **T0 only** (two RPC handlers 8416/8775) | blocking | no |
| `_boltz_auto_cycle_run_lock` | cro:805 | boltz cycle single-flight | T0 (RPC) + T6 (daemon) | **non-blocking** (1676) | held across whole cycle |
| `_boltz_auto_cycle_state_lock` | cro:806 | `_boltz_auto_cycle_state` dict | T0, T6 | blocking, trivial | no |
| `_scid_cache_lock` | cro:828 | `_scid_to_peer_cache`, `_scid_cache_last_cleared` | T0 (fwd/RPC) + any resolver | blocking, brief | no (RPC done outside, see below) |
| `_scid_cache_fetch_lock` | cro:829 | serialize cache-miss RPC | same | blocking | **yes** — RPC held (M-2 design, 5428) but `_scid_cache_lock` released during RPC |
| `_spend_reserve_lock` | cro:6367 | unified-budget check + generic-ledger reserve (TOCTOU) | T0 (spend RPCs) | blocking | **yes** — wraps `_total_cost_budget_status()`+`reserve_spend` (6388-6415) |
| `_total_cost_budget_memo_lock` | cro:6835 | `_total_cost_budget_memo` cache | any caller | blocking, brief | no |
| `Config._lock` | config.py:614 | config fields on the transactional path | T0 (update_runtime) + all snapshot callers | blocking | no |
| `Database._thread_conn_lock` | database.py:276 | `_thread_connections` registry ONLY (not writes) | all (register at 347) + shutdown (399) | blocking, brief | no |
| `FeeController._state_lock` (RLock) | fee_controller.py:2591 | all DTS/PID fee state | T2 (adjust_all_fees) + T0 (forward_event, set_channel_fee) | **non-blocking** (4383) for the cycle; **2.0s timeout** from T0 forward_event (cro:5609) | **YES — serial listchannels RPCs + DB under the lock for the whole cycle** (3144, 4454) |
| `HiveHintAdapter._lock` (RLock) | hive_hints.py:217 | hint cache dict | T2 (fee inputs) + T0/T3 | blocking | no — `json.loads` done BEFORE the lock (252-271 vs store 300) |
| `FlowAnalyzer._kalman_lock` | flow_analysis.py:777 | Kalman posterior writes | T1 (analyze_all) + T0 (analyze RPC) | blocking, per-write | no |
| `FlowAnalyzer._analysis_lock` | flow_analysis.py:791 | analysis stampede guard | T1 only takes it (1313) — **T0 analyze_channel does NOT** | **non-blocking** (1313) | held across analysis |
| `ChannelProfitabilityAnalyzer._analysis_lock` | profitability_analyzer.py:592 | profit-analysis stampede | T0 (analyze RPC) + daemons via callers | non-blocking | held across analysis |
| `PolicyManager._cache_lock` | policy_manager.py:208 | `_cache` policy dict | T0 (policy RPCs) + T1 (cleanup) | blocking | no |
| `PolicyManager._callback_lock` | policy_manager.py:212 | callback list | registration + fire | blocking | no |
| `EVRebalancer._source_failures_lock` | rebalancer.py:201 | `_source_failures` | T3 + T0 (rebalance RPCs) | blocking | no |
| `EVRebalancer._pending_lock` | rebalancer.py:303 | `_pending` dict (all sites 1883-2368) | T3 + T0 | blocking, brief | no |
| `RebalanceEngine._cycle_lock` | rebalance_engine_v2.py:228 | rebalance cycle single-flight | T3 (daemon) + T0 (revenue-rebalance) | **non-blocking** (3120/3247) | held across whole cycle |
| `DataService._lock` | data_service.py:49 | TTL cache dict | any thread | blocking, brief | no — RPC fetched outside |
| `DataService._forever_lock` | data_service.py:52 | forever cache (getinfo/configs) | any thread | blocking | `_ensure_getinfo` (115) may fetch under lock (minor) |
| `SegmentObservationStore._lock` | segment_observations.py:36 | segment obs dict | T3 engine | blocking, brief | no |
| `RebalanceHiveRouter._local_disable_lock` | rebalance_hive_router.py:99 | local-disable flag | T3 + T0 | blocking, brief | no |
| `BoltzCliManager._journal_lock` | boltz_manager.py:245 | swap journal | T0/T6 | blocking | no |
| `BoltzCliManager._ignored_swaps_lock` | boltz_manager.py:246 | ignored-swaps set | T0/T6 | blocking | no |
| `BoltzCliManager._swap_creation_lock` | boltz_manager.py:248 | swap-create single-flight | T0/T6 | blocking (1575/1656/2053) | held across swap create |

### Deadlock adjudication

**No lock-ordering deadlock found.** Evidence:

- **No nested cross-module lock acquisition** (A-then-B / B-then-A) exists. Each module's lock
  guards only that module's state; a thread holding one module's lock does not reach in and
  acquire another module's lock. The only lock a foreign thread reaches for is
  `FeeController._state_lock` from T0's `forward_event`, and that uses a **2.0s timeout**
  (`FORWARD_EVENT_LOCK_TIMEOUT_SECS`, cro:5609) — it fails open rather than blocks.
- **Every cross-thread-contended coarse lock uses a non-blocking or bounded acquire**:
  `_cycle_lock` (3120/3247 `blocking=False`), fee `adjust_all_fees` `_state_lock` (4383
  `blocking=False`), `_boltz_auto_cycle_run_lock` (1676 `blocking=False`), `_analysis_lock`
  (1313 `blocking=False`), RPC semaphores (`timeout≤1.0s` / `blocking=False`).
- The one lock **held across blocking I/O** is fee `_state_lock` (serial `listchannels` RPCs +
  DB writes for the whole cycle, 3144/4454; the `setchannel` RPC at 7186 and `record_fee_change`
  at 7220 also run under it when called from the cycle via RLock re-entry). This is a
  **latency/liveness** hazard (T0 `forward_event` DTS nudges get dropped after the 2s wait
  during a long fee cycle — see NC-6 / DEF-051), **not a deadlock**, because the RPC worker pool
  never re-acquires `_state_lock`.
- **Boltz nested lock order `_swap_creation_lock` → `_journal_lock`** (boltz_manager.py: creation
  holds `_swap_creation_lock` at 1575/1656/2053 then calls `_record_swap_result` → `_journal_lock`
  at 1619/1754/1811/1827/1898/2073; `swap_status` RPC also runs under `_swap_creation_lock` at
  ~1892). **No reverse order (`_journal_lock` → `_swap_creation_lock`) exists**, so no deadlock
  cycle — but it is a blocking global mutex held across the Boltz CLI subprocess + RPC. Latency
  only. `_ignored_swaps_lock` also holds `swap_status` RPC under it (1295).
- `RebalanceHiveRouter._local_disable_lock` (357) and `DataService._forever_lock` (119/141) each
  hold askrene / getinfo-listconfigs RPCs under the lock — single-flight startup/rebuild paths,
  no cross-lock nesting, latency only.

`REC:` shorten the `_state_lock` critical section (issue gossip `listchannels` before acquiring)
— tracked as DEF-051; do not fix here.

---

## 3. Shared mutable state table

Every module-level or instance mutable touched by >1 thread. **UNPROTECTED** = read/written
across threads with no lock (relying on GIL atomicity or on a higher single-flight).

| State | file:line | Readers/Writers | Lock | Verdict |
|---|---|---|---|---|
| `_scid_to_peer_cache` / `_scid_cache_last_cleared` | cro:824/825 | T0 (fwd + RPC resolvers) | `_scid_cache_lock` + `_scid_cache_fetch_lock` (double-checked, 5415-5454) | PROTECTED |
| `_boltz_balance_last_action` | cro:803 | T0 only (2 RPC handlers) | `_boltz_balance_lock` (8416/8507/8524/8775/8801/8807) | PROTECTED (also T0-serial) |
| `_boltz_auto_cycle_state` | cro:807 | writers T0/T6 under lock (1538, 1741, 1868, 1887); **reads at 5168 & 8237 via `dict(...)` WITHOUT the lock** | `_boltz_auto_cycle_state_lock` | **PARTIAL** — see NC-5 (benign: writers only `.update()` existing keys, no size change) |
| `_total_cost_budget_memo` | cro:6834 | any (budget status callers) | `_total_cost_budget_memo_lock` | PROTECTED |
| `_async_fail_count` | cro:629 | RPC callers + async workers | `_async_lock` | PROTECTED |
| Config fields (transactional path) | config.py | T0 `update_runtime` writer; all daemons read via `snapshot()` | `Config._lock` (writer 804, snapshot copy 1068) | PROTECTED |
| Config fields (dynamic path) | — | writers `_on_rebalance_tuning_change` (cro:1048), `_on_rebalance_router_change` (cro:1509), `_refresh_dynamic_config` (cro:5602) on T6/T7 mutate live config; readers = snapshot callers | **NO `_lock`, no `_version` bump** | **UNPROTECTED-BUT-BENIGN** — each writes ONE independent GIL-atomic scalar (P1-020); latent if a future on_change writes a pair. See C-3 verdict. |
| `_thread_connections` | database.py:273 | all threads register; shutdown clears | `_thread_conn_lock` | PROTECTED |
| SQLite (per-thread conns, autocommit+WAL) | database.py:307 | up to 8 concurrent writers | WAL single-writer + `BEGIN IMMEDIATE` on money path | See C-1 verdict |
| FeeController DTS/PID state | fee_controller.py | T2 (cycle) + T0 (forward_event, set_channel_fee) | `_state_lock` (RLock) | PROTECTED (concurrent); crash-consistency of memory-before-DB save remains (DEF-053) |
| Neighbor gossip cache | fee_controller.py | T2 + T0 | `_state_lock` | PROTECTED |
| `self.config.max_fee_ppm` live reads | fee_controller.py:6187/6200 | T2/T0 amid snapshot-based compute | none (live) | **UNPROTECTED-BUT-BENIGN** — single field, snapshot-vs-live mix (NC-7) |
| **`_channel_fee_states` lock-free READS** | fee_controller.py:3671/4969/6559/6651/6706/6729/6787/6867 | T2 (cycle mutates under lock) + T0 (status/report RPCs read w/o lock) | **NONE on the read side** | **UNPROTECTED (stale-read)** — `.get()` won't crash on CPython but returns a value the cycle may be mid-updating (NC-9) |
| **`_hive_member_set_at` / `_hive_member_advisory_peers` / `_hive_member_released_peers`** | fee_controller.py:2604/2606/2605 (write 2745/2747/2752, discard 2753/2754/3478) | T2 (cycle) + T0 (hint-update / forward_event via `_remember_hive_member`/`_clear_hive_member_cache`) | **NONE** | **UNPROTECTED** — lock-free set/dict mutation across threads (NC-10) |
| Hint cache | hive_hints.py | T2 + T0/T3 | `_lock` (RLock) | PROTECTED |
| Kalman posterior | flow_analysis.py | T1 (analyze_all) + T0 (analyze_channel RPC) | `_kalman_lock` per-write | PROTECTED against corruption; **NOT against double-application** (NC-1 / DEF-027/029) |
| Flow analysis stampede | flow_analysis.py | T1 takes `_analysis_lock`; **T0 `analyze_channel` bypasses it (1675)** | `_analysis_lock` (T1 only) | **GAP** — see NC-1 |
| `_cache` (policies) | policy_manager.py | T0 + T1 | `_cache_lock` | PROTECTED (logical load-vs-write ordering DEF-036 remains, not lock-absence) |
| `_pending` (rebalance) | rebalancer.py | T3 + T0 | `_pending_lock` | PROTECTED |
| `_source_failures` | rebalancer.py | T3 + T0 | `_source_failures_lock` | PROTECTED |
| **`_fee_cache`** | rebalancer.py:305/1697/1740; reset 1387 | T3 + T0 (manual_rebalance) | **NONE** | **UNPROTECTED** (DEF-061) — mitigated by `_cycle_lock` single-flight on the auto path; manual path + GIL. Low. |
| **`_peer_inbound_fees`** | rebalancer.py:1704/1816/1820 | T3 + T0 | **NONE** | **UNPROTECTED** (DEF-067-S9) — same mitigation caveat. Low. |
| TTL cache / forever cache | data_service.py | all threads | `_lock` / `_forever_lock` | PROTECTED (get-or-fetch dup RPC on miss, benign) |
| Segment observations | segment_observations.py | T3 | `_lock` | PROTECTED |

### Unprotected cross-thread mutable list (the headline)

1. **`rebalancer._fee_cache`** (rebalancer.py:305/1697/1740, reset 1387) — no lock. DEF-061.
2. **`rebalancer._peer_inbound_fees`** (rebalancer.py:1704/1816/1820) — no lock. DEF-067-S9.
3. **Config dynamic-path fields** (cro:1048, 1509, 5602) — mutated live off `Config._lock`, `_version` never bumped. P1-020 / C-3 residual.
4. **`fee_controller` live `self.config.max_fee_ppm`** (6187/6200) — read live amid a snapshot-based computation (single-field, benign).
5. **`_boltz_auto_cycle_state` reads** (cro:5168, 8237) — `dict(...)` snapshot taken without `_boltz_auto_cycle_state_lock` (benign; no key-set change).
6. **`FlowAnalyzer._analysis_lock` bypass** by RPC `analyze_channel` (flow_analysis.py:1675) — not a mutable per se, but the missing stampede guard exposes the Kalman posterior to concurrent double-application.

7. **`fee_controller._hive_member_set_at` / `_hive_member_advisory_peers` /
   `_hive_member_released_peers`** (fee_controller.py:2604/2606/2605) — lock-free set/dict
   mutated by the fee cycle (T2) and by hint-update / forward_event paths (T0). NC-10.
8. **`fee_controller._channel_fee_states` lock-free reads** (3671/4969/6559/6651/6706/6729/6787/6867)
   — read without `_state_lock` while the cycle mutates the dict under it. NC-9.

Items 1, 2, 6, 7 are the substantive ones (financial-signal / rebalance-decision / hive-gate
state). Item 8 is stale-read (no crash). Items 3–5 are GIL-atomic-scalar / benign-today but
latent.

---

## 4. DB concurrency — table→thread writer map + C-1 adjudication

**Model (re-derived, database.py):** `sqlite3.connect(..., isolation_level=None,
check_same_thread=False)` (307) = **autocommit**; `threading.local()` per-thread connections
(273/289/348); `PRAGMA journal_mode=WAL` (313); `PRAGMA busy_timeout=5000` (322);
`synchronous=NORMAL` (324); `foreign_keys=ON` (326). Up to **8 concurrent writers** (T0 + 7
daemons), each on its own connection. `_thread_conn_lock` guards only the connection registry,
**not** writes — write serialization is delegated entirely to SQLite's WAL single-writer rule.

**Transaction discipline:**
- **`BEGIN IMMEDIATE` (grabs the WAL write lock up-front, serializes writers):**
  `_reserve_budget_atomic` (119→173), batch upserts `update_channel_states_batch` (1615),
  `save_kalman_states_batch` (1919), `update_fee_strategy_states_batch` (2175),
  `bulk_insert_forwards` (4441), `record_forward_and_reputation` (4781), `cleanup_old_data`
  (6272→6377), `clear_all_reservations` (3694), migrations (1312/1394), and the version-bump
  helpers.
- **Single-statement bare autocommit (atomic — NOT torn):** `record_fee_change`/fee_changes
  (2249), `record_rebalance`/rebalance_history (2319), `record_spend_event`/spend_events
  (INSERT OR REPLACE, 3835), `record_rebalance_cost`/rebalance_costs (5154),
  `record_forward`/forwards (INSERT OR IGNORE, 4732), `increment_failure_count` (atomic upsert
  INSERT…ON CONFLICT DO UPDATE RETURNING, 6015), `release_budget_reservation` (3601),
  `mark_budget_spent` (3623).
- **Multi-statement, NOT wrapped in BEGIN (windows):** `reserve_spend` (SELECT status 3757 →
  INSERT OR REPLACE 3770), `mark_spend_reservation_spent` (SELECT 3796 → UPDATE 3801 →
  `record_spend_event` 3809), `release_spend_reservations` (SELECT 3904 → bulk UPDATE 3921).
- **One deferred `BEGIN` (weaker than IMMEDIATE):** line 4204.

**Table → writer-thread map** (33 CREATE TABLEs; thread inferred from the writer method's
caller). "init" = one-shot on T0 at `initialize()` (562, called cro:2223) before daemons start.

| Table | Written by (method) | Thread(s) |
|---|---|---|
| schema_version | seed only (575) | init |
| channel_states | update_channel_state(s)_batch (1559/1615) | T1/T2/T5 + T0 (connect/disconnect/channel_state_changed subs) + T4 snapshot |
| fee_strategy_state | update_fee_strategy_states_batch (2175) | T2 + T0 (analyze/set-fee RPC) |
| fee_changes | record_fee_change (2249) | T2 (cycle) + T0 (revenue-set-fee) |
| rebalance_history | record_rebalance (2319) / update_rebalance_result (2360) | T3 + T0 (revenue-rebalance) |
| forwards | record_forward (4732) / bulk_insert_forwards (4441) / record_forward_and_reputation (4781) | T0 (forward_event) + init hydration (T0) |
| channel_costs | cost writers | T3 + T0 |
| rebalance_costs | record_rebalance_cost (5154) | T3 + T0 |
| channel_failures | increment/reset_failure_count (6015/6038) | T3 + T0 |
| pair_rebalance_failures | failure writers | T3 + T0 |
| peer_reputation | update_peer_reputation / apply_reputation_decay (6213) | T0 (forward_event) + T1 (decay) |
| peer_connection_history | connection writers | T0 (connect/disconnect subs) + T4 |
| lifetime_aggregates | cleanup_old_data rollup (6272) | T1 |
| channel_probes | probe writers | T3/T0 |
| ignored_peers / peer_policies | policy_manager + migration (1394) | T0 (policy RPC) + T1 (cleanup) |
| hot_channel_protection_overrides | override RPC | T0 |
| config_overrides | update_runtime (config.py) | T0 (setconfig/RPC) |
| mempool_fee_history | mempool writer | T2 |
| daily_forwarding_stats(_inbound) | rollup (cleanup_old_data) + forward path | T1 + T0 |
| budget_reservations | _reserve_budget_atomic (119) / release / mark_spent | T3 + T0 (reserve/release RPC) |
| spend_reservations | reserve_spend (3736) / mark_spent (3789) / release (3881) | T0 (spend RPC) + T6 (boltz auto reserves) |
| spend_events | record_spend_event (3835) | T0 + T6 |
| financial_snapshots | record_financial_snapshot (2843) | T5 |
| channel_closure_costs / closed_channels | closure writers | T7 (planner) + T0 |
| planner_candidates / planner_actions / dead_capital_stage / planner_recycle_ops | capacity_planner | T7 + T0 (planner RPCs) |
| kalman_state | save_kalman_states_batch (1919) | T1 + T0 (analyze RPC) |
| plugin_flags | flag writers | various |

Money-path tables with **two concurrent writer threads** (T0 dispatch + a daemon):
`fee_changes`, `rebalance_history`, `rebalance_costs`, `budget_reservations`,
`spend_reservations`, `spend_events`, `channel_failures`, `kalman_state`, `channel_states`.

### C-1 verdict (DEF-019 — autocommit + manual BEGIN corruption)

**CLOSED-BY-MECHANISM for corruption / torn / lost writes; NEEDS-FIX (new, Medium) for a
"database is locked" liveness window; STILL-OPEN (Low) for spend-ledger crash-consistency.**

Re-derivation:
- The audit hypothesis that **`fee_changes` / `rebalance_history` / `spend_events` use bare
  autocommit with no BEGIN and are thus exposed to interleaving is REFUTED**: all three are
  **single-statement** writes (2249, 2319, 3835 — the last an `INSERT OR REPLACE`), which are
  **atomic under `isolation_level=None`**. No torn write is possible across connections; a
  concurrent writer on another connection is **blocked by the WAL single-writer lock**, it does
  not interleave.
- `_reserve_budget_atomic`'s `BEGIN IMMEDIATE` (119) grabs the write lock up-front, so the
  daily/weekly `SUM` checks (122-155) and the INSERT (167) are all inside the writer lock —
  the money-path budget reserve **is genuinely serialized** against every other writer (WAL
  permits one writer; others block on `busy_timeout`). TOCTOU-safe.
- **"database is locked" is reachable (NEW, NC-3, Medium):** `busy_timeout=5000` means a
  blocked writer waits only 5s. If any long `BEGIN IMMEDIATE` batch holds the writer >5s while
  another writer contends, the contender raises `sqlite3.OperationalError: database is locked`.
  The exposed batches are `bulk_insert_forwards` (4441, up to 10M forwards at startup),
  `cleanup_old_data` (6272 — prune + rollup + `incremental_vacuum`, runs on T1 **every flow
  cycle**), and the batch upserts (1615/1919/2175). Realistic interleave: startup hydration
  (T0) `bulk_insert_forwards` vs a `forward_event` write (T0) — same thread, safe — but
  `cleanup_old_data` on T1 vs `record_fee_change` on T2 / `record_forward` on T0 is a genuine
  cross-thread contention that can breach 5s on a large DB. **Hand to Phase 2B** (assert no
  `OperationalError` under soak) and fix stage. `REC:` raise `busy_timeout` and/or chunk
  `cleanup_old_data`.
- **Spend-ledger crash-consistency (STILL-OPEN, Low, NC-2):** `reserve_spend` (3736),
  `mark_spend_reservation_spent` (3789), `release_spend_reservations` (3881) do read-then-write
  across **separate autocommit statements** with no enclosing `BEGIN`. Cross-thread interleave
  is limited because these run on **T0 only** (spend RPCs; T6 boltz auto also reserves) — but a
  **crash between the UPDATE and the `record_spend_event`** leaves the reservation `spent` with
  no event (the guards `status='active'` and event-id `resv:{rid}` keep it idempotent, so it is
  crash-consistency, not corruption). `REC:` wrap in `BEGIN IMMEDIATE`.
- Original DEF-019 "rollback failure" residual: `_reserve_budget_atomic` ROLLBACKs on except
  (139/161); a failing ROLLBACK would leave the connection in a transaction — Low, unmitigated.
- `schema_version.version` is **dead** (only written 575, never read) — migrations are
  idempotent-by-inspection (`PRAGMA table_info`), run once on T0 at init before daemons start;
  no concurrency exposure. (Hand DEF to Phase 3A.)

---

## 5. C-3 verdict (config torn read — DEF-032 / DEF-044 / DEF-010 / P1-020)

**CLOSED-BY-MECHANISM for the multi-field torn read; STILL-OPEN (Low) for the P1-020 residual
(unlocked dynamic writers + stale `_version`).**

Re-derivation (config.py + cro):
- `Config` is a `@dataclass` with `_version` (613) and `_lock` (614). `snapshot()` (656) →
  `ConfigSnapshot.from_config` (1060), which **acquires `_lock`** (1068) and copies every field
  (1070-1074) into a **`frozen=True`** snapshot (865) — an immutable, mutually-consistent copy.
- **`update_runtime` (751) is the only writer of the must-be-consistent pairs**: it holds
  `_lock` (804), validates the pairs **inside** the lock (805-835), read-back-verifies the DB
  (842-850), then `setattr`s and **bumps `_version`** (854). It writes exactly one field per
  call but validates dependent pairs atomically.
- **No daemon reads both halves of any invariant pair as two separate live reads.** Every
  pair (`min_fee_ppm`/`max_fee_ppm`, `low`/`high_liquidity_threshold`,
  `rebalance_min`/`max_amount`, budget pairs, etc.) is read off a single `snapshot()` in the
  daemons and in `rebalancer.py` (snapshot at 1158/1258/1415/1887/2665) and `fee_controller`.
  So the original C-3 torn-pair read **cannot occur** on the snapshot path.
- **Residual (P1-020, STILL-OPEN Low):** three writers mutate **live** `Config` bypassing
  `_lock` and never bumping `_version`: `_on_rebalance_tuning_change` (cro:1048),
  `_on_rebalance_router_change` (cro:1509), `_refresh_dynamic_config` (cro:5602, on T6/T7).
  Because they skip `_lock`, such a write can interleave with an in-progress `from_config` copy
  loop, and `_version` stops reflecting reality. Each writes **one independent GIL-atomic
  scalar**, so no pair is torn today — but the "snapshot is atomic" guarantee is technically
  broken for those fields, and a future multi-field `on_change` would reintroduce a true torn
  read. `REC:` route all config mutation through `update_runtime` (or take `_lock` + bump
  `_version`). Do not fix here.
- **`_version` provides NO torn-read protection** — there is no versioned-read/retry loop
  anywhere; `_version` is a monotonic tag stamped onto snapshots, never compared by consumers.
- **DEF-008 is MOOT:** `hive_fee_ppm` / `hive_rebalance_tolerance` do not exist in the current
  `Config`. The only snapshot-coverage gap is `hive_hints_allow_all_hints_m2_scope`
  (config.py:595), absent from `ConfigSnapshot`; its sole reader is init-time
  `HiveHintAdapter` construction (cro:2399, live read before daemons start) — benign, but any
  daemon needing it would be forced to read live.

---

## 6. Re-adjudication of deferred items whose sole mitigation was "single-threaded"

| Item | Original mitigation | Verdict | Evidence |
|---|---|---|---|
| **C-1 / DEF-019** | single-threaded write pattern | **CLOSED-BY-MECHANISM** (WAL single-writer + BEGIN IMMEDIATE + single-stmt autocommit atomicity); **NEEDS-FIX** new "db locked" (NC-3); Low crash-consistency (NC-2) | §4 |
| **C-3 / DEF-032 / DEF-044 / DEF-010** | ConfigSnapshot / single-threaded | **CLOSED-BY-MECHANISM** for torn pairs; **STILL-OPEN Low** P1-020 residual | §5 |
| DEF-024 increment_failure_count non-atomic read-after-write | (implicit) | **CLOSED-BY-MECHANISM** — now a single-statement atomic upsert `INSERT…ON CONFLICT DO UPDATE RETURNING` (database.py:6015) | 6015 |
| DEF-027 flow analyze_channel vs analyze_all_channels race | single-threaded timer | **STILL-OPEN** (NC-1) — mitigation false; `analyze_all_channels` single-flights via `_analysis_lock` (1313) but RPC `analyze_channel` (1675) does NOT take it → concurrent with T1 | 1313 vs 1675 |
| DEF-029 Kalman double-update | single-threaded timer | **STILL-OPEN** (NC-1) — `_kalman_lock` prevents corruption, not double-application of the same window | 821/1503 |
| DEF-053 / FC-I16 non-atomic state save | single-threaded fee cycle | **CLOSED-BY-MECHANISM** for concurrent corruption (guarded by `_state_lock` RLock, single-flight 4383); **STILL-OPEN Low** crash-consistency (memory-before-DB) | 2591/4383 |
| DEF-051 / FC S-3 `_state_lock` across DB/RPC blocks set_channel_fee | (n/a) | **STILL-OPEN** (NC-6) — confirmed serial `listchannels` RPCs under `_state_lock` (3144); T0 forward_event waits 2s then drops nudge | 3144/4454/5609 |
| DEF-061 `_fee_cache` unprotected | GIL | **STILL-OPEN Low** — no lock; mitigated by `_cycle_lock` single-flight on auto path only | rebalancer 305/1697/1740 |
| DEF-067-S9 `_peer_inbound_fees` unlocked | (n/a) | **STILL-OPEN Low** — no lock | rebalancer 1704/1816/1820 |
| DEF-060 `_budget_hot_channel_only` unprotected | (n/a) | **CLOSED-BY-REMOVAL** — symbol no longer present in rebalancer.py (re-verify in Phase 4) | grep: absent |
| DEF-004 clboss TOCTOU / DEF-040 `_clboss_available` | single-writer | **CLOSED-BY-REMOVAL** — `modules/clboss_manager.py` deleted | file absent |
| DEF-036 policy `_load_cache` overwrites `_update_cache` | single-threaded | **STILL-OPEN (logical, not lock-absence)** — both take `_cache_lock` (270/329/339); the load-vs-write-through ordering hazard survives but cannot corrupt | policy_manager 270/329/339 |
| P1-002 no single-flight at run_* layer | — | **WONTFIX confirmed** — money paths single-flighted one layer down (`_cycle_lock` 3120/3247, fee `_state_lock` 4383, `_analysis_lock` 1313); **exception = analyze_channel RPC (NC-1)** | as cited |
| P1-003 cross-category budget TOCTOU | — | **STILL-OPEN Medium** (NC-8) — rebalance budget uses `BEGIN IMMEDIATE`/`reserve_budget`; generic ledger uses `_spend_reserve_lock` (cro:6367); the two categories are reconciled under **different locks at different instants** → latent cross-category TOCTOU, bounded by daemon cadence | cro:6367 / rebalancer external-liquidity read |
| P1-013 proxy timeout not cancellation | — | **STILL-OPEN Low** — timed-out future abandoned not cancelled (cro:662); may execute later; known callers idempotent | cro:662 |
| P1-014 atexit closes DB conns w/ daemons unjoined | — | **STILL-OPEN Low** — `close_all_connections` (399) can close a thread-local conn under a still-running daemon at exit; bounded to process exit | cro:1627 / database 399 |
| P1-020 config unlocked writers | — | **STILL-OPEN Low** — see C-3 residual | cro:1048/1509/5602 |
| DEF-008 snapshot omits hive fields | ConfigSnapshot | **MOOT** — fields don't exist; only gap is `hive_hints_allow_all_hints_m2_scope` (init-only) | config 595 |

---

## 7. NEW concurrency findings (for the coordinator to add to the ledger)

Structured `severity | file:line | defect | interleaving scenario`:

- **Medium | modules/flow_analysis.py:1675 (analyze_channel) vs :1313 (analyze_all_channels)** |
  RPC `analyze_channel` does not acquire `_analysis_lock`; only `analyze_all_channels` does.
  Worse, `_kalman_lock` guards ONLY the `_get_kalman_filter` dict lookup (821/838) — once the
  `kf` reference is returned, `kf.predict()` (961) and `kf.update()` (968) mutate the filter
  **with no lock at all** | T0 (`revenue-analyze` at cro:3820) and T1 (flow daemon) both hold
  the same `kf` for a channel and both predict+update it → double-consumed observations and
  artificially shrunk uncertainty. Confirms DEF-027/DEF-029.
- **Medium | modules/database.py:6272 (cleanup_old_data) / :4441 (bulk_insert_forwards)** |
  Long `BEGIN IMMEDIATE` batch can hold the WAL writer > `busy_timeout=5000ms` |
  `cleanup_old_data` on T1 (every flow cycle: prune + rollup + incremental_vacuum) contends
  with `record_fee_change`(T2)/`record_forward`(T0)/`record_rebalance_cost`(T3); on a large DB
  the batch exceeds 5s → the contender raises `sqlite3.OperationalError: database is locked`.
  Real "database is locked" exposure — Phase 2B soak target.
- **Low | modules/database.py:3789 (mark_spend_reservation_spent), :3736 (reserve_spend), :3881
  (release_spend_reservations)** | multi-statement read-then-write with no enclosing `BEGIN` |
  crash between the UPDATE and `record_spend_event` (3809) leaves the reservation `spent` with
  no spend_event (or the reverse); idempotency guards keep it consistent-not-corrupt, so this
  is a crash-consistency window (spend ledger).
- **Low | modules/database.py:4204** | deferred `BEGIN` (not `BEGIN IMMEDIATE`) inconsistent
  with every other transaction | lazily upgrades to a writer mid-transaction → higher
  `SQLITE_BUSY`/upgrade-failure probability than the IMMEDIATE paths under contention.
- **Low | modules/rebalancer.py:1697/1740 (`_fee_cache`), 1704/1816/1820 (`_peer_inbound_fees`)** |
  instance dicts read/written with no lock | T3 (rebalance daemon) and T0 (`revenue-rebalance`,
  esp. force=false manual path that bypasses the engine cycle — P1-001) touch them concurrently;
  GIL keeps individual ops safe but `_fee_cache = {}` reset (1387) can clear mid-read →
  stale/partial fee memoization. Confirms DEF-061 / DEF-067-S9.
- **Low | modules/fee_controller.py:3144/4454 (`_state_lock` held across serial listchannels
  RPCs + DB for the whole fee cycle)** | long lock hold | during a fee cycle on T2, T0's
  `forward_event` reputation/DTS nudge waits `FORWARD_EVENT_LOCK_TIMEOUT_SECS=2.0s`
  (cro:5609) then is **dropped** → lost negative-flow signal under load. Liveness, not
  deadlock. Confirms DEF-051.
- **Low | cl-revenue-ops.py:5168, 8237** | `dict(_boltz_auto_cycle_state)` read without
  `_boltz_auto_cycle_state_lock` | a status RPC on T0 snapshots the dict while T6 mutates it;
  benign today (writers only `.update()` existing keys, no size change) but unsynchronized.
- **Info | modules/fee_controller.py:6187/6200** | live `self.config.max_fee_ppm` read amid an
  otherwise snapshot-based computation (snapshot uses `cfg` at 6175) | single field, not a torn
  pair — snapshot-vs-live inconsistency only.
- **Low | modules/fee_controller.py:2604/2605/2606 (`_hive_member_set_at`,
  `_hive_member_released_peers`, `_hive_member_advisory_peers`)** | lock-free set/dict mutation
  (`_remember_hive_member` 2745-2752, `_clear_hive_member_cache` 2753/2754, discard 3478) |
  T2 fee cycle and T0 (hint-update / forward_event) mutate these hive-gate collections
  concurrently with no lock → a set resized during iteration on another thread can raise
  `RuntimeError`, and the zero-fee hive gate can read a half-updated membership set.
- **Low | modules/fee_controller.py:3671/4969/6559/6651/6706/6729/6787/6867** | lock-free reads
  of `_channel_fee_states` while the cycle mutates it under `_state_lock` | T0 status/report
  RPCs read fee-state values the T2 cycle is mid-writing → stale/torn field reads in operator
  output (no crash on CPython `.get()`).
- **Medium (carry-over, P1-003) | cl-revenue-ops.py:6367 (`_spend_reserve_lock`) vs
  rebalancer external-liquidity read** | cross-category budget reconciled under two different
  locks at two different instants | T3 rebalance reserve (BEGIN IMMEDIATE budget) and T0/T6
  generic spend-reserve (`_spend_reserve_lock`) each pre-subtract the other's category from a
  read taken outside the other's lock → the unified total-cost budget can be jointly exceeded
  across categories. Latent, bounded by daemon cadence.

---

## 8. Recommendations (for 2B/2C and the fix stage — NOT applied here)

- `REC:` Phase 2B stress must assert **no `sqlite3.OperationalError`** under concurrent daemon
  + RPC-firehose (NC-3), plus `PRAGMA integrity_check` and no thread death.
- `REC:` add `_analysis_lock` (or a shared single-flight) to the RPC `analyze_channel` path
  (NC-1) — behavioral, escalate.
- `REC:` wrap the three spend-ledger multi-statement methods in `BEGIN IMMEDIATE` (NC-2);
  normalize the deferred `BEGIN` at 4204 to `BEGIN IMMEDIATE`.
- `REC:` guard `_fee_cache`/`_peer_inbound_fees`, or document reliance on `_cycle_lock` and
  gate the manual force=false path behind it.
- `REC:` route all config mutation through `update_runtime` (`_lock` + `_version` bump) —
  closes P1-020 / the C-3 residual.
- `REC:` shorten fee `_state_lock` critical section (DEF-051) — behavioral, escalate.
