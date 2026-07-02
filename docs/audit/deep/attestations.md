# Deep-Audit Attestations

Per P1, a chunk is COVERED without a finding only by a **structured
attestation**. A bare "clean" is rejected by the refuter (P2). Each attestation
block MUST contain, for its chunk:

1. **chunk_id** — must match a row in `coverage-manifest.md`.
2. A **3-5 line control-flow / state-write summary** — what the code in the
   chunk does, and specifically what external state it writes (DB rows, files,
   RPC calls, in-memory shared caches). Pure/no-write chunks say so explicitly.
3. The **invariant checklist applied** — tick which invariants you actively
   checked (msat/sat rounding direction, sign/off-by-one, budget/cap arithmetic,
   None/missing handling, SQL/transaction boundaries, AGENTS.md never-call
   compliance, concurrency/shared-state). Mark N/A where a class cannot apply.
4. The **single most-suspicious line + why it is acceptable** — the one line you
   would flag if forced to pick, and the concrete argument for why it is safe.

## Format

Use a level-3 heading whose first token is the exact `chunk_id`, then the four
required fields as a bullet list. The coverage tool reads the `chunk_id:` field
(preferred) or the heading token. Any block containing the word `EXAMPLE` is
ignored by `deep_manifest.py --coverage`, so the worked example below never
counts toward real coverage.

```
### <chunk_id>
- chunk_id: <chunk_id>
- summary: <3-5 lines of control flow / state writes>
- invariants: [rounding: OK, sign: OK, caps: N/A, none-handling: OK,
  sql: N/A, never-call: OK, concurrency: N/A]
- most_suspicious: L<NNN> — <the line> — <why acceptable>
- auditor: <name/agent>  date: <YYYY-MM-DD>
```

---

## EXAMPLE (worked, non-counting)

### EXAMPLE modules/utils.py#1
- chunk_id: modules/utils.py#1
- summary: Pure helper module (lines 1-114). `normalize_scid` maps ':'→'x';
  `parse_msat` coerces heterogeneous msat representations to int; the
  `base_to_sats_*` family converts between msat base units and sats with an
  explicit rounding direction; module tail defines constants and backward-compat
  aliases. **No I/O, no DB, no RPC, no shared mutable state** — only a
  module-level logger used for debug messages.
- invariants: [rounding: OK — `base_to_sats_ceil` uses `-(-base // 1000)` so
  fees/budgets round UP (never undercharge); `base_to_sats_floor` rounds DOWN for
  spendable balances; `base_delta_to_sats_toward_zero` handles the signed case;
  matches README rounding contract], [sign: OK — negative deltas explicitly
  handled at L93-95], [caps: N/A], [none-handling: OK — `parse_msat(None)`→0,
  `normalize_scid(None)`→""], [sql: N/A], [never-call: OK — no RPC], [concurrency:
  N/A — stateless, thread-safe by construction]
- most_suspicious: L42 `if isinstance(msat_val, (int, float))` — a bool would
  match `int` and coerce (True→1), but L39-41 intercepts bool first and returns
  0 (the "U-1 FIX"), so the ordering makes this acceptable.
- auditor: EXAMPLE  date: 2026-07-01

---

## Attestations

<!-- Real attestation blocks go below this line. -->

### cl-revenue-ops.py#4
- chunk_id: cl-revenue-ops.py#4
- summary: Pure declarative block of `plugin.add_option(...)` registrations (budgets, Boltz, planner, hive-hints options) ending at the start of `_on_rebalance_router_change`. No control flow beyond registration; writes only plugin option metadata at import/init time. No DB rows, no RPC calls, no money movement — all defaults are strings/ints parsed later elsewhere.
- invariants: [rounding: N/A no arithmetic, only default literals; sign: N/A; caps: N/A defaults only (e.g. daily-budget 5000, max-withdraw 10000000) — enforcement is elsewhere; none-handling: N/A no runtime values; sql: N/A; never-call: OK no RPC invoked; concurrency: OK registration runs single-threaded at load]
- most_suspicious: L1423 — `description='Hard cap on a single Boltz on-chain withdraw in sats (default: 10000000; 0 disables). A sweep withdraw bypasses this cap and requires confirm_sweep=true.'` — this only sets a description string; the documented cap-bypass is enforced in the withdraw path, not here, so the option declaration itself moves no funds and is safe.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#8
- chunk_id: cl-revenue-ops.py#8
- summary: Background daemon loop bodies (rebalance-check, boltz-auto-cycle, capacity-planner, startup-snapshot, financial-snapshot) plus `_take_financial_snapshot`, `run_flow_analysis`, `run_fee_adjustment`, `_push_dashboard_to_datastore`, and thread spawns. Each loop is wrapped in a canonical try/except with heartbeat + interruptible `shutdown_event.wait`. External writes: `database.record_financial_snapshot(...)`, `database.decay_reputation(...)`, `data_service.datastore_push(...)`. RPC via wrapped safe proxies only.
- invariants: [rounding: OK L3049 `revenue_msat // 1000` floors msat→sat for a snapshot (conservative, no money moved); sign: OK jitter uses symmetric `randint(-jitter,jitter)` with `max(60,...)`/`max(600,...)` interval floors; caps: OK sleep/interval floored, mid_fee `(min+max)//2` is display only; none-handling: OK `getattr(config,...,default)`, `if database`, `recent[0].get(...,0)`; sql: OK all writes go through database methods (parameterized layer), no inline SQL; never-call: OK uses `refresh_hive_runtime`, `run_rebalance_check`, dashboard helpers — no forbidden lightning RPC; concurrency: OK config.snapshot() to avoid mid-loop mutation, daemon threads, shutdown_event coordinated]
- most_suspicious: L3058 — `capacity_sats=local_bal + remote_bal,` — capacity is derived as the sum of two `.get(...,0)` sat values already floored upstream; both default to 0 so no None can enter the addition, and this feeds a historical snapshot row only (not a spend decision), so the derived capacity is acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#12
- chunk_id: cl-revenue-ops.py#12
- summary: Tail of `revenue-profitability` (per-channel + all-channel aggregation) and the deprecated ignore/unignore/list-ignored methods plus start of `revenue-policy`. Read-only reporting: builds flow-profile classification and summed totals from `profitability_analyzer` results; deprecated methods gate writes behind `_policy_write_override` (internal/admin flags) and validate peer_id hex before calling `policy_manager.set_policy/delete_policy`. Writes: policy rows only via policy_manager when override present.
- invariants: [rounding: OK L4508 `total_revenue = -(-total_revenue_msat // 1000)` is deliberate ceiling of msat→sat for a report total (rounds revenue up in a display aggregate, not a spend/budget path); sign: OK ratio thresholds (3.0 / 0.33) consistent, `outbound+inbound==0` guarded before division; caps: N/A reporting; none-handling: OK `next(...,default)`, `tracked_state.get(...) if tracked_state`, peer_id regex-validated; sql: OK via policy_manager; never-call: OK no lightning RPC here; concurrency: OK read-mostly, policy writes serialized in manager]
- most_suspicious: L4524 — `"overall_roi_pct": round((total_profit / total_costs * 100) if total_costs > 0 else 0, 2),` — division is guarded by `total_costs > 0`, so no ZeroDivisionError; the value is a reporting percentage only, so even the ceiling-rounded revenue feeding it cannot move money.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#14
- chunk_id: cl-revenue-ops.py#14
- summary: Tail of `revenue-config` (set/reset/list-mutable, gated by `is_public_runtime_key`), then `revenue-dashboard` and `revenue-health` reporting methods, and start of `revenue-cleanup-closed`. Reads profitability/pnl/roc/bleeders and loop-liveness; writes config overrides only via `config.update_runtime(database, ...)` / `database.delete_config_override`. Health method is read-only aggregation across subsystems with per-section try/except.
- invariants: [rounding: OK L5511 `int(p.get("total_fee_msat",0))//1000` floors msat→sat for a route-report field; sign: OK `max(0,...)`, `max(1,...)` guards on utilization denominator; caps: OK L5290 `max(1, min(int(window_days),365))` clamps window; none-handling: OK dict `.get` with defaults everywhere, `isinstance(budget,dict)` guard, int coercion wrapped in try; sql: OK through database methods; never-call: OK no forbidden RPC; concurrency: OK `with _boltz_auto_cycle_state_lock` snapshot, `list(...)` copies of fee_controller state dicts before iteration]
- most_suspicious: L5467 — `100.0 * actual_spent / max(1, budget.get("effective_budget_sats", 1)), 1` — the denominator is guarded by `max(1, ...)` so a zero or missing effective budget cannot divide-by-zero; utilization is a display percentage, acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#15
- chunk_id: cl-revenue-ops.py#15
- summary: Tail of `revenue-cleanup-closed` (diff tracked vs open SCIDs, archive closed channels via `_archive_closed_channel`), `revenue-clear-reservations` (releases budget reservations), `_resolve_scid_to_peer` (cached SCID→peer with negative caching), `_looks_like_scid`, `_resolve_event_channel_scid`, `_parse_msat`, `_refresh_dynamic_config`, and forward_event handler start. Writes: closed_channels archive rows, reservation clears, peer reputation upserts; RPC via data_service/safe_plugin proxies.
- invariants: [rounding: N/A this range parses msat but does no sat conversion here (`_parse_msat` delegates); sign: OK `max(0, cfg.daily_budget_sats - daily_spent)` floors budget at 0; caps: OK negative-cache bounded by `_SCID_NEGATIVE_CACHE_MAX_ENTRIES`; none-handling: OK `peer_id or ch_info.get('peer_id')`, `resolved is None` handled, `if not isinstance(raw_channel_id,str)` guard; sql: OK via database methods; never-call: OK uses `listpeerchannels`/`listclosedchannels`/`listconfigs` through data_service or safe_plugin proxy — read-only allowed RPCs, no forbidden fee/pay call; concurrency: OK double-checked locking with `_scid_cache_lock` + `_scid_cache_fetch_lock` serializes cache-miss RPC (M-2 fix)]
- most_suspicious: L5703 — `budget_available = max(0, cfg.daily_budget_sats - daily_spent)` — subtraction could theoretically underflow if daily_spent exceeds budget, but `max(0, ...)` clamps it; this is a reported "available" figure after clearing reservations, so a clamped-to-zero result is correct and cannot over-authorize spend.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#21
- chunk_id: cl-revenue-ops.py#21
- summary: Core of `_build_boltz_balance_plan` phase-1 candidate scan + phase-2 bounded quoting. Per-channel: computes local_pct, depletion estimate, dynamic thresholds, direction (loop_in/loop_out), amount with safety caps, policy gate; defers boltzcli quotes to top-ranked candidates; builds economics (expected uplift, DTS/structural credit, profit guard). Reads DB rebalance-success and profitability; issues boltzcli `bm.quote(...)` subprocesses (quoting only, no swap execution here). No money moved — planning only.
- invariants: [rounding: OK L8032/8033 `parse_msat(...)//1000` floors spendable/receivable msat→sat (conservative for amount sizing); sign: OK severity uses `max(0.0,...)` and guarded denominators `max(...,1.0)`; caps: OK L8174 amount clamped to `min(cap, 25% capacity, 5_000_000)`, quote_budget `max(0,...)*MULT`; none-handling: OK extensive `(dts_summary or {}).get`, `float(... or 0.0)`, `prof is None` skip; sql: OK via database methods, try-wrapped; never-call: OK only boltzcli quote subprocess + read RPCs, no forbidden lightning fee/pay RPC; concurrency: OK operates on passed-in channel snapshot, no shared-state mutation in this range]
- most_suspicious: L8300 — `amount_fraction = min(1.0, amount_sats / max(1, raw_amount))` — division guarded by `max(1, raw_amount)` (raw_amount is already `max(0,...)`), so no divide-by-zero; the fraction pro-rates the uplift heuristic for cap-bound partial swaps and is clamped to 1.0, so it cannot inflate expected profit.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#22
- chunk_id: cl-revenue-ops.py#22
- summary: Tail of `_build_boltz_balance_plan` (loop-out multi_goal scoring, candidate dict assembly, final sort, return payload with thresholds/structural-credit/planner-coordination), then `revenue-boltz-balance-recommendations` RPC wrapper, `revenue-boltz-auto-cycle-status/-run-now`, and start of `_execute_boltz_balance_cycle` (pending-swap short-circuit, plan build, per-rec profit+budget gates). Planning + first execution gates; hive_router.score used as bounded multiplier. No swap fired in this range except entry into the execution loop.
- invariants: [rounding: OK scoring is float ratios, no msat/sat conversion loses money; sign: OK signals clamped `max(0.0,min(1.0,...))`, `max(0,int(max_candidates))`; caps: OK recommendations sliced `[:max(0,int(max_candidates))]`, cooldown `max(0.5,...)`; none-handling: OK `float(... or 0.0)`, `isinstance(...,dict)` guards, `getattr(config,...,default)`; sql: N/A no direct SQL; never-call: OK hive_router scoring + boltz manager only, no forbidden RPC; concurrency: OK `with _boltz_auto_cycle_state_lock` for status snapshot; execution cooldown locks appear in chunk #23]
- most_suspicious: L8738 — `if est_fee > remaining_budget:` — a strict `>` allows a swap whose estimated fee exactly equals remaining budget, but est_fee and remaining_budget are both non-negative ints and equality-spend is intended (fully consuming the 24h estimate is acceptable), so no over-spend beyond the budget occurs.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#23
- chunk_id: cl-revenue-ops.py#23
- summary: Execution loop of `_execute_boltz_balance_cycle` (structural-envelope gate, TOCTOU-safe cooldown pre-claim under `_boltz_balance_lock`, dry-run vs live `bm.loop_in/loop_out`, budget decrement, cooldown restore on reject/exception), its RPC wrapper `revenue-boltz-balance-cycle`, `revenue-boltz-expansion-treasury-status/-recommendations`, and start of `_execute_boltz_expansion_treasury_cycle`. Writes: real Boltz swaps via `bm.loop_out/loop_in` (spends fees), cooldown map mutation, budget accounting. Structural spend gated by 24h envelope, fail-closed on unknown spend.
- invariants: [rounding: OK est_fee/amount are ints; deficit `max(0, target-onchain)`; sign: OK budget decrements `max(0, remaining_budget - est_fee)` cannot go negative; caps: OK structural envelope `spent_24h + est_fee > envelope` blocks, `len(executed) >= max_actions` breaks; none-handling: OK `onchain is not None` distinguishes listfunds-failure (F9) from zero, `int(... or 0)` throughout; sql: OK `database.get_category_spend_sats` fail-closed to None→skip; never-call: OK swaps go through boltz manager, no forbidden CLN pay/fee RPC directly; concurrency: OK `with _boltz_balance_lock` pre-claims slot before execute and restores on reject/exception (H-5/C1 TOCTOU fixes)]
- most_suspicious: L8766 — `if envelope <= 0 or spent_24h is None or spent_24h + est_fee > envelope:` — this fail-closed condition treats a None spend query (DB error) as envelope-exhausted, correctly refusing the structural swap rather than spending blind; the ordering short-circuits `spent_24h is None` before the arithmetic so no None enters the addition — safe.
- auditor: fable-phase8  date: 2026-07-02

---

### cl-revenue-ops.py#24
- chunk_id: cl-revenue-ops.py#24
- summary: Closing brace of the treasury-cycle return dict, the `revenue-boltz-expansion-treasury-cycle` RPC wrapper (thin delegate to `_execute_boltz_expansion_treasury_cycle` passing args through), and the `__main__` entry point calling `plugin.run()`. Pure dispatch/pass-through; no arithmetic, no direct DB/RPC writes in this range (delegated).
- invariants: [rounding: N/A no arithmetic; sign: N/A; caps: N/A; none-handling: N/A wrapper forwards operator args verbatim to the executor which coerces/defaults them; sql: N/A; never-call: OK only `plugin.run()` and internal delegate, no lightning RPC; concurrency: OK `plugin.run()` is the single blocking main-thread entry]
- most_suspicious: L9248 — `plugin.run()` — this is the standard pyln-client blocking event loop entry under `if __name__ == "__main__"`; it starts the plugin only when executed directly and performs no financial action itself, so it is safe.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#3
- chunk_id: modules/database.py#3
- summary: Pure DDL/migration section of `_init_db`. Creates tables (channel_probes, ignored_peers, peer_policies, budget/spend reservations, spend_events, financial_snapshots, closure/closed channels, daily aggregates), builds indexes, and runs additive `ALTER TABLE ... ADD COLUMN` migrations each guarded by try/except on `sqlite3.OperationalError`. Writes schema objects only; no row-level money writes. Runs inside the caller's init connection.
- invariants: [rounding: N/A no arithmetic; sign: N/A; caps: N/A; none-handling: OK columns default NOT NULL DEFAULT 0 for money fields; sql: OK all statements are static string literals, zero interpolation/params; never-call: OK no RPC/forbidden API; concurrency: OK idempotent DDL (IF NOT EXISTS / OperationalError-swallowed ALTERs) safe under WAL re-init]
- most_suspicious: L845 — `conn.execute("ALTER TABLE hot_channel_protection_overrides ADD COLUMN min_depletion_trigger_pct REAL")` — bare ALTER that will raise on a DB where the column already exists, but it is wrapped by the `except sqlite3.OperationalError: pass` at L847, making the migration idempotent; static SQL, no injection surface.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#5
- chunk_id: modules/database.py#5
- summary: channel_states upsert (single + batch), temporal profile save/load, hourly histogram batch, and kalman_state get/save/batch/reset. Writes channel_states and kalman_state rows. Single-row upserts run under autocommit; batch upserts and `save_temporal_profiles_batch` use `BEGIN IMMEDIATE`/COMMIT with ROLLBACK-on-exception. Kalman saves reject NaN/Inf before persisting.
- invariants: [rounding: OK histogram uses integer floor division for per-day averages (display metric, not money); sign: OK; caps: N/A; none-handling: OK `.get(...)` defaults for optional cols, NaN/Inf guard via `math.isfinite`; sql: OK fully parameterized, no interpolation; never-call: N/A no RPC; concurrency: OK BEGIN IMMEDIATE on all multi-row writes, single-statement upserts atomic under autocommit]
- most_suspicious: L1860 — `histogram[h]["out_sats"] = int(row[2] or 0) // days_with_data` — floor division discards a sub-sat remainder, but this is a per-hour flow histogram used for temporal profiling (not an accounting figure), `days_with_data` is guaranteed `>= 1` by L1815 `max(1, ...)`, so no divide-by-zero and no money is lost.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#6
- chunk_id: modules/database.py#6
- summary: channel_probe set/get/clear, fee_strategy_state get/update (single+batch)/reset, record_fee_change, record_rebalance, and start of update_rebalance_result. Writes channel_probes, fee_strategy_state, fee_changes (audit), rebalance_history. Single writes autocommit; `update_fee_strategy_states_batch` uses BEGIN IMMEDIATE. record_rebalance uses named-parameter binding and returns `cursor.lastrowid`.
- invariants: [rounding: N/A integer ppm/sats passed through; sign: OK trend_direction ±1 stored verbatim; caps: N/A; none-handling: OK get_fee_strategy_state backfills every missing column with defaults; sql: OK parameterized incl. named `:from_channel` dict binding at L2380; never-call: N/A; concurrency: OK batch write transactional; get_channel_probe's inline DELETE at L2070 is a single autocommit statement]
- most_suspicious: L2070 — `conn.execute("DELETE FROM channel_probes WHERE channel_id = ?", (channel_id,))` — a write performed inside the read method `get_channel_probe` (lazy expiry of stale flags); acceptable because it is a single idempotent parameterized DELETE that only fires when the probe already exceeded `max_age_seconds`, and re-deleting an absent row is a no-op under concurrency.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#7
- chunk_id: modules/database.py#7
- summary: Tail of update_rebalance_result (msat/sat reconciliation + COALESCE update of rebalance_history), pending-settlement/anchor getters, pair-rebalance-failure cooldown upsert (atomic increment under BEGIN IMMEDIATE), clear, cost getters, and node-realized-fee-ppm queries. Money-relevant writes: rebalance_history result update and pair_rebalance_failures upsert. Validates channel ids before writes.
- invariants: [rounding: OK fee_sats derived via `base_to_sats_ceil` (conservative, never underbills), ppm via floor `//`; sign: OK amounts clamped `max(0, int(...))` L2413, ratio clamped [0,1] L2408; caps: OK backoff multiplier bounded `MIN(MAX(...),6)`; none-handling: OK COALESCE preserves existing cols when args None; sql: OK parameterized, OperationalError fallback branches also parameterized; never-call: N/A; concurrency: OK atomic upsert + follow-up SELECT inside one BEGIN IMMEDIATE]
- most_suspicious: L2604 — `cooldown_until = excluded.last_failure_at + (? * MIN(MAX(pair_rebalance_failures.failure_count + 1, 1), 6))` — computes exponential-ish backoff off the post-increment count inside the same upsert; acceptable because the increment and this arithmetic are one atomic statement (no read-modify-write race), the multiplier is capped at 6, and `base_cooldown` is `max(1, ...)` so cooldown_until always advances.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#8
- chunk_id: modules/database.py#8
- summary: get_node_realized_fee_ppm_30d (forwards+daily rollup blend), get_total_routing_revenue, financial_snapshot record/get, and per-channel P&L (get_channel_pnl / inbound_contribution / full_pnl). Writes only financial_snapshots (INSERT OR REPLACE, autocommit, try/except returns bool). Rest are pure reads. Uses f-string `{placeholders}` built from `_sql_placeholders`.
- invariants: [rounding: OK fees via `base_to_sats_ceil`, volume via floor, net via `base_delta_to_sats_toward_zero`; sign: OK net_pnl can be negative, toward-zero conversion preserves sign; caps: N/A; none-handling: OK COALESCE(...,0) and `or 0` guards on every SUM; sql: OK `{placeholders}` is only `?`-chars from `_sql_placeholders`, all real values bound as params incl. `*aliases`; never-call: N/A; concurrency: OK read-only + single-statement snapshot write; day-boundary `date < today_start` prevents rollup/forwards double-count]
- most_suspicious: L3006 — `WHERE channel_id IN ({placeholders}) AND date >= ? AND date < ?` — an f-string interpolated directly into SQL; safe because `placeholders` is produced solely by `_sql_placeholders` (L54: `",".join("?" for _ in values)`) so it can only ever emit `?` separators, and the SCID alias strings are passed positionally through the params tuple — no user data reaches the SQL text.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#9
- chunk_id: modules/database.py#9
- summary: Batched all-channel P&L (get_all_channels_full_pnl), last-forward-time, and per/all-channel revenue totals. Pure read methods that aggregate forwards + daily rollups via `REPLACE(col,':','x')` GROUP-BY key normalization (equivalent to normalize_scid), then assemble dicts identical in shape to the single-channel methods. No external state writes.
- invariants: [rounding: OK identical helpers to #8 — ceil for fees/contribution, floor for volume, toward-zero for net_pnl; sign: OK signed net preserved; caps: N/A; none-handling: OK `int(x or 0)` on every field, `if not cid: continue` skips empty scids; sql: OK static SQL literals here, no interpolation, single `?` param bound; never-call: N/A; concurrency: OK read-only, `date < today_start` boundary avoids double-count with unpruned forwards]
- most_suspicious: L3352 — `'net_pnl_sats': base_delta_to_sats_toward_zero(net_pnl_msat)` — signed msat->sat conversion where net_pnl may be negative (contribution < rebalance cost); acceptable because toward-zero rounding is the correct convention for a signed delta (it neither overstates a profit nor a loss), matching the single-channel `get_channel_full_pnl` at L3194 so batched and per-channel results agree.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#12
- chunk_id: modules/database.py#12
- summary: Tail of budget-summary, get_budget_status (transactional two-SUM snapshot), rebalance-history-by-peer, historical inbound fee ppm, latest-forward-timestamp, bulk_insert_forwards (chunked BEGIN IMMEDIATE idempotent inserts), and start of get_daily_flow_buckets. Money-relevant read: budget snapshot (spent+reserved). Write: bulk forward insert under INSERT OR IGNORE.
- invariants: [rounding: OK ppm via floor `//`, median via `//`; sign: N/A non-negative; caps: OK limits clamped `max(1,min(limit,1000/10000))`; none-handling: OK COALESCE/`or 0`, NULLIF(amount_sats,0) guards divide-by-zero, `total_volume==0` early-return; sql: OK `{placeholders}` from `','.join('?'*len)`, all ids bound positionally; never-call: N/A; concurrency: OK budget snapshot in BEGIN IMMEDIATE, bulk insert chunked to release WAL writer, ROLLBACK on error]
- most_suspicious: L4607 — `avg_fee_ppm = (total_fees_msat * 1000) // total_volume` — volume-weighted ppm via floor division; acceptable because `total_volume == 0` is explicitly guarded at L4604 (returns None), the multiply-before-divide preserves precision, and a floored ppm cost estimate is conservative (never overstates how cheap inbound is).
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#14
- chunk_id: modules/database.py#14
- summary: Reputation-weighted volume query, daily/total volume + forward-count getters, peer latency stats, channel-open/rebalance cost record+read, cost-history, and rebalance success-rate helpers. Writes channel_costs (INSERT OR REPLACE) and rebalance_costs (INSERT), both single-statement autocommit. record_rebalance_cost reconciles cost_sats/cost_msat symmetrically.
- invariants: [rounding: OK msat->sat via `base_to_sats_ceil` for costs (conservative), `base_to_sats_floor` for volume; sign: N/A non-negative costs; caps: N/A; none-handling: OK cost_sats/cost_msat each backfilled from the other when None (L5400-5405); sql: OK `{placeholders}` from `_sql_placeholders`, params bound; never-call: N/A; concurrency: OK single-statement writes atomic; success-rate query counts only terminal statuses]
- most_suspicious: L5405 — `persisted_cost_sats = base_to_sats_ceil(persisted_cost_msat)` — derives the stored sat cost by rounding msat up; acceptable and intentional: rounding a rebalance cost toward the ceiling never understates spend, so budget accounting that later sums cost_sats stays conservative rather than silently leaking fractional-sat spend.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#17
- chunk_id: modules/database.py#17
- summary: Tail of peer-reputation getters, decay_reputation (transactional UPDATE+DELETE), cleanup_old_data (chunked forwards prune with atomic aggregate-into-daily-stats + delete, then autocommit secondary prunes + incremental_vacuum), and peer connection-history record/uptime. Writes: peer_reputation decay, daily_forwarding_stats(_inbound) rollups, and DELETEs across forwards/fee_changes/rebalance_history/reservations.
- invariants: [rounding: OK decay via `CAST(... AS INTEGER)` floors counts; sign: N/A; caps: OK uptime clamped `min(100,max(0,...))`; none-handling: OK `int(r["x"] or 0)` throughout, cold-start returns 100.0; sql: OK all parameterized static SQL; never-call: N/A; concurrency: OK per-chunk BEGIN IMMEDIATE keeps forwards->daily aggregation+delete atomic (no double-count), decay UPDATE+DELETE wrapped so fresh writes aren't dropped, secondary prunes intentionally autocommit to release writer]
- most_suspicious: L6477 — `SET success_count = CAST(success_count * ? AS INTEGER),` — multiplicative decay that floors to integer each cycle and can drive a low count to 0; acceptable because it is the intended time-windowing (rows hitting (0,0) are then garbage-collected by the L6483 DELETE in the same transaction), and the whole UPDATE+DELETE runs inside BEGIN IMMEDIATE so a concurrent `update_peer_reputation` cannot be silently erased.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#18
- chunk_id: modules/database.py#18
- summary: Tail of uptime calc, hot-channel-protection override CRUD, config_overrides get/set/delete (versioned, BEGIN IMMEDIATE to avoid version TOCTOU), planner candidate/action CRUD, dead-capital-stage upsert, mempool-fee record (self-pruning) and MA, and recycle-op CRUD. Writes many operational tables; config version writes and deletes are transactional, others single-statement autocommit. Dynamic UPDATE column lists built for planner/recycle updates.
- invariants: [rounding: N/A; sign: N/A; caps: OK min_depletion_trigger_pct validated `0 < x <= 100`; none-handling: OK optional args gate each SET clause, mempool MA returns 1.0 fallback; sql: OK dynamic `SET {', '.join(updates)}` uses only hardcoded column-name literals with values bound as params — no user input in SQL text; never-call: N/A; concurrency: OK set/delete_config_override compute version inside BEGIN IMMEDIATE and return the in-transaction value]
- most_suspicious: L7028 — `conn.execute(f"UPDATE planner_actions SET {', '.join(updates)} WHERE id = ?", params)` — SQL text assembled via f-string; safe because every element appended to `updates` is a fixed literal like `"status = ?"` / `"actual_cost_sats = ?"` (L7012-7024) with the actual values pushed onto the bound `params` list, so the interpolated fragment is attacker-uninfluenced and `action_id` is parameterized.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/database.py#19
- chunk_id: modules/database.py#19
- summary: Policy CRUD (get/upsert/delete/batch/expired-sweep/changes-since), get_total_capex_by_channel (sums rebalance_costs.cost_sats + spend_events.amount_sats), and cleanup_orphaned_rebalances (transactional SELECT+UPDATE marking stale pendings failed). Writes peer_policies and rebalance_history status. delete_expired_policies and orphan cleanup wrap SELECT+DELETE/UPDATE in BEGIN IMMEDIATE so returned id lists match rows mutated.
- invariants: [rounding: N/A sats-native sums, no msat conversion; sign: N/A non-negative; caps: N/A; none-handling: OK `int(r["total"] or 0)`, `channel_id IS NOT NULL` filter on spend_events, `(row["max_ts"] or 0)`; sql: OK all parameterized static SQL, executemany with bound tuple rows; never-call: N/A; concurrency: OK expired-policy sweep and orphan cleanup are atomic SELECT+mutate under BEGIN IMMEDIATE with ROLLBACK]
- most_suspicious: L7315 — `SELECT channel_id, COALESCE(SUM(cost_sats), 0) as total FROM rebalance_costs WHERE timestamp >= ?` — capex aggregation sums the sat-native `cost_sats` column rather than the msat-precise `cost_msat` used by the P&L paths; acceptable because capex budgeting operates in whole sats and `cost_sats` is written via `base_to_sats_ceil` (chunk #14), so this sum is a conservative (ceiling) capex figure consistent with the budget envelope's sat granularity.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#1
- chunk_id: modules/fee_controller.py#1
- summary: Module docstring, FeeReasonCode enum, and GaussianThompsonState dataclass definition (constants + fields) plus pure math helpers (_mat3_det/invert/vec_mul, _cholesky3), predict_optimal_fee, scale_variance, _resolve_exploration_boost. All pure in-memory computation over the posterior state; no DB rows, files, or RPC. predict_optimal_fee returns a clamped posterior mean; scale_variance arms an exploration multiplier and widens posterior_std.
- invariants: [rounding: L393 `int(max(floor_ppm, min(ceiling_ppm, round(optimal_fee))))` rounds-then-clamps, correct order; sign: _mat3_invert uses relative tol (L327) scaled by cube of max element — singular guard sound; caps: scale_variance clamps factor to [0.75,2.0] (L415-416) and posterior_std to max_std (L419-421); none-handling: predict_optimal_fee returns None on <MIN_OBSERVATIONS or non-finite mean (L386-391); _mat3_invert/_cholesky3 return None on singular (L328-329, L369/373); sql: N/A pure; never-call: N/A no RPC; concurrency: N/A no shared-state writes (instance-local fields, caller holds lock)]
- most_suspicious: L418 — `max_std = math.sqrt(1.0 / self.MIN_PRECISION)` — MIN_PRECISION is a class attribute defined far below at L1518 (0.000025) inside the same class body; at runtime the attribute is resolved on the fully-built class so the forward textual reference is safe (never evaluated at class-def time), yielding max_std≈200; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#2
- chunk_id: modules/fee_controller.py#2
- summary: Sampling and observation ingestion — sample_fee, sample_fee_contextual, _sample_from_polynomial_posterior, update_posterior, trickle/zero-probe helpers, _positive_revenue_mass, _earning_region_fee, supported_fee_ceiling. Pure in-memory: draws from posterior, appends observation tuples to self.observations (bounded to MAX_OBSERVATIONS), recomputes posterior. No DB/file/RPC.
- invariants: [rounding: sampled fees floored via `int(max(floor,min(ceiling,sampled)))` (L484,495,564) — int() truncates toward zero but result is re-clamped to floor so cannot drop below floor; sign: zero-probe injects fee*0.9 strictly below charged fee, guarded `if probe_fee < fee` (L709); caps: sample clamps to [floor,ceiling] every path; ctx offset clamped to ±CTX_OFFSET_CAP_FRAC*|base| then *confidence (L561-562); none-handling: update_posterior skips non-finite/negative fee (L656-657), coerces bad hours/rate (L652-655); polynomial sampler returns None->Gaussian fallback (L589,594,598,621); sql: N/A; never-call: N/A no RPC; concurrency: mutates self.observations/streak fields — caller (_state_lock) serializes]
- most_suspicious: L659 — `weight = min(1.0, hours / 6.0)` — exposure-time-only weighting (no revenue term), deliberately caps a window's weight at 1.0 regardless of duration; matches the documented WEIGHT_SCHEME "exposure_v2" fix so a long window cannot dominate; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#3
- chunk_id: modules/fee_controller.py#3
- summary: Contextual posterior updates (update_contextual, _update_related_time_contexts), durable nudge machinery (record_posterior_nudge, _blend_posterior_toward, _posterior_bias_shift, _apply_posterior_bias), and _recompute_posterior/_recompute_posterior_core through the zero-regime anchor branch. Pure in-memory Bayesian updates; prunes contextual dict to 104 entries and posterior_bias to MAX_BIAS_NUDGES. No DB/file/RPC.
- invariants: [rounding: N/A (float posterior math, ppm rounding done downstream); sign: precision-weighted means divide by new_precision which is strictly positive (ctx_precision floored at 1/200² L888, obs_precision ≥0); nudge blend frac=weight/(1+weight)∈(0,1) moves toward target, correct sign (L1039-1041); caps: contextual precision floored (L888), pruned to 104 (L927), bias list capped (L1017-1018), std widening bounded by max_std (L1175,1220); none-handling: legacy 3-tuple vs 4-tuple ctx handled (L541-544,869-878,973-978); empty observations → prior (L1113-1117); non-finite entries skipped in bias loops (L1064,1084); sql: N/A; never-call: N/A; concurrency: instance-local mutation under caller lock]
- most_suspicious: L920 — `if len(self.contextual_posteriors) > 130:` prunes to `sorted_contexts[:104]` — magic thresholds (130→104) rather than named constants, but purely a memory bound on advisory contexts keyed by usage count (index 2), so over-pruning only drops least-used contexts and cannot corrupt fee math; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#4
- chunk_id: modules/fee_controller.py#4
- summary: Tail of _recompute_posterior_core (Bayesian quadratic regression, LCB-bucketed non-concave fallback, delta-method std), _recompute_posterior_legacy, get_exploitation_fee, check_for_discovery, apply_vegas_adjustment, apply_dts_discount, and start of to_dict. Pure in-memory posterior math and serialization; no DB/file/RPC (to_dict only builds a dict).
- invariants: [rounding: get_exploitation_fee int()-truncates posterior_mean (L1414) — used as estimate not a broadcast fee, benign; sign: f_star extrapolation clamped [-0.5,1.5] (L1303), var_fstar computed as quadratic form ≥0 then sqrt(max(0,...)) (L1345); caps: noise_variance floored at 10.0 (L1289), posterior_std floored MIN_STD and capped max_std (L1345,1359-1361), dts discount precision floored MIN_PRECISION (L1546); none-handling: Sigma_n singular->legacy fallback (L1272-1274), empty obs->prior (L1373-1377); apply_dts_discount ignores gamma outside (0,1) (L1542-1543); sql: N/A; never-call: N/A; concurrency: instance-local under caller lock]
- most_suspicious: L1568 — `max(min(float(base_weight), self.DISCOUNT_WEIGHT_FLOOR), float(base_weight) * gamma)` — the min() caps the floor at the weight itself so a below-floor weight is never raised, and gamma-decayed weight is taken when it exceeds the floor; correctly implements "never decay below floor, never raise an already-sub-floor weight"; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#5
- chunk_id: modules/fee_controller.py#5
- summary: GaussianThompsonState.from_dict (validated deserialization with legacy weight-rescale), _PID_TARGET_RATIOS, PIDState.calculate_multiplier/to_dict/from_dict, and ChannelFeeState dataclass (fields, __post_init__, __setattr__ shared-field tracking, to_v2_dict). from_dict parses persisted dict; PID computes a bounded balance multiplier; to_v2_dict serializes. No RPC; DB interaction only via caller passing/consuming these dicts.
- invariants: [rounding: N/A (multiplier is a float scalar applied upstream); sign: raw_error = target - outbound_ratio (L1822) drives P-term in correct direction; sign preserved through 1.5**output; caps: multiplier clamped [0.5,2.0] (L1847), integral anti-windup clamped ±integral_clamp (L1836-1839), exploration_boost clamped [MIN,MAX] on load (L1754-1757); none-handling: from_dict validates coeffs/precision shape and positive diagonals with defaults (L1657-1691), non-finite guards on charged_fee_mean/streak/refs (L1716-1745), NaN outbound->target (L1820-1821); sql: N/A (dict only); never-call: N/A; concurrency: __setattr__ records explicit shared-field writes for the shared-state merge protocol — instance-local, no lock needed here]
- most_suspicious: L1633 — `w = w / cls.ZERO_REVENUE_WEIGHT_FACTOR` (÷0.15 ≈ ×6.67) when rescaling legacy zero-window weights — a large upscale, but immediately clamped by `w = min(1.0, w)` (L1634) and only runs on legacy payloads lacking the weight_scheme marker; cannot exceed 1.0; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#6
- chunk_id: modules/fee_controller.py#6
- summary: ChannelFeeState.from_v2_dict (deserialize with legacy-version migration), ChannelCycleState dataclass (congestion/pending-target/shared-field tracking), VegasReflexState (update/get_floor_multiplier), FeeAdjustment dataclass, FeeProfileSettings, and start of FeeController class (docstring, class constants, FEE_PROFILES). Mostly dataclass definitions + Vegas intensity state machine; no DB/file/RPC writes (from_v2_dict reads passed dicts).
- invariants: [rounding: N/A here (fee constants only, no ppm arithmetic in range); sign: Vegas decays intensity before spike check (L2192) so a fresh 1.0 isn't halved same cycle — ordering correct; caps: intensity boost capped min(1.0,...) (L2209), floor multiplier bounded 1.0–3.0 via sqrt curve (L2220-2223), spike_ratio div-by-zero guarded (L2185-2186); none-handling: from_v2_dict defaults every field, unknown algorithm_version->fresh state (L2013-2015), legacy_state optional (L2038); sql: N/A; never-call: N/A no RPC; concurrency: VegasReflexState.update mutates instance fields — single-threaded cycle per docstring; ChannelCycleState __setattr__ tracks shared fields]
- most_suspicious: L2208 — `if self.consecutive_spikes >= 2 or random.random() < boost * 0.5:` — nondeterministic probabilistic trigger for 200–400% spikes; the randomness only *accelerates* floor-raising (a defensive action) and the deterministic ≥2-consecutive path guarantees eventual trigger on a sustained spike, so a missed random draw cannot leave the node unprotected; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#9
- chunk_id: modules/fee_controller.py#9
- summary: Competitor/gossip fee helpers — deprecated market-boundary stubs (return None), _get_neighbor_fee_median, _is_cln_default_fee, _get_neighbor_fee_percentile, _get_competitive_undercut_pct, _get_channel_rebalance_cost_ppm, hive-member fee/release/advisory consumers, _get_context_with_values, _load_persisted_fee_strategy_row/_extract_fee_state_payload. Reads gossip cache + DB rows (get_last_rebalance_cost, get_fee_strategy_state); writes to self._neighbor_fee_cache (outside _state_lock by design) and _persisted_shared_fields memo. No setchannel/RPC fee writes.
- invariants: [rounding: L3479 `cost_ppm = int((cost_sats*1_000_000)/amount_sats)` truncates the break-even cost downward (<1ppm under true cost) then caps at 5000 (L3481) — mild under-estimate of a floor, negligible; L3395 nearest-rank percentile idx clamped to [0,len-1]; sign: undercut fraction bounded min(0.20,max(0.03,...)) (L3458); caps: fee filter 1≤ppm≤10000 (L3285,3377), cost_ppm cap 5000; none-handling: returns None/defaults on missing data and broad try/except (L3320-3321,3399-3400,3435-3436), guards amount_sats/cost_sats ≤0 (L3477); sql: read-only DB access via database accessors, no transaction boundary crossed here; never-call: compliant — no setchannel/RPC-fee mutation; concurrency: _neighbor_fee_cache eviction iterates a `list()` snapshot (L3258) precisely because writes happen outside _state_lock — documented and correct]
- most_suspicious: L3287 — `if self._is_cln_default_fee(ch): continue` — silently drops competitors sitting at exactly (10 ppm / 1000 msat) from the median pool; _is_cln_default_fee is conservative (missing base field ⇒ not default, L3342-3345) so it only excludes unambiguous untouched-CLN nodes, preventing the neighbor median from being dragged to the floor; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#11
- chunk_id: modules/fee_controller.py#11
- summary: _get_rebalance_cost_floor (per-channel then per-peer cost-recovery floor), _get_flow_adjusted_ceiling (zero-flow ceiling reduction), _prune_stale_states (GC of in-memory + DB fee states), wake_all_sleeping_channels, _maybe_wake_for_vegas_spike, and the head of adjust_all_fees (pre-lock config snapshot, profitability warm-up, gossip prefetch, non-blocking _state_lock acquire). Reads DB cost history; writes DB via reset_fee_strategy_state (prune) and _save_cycle_state/_save_channel_fee_state (wake); issues listchannels-class RPC prefetch (no setchannel).
- invariants: [rounding: L4060 `cost_ppm = (total_cost*1_000_000)//total_volume` floor division, L4068 `int(cost_ppm*REBALANCE_FLOOR_MARGIN)` truncates the floor downward (<1ppm) — under-estimates a break-even floor by a rounding step, negligible and margin-cushioned; sign: days_since_forward computed now−ts, thresholds SEVERE≥7 before MODERATE≥3 (L4134,4143) correct ordering; caps: ceiling reductions floored at max(1,...) (L4136,4145), P3 fix removed the success-rate division that overcharged up to 12x (L4062-4068); none-handling: get_channel_cost_history TypeError fallback for older DB (L4050-4052), last_forward None/0->base_ceiling (L4126-4129), broad except returns base_ceiling (L4155-4160); sql: prune calls reset_fee_strategy_state per closed channel inside try/except (L4211); wake mutations wrapped in `with self._state_lock` (L4253); never-call: compliant — adjust_all_fees prefetch is read-only RPC, no setchannel in range; concurrency: gossip/profitability prefetch deliberately outside _state_lock (L4374-4395), then non-blocking acquire (L4398) prevents overlapping cycles; wake_all mutates state only under _state_lock (L4253)]
- most_suspicious: L4398 — `if not self._state_lock.acquire(blocking=False):` — non-blocking acquire silently skips the whole fee cycle when another cycle holds the lock; this is the documented H-2 single-cycle guard (a skipped tick simply retries next timer interval, no fee corruption) and the pre-lock prefetch writes only to caches designed for lock-free access; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#13
- chunk_id: modules/fee_controller.py#13
- summary: Tail of the per-channel adjust loop's exception handler plus skip-reason classifier, then gossip-refresh eligibility/execution and DTS-summary reader, and the step-cap / sparse / blend-ratio / exploration-target math helpers. State writes: `_create_gossip_refresh_adjustment` calls `set_channel_fee` (real +/-1ppm nudge) then mutates+persists both cycle and ts_state and the atomic `_channel_fee_states.get` snapshot. `get_dts_summary` reads shared state under a bounded `_state_lock.acquire(timeout=...)`, caching `_last_dts_summaries`. The math helpers are pure.
- invariants: [rounding: `math.ceil` on step cap L5068 rounds the *allowed move* up (permissive cap, safe); `round`/`ceil` in exploration+blend bias toward a nonzero move; sign: blend forces +/-1 when ratio zeroes a nonzero delta (L5153-5154) preserving direction; caps: nudge clamped to `[min_fee,max_fee]` L4935, returns None if no clamp differs (min==max pinned); none-handling: `getattr(state,"last_broadcast_at",0) or 0`, `get_last_forward_time` None treated as idle; sql: read-only `get_last_forward_time`; never-call: N/A (gossip refresh routes through `set_channel_fee`, no raw in-cycle listchannels); concurrency: `get_dts_summary` bounded-acquire returns cached snapshot on contention (7caf3dd discipline), refresh writes assume cycle-held lock (DEF-051)]
- most_suspicious: L5068 — `scaled_delta = int(math.ceil(max(current_fee_ppm, 1) * ratio))` — ceil rounds the per-cycle delta cap upward, so the permitted move is slightly larger than ratio*fee; acceptable because it is an upper bound on movement (not a fee value) and guarantees at least `min_delta`, preventing a 0-ppm cap that would freeze low fees.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#14
- chunk_id: modules/fee_controller.py#14
- summary: End of `_apply_damped_fee_target` (per-cycle delta clamp), then `_apply_zero_flow_ratchet_guard` (silence-period fee holddown/downshift), `_detect_congestion` (live HTLC util vs stale snapshot TTL), and the head of `_adjust_channel_fee` through the sleep/wake hysteresis. State writes: sleep-wake branches mutate and persist both `_ts_sleep_state` and `cycle` (save fee-state + cycle-state); guard/damp/congestion helpers are pure. Sleeping-with-no-wake path returns None early.
- invariants: [rounding: `math.floor(current * ZERO_FLOW_DOWNSHIFT_RATIO)` L5266 rounds downshift cap down (more aggressive cut, safe for a downshift); sign: damp applies `+max_delta` or `-max_delta` per `requested_delta>0` L5216-5218 (direction correct); caps: `guarded = max(floor, min(target, downshift_cap))` keeps floor as hard lower bound, floor-override tag when floor forces a raise; none-handling: guard wraps int/float coercion in try/except -> passthrough `target,None`; `_detect_congestion` treats missing `updated_at` as fresh, malformed live HTLC data falls back to snapshot; sql: none (reads dict state only); never-call: N/A; concurrency: sleep-entry mirrors ts_state->cycle so a restart won't wake all sleepers, mutations under cycle `_state_lock`]
- most_suspicious: L5252 — `if rate != 0.0 or forwards != 0 or streak < self.ZERO_FLOW_GUARD_STREAK:` — exact float `!= 0.0` compare means any nonzero revenue_rate disables the ratchet; acceptable because revenue_rate is `volume*fee/1e6/hours` and a truly zero-flow window yields exactly 0.0, while any real forward both sets `forwards!=0` and produces nonzero rate, so the guard only engages on genuine silence.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#15
- chunk_id: modules/fee_controller.py#15
- summary: Revenue-rate signal computation, floor/ceiling assembly (chain/vegas/rebalance floors, flow-adjusted ceiling, inversion guard), then Priority-1 congestion block and Priority-2 exploration block of `_adjust_channel_fee`. State writes: congestion path feeds `thompson.update_posterior(congested=True)` and mutates `cycle.congestion_active/entry_fee`; exploration success path calls `database.clear_channel_probe` (DB write). Fee targets computed but not yet broadcast in this range.
- invariants: [rounding: `int(base_floor_ppm * flow_state_multiplier)` L5644 and `int(current_fee*MULT)` truncate toward zero, each re-clamped to `min_fee_ppm`/ceiling; sign: `new_direction`/`step_ppm` via abs+comparison L5824; caps: congestion capped by min(ceiling, episode_cap, per-cycle cap) then `max(...,current)` L5782 so it never drops below current; floor-inversion guard forces floor<ceiling (ceiling wins, P3 fix); none-handling: `capacity or 2_000_000`, `spendable_msat` via `parse_msat`, `outbound_ratio` guarded for capacity>0; sql: `get_volume_since`/`clear_channel_probe` (probe clear wrapped in try/except); never-call: N/A; concurrency: posterior mutation under cycle lock]
- most_suspicious: L5610 — `revenue_sats = (volume_since_sats * raw_chain_fee) / 1_000_000` — uses `raw_chain_fee` not the seeded `current_fee_ppm`; acceptable and in fact required (P7 rationale in comment): pairing revenue with the min-fee-seeded value would inject phantom revenue at min_fee and bend the posterior, so attributing to the truly-advertised fee is correct.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#16
- chunk_id: modules/fee_controller.py#16
- summary: DTS+PID core: Kalman demand factor, posterior+contextual update (skipped at 0-fee), vegas adjustment, contextual sample, PID multiplier, composed hive/temporal bias (single clamp), drain discount, neighbor-median market modes (undercut/match/premium/competition_aware), rebalance-cost nudge, supported-fee ceiling, and final `bounded_target_ppm` clamp. State writes: multiple `thompson.update_*`/`record_posterior_nudge`/`scale_variance`/`apply_vegas_adjustment` mutations plus `ts_state.last_*` fields (under cycle lock). No RPC here.
- invariants: [rounding: pervasive `int(x*mult)` truncation on target blends — each downstream re-clamped to floor/ceiling; sign: undercut/premium only move in the intended direction (undercut pulls down only when `post_pid>undercut_target`); caps: composite hint bias clamped to `[HIVE_HINT_TOTAL_BIAS_MIN,MAX]` once (F2 fix) then applied; `bounded_target=max(floor,min(ceiling,post_pid))` L6380; none-handling: `neighbor_median is None` skip, `math.isfinite` on kalman, ctx_tuple None guard; sql: none (reads cached state row L6004); never-call: N/A; concurrency: `ts_state.thompson` mutated under cycle `_state_lock`]
- most_suspicious: L6204 — `target = min(self.config.max_fee_ppm, target)` — reads live `self.config.max_fee_ppm` instead of the thread-safe `cfg` snapshot used everywhere else in this method (also L6217); acceptable because the premium/match target is unconditionally re-clamped by `bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, post_pid_target_ppm))` at L6380 where `ceiling_ppm` derives from the `cfg` snapshot, so a torn read here cannot exceed the snapshot ceiling.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#17
- chunk_id: modules/fee_controller.py#17
- summary: Pending-target blend anchor (P2), zero-flow ratchet guard call, damped delta cap, dynamic HTLC-max/base-fee policy diffing, Alpha Guard (min-change skip), Gossip 5% gate with gossip-refresh fallthrough, reason-string assembly, and idempotency guard before broadcast. State writes: skip/hysteresis branches persist `cycle`+`ts_state` with `last_update=now` (observation-cursor reset to avoid double-count) and `pending_target_ppm`; gossip-refresh path may return a real broadcast; the final broadcast `set_channel_fee` begins at end of range.
- invariants: [rounding: `min_change = max(5,(current_fee*3+99)//100)` L6549 is exact integer ceil of 3%; sign: `applied_delta` up/down/flat via sign; caps: alpha-guard suppresses moves `< min_change` unless congested/policy/zero-fee-recovery; none-handling: `int(cycle.pending_target_ppm or 0)`, `cycle.last_broadcast_fee_ppm or 0`; sql: `_save_cycle_state`/`_save_channel_fee_state` writes; never-call: N/A; concurrency: `_channel_fee_states.get(...)` atomic snapshot (P2-007), cursor-reset invariant preserved across every early-return so posterior data is never re-ingested]
- most_suspicious: L6734 — `if new_fee_ppm == raw_chain_fee and not channel_policy_change:` — idempotency compares the target to `raw_chain_fee` (true on-chain value) rather than the min-fee-seeded `current_fee_ppm`; acceptable and correct because skipping the RPC must be decided against what is physically advertised, and the branch still resets the observation cursor so a seeded-vs-actual mismatch does not double-count.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#19
- chunk_id: modules/fee_controller.py#19
- summary: Body of `set_channel_fee`: extracts effective post-clamp values from the `data_service.set_channel` RPC result, marks `success=True` once the fee is live, records the change to DB (failure demoted to warning), and conditionally re-syncs cycle+ts_state under `_state_lock` for manual/policy/exploration/refresh/open reasons. Then RpcError/Exception handlers demote post-broadcast errors to warnings. Followed by `_select_best_fee_prior` (fleet>network>None, `allow_rpc` gate) and `_maybe_reseed_skewed_prior` (one-time prior repair, persists) and start of `set_initial_fee`.
- invariants: [rounding: N/A (integer ppm passthrough from RPC readback); sign: N/A; caps: effective fee read back from RPC (`applied_fee_ppm`) reflects CLN's own clamp; none-handling: `applied_*` None checks before overwrite, `_get_channel_fee_state` fallbacks; sql: `record_fee_change` wrapped in try/except -> warning, not failure; state-sync mutations in try/except; never-call: `_select_best_fee_prior(allow_rpc=False)` deliberately skips the uncached listchannels gossip fallback for in-cycle callers (AGENTS in-cycle-RPC discipline honored); concurrency: STATE_SYNC block explicitly `with self._state_lock` L7269, reseed persists under caller lock]
- most_suspicious: L7226 — `result["success"] = True` — success is asserted immediately after the RPC read-back and before `record_fee_change` bookkeeping runs; acceptable because the fee is already live on-chain (documented rationale L7220-7225): reporting a bookkeeping failure as success=False would leave `last_broadcast_fee_ppm` stale and make the optimizer fight its own applied change every cycle.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#20
- chunk_id: modules/fee_controller.py#20
- summary: Tail of `set_initial_fee` (seed persistent Thompson prior under `_state_lock`, sample initial fee, broadcast via `set_channel_fee`), `_calculate_floor` (replacement-cost base + stall multiplier + congestion risk premium), `_get_dynamic_chain_costs` (feerates RPC -> open/close sat costs, sanity-bounded), `_get_cycle_state` (cache/DB load with Issue-#32 desync resync + P2 pending sanitize), `_save_cycle_state`, and `_get_channels_info` (listpeerchannels -> per-channel dict, live HTLC slot usage).
- invariants: [rounding: `int(base_floor)`/`int(risk_premium_ppm)` truncate then `max(1,...)` floor; sat costs `int(sat_per_vbyte*vbytes)`; sign: N/A; caps: chain costs clamped `max(500,min(50000,...))`/`max(300,min(50000,...))`; pending target `max(0,min(...,ABS_MAX_FEE_PPM))`; desync resync when `abs(actual-tracked) > max(100, tracked*0.5)`; none-handling: `dynamic_costs` None -> static defaults, `total_msat` fallback to spendable+receivable, `_safe_entry_fee` try/except; sql: `get_peer_latency_stats` memoized per cycle, `_load_persisted_fee_strategy_row`; never-call: `set_channel` only via `set_channel_fee`; concurrency: initial-prior seed uses `with self._state_lock` + `_get_channel_fee_state_locked` L7602]
- most_suspicious: L7721 — `floor_ppm = int(floor_ppm * 1.2)` — the 20% stall markup is applied to the running `floor_ppm` (still just the replacement-cost base) BEFORE the congestion risk-premium `max()` at L7747, diverging from the docstring formula `max(base_floor, risk_premium) * stall_multiplier`: when risk_premium dominates, the stall multiplier is silently dropped. NOT presented as fully clean — this is a real docs/impl divergence escalated to the Phase-8 defect report; acceptable to ship in-place only because the result is always a valid lower-bound floor >= each component and the effect is under-charge (never over-charge), Low severity.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/fee_controller.py#21
- chunk_id: modules/fee_controller.py#21
- summary: End of file: `is_fee_relevant_failure` (static, WIRE_FEE_INSUFFICIENT-only gate; no failcode/reason -> False) and `record_failed_forward` (records a fee-relevant failed forward as a weak negative posterior nudge under `_state_lock`). State write: `state.record_posterior_nudge(implied_fee, base_weight)` on the outgoing channel's Thompson state; whole mutation inside `with self._state_lock`.
- invariants: [rounding: `implied_fee = int(current_fee_ppm*0.8)` truncates down (weaker fee evidence, safe); sign: negative signal implemented as a nudge toward a lower implied fee; caps: `amount_boost = min(3.0, 1.0 + log10(max(1,amount_sats))/3.0)` bounded to [1,3]; base_weight 0.1xboost; none-handling: early return on falsy channel_id or `current_fee_ppm<=0`; `is_fee_relevant_failure` drops payloads with no usable failcode/failreason; `fee_state`/GaussianThompsonState type guards; sql: none; never-call: N/A; concurrency: forward_event-thread mutation guarded by `_state_lock` vs fee loop (correct)]
- most_suspicious: L8069 — `amount_boost = min(3.0, 1.0 + math.log10(max(1, amount_sats)) / 3.0)` — `max(1, amount_sats)` guards `log10(0)`/negative; acceptable and correct: at 1M sats log10=6->boost=3.0 (capped), sub-1-sat amounts floor to boost 1.0, so the weight stays in the documented [0.1, 0.3] band.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalancer.py#1
- chunk_id: modules/rebalancer.py#1
- summary: Module header, RebalanceReasonCode enum, RebalanceCandidate dataclass + to_dict (pure serialization, rounds floats for display only), JobManager stripped stub, and EVRebalancer.__init__ wiring. No money writes here. JobManager retains only source-failure tracking under a lock; all job/slot methods are no-op stubs. __init__ sets up `_pending_lock`, `_cache_lock` (guards `_fee_cache`/`_peer_inbound_fees`), and injects engine references.
- invariants: [rounding: to_dict round() is display-only, no state; sign: N/A pure; caps: none applied here; none-handling: `from_channel` guards empty `source_candidates`; sql: N/A; never-call: no pay/sendpay; concurrency: `_source_failures_lock`, `_pending_lock`, `_cache_lock` all declared; hive_router setter propagates to job_manager (L353-356) — single-writer at init]
- most_suspicious: L217 — `        return 999` — `slots_available` stub always reports slots free, so the `available_slots <= 0` gate in find_rebalance_candidates never fires; acceptable because concurrency is now enforced by the engine's single-flight cycle, not the removed async job queue (documented at L279-284).
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalancer.py#2
- chunk_id: modules/rebalancer.py#2
- summary: Decision-summary setters, `_derive_hold_reason`, Boltz coordination getter, and the hive liquidity-state payload builders/reporters plus coordination-segment normalization helpers. Pure/derive-only except `_report_hive_liquidity_state` which does a datastore push (not money). Reads snapshot channel ratios to bucket depleted/source. No sendpay, no budget reservation.
- invariants: [rounding: `capacity_msat = capacity_sats * 1000` exact scale-up (L487,527); sign: band compares `< band_low`/`> band_high` symmetric (L596-599); caps: hive bias clamp deferred to #1; none-handling: `str(... or "").strip()` guards throughout, skips empty peer_ids (L482-483,510-515); sql: N/A datastore only; never-call: none; concurrency: cycle-local, no shared state]
- most_suspicious: L487 — `                    "capacity_msat": capacity_sats * 1000,` — sats->msat scale-up feeding cl-hive's coordinator; exact x1000 with no rounding loss, and the value is advisory liquidity-state telemetry, not a payment amount.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalancer.py#3
- chunk_id: modules/rebalancer.py#3
- summary: Coordination execution-context assembly, `_report_coordination_intent`/`_report_coordination_outcome` (hive RPC calls, advisory only), futility/error classifiers, `_apply_fee_escalation`, success-signal normalizer, and budget-provider helpers `_get_external_liquidity_costs`/`_get_global_budget_limit`. The RPC calls report intent/outcome to cl-hive; they do NOT move funds. Budget helpers clamp to `max(0, int(...))` and fall back to `daily_budget_sats`.
- invariants: [rounding: int() truncation on provider values, floored non-negative (L1172-1173,1186-1193); sign: `max(0, ...)` guards negative budgets; caps: fee escalation capped at `ev_max_fee_ppm` (L1138-1140); none-handling: `or 0`/`isinstance` guards on every provider field, try/except returns safe zeros (L1175-1177); sql: N/A; never-call: hive-report-* RPCs are advisory, compliant; concurrency: no shared mutable state]
- most_suspicious: L1140 — `        return min(int(last_attempted_ppm * 1.5), ev_max_fee_ppm)` — fee-budget escalation after failures; the 1.5x bump is hard-capped at the EV-derived max so escalation can never exceed the profitability envelope, and int() floors (conservative).
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalancer.py#4
- chunk_id: modules/rebalancer.py#4
- summary: `_get_our_node_id` (caches success only, F10), `_get_channel_age_days` (SCID block-height parse), `find_rebalance_candidates` (delegates entirely to engine.run_cycle/find_candidates; clears `_fee_cache` under `_cache_lock` in try/finally; cleans stale reservations first), turnover/contribution estimators, and `_compute_hot_channel_protection`. The find path performs no direct pay — it calls `engine.run_cycle()` and sets decision summaries. Hot-channel protection reads DB override peers and closed-channel profit inheritance; computes budgets/multipliers.
- invariants: [rounding: `channel_profit_budget_sats` uses `math.ceil` then `max(1,...)` to avoid truncation-to-0 (L1540, I-12 fix); sign: scores clamped `max(0,min(1,...))`; caps: chunk_multiplier `>=1.0`, cooldown floored at `min_cd` (L1542-1547); none-handling: `cleanup_stale_reservations` before cycle prevents budget leak (L1301-1306); `or 0`/try-except around DB inheritance (L1485-1506); sql: `cleanup_stale_reservations` releases orphaned reservations, `list_hot_channel_protection_override_peers` read-only; never-call: no pay; concurrency: `_fee_cache={}` reset guarded by `_cache_lock` in both entry (L1290) and finally (L1423)]
- most_suspicious: L1540 — `        channel_profit_budget_sats = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))` — ceil-up of a spend budget is the anti-conservative direction, but it is a per-channel *profit-reinvestment* cap (max 1 sat floor to avoid zeroing small contributors), still bounded downstream by the daily budget reservation; acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalancer.py#8
- chunk_id: modules/rebalancer.py#8
- summary: Tail of `_check_capital_controls`: sums confirmed on-chain + channel-spendable as wallet reserve (msat->sat floor), enforces `min_wallet_reserve`; on `RpcError` skips reserve check but still enforces budget. Then DB-only budget gate: rebalance fees + external spend vs daily `effective_budget` (`>=` blocks), plus weekly gate. Sets `_capital_control_blocker`. Returns bool; blanket except -> False (fail-closed). No pay here — this is the coarse pre-cycle gate; atomic reservation lives in engine.
- invariants: [rounding: `base_to_sats_floor` on msat (L2804,2811) — floors reserve, conservative; sign: `>=` on both budget gates (L2846,2871); caps: daily gate on actual spend only (reservations enforced per-execution in engine, documented L2843-2844); wallet reserve `total_reserve < min_wallet_reserve` blocks (L2816); none-handling: `int(... or 0)` on external costs, RpcError caught (L2823); sql: read-only aggregates `get_total_rebalance_fees`; never-call: none; concurrency: no shared state, snapshot cfg passed in]
- most_suspicious: L2846 — `            if total_actual_spent >= effective_budget:` — the gate uses actual spend (fees_spent_24h + ext_spent) not including pending reservations; intentional (comment L2843-2844: each subsystem enforces its own reservation cap atomically), so this coarse gate cannot double-block and the authoritative check is the BEGIN IMMEDIATE reservation in `_reserve_execution_budget`.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_engine_v2.py#3
- chunk_id: modules/rebalance_engine_v2.py#3
- summary: Debug serializers (metabolic/immune influence aggregation, pure), `_get_our_id` (cache success only), `_get_post_rebalance_anchor`, `_effective_dest_cooldown_secs` (fill-fraction cooldown scaling, F7), and `_build_snapshot` — normalizes live listpeerchannels into channel dicts (msat->sat floor), computes historical fee ppm, cooldown + drift-override using a single shared anchor row. Read-only: builds StateSnapshot, no pay, no reservation.
- invariants: [rounding: `capacity_sats = capacity_msat // 1000` and `local_sats = local_msat // 1000` floor (L1038,1045) — understates local, conservative for depletion; fill_fraction int-floored (L969); sign: `anchor_ratio - current_ratio >= drift_threshold` (L1168), band compares consistent; caps: `historical_fee_cap_ppm` bounds historical rates (L1108-1117); none-handling: `.rstrip("msat")` on str msat, `or 0` guards, try/except around profitability + cooldown + anchor (L1118,1147); sql: batched `get_last_rebalance_times` preferred, point-query fallback, all read-only; never-call: none; concurrency: cycle-local build, no shared writes]
- most_suspicious: L1038 — `            capacity_sats = capacity_msat // 1000` — floor-division drops sub-sat remainder on capacity; correct direction (never overstates capacity/local ratio), and total_msat is always a whole-sat multiple in practice, so no material loss.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_engine_v2.py#4
- chunk_id: modules/rebalance_engine_v2.py#4
- summary: `_get_hive_rebalance_bias` (clamp 0.85-1.15), `_hive_equalization_overlay` (builds zero-budget PairCandidates from hive-high->hive-low, amount = min(excess, need, max_chunk), cooldown-gated, dedupes sources/dests), and the start of `find_candidates` (clears memo, captures cycle router, builds snapshot, runs planner + coordination overlay + merge + equalization fallback + lease suppression + in-flight-dest guard + drain-demand netting). No pay yet — selection/planning only. `pair_budget_sats=0` on equalization pairs preserves free-route-only invariant.
- invariants: [rounding: `int((ratio - pct) * capacity)` floors excess/need (L1284,1293) — conservative amount; sign: `low_pct - dest.local_ratio` / `source.local_ratio - high_pct` correct polarity; caps: `amount = min(source_excess, dest_need, max_chunk_sats)` triple-bound (L1295), pair_budget 0; none-handling: `getattr(...,default) or default` throughout, DB cooldown try/except (L1260-1268); sql: `get_last_rebalance_time` read-only; never-call: none; concurrency: `_cycle_router = self._active_router()` captured once so a mid-cycle config flip cannot split routers (L1372), `inflight_dests` snapshot prevents re-selecting an in-flight dest (P4-008, L1469-1489)]
- most_suspicious: L1295 — `                amount_sats = min(source_excess, dest_need, max_chunk_sats)` — the sole amount cap for equalization moves; bounded by both endpoints' band gaps and the global chunk cap, and paired with `pair_budget_sats=0` so only free routes execute — no budget bypass.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_engine_v2.py#5
- chunk_id: modules/rebalance_engine_v2.py#5
- summary: Tail of `find_candidates` route-pricing loop (persisted pair-cooldown skip, in-memory futility skip pre-pricing, `_route_pair`, probability-adjusted budget then per-attempt fee ceiling stored on `effective_budget_sats`, do-nothing hold-margin gate, over-budget/no-route skips) then caches CycleResult. Followed by `_route_decision_for_pair` and metabolic/immune score-bias appliers (clamped 0.85-1.15). Pricing only — no pay; sets the ceiling execution will honor.
- invariants: [rounding: `int(effective_budget)` stored (L1634); sign: acceptance `route_cost <= effective_budget` (L1652), hold-margin `final_score < hold_margin` rejects (L1671-1675); caps: `effective_budget` = per-attempt ceiling bounding pair budget (F1, L1628); zero-cost routes bypass hold gate (`route_cost > 0` guard, L1673) preserving zero-budget equalization; none-handling: `getattr(...,0) or 0`, `final_score_present` guard before applying margin (L1667,1672); sql: `_get_persisted_pair_cooldown` read-only; never-call: unpriced routes rejected as no_route (L1752-1774), never submitted; concurrency: `router.begin_cycle`/`end_cycle` bracket in try/finally (L1542-1546,1776-1781), cycle-local `priced` list]
- most_suspicious: L1652 — `                        if route_result.route_cost_sats <= effective_budget:` — the money-gate accepting a priced route; `<=` (not `<`) admits a route costing exactly the ceiling, which is correct since `effective_budget` already IS the maximum authorized spend and the reservation will hold that same amount.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_engine_v2.py#6
- chunk_id: modules/rebalance_engine_v2.py#6
- summary: `_hybrid_choice`/`_fail_closed_on_route_failure`/`_route_pair` (policy-driven hive/market/hybrid pricing, fail-closed for HIVE_ONLY/HYBRID without market fallback), pair-failure in-memory tracking, `_persist_pair_failure` (DB cooldown by failure kind), `_probability_adjusted_budget`, `_per_attempt_fee_ceiling` (F1 ppm cap), `_pair_max_fee_sats`, external/global budget helpers, and `_reserve_execution_budget` — the atomic money gate: computes max_fee, calls `database.reserve_budget(...)` with daily+weekly limits inside its BEGIN IMMEDIATE. Returns (reserved, error_result).
- invariants: [rounding: `ppm_cap_sats = (amount*ppm + 999_999)//1_000_000` ceiling for the cap (L2243), `int(pair_budget*(1+bonus))` floors bonus (L2216); sign: `max(0,...)` on max_fee/budget throughout; caps: per-attempt fee = min(budget, ppm_cap) (L2244); zero-budget path admits only free routes when flag set, else blocks (L2319-2329); none-handling: `getattr(...,None)` callable checks, reserve_budget-unavailable -> error result (L2330-2336); sql: `reserve_budget` passes FULL unified budget + since_ts + weekly (L2356-2364) so the in-txn cross-category SUM counts each category once (P4-016) — no pre-subtract double-count; never-call: no direct pay; concurrency: reservation atomicity delegated to DB BEGIN IMMEDIATE, no TOCTOU in-process]
- most_suspicious: L2243 — `        ppm_cap_sats = (amount * ppm + 999_999) // 1_000_000` — ceiling-rounds the per-attempt fee CAP upward (slightly more permissive); acceptable because it only widens route *acceptance*, while actual spend is still bounded by the reserved `max_fee_sats` and the settled fee recorded on completion.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/hive_router.py#1
- chunk_id: modules/hive_router.py#1
- summary: HiveRouter manages transient askrene layers (hive-fleet zero-fee, revenue-local profitability biases). refresh_layer detects cl-hive-managed layers else self-creates; _create_standalone_layer writes zero-fee channel overrides + node biases via data_service/RPC. discover_route calls getroutes, computes fee from first_hop-dest amounts, caches per-peer 60s. Fleet balance helpers read hive_hints gossip blobs. State writes: askrene layer create/update/bias/age RPCs; in-memory caches (_route_cache, _fleet_balances, _member_ids).
- invariants: [rounding: L360 fee_ppm floor-divides `total_fee_msat*1_000_000 // dest_amount` — floor rounds fee down, understates ppm slightly, acceptable for a discovery estimate; L294 `amount_msat // 100` 1% cap floor — OK; sign: L359 `max(0, first_hop-dest)` clamps negative fee to 0, correct; L459 `max(0, available-healthy_floor)` excess clamp OK; caps: L455 `capacity//4` 25% cap, L458 `int(capacity*0.4)` floor, L513 `min(remaining,max_amt,max_chunk_sats)` triple-min then L514 <10k skip — sound; none-handling: L345-358 explicit None guard on first_hop/dest amounts with warn+fallback; L449 `.get(...,0)` defaults; L347 dest_amount>0 guard before divide L360; sql: N/A no SQL; never-call: askrene-create/update/bias/age/reserve only — no setchannel/fundchannel/close/sendpay here, compliant; concurrency: _route_cache/_fleet_balances mutated without lock but HiveRouter is single-cycle-owned per design, shared datastore reads fail-open]
- most_suspicious: L360 — `fee_ppm = (total_fee_msat * 1_000_000) // dest_amount if dest_amount > 0 else 0` — floor division understates fee_ppm by <1ppm and the `dest_amount>0` guard prevents ZeroDivisionError even when the L357-358 fallback sets dest_amount to amount_msat; acceptable because amount_sats>0 in all call paths and underestimate is conservative for a 1%-capped discovery probe.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/hive_router.py#2
- chunk_id: modules/hive_router.py#2
- summary: reserve_path/unreserve_path normalize a path (via _normalize_path) and call askrene-reserve/askrene-unreserve through data_service or raw RPC. Both fail-open (return False on empty path or any exception). Pure wrappers over external askrene state writes; no arithmetic.
- invariants: [rounding: N/A no msat math here; sign: N/A; caps: N/A; none-handling: L707/L723 empty-normalized guard returns False before RPC; _normalize_path (L582-594) drops hops with missing scid_dir or amount<=0; sql: N/A; never-call: only askrene-reserve/unreserve, compliant; concurrency: askrene reserve/unreserve mutate shared askrene state — reserve/unreserve must be paired by caller; exceptions swallowed so a failed unreserve leaks a reservation until askrene-age cutoff, acceptable given age sweeps]
- most_suspicious: L727 — `self.data_service.askrene_unreserve(path=normalized)` — a swallowed exception here (L730-731 returns False) leaks an askrene reservation, but this is bounded by the periodic `_age_layer` cutoff and reservations are advisory, so acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/hive_hints.py#4
- chunk_id: modules/hive_hints.py#4
- summary: HiveHintAdapter diagnostic/status surface. _cache_debug_view/_status_from_cache build debug dicts under self._lock from cached snapshot. refresh_status_for_debug re-reads datastore raw + live hive-export, validates/normalizes, stores fresh snapshot or falls back to stale datastore; classifies hive-unavailable. get_status optionally triggers refresh then merges diagnostics. State writes: in-memory snapshot store/clear only (no DB writes); reads datastore + hive-export RPC.
- invariants: [rounding: L2121/L2275-2276 `int(time.time()) - int(...)` age math, integer seconds, no msat — N/A rounding direction; sign: age can be negative if generated_at is future-dated (clock skew) — not clamped, but only used for display/freshness which uses TTL comparison elsewhere; off-by-one N/A; caps: N/A budget arithmetic not here; none-handling: extensive — L2131 `str(probe.get(...) or "")`, L2275 `int(self._snapshot.get("generated_at",0))`, L2288 `... or 0`, L2319-2326 isinstance-dict guards on diagnostics sub-blobs; L2258 snapshot-None branch; sql: N/A; never-call: read-only diagnostics, validation via _validate_* callbacks, compliant; concurrency: all snapshot reads/mutations under `with self._lock` (L2105,2257,2355); get_status L2355 takes lock only for freshness check then refresh_status_for_debug re-locks internally — non-atomic across the two but fail-safe]
- most_suspicious: L2275 — `age = int(time.time()) - int(self._snapshot.get("generated_at", 0))` — if a malicious/corrupt datastore blob carried a future generated_at, age goes negative; acceptable because freshness is gated by _snapshot_is_fresh/TTL logic (not by this display age) and generated_at passed _validate_and_normalize_snapshot before storage.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/flow_analysis.py#1
- chunk_id: modules/flow_analysis.py#1
- summary: Module constants + estimate_depletion_hours + KalmanFlowState/Filter + TemporalProfile + ChannelState enum. estimate_depletion_hours computes net drain sats/day and hours to depletion. KalmanFlowFilter predict/update do bounded state estimation with NaN guards, positive-definite covariance, physical clamps. update_temporal_profile EMA-blends hourly histograms, advances observation_days once per epoch day. Mostly pure/in-memory; no DB writes in this range.
- invariants: [rounding: float math throughout, no msat/sat integer rounding — N/A direction; L435 `now // 86400` epoch-day floor is intentional; sign: L202 `max(0.0, ratio)*capacity` drops negative (filling) drain, L204 clamps net>=0, correct; L513 velocity*dt could push ratio out of range but L517 clamps to [-1,1]; caps: L517-518,606-607 physical clamps; L522/L525 process-noise clamps; L565-566 measurement-noise clamp; velocity clamp uses KALMAN_MIN/MAX_VELOCITY (~±0.021/hr); none-handling: L189-195 try/except TypeError/ValueError->None, L197 isfinite-all guard, L199 capacity<=0/local<0 guard, L206 min-drain->None, from_dict L257-259 `is not None` (I-7 fix) preserves stored 0.0; L372-374 histogram sliced `[:24]` then padded; sql: N/A; never-call: pure computation, compliant; concurrency: filter state is per-channel object, caller holds _kalman_lock per module docstring]
- most_suspicious: L413 — `new_out = float(histogram[h].get("out_sats", 0))` — indexes histogram[h] for h in range(24) without checking len(histogram)>=24; if a caller passes a short histogram list this raises IndexError. Acceptable because the sole documented producer (_hourly_forward_histogram_sql) always yields 24 buckets and the KeyError-vs-IndexError distinction is contained, but it is the least-defended external-shape assumption in range.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/flow_analysis.py#4
- chunk_id: modules/flow_analysis.py#4
- summary: Tail of EMA aggregation (divides ema_in/ema_out by total_weight) plus _get_channels/_get_channel. _get_channels reads listpeerchannels via data_service, filters CHANNELD_NORMAL, annotates HTLC limits and active_htlcs count, returns list. _get_channel normalizes scid and linear-scans for a match. Reads only; no writes.
- invariants: [rounding: N/A no msat rounding; sign: N/A; caps: L2138 max_accepted_htlcs default 483 (BOLT#2) OK; none-handling: L2103 `bucket.get('last_ts',0) or 0`, L2133-2142 `.get(...,default)` guards, L2142 `len(htlcs) if htlcs else 0`; sql: N/A here; never-call: read-only listpeerchannels, compliant; concurrency: reads shared data_service cache, no mutation]
- most_suspicious: L2109 — `ema_in /= total_weight` — divides without a total_weight>0 guard in this excerpt; a ZeroDivisionError would occur if the weighted loop produced total_weight==0 (all-zero/empty buckets). Acceptable only if the caller guarantees ≥1 positive-weight bucket before reaching here; the surrounding function (above L2101) gates on forward_count/empty-bucket — flagged as the load-bearing precondition not visible in range.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/hive_runtime.py#1
- chunk_id: modules/hive_runtime.py#1
- summary: refresh_hive_runtime orchestrates best-effort refresh of shared hive state: hive_hints.poll(), hive_router.refresh_layer(), refresh_fleet_balances(), clear_route_cache(). _safe_log tolerates log callables with/without level kwarg. Every step wrapped in try/except and fails open; returns early if router refresh returns falsy. Pure orchestration; state writes delegated to callees.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: L28 `hive_hints is not None` guard, L34 `hive_router is None` early return, L8-9 log-None guard; sql: N/A; never-call: only calls documented refresh methods, compliant; concurrency: sequential best-effort, no shared-state mutation of its own; relies on callee thread-safety]
- most_suspicious: L38 — `refreshed = bool(hive_router.refresh_layer())` — the only non-swallowed logical gate; if refresh_layer returns False the fleet-balance/cache-clear steps are skipped, so a transient false keeps a stale route cache for a cycle. Acceptable: cache is TTL-bounded (60s) and design is explicitly fail-open.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/data_service.py#1
- chunk_id: modules/data_service.py#1
- summary: DataService tiered RPC cache. _get_cached deletes-on-expiry; _set_cached evicts oldest by ts past _CACHE_MAX_ENTRIES (256). Forever tier (getinfo/listconfigs) under separate lock. Medium/long getters wrap listpeerchannels/funds/peers/channels/forwards/nodes/feerates. Never-cached transactional ops (setchannel/fund/close/sendpay/askrene mutations) invalidate relevant keys. datastore_push validates dict/no-error/size<60KB then writes create-or-replace.
- invariants: [rounding: N/A no msat math; sign: N/A; caps: L40 _CACHE_MAX_ENTRIES=256 bounds cache, L83-89 evicts `len-max` oldest; L431/L450 _DATASTORE_MAX_BYTES=60000 under 65KB CLN limit, checks encoded utf-8 byte length correctly; none-handling: L442 isinstance dict, L444 "error" in payload reject, L446 timestamp default, L67 entry None guard, L230 `.get("blockheight",0)`; sql: N/A (datastore/RPC, no raw SQL); never-call compliance: transactional methods correctly bypass cache and invalidate — set_channel/fund/close invalidate listpeerchannels/listfunds; get_askrene_layers never cached (L248-255 documented shared mutable state) — compliant; concurrency: cache ops under self._lock, forever under _forever_lock; get_block_height L223-232 does a fresh getinfo RPC rather than reusing forever getinfo — minor redundancy not a bug]
- most_suspicious: L152 — `return self._plugin.rpc.listpeerchannels(peer_id)` — the per-peer branch bypasses the cache entirely (documented "per-peer uncached"), so callers passing peer_id get an uncached live RPC every call while the broadcast path is 30s-cached; acceptable because per-peer keys would balloon the bounded cache and per-peer freshness is desirable, but it is an asymmetric-cost path worth noting.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_execution.py#1
- chunk_id: modules/rebalance_execution.py#1
- summary: Pure module. Defines `ExecutionResult` dataclass (fee/amount/hops/parts/error/excluded_channels/failure_data/payment_pending) and `stable_failure_reason`, a pure string classifier mapping executor-local error strings to stable coordination reason codes. No DB/RPC/shared state; mutable dataclass defaults use `field(default_factory=...)` correctly.
- invariants: [rounding: N/A (no arithmetic); sign: N/A; caps: N/A; none-handling: `str(error or "")` guards None at L30; sql: N/A; never-call: OK (no RPC); concurrency: N/A (pure, per-call locals)]
- most_suspicious: L41 — `if "temporary_channel_failure" in normalized or "fee_insufficient" in normalized:` — substring matching on lowercased error text can misclassify an unrelated message containing the token, but the token set is CLN-specific and the fallback is the same generic reason, so blast radius is nil.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_executor_v2.py#1
- chunk_id: modules/rebalance_executor_v2.py#1
- summary: Pure compatibility shim. Re-exports `ExecutionResult`/`stable_failure_reason` from `rebalance_execution` and aliases `NativeRouteExecutor` as `RebalanceExecutor`. No logic, no state.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: N/A; sql: N/A; never-call: OK; concurrency: N/A]
- most_suspicious: L11 — `from .rebalance_native_executor_v2 import NativeRouteExecutor as RebalanceExecutor` — import-time coupling to the native executor module; acceptable because it is the intended single executor and the removed external executor is documented in the module docstring.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_native_executor_v2.py#1
- chunk_id: modules/rebalance_native_executor_v2.py#1
- summary: `NativeRouteExecutor` helpers: RPC wrapper with optional timeout kwarg (TypeError fallback), node-id caching, failure classification, exclusion inference, and failure-observation recording. `_validate_route` enforces route shape, pins first/final hops to source/dest SCIDs and final hop to our node, verifies final amount == amount_sats*1000, non-increasing hop amounts, and fee budget. Writes: RPC `getinfo` (id cache); observation store `record_failure`.
- invariants: [rounding: `base_to_sats_ceil` at L329 rounds fee up (conservative over-report vs budget); sign: `fee_msat = max(0, first-final)` L324 floors at 0; caps: `max_fee_msat = max(0,int(max_fee_sats))*1000` L325, `fee_msat > max_fee_msat` reject L326 correct; none-handling: `parse_msat(hop.get(...))` and `str(x or "")` throughout; sql: observation writes wrapped in try/except (L274) — never aborts execution; never-call: only getinfo here, compliant; concurrency: `_our_id` lazily cached, benign duplicate getinfo under race, no shared mutation]
- most_suspicious: L253 — `confidence = max(self.INFERRED_CONFIDENCE_FLOOR, self.ATTRIBUTED_CONFIDENCE / len(directed))` — divides blame across inferred suspects with a 0.2 floor; acceptable because `directed` is guaranteed non-empty (early `return` at L251) so no ZeroDivisionError.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_native_executor_v2.py#2
- chunk_id: modules/rebalance_native_executor_v2.py#2
- summary: `execute()` body. Validates route, then creates a self-invoice, sendpay on the exact route, waitsendpay with timeout. On success computes actual fee from `amount_sent_msat`. On exception, distinguishes unresolved-payment (code 200 / proxy timeout / pending) — holds budget, does NOT delete invoice/payment — from terminal failure, which records exclusions/observations and cleans up invoice+payment via delpay/delinvoice. External writes: `invoice`, `sendpay`, `waitsendpay`, `delpay`, `delinvoice` RPCs; observation store.
- invariants: [rounding: `base_to_sats_ceil` for fee_sats (up); `fee_ppm = actual_fee_msat*1e6 // (amount*1000)` L485 floors (slight under-report, benign); sign: `actual_fee_msat = max(0, sent-amount)` L476 floors at 0; caps: planned_fee surfaced on over-budget/pending so engine holds budget; none-handling: `paid.get(...)` guarded by isinstance dict L470/L474, `actual_sent_msat or route[0] amount` fallback L478; sql: N/A (RPC-only), cleanup best-effort try/except; never-call: uses only invoice/sendpay/waitsendpay/delpay/delinvoice — compliant; concurrency: pending-detection at L494 prevents double-spend by not releasing an unresolved HTLC]
- most_suspicious: L478 — `(actual_sent_msat or parse_msat(route[0].get("amount_msat")))` — when `amount_sent_msat` is 0/absent it falls back to the planned first-hop amount; acceptable because a 0 sent value only occurs when CLN omitted the field, and the planned first-hop amount is the correct upper bound for fee derivation.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_coordination_overlay.py#1
- chunk_id: modules/rebalance_coordination_overlay.py#1
- summary: Builds coordination `PairCandidate`s from hive recommendation/campaign hints. Resolves hint endpoints to local channels (SCID exact-match or peer-id best viable, never wildcard on unknown SCID), gates on designated-executor, scores in planner units (0.30 urgency + 0.20 drain) with bounded priority multiplier, sizes amount by min(chunk, excess, need, hinted), suppresses pairs conflicting with foreign active leases, and merges overlay+planner pairs under slot caps with reserved coordination slots. Pure w.r.t. RPC; reads hive_hints getters, mutates only local PlanResult/sets.
- invariants: [rounding: `int(...)` truncation in `_channel_excess/need_sats` L69/L73 (floor, benign sizing); fee_cap `(amount*ppm+999_999)//1_000_000` L305 ceil-divides correctly; sign: excess/need `max(0,...)` clamped; caps: `min(priority, MAX_HINT_PRIORITY_SCORE)` L282 and reserved-slot arithmetic in merge L534-542 consistent; none-handling: extensive `str(x or "")`, `get_segment_score` wrapped in try/except; sql: N/A; never-call: no RPC; concurrency: operates on caller-owned snapshot/lists, `seen_pairs` local — no shared state]
- most_suspicious: L289 — `hinted_amount or min(source_excess, sink_need)` — a hinted amount of 0 (falsy) silently falls back to computed sizing; acceptable because a 0 hint carries no authorization and the min() of local excess/need is the correct conservative bound, and the subsequent `amount_sats <= 0` guard (L291) rejects anything non-viable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_planner_v2.py#1
- chunk_id: modules/rebalance_planner_v2.py#1
- summary: `RebalancePlanner.plan` classifies channels into over_local/over_remote by band, applies role eligibility, generates all cross-peer pairs, scores via additive role terms (0.30 urgency + 0.20 drain + 0.20 dest_value + cheap_return), applies bounded hive bias multiplier, selects top `max_pairs` with disjoint source/dest, emits explicit skip records for every unpaired eligible channel, and publishes residual DrainDemand. Pure — no RPC/DB, consumes StateSnapshot, returns PlanResult.
- invariants: [rounding: `int(round(...))` in `_sats_from_ratio_delta` L34 and fee_cap ceil-div `(amount*ppm+999_999)//1_000_000` L275 correct; sign: `max(0,...)` on excess/need/deltas; caps: `pair_budget = max(dest_budget, fee_cap)` L278, `_normalize_rebalance_bias` clamps [0.85,1.15] L396, `_pair_hint_multiplier` re-clamps L411; none-handling: `getattr(...,default) or 0` guards on all optional fields; sql: N/A; never-call: no RPC; concurrency: pure per-call, no shared mutable state]
- most_suspicious: L410 — `multiplier = 1.0 + (dest_bias - 1.0) - (source_bias - 1.0)` — intentionally inverts the source bias so source-preference peers rank higher as drain sources; acceptable and re-clamped to [0.85,1.15] at L411 so no unbounded amplification, matching the documented HiveHintAdapter semantics.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_route_policy.py#1
- chunk_id: modules/rebalance_route_policy.py#1
- summary: `decide_route_policy` classifies a pair into HIVE_ONLY/HYBRID/MARKET_ONLY with a priority tier. Order: strict hive-equalization between two hive members -> fresh hint/campaign metadata (honoring route_policy/allow_market_fallback/priority_score) -> endpoint-membership heuristic -> market fallback. Helpers normalize segments, clamp priority scores to [0,MAX], and gate all hint reads behind `_hints_fresh`. Pure — reads hive_hints getters, all wrapped in try/except.
- invariants: [rounding: `//1000` msat->sat floor in `_entry_amount_sats` L177 (benign, sizing); sign: N/A; caps: `_priority_score` clamps to [0,MAX_HINT_PRIORITY_SCORE] and rejects non-finite L188; none-handling: freshness gate returns [] on missing/stale/exception, `str(x or "")` throughout; sql: N/A; never-call: no RPC; concurrency: pure, per-call locals]
- most_suspicious: L234 — `allow_market_fallback = bool(hinted.get("allow_market_fallback", route_policy != "hive_only"))` — an untrusted producer can set `allow_market_fallback=True` even under a hive_only policy, permitting market escape; acceptable because market fallback is an advisory relaxation (never a safety gate) and the strict no-fallback path is reserved for the internally-derived `hive_equalization` branch at L221-227 which ignores producer flags.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_router_v2.py#1
- chunk_id: modules/rebalance_router_v2.py#1
- summary: `RebalanceRouter.price_pair` builds a sendpay-ready circular route using only official CLN RPCs (listpeerchannels/listchannels/getroute/listconfigs). Resolves final-hop policy (base+ppm+cltv), gets middle path via getroute (fromid=source_peer->dest_peer), reprices middle hop amounts bottom-up from live forwarding policies, prepends our first hop (adding source-peer fee+cltv) and appends final hop. Reads RPCs (or cached data_service); no writes. NOTE: contains a latent dead-code defect (see most_suspicious) that is escalated as a finding.
- invariants: [rounding: `math.ceil` on all fee msat->sat conversions (L198-201, L217, L299, L524) — conservative over-estimate, correct for not under-paying hops; sign: `max(0, ...)` on route/total cost; caps: N/A (pricing only, budget enforced in executor); none-handling: policy lookups return None -> early failure RouteResult; `.get(...,0) or 0` guards; sql: N/A; never-call: read-only official RPCs only, compliant; concurrency: `_invoice_final_cltv` cached once, idempotent under race]
- most_suspicious: L184 — `def _get_final_hop_fee_ppm(self, dest_peer_id: str)` decorated `@staticmethod` (L183) yet declaring `self` — invoked as an instance method it would bind `dest_peer_id` to `self` and AttributeError. This is a genuine bug but PROVABLY UNREACHABLE: grep confirms zero callers anywhere in source or tests (dead legacy shim). Not clean — escalated to the Phase-8 defect report for a follow-up finding; acceptable to ship in-place only because it can never execute.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_router_v3.py#1
- chunk_id: modules/rebalance_router_v3.py#1
- summary: `RebalanceRouterV3.price_pair` prices via askrene `getroutes` with layer biasing. Reuses v2 helpers for fee/cltv lookups. Probes live layers each call, always appends `auto.no_mpp_support`, excludes local source/dest SCIDs from the middle search plus retry excludes via a throwaway/cached exclude layer, picks cheapest route, validates middle path (non-empty, terminates at dest_peer, never loops through us), translates hops to sendpay format, reprices, wraps with our first/final hops. Cycle-scoped thread-local caches for listlayers and exclude layers. Writes: askrene-create/update/remove-layer RPCs (throwaway layers).
- invariants: [rounding: `math.ceil` on fee conversions (L524, L563), `_route_fee_msat` uses raw msat diff; sign: `max(0, first-delivered)` L588/L561; caps: `maxfee_msat=route_amount_msat` (permissive, real cap enforced downstream in executor validate_route); none-handling: `_parse_msat` handles int/str/None, `.get(...,0)` fallbacks, probability defaults 0; sql: N/A; never-call: askrene layer RPCs are the intended mechanism, compliant; concurrency: `itertools.count` (L182/L600) atomic under GIL for unique layer names, cycle caches are `threading.local` so RPC-thread pricings never clobber the background cycle]
- most_suspicious: L570 — `probability_ppm = int(cheapest.get("probability_ppm", 0))` — a missing field defaults to 0 which the engine reads as "no probability relaxation"; acceptable because 0 is the documented conservative sentinel (matches v2 behavior) and cannot loosen the budget gate.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_router_v3.py#2
- chunk_id: modules/rebalance_router_v3.py#2
- summary: Tail of `_exclude_layer` contextmanager: the outside-a-cycle (ephemeral) branch — builds a throwaway exclude layer, yields its name, and removes it in a `finally` so it is torn down even on exception. External write: askrene-remove-layer.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: layer removal is best-effort (try/except in `_remove_exclude_layer`); sql: N/A; never-call: OK; concurrency: ephemeral layer name is per-call unique via itertools.count, no shared state leak]
- most_suspicious: L704 — `self._remove_exclude_layer(layer_name)` — the `finally`-guaranteed cleanup; acceptable and correct: it ensures no live askrene layer leaks when the yielded pricing raises, matching the docstring contract.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_state_v2.py#1
- chunk_id: modules/rebalance_state_v2.py#1
- summary: Pure state builder. `build_state_snapshot` normalizes channel inputs, computes clamped local_ratio, resolves capex budget (with hive-bootstrap fallback), assigns value_class, evaluates source/dest eligibility (destination requires value+budget+not-cooldown unless drift/emergency override), computes drain/urgency scores, and returns an immutable StateSnapshot. No RPC/DB — explicitly plugin-free.
- invariants: [rounding: `base_to_sats_ceil` for budget msat->sat (L161/L165, up = generous budget, benign); `round(...,6)` on ratios/scores; sign: `max(0,...)` on capacity/local/budget, ratio clamped [0,1] L289, `_as_nonnegative_float` floors at 0; caps: `_as_rebalance_bias` clamps [0.85,1.15] L122; none-handling: `_as_int/_as_bool/_as_nonnegative_float` all default-guard None/invalid; sql: N/A; never-call: no RPC; concurrency: pure, frozen dataclasses, builds fresh tuple]
- most_suspicious: L293 — `if channel.is_hive_member and remaining_budget_sats <= 0:` grants `hive_bootstrap` budget to any hive member lacking capex; acceptable because the bootstrap amount is an explicit caller-supplied parameter (default 0 = disabled) and is `max(0,...)` clamped, so it cannot fabricate spend authority unless the operator opts in.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/rebalance_types_v2.py#1
- chunk_id: modules/rebalance_types_v2.py#1
- summary: Pure dataclass definitions: `PairCandidate` (scored pair with fee/budget/route/decision fields), `SkipRecord`, `DrainDemandEntry`, `DrainDemand`, `PlanResult`. Mutable collection fields use `field(default_factory=...)`. No logic, no state writes.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: Optional-typed fields default None/factory; sql: N/A; never-call: OK; concurrency: N/A (plain data carriers)]
- most_suspicious: L52 — `metabolic_rebalance_influence: Dict[str, Any] = field(default_factory=dict)` — a mutable default correctly guarded by default_factory (a bare `= {}` would share state across instances); this is the right pattern, so the line is safe.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/boltz_manager.py#2
- chunk_id: modules/boltz_manager.py#2
- summary: boltzcli subprocess plumbing plus first-hop/route resolution helpers. `_base_cmd`/`_run` assemble argv as a Python list (optionally prefixed `sudo -n -u <user>`, then `cli_path --datadir <datadir>` + args) and invoke `subprocess.run(cmd, ...)` with no `shell=True`; TimeoutExpired/FileNotFound wrapped as BoltzCliError. Parsing helpers (`_parse_int`, `_parse_timestamp`, `_decodepay_amount_msat`) and route-forcing exclude builders. State writes: none here except reads via data_service/rpc; `_pay_invoice_via_first_hop` submits a CLN `pay`.
- invariants: [rounding: L797 sat->msat `*1000` exact, no loss; sign: L803 rejects only invoice_msat > expected (overpay), underpay allowed — correct guard direction; caps: expected-amount ceiling on externally-produced boltzd invoice enforced L795-807; none-handling: `_parse_int`/`_parse_timestamp` default-safe, L798 None amount -> refuse; sql: N/A (no SQL); never-call: get_boltz_cost_components contract not in this range, no provider recursion here; concurrency: `_reverse_chanids_supported` cached without lock but idempotent bool memo, benign]
- most_suspicious: L797 — `expected_msat = int(expected_amount_sats) * 1000` — multiplication is exact sat->msat with no truncation, and the surrounding block refuses to pay when the invoice has no decodable amount or exceeds this bound, so the only failure mode is a safe refusal.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/boltz_manager.py#3
- chunk_id: modules/boltz_manager.py#3
- summary: swap-list normalization/dedup, fee estimation, terminal/completed classification, and `get_boltz_cost_components` which sums completed-swap fees in a 24h window and reserves pending-swap fee estimates, capping reserved at remaining budget. Pure computation over boltzcli JSON + swap journal; no DB writes. `_get_global_budget_limit`/`_get_external_liquidity_costs` call injected providers defensively.
- invariants: [rounding: fee sums are integer sats via `_parse_int`, no fractional loss; sign: reserved/spent clamped `max(0, ...)` L1102/L1122; caps: L1134-1136 reserved = min(reserved, cap_budget - boltz_spent) with tighter of boltz vs global budget L1129-1133 — cap skipped only when cap_budget==0 (no budget configured), reserved then informational; none-handling: swap ts None -> skipped/counted as unknown L1094-1096; sql: N/A; never-call: explicit contract L1075-1080 — this method must NOT call global_budget_limit_provider (mutual recursion); confirmed it uses the passed `global_budget_cap_sats` instead — compliant; concurrency: reads only]
- most_suspicious: L1135 — `max_reservable = max(0, cap_budget - boltz_spent)` — if already-spent exceeds the cap this floors to 0 (reserved fully suppressed) rather than going negative, which is the conservative/correct direction for a capital-control gate.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/boltz_manager.py#7
- chunk_id: modules/boltz_manager.py#7
- summary: tail of a backup export (`swapmnemonic get` optionally embedded with a plaintext warning) plus `backup_verify`, which fetches the current swap mnemonic via boltzcli and compares whitespace-normalized word strings, returning match bool and word counts. boltzcli invoked with fixed argv (`["swapmnemonic", "get"]`) — no user data in argv. No DB/budget writes.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: `str(swap_mnemonic or "")` guards None L2412; sql: N/A; never-call: N/A; concurrency: read-only external cmd]
- most_suspicious: L2415 — `"matches": provided == actual` — a non-constant-time string equality on a secret mnemonic; a timing side-channel is theoretically present but this is a local operator-invoked verify against a mnemonic the operator already supplied, so it leaks nothing an attacker with that access lacks — acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capex_budget.py#1
- chunk_id: modules/capex_budget.py#1
- summary: CapexBudgetEngine.compute_allocations reads profitability cache + DB spend history, computes per-channel/exploration/tactical budgets entirely in msat, enforces a global envelope with daily/weekly 30d-projected caps, scales all budgets down proportionally if over envelope, and fails closed (zeros every budget) when DB reads return None. `attribute_boltz_cost` splits a sat cost 50/50 (informational). No DB writes in this range (record_boltz_spend body continues past L400).
- invariants: [rounding: sat->msat via `*MSAT_PER_SAT` exact L197/L244/L290; msat->sat exposed only via `base_to_sats_ceil` properties (ceiling, no false-zero) L79-121; scale-down uses `int(...)` truncation L305-308 — conservative (never over-grants); sign: contribution/tactical clamped `max(0,...)` L246; caps: envelope = min(configured, daily_30d, weekly_30d) L289-300 correctly tightened; split floor-div L377-378 tactical takes remainder — no sat lost; none-handling: capex_by_channel None -> db_degraded fail-closed L167-169/L275-284; spend_summary None -> db_degraded L249-251; getattr defaults on prof fields; sql: reads only, wrapped; never-call: N/A (no RPC, doc L4); concurrency: single-cycle compute, `_last_allocations` overwritten atomically by reference]
- most_suspicious: L245 — `tactical_msat = min(reserve_deficit_msat, int(total_fleet_contribution_msat * cfg.capex_tactical_rate))` — tactical is capped by the reserve deficit so it can never exceed the actual on-chain shortfall, and the `int()` truncates downward; the follow-up `max(0, ...)` L246 forbids a negative deficit from inverting the budget.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capex_budget.py#3
- chunk_id: modules/capex_budget.py#3
- summary: `_get_spend_ledger_summary` wraps `database.get_spend_ledger_summary` returning None on any exception (CB-4 fail-closed, never empty dict). `_apply_category_spend_remaining` subtracts (spent+reserved) sats for one category, converted to msat, from a nominal msat budget, flooring at 0. Pure arithmetic; DB read only.
- invariants: [rounding: L831 `consumed_sats * MSAT_PER_SAT` exact sat->msat, no loss; sign: L832 `max(0, remaining_msat)` prevents negative budget; caps: subtracts both spent and reserved so budget cannot be double-spent; none-handling: `summary.get(...) or {}` and `.get(category,0) or 0` guard missing keys/None L826-829; None summary handled by caller (returns None -> db_degraded); sql: single wrapped read, no write; never-call: N/A; concurrency: stateless]
- most_suspicious: L831 — `remaining_msat = budget_msat - (consumed_sats * MSAT_PER_SAT)` — mixes a msat budget with sat-denominated ledger consumption, but the `* MSAT_PER_SAT` correctly lifts consumed sats into msat before subtraction, so units are consistent and the result cannot silently under-deplete.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/policy_manager.py#1
- chunk_id: modules/policy_manager.py#1
- summary: PolicyManager scaffolding — dataclasses, fee-multiplier bounds clamping, peer_id regex validation, thread-safe write-through cache with rate-limit bookkeeping, and `_load_cache`/`_row_to_policy` DB->object conversion with per-row isolation. Cache guarded by `_cache_lock`; callbacks by `_callback_lock`. DB reads only in this range.
- invariants: [rounding: N/A; sign: rate-limit window `now-60` correct; caps: fee multipliers clamped to [0.1,5.0] and swapped if inverted L118-129; MAX_POLICY_CHANGES_PER_MINUTE gate L285; none-handling: `_validate_peer_id` handles None/non-str L310; corrupt row skipped per-row L356-371, tags JSON fallback L386-392; sql: reads via database methods, no raw SQL here; never-call: N/A; concurrency: L343-379 reads DB outside lock then re-checks `_cache_valid` under lock before publishing — correct double-check; `_check_rate_limit` mutates `_change_timestamps` under `_cache_lock`]
- most_suspicious: L151 — `PEER_ID_PATTERN = re.compile(r'\A[0-9a-fA-F]{66}\Z')` — deliberately uses `\A...\Z` rather than `^...$` because Python `$` matches before a trailing newline (the PM-I1 defect that let a 67-char `...\n` id persist); this is the fixed, correct anchoring.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/policy_manager.py#2
- chunk_id: modules/policy_manager.py#2
- summary: `_row_to_policy` completion, `get_policy` (cache hit with expiry eviction + out-of-lock DB cleanup), `set_policy` (full validation -> rate-limit -> `upsert_policy` DB write -> record rate-limit -> write-through cache -> callback), plus delete/tag helpers. Writes go through `database.upsert_policy`/`delete_policy` with parameterized args; `json.dumps(new_tags)` serialized as a bound value.
- invariants: [rounding: N/A; sign: fee_ppm non-negative enforced L621; caps: fee_ppm ≤100000 L623, multiplier bounds re-validated + cross-checked L640-659, expiry ≤30d L667-669; none-handling: strategy=static requires target L625-628, tags None->existing L631; sql: L682-686 upsert via DB layer parameterized (peer_id/tags passed as values, not interpolated) — injection-safe; never-call: N/A; concurrency: rate-limit recorded only AFTER successful DB write L688-689 (B2 fix), avoiding counter pollution; expiry DB delete done outside `_cache_lock` L475-477 avoiding I/O under lock]
- most_suspicious: L676 — `if not self._check_rate_limit(peer_id):` — placed after all validation but before the DB write, so a rejected/invalid request never consumes a rate-limit slot; the matching `_record_rate_limit_change` fires only post-write, keeping the counter consistent with committed changes.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/policy_manager.py#3
- chunk_id: modules/policy_manager.py#3
- summary: tag removal, `get_policy_suggestions` (bleeder/zombie/high-velocity-source heuristics reading profitability + DB success-rate trends, read-only — emits suggestion dicts, never mutates policy), and `apply_corridor_policies` which auto-sets/deletes only automation-owned (`auto_corridor`) policies, explicitly refusing to overwrite operator-owned or `manual`-tagged policies. DB writes only via `set_policy`/`delete_policy` for automation-created entries.
- invariants: [rounding: `flow_ratio` formatting only, no money math; sign: `abs(net_pnl)` for display, `net_pnl < 0` loss test L993/L1008; caps: N/A; none-handling: peer_id/channel_id empty-guarded L959/L1041, trend query try/except L989; sql: reads via database methods; never-call: hive corridor_role is external input and is explicitly barred from overriding local policy L1139-1141; concurrency: operates through set_policy/delete_policy which lock internally]
- most_suspicious: L1133 — `has_stored_policy = int(current.updated_at or 0) > 0` — this is the guard that distinguishes an operator-stored policy (updated_at>0) from the synthetic default (updated_at=0) so automation won't clobber real operator config; relying on updated_at is sound because `get_policy` returns defaults with updated_at=0 and every real write sets `now`.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/config.py#2
- chunk_id: modules/config.py#2
- summary: `__post_init__` clamp tail, `_apply_override` (typed env/DB override with finite-check, range, and enum validation, failures appended to warnings not raised), `update_runtime` (transactional Validate->Write->Read-Back->Update-memory under `_lock` with cross-field consistency and Ghost-Config rollback), the frozen ConfigSnapshot dataclass + `from_config` mapper, and static chain-cost/liquidity-bucket helpers. DB writes via `set_config_override`/`delete_config_override`.
- invariants: [rounding: floor_ppm `int(...)` with `max(1,...)` floor L1127; sign: NaN/Inf rejected for floats L728/L779; caps: range check L734-738/L787-790, cross-field min/max ordering enforced inside lock L808-835; none-handling: unknown/private/immutable keys rejected L763-767, override conversion errors caught L748; sql: L840-850 write then read-back verify, rollback via delete_config_override on mismatch — Ghost-Config defense, all through parameterized DB layer; never-call: N/A; concurrency: L804 all cross-field checks + write + read-back + setattr under `self._lock` (M-R5-1 TOCTOU fix); snapshot taken under `config._lock` L1068]
- most_suspicious: L844 — `if read_back != value:` — compares the DB read-back against the raw string `value` (not the typed value), which is correct because `set_config_override` stores the raw string; a mismatch triggers rollback so a half-applied override can never survive into memory or restart.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capacity_planner.py#1
- chunk_id: modules/capacity_planner.py#1
- summary: Module constants (forecast/normalization anchors, backoff caps) and `CapacityPlanner` core: status/report accessors, per-cycle cache init, portfolio-balance gate, and the large `execute_cycle` orchestration. `execute_cycle` fetches profitability/flow, identifies winners/losers, prices closes, defibrillates/closes/opens under budget+portfolio gates, and updates the candidate pool. State writes are indirect (via `_execute_open/close/defibrillation`, capex engine, DB planner-action records); this chunk itself mutates only in-memory coordination fields (`_last_*`).
- invariants: [rounding: `base_to_sats_floor(parse_msat(...))` for confirmed funds/capacity (L502-505,L563) — floor is correct for spendable; sign: `available_sats = max(0, confirmed - min_reserve)` (L507) and `max(0, available_sats - channel_size)` (L660) clamp non-negative; caps: exploration budget decremented and re-checked per open (L601-605,L662-664), max_opens/close_limit honored (L595,L429); none-handling: `.get()` defaults throughout, `max_closes` type-guarded (L188), config via `getattr` defaults; sql: none direct here — DB via injected services; never-call: production executor path, close/open gated by dry_run+execute_closes downstream (compliant); concurrency: per-cycle caches cleared at L345/`_init_cycle_cache`, single-threaded timer cycle]
- most_suspicious: L513 — `top = max(candidates, key=lambda c: c.get("score", 0))` — `max()` on empty raises ValueError, but it is guarded by `if candidates:` at L512, so unreachable when empty.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capacity_planner.py#2
- chunk_id: modules/capacity_planner.py#2
- summary: Cycle summary logging tail, `_get_mempool_recommendation`, `_identify_winners`, `_identify_losers`, hive-member protection, `_close_protection_reason`, inactivity-signal helpers, and `_build_dead_capital_loser` staging state machine. Pure analytics except `_record_fee_reduce_delegation`/`_build_dead_capital_loser` which write DB rows (`record_planner_action`, `upsert_dead_capital_stage`, `delete_dead_capital_stage`). Loser classification via profitability class + flow + rebalance-difficulty; dead-capital advances FEE_REDUCE->DEFIBRILLATE->CLOSE on stage timeouts with protection/attempt gates.
- invariants: [rounding: `round(...,2)` on ROI/difficulty display only (L1037-1048), no msat/sat math; sign: `rebal_penalty=(0.5-sr)*50` non-negative since gated `sr<0.5` (L758-759); caps: penalty capped at 25% ROI by construction; none-handling: extensive `getattr`/`.get` guards, `flow_metrics` None-checked before use (L741,L960,L998), confidence coerced non-numeric->1.0 (L1114-1116); sql: DB reads/writes wrapped in try/except, membership fails CLOSED (L1064-1078); never-call: N/A production; concurrency: dead-capital stage timer anchored only on successful delegation record (L1310-1311) preventing silent aging]
- most_suspicious: L1312 — `elif stage == "fee_reduction" and entered_at and now - entered_at >= stage_timeout:` — advancing to DEFIBRILLATE requires truthy `entered_at`; a stored `entered_at==0` correctly falls through to the plain `fee_reduction` branch (L1317) rather than instantly aging, so the guard is deliberate and safe.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capacity_planner.py#3
- chunk_id: modules/capacity_planner.py#3
- summary: Dead-capital loser dict tail, `_generate_recommendations`, `_apply_redeployment_ev_demotion` (mutates loser action/reason in place), and peer-discovery strategies (winners, neighbors incl. 1st/2nd degree, graph centrality, route-pairs, hive). Pure analytics — reads cached channels/profitability/DB route-pairs, produces scored candidate dicts; no external state writes. Scores are heuristic sums with capacity/fee bonuses.
- invariants: [rounding: `base_to_sats_floor(parse_msat(...))` for capacities (L1704,L1751,L1815), `base_to_sats_ceil` for route-pair fees (L1878) — ceil correct for fee/revenue; sign: EV demotion only on `ev <= 0` (L1505); caps: candidate lists sliced `[:10]`/`[:5]`/`[:3]`; none-handling: `.get`/`getattr` defaults, `channels.get("destination")` skip-on-missing, `median(capacities) if capacities else 0` (L1757); sql: `get_top_route_pairs`/`get_channel_...` in try/except returning [] on failure; never-call: N/A; concurrency: reads per-cycle `_cycle_channels_source` cache populated earlier same cycle]
- most_suspicious: L1819 — `capacity_btc = total_capacity_sats / 100_000_000 if total_capacity_sats > 0 else 0.001` — the `else 0.001` floor keeps `math.sqrt` finite and non-zero for zero-capacity nodes; combined with `channel_count < 5` skip (L1811) it cannot produce a div-by-zero or NaN score.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capacity_planner.py#6
- chunk_id: modules/capacity_planner.py#6
- summary: Capex settle-failure log tail, `_check_close_allowed` (policy gate), `_execute_defibrillation`, `_execute_close`, `_rpc_close`. These are the real executor/mutation paths: record planner actions, honor dry_run/recommendation gates, reserve budget (`reserve_spend` under BEGIN IMMEDIATE), stop rebalancer jobs, call close RPC, then settle/release the reservation. Records ACTUAL shock/close outcomes rather than optimistic completed.
- invariants: [rounding: `int(...)` on fee-plan cost/reserve (L3654-3655), actual_fee cast `int(actual_fee_sats)` (L3619); sign: reservation released on failure (L3826-3830), settled with actual-or-cap on success (L3796); caps: close gated by `_close_execution_enabled` (max_closes>0) and unified budget check + reservation (L3702-3781); none-handling: `db`/`action_id` truthiness-guarded before every DB call, `result` isinstance-checked before `.get` (L3605,L3612,L3632); sql: reserve-before-spend, settle-on-success, release-on-failure via `release_spend_reservation`; effective_budget passed so cross-category BEGIN IMMEDIATE rejection runs (L3736-3761); never-call: close RPC is the intended executor, gated by dry_run+execute_closes+policy (compliant); concurrency: atomic budget reservation prevents concurrent close/rebalance/open joint overshoot (documented DD1/P4-018)]
- most_suspicious: L3796 — `close_cost = actual_close_fee_sats if actual_close_fee_sats is not None else reserved_close_fee_sats` — settling with the reserved cap when the RPC reports no fee could over-count spend, but that is the conservative direction (never under-counts the budget) and is corrected to actuals whenever the RPC surfaces them.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/profitability_analyzer.py#1
- chunk_id: modules/profitability_analyzer.py#1
- summary: Module docstring/imports, `BookkeeperCache` (bulk bkpr-listincome index by txid), enums (`ProfitabilityClass`, `ChannelRole`), dataclasses (`ChannelCosts`, `ChannelRevenue`, `ChannelProfitability` with role/marginal-ROI properties), and `ChannelProfitabilityAnalyzer.__init__`/`analyze_all_channels`. Mostly pure aggregation; `analyze_all_channels` writes the in-memory cache and pushes a datastore summary. Revenue dataclass exposes msat-native fields with ceil-on-sat properties.
- invariants: [rounding: fees->sats use `base_to_sats_ceil` with non-zero->>=1 sat (L224,L236,L261) preventing sub-sat truncation; volume uses floor (L229,L241); onchain-fee cost uses `base_to_sats_floor(net_msat)` (L107,L117); sign: only `net_msat>0` fees indexed (L106,L116), `marginal_roi` returns 1.0/0.0 when no rebal cost (L340-342); caps: MAX_OPEN_FEE_SATS=10 BTC ceiling; none-handling: `parse_msat` None-safe, `total_forwards<10`->DORMANT avoids div-by-zero (L375-379,L417); sql: BookkeeperCache single bulk RPC, all-or-None via `_fetch_ok`; never-call: N/A; concurrency: non-blocking `_analysis_lock.acquire` returns stale cache on contention, timestamp bumped early to prevent stampede (L633-639), documented GIL-atomic dict reads]
- most_suspicious: L107 — `self._onchain_fees[txid] = base_to_sats_floor(net_msat)` — a channel-open on-chain fee (a cost) is floored, marginally understating cost vs the ceil convention utils prescribes for fees; the loss is <1 sat per tx on historical accounting and does not affect classification thresholds, so acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/profitability_analyzer.py#2
- chunk_id: modules/profitability_analyzer.py#2
- summary: `analyze_all_channels` finally-release, `_push_profitability_summary` (datastore write), `analyze_channel` (single-channel ROI/classification build), and accessor methods (`get_profitability`, `get_profitability_by_peer`, fee/rebalance multipliers, `should_rebalance`, `record_*_cost`, zombie/prune/summary getters). `analyze_channel` is pure compute over prefetched revenue/costs/30d-PnL; `record_*` write DB and invalidate cache entries.
- invariants: [rounding: `base_to_sats_ceil` for revenue aggregation (L954-957,L1310-1312), `base_delta_to_sats_toward_zero` for signed 30d marginal profit (L858); sign: `roi=net/max(1,cost)` and `total_contribution/max(1,capacity)` guard zero (L818,L828), overall_roi guarded `if total_cost>0` (L960,L1315); caps: fee multipliers bounded 0.95-1.15, rebalance budget `min(1.5,...)` (L1114); none-handling: `get_profitability` returns None->callers default to 1.0/neutral, `pnl_30d.get(...,0)` defaults, `window_30d_available=True` set so consumers trust 30d fields; sql: `record_rebalance_cost`/`record_channel_open_cost` delegate to DB then `pop` cache; never-call: N/A; concurrency: `_analysis_lock.release()` in finally (L701), single-key `dict.pop` atomic under GIL (L1175)]
- most_suspicious: L856 — `int(pnl_30d.get('rebalance_cost_msat') or sats_to_base(rebalance_cost_30d))` — if `rebalance_cost_msat` is a legitimate 0 it is falsy and falls back to `sats_to_base(rebalance_cost_30d)`, but that sats value is itself 0 in that consistent case, so the fallback yields the same 0; only inconsistent DB rows (msat=0, sats>0) would diverge, and then it favors the sats figure, which is acceptable.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/profitability_analyzer.py#3
- chunk_id: modules/profitability_analyzer.py#3
- summary: `get_lifetime_report`, `get_pnl_summary`, `identify_bleeders` (v1, F7 pure-bleeder inclusion), `identify_bleeders_v2` (hard/soft classification with hysteresis + materiality floor, writes `_bleeder_cache`), `get_bleeder_status`, `calculate_roc`, `get_tlv`, and cost helpers `_get_all_channels`/`_get_channel_open_timestamp`/`_get_channel_costs` (self-healing/retroactive DB writes of open cost). Analytics plus DB reads; `_get_channel_costs` writes corrected open costs.
- invariants: [rounding: lifetime/pnl revenue `base_to_sats_ceil` (L1411,L1478), bleeder revenue/net fall back to `base_to_sats_floor` of msat (L1677-1679) — floor understates revenue, conservative for bleeder detection; sign: ROI/margin guarded `if cost>0 / revenue>0` (L1427,L1495,L1848); caps: hard-bleeder needs net<-1000, hysteresis exit above -500, materiality floor <100 sats->none (L1696-1721); none-handling: `.get(...,default)` throughout, per-channel PnL fallback when batch None (L1558-1560,L1666-1673), `window_days` clamped to min (L1471,L1541,L1634,L1832); sql: batch `_get_all_full_pnl_batch` with per-channel fallback; retroactive/self-heal writes wrapped; never-call: N/A; concurrency: `_bleeder_cache` set before timestamp (M10, L1811-1813) and only default 30d window overwrites (L1785)]
- most_suspicious: L1685 — `effective_rebalance_cost_30d = int(rebalance_cost_30d / sr)` — a raw division, but `sr = max(success_data['success_rate'], 0.10)` at L1684 floors the divisor at 0.10, so a 0% success rate cannot cause a ZeroDivisionError; upper bound is 10x cost, which is the intended inflation for failure-prone channels.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/__init__.py#1
- chunk_id: modules/__init__.py#1
- summary: Package wiring only — re-exports FlowAnalyzer/FeeController/EVRebalancer/Config/Database/DataService/PolicyManager and related types via `from .x import ...`, defines `__all__`. No control flow, no state.
- invariants: [rounding: N/A no arithmetic; sign: N/A; caps: N/A; none-handling: N/A; sql: N/A; never-call: N/A no RPC; concurrency: N/A import-time only]
- most_suspicious: L15 — `from .rebalancer import EVRebalancer, RebalanceCandidate` — an import of names that must exist in `rebalancer.py`; if absent the package fails to import at load, which is fail-fast and acceptable for package wiring.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/utils.py#1
- chunk_id: modules/utils.py#1
- summary: Pure shared helpers — `normalize_scid`, `parse_msat` (None/bool/numeric/str/`.millisatoshis` safe coercion to int msat), base-unit conversions (`base_to_sats_ceil`/`floor`, `base_delta_to_sats_toward_zero`, `sats_to_base`), unit constants, and backward-compat aliases. No I/O, no state, only a module logger.
- invariants: [rounding: ceil `-(-base//1000)` for fees/budgets (L78), floor `base//1000` for balances (L88), signed delta rounds toward zero (L91-95) — directions match documented never-underbudget/never-overstate rules; sign: bool explicitly rejected as invalid msat (L39-41, U-1 fix), negative delta handled symmetrically; caps: N/A; none-handling: `parse_msat(None)->0` (L31), every conversion in try/except->0 with debug log; sql: N/A; never-call: N/A; concurrency: N/A stateless pure functions]
- most_suspicious: L78 — `return -(-base // BASE_UNITS_PER_SAT)` — ceiling-division idiom; for negative `base` this rounds toward zero rather than up, but callers pass this only non-negative fee/budget msat (guarded upstream by `if ... <= 0: return 0` in the revenue properties), so the negative-input edge is never exercised.
- auditor: fable-phase8  date: 2026-07-02

---

### modules/capital_efficiency.py#1
- chunk_id: modules/capital_efficiency.py#1
- summary: `ChannelEfficiency`/`FleetEfficiency` dataclasses and `CapitalEfficiencyAnalyzer.analyze` — pure analytics building a fleet snapshot from profitability+flow snapshots and DB dead-capital stages. Computes lifetime-gross RPSD (ppm), optional 30d-net RPSD blended 50/50 into percentile ranks, dead-capital classification, and per-channel efficiency records. Only DB read is `get_dead_capital_stages` (try/except-guarded); no writes.
- invariants: [rounding: RPSD `fees_msat*1000/capacity` and windowed `sats*1e6/capacity` both ppm-correct (L164,L182); sign: `max(0,int(...))` on capacities (L115,L136,L153,L179); caps: percentile ranks bounded [0,1], `forward_count/max(flow_window_days,1)` guards zero window (L143); none-handling: windowed rank only activates when ALL channels expose numeric `marginal_profit_30d_sats` else fully discarded (L92-97, never synthesized), bool rejected as non-numeric (L177), `flow_metrics None->not dead capital` (L211); sql: single guarded DB read, `median` guarded `if rpsd_by_channel else 0.0` (L113); never-call: N/A; concurrency: stateless per-call analysis]
- most_suspicious: L164 — `return float(fees_earned_msat) * 1000.0 / capacity_sats` — a bare division, but `capacity_sats <= 0` returns 0.0 at L154 before this line, so the divisor is guaranteed positive and no ZeroDivisionError is reachable.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/check_hermes_forwards_chain.py#1
- chunk_id: tools/audit/check_hermes_forwards_chain.py#1
- summary: Pure read-only audit. `iter_windows` gzip-opens `listforwards-window.json.gz` files under `$CL_MYCELIUM_HERMES_ROOT/<node>`, parsing JSON per file. `audit_node` walks windows in sorted path order, checking chain contiguity (`start == prev_next_start`), overlap/gap classification, truncation-retry adjacency, and forward dedup by `(created_index, updated_index, in_channel, in_htlc_id, status)` key, summing settled fee_msat. `main` prints a report and returns 0/1. No files written, no subprocess calls, no RPC.
- invariants: [rounding: N/A — settled_fee_msat/1000 sat conversion is display-only at print time (L140), not persisted; sign/off-by-one: gap-vs-overlap split correctly uses `start < prev_next_start` for overlap (L74) vs anything else as gap; caps: N/A no budget logic; none-handling: fee parse guarded by try/except TypeError/ValueError (L117-120), meta.get() defaults to None safely; sql: N/A no DB access; never-call: N/A no RPC/subprocess calls anywhere in file; concurrency: N/A single-threaded sequential walk, no shared mutable state across processes]
- most_suspicious: L119 — `except (TypeError, ValueError):` — silently drops a malformed `fee_msat` from the `settled_fee_msat` reconciliation total without recording an error; acceptable because this sum is informational (for cross-checking against revenue-dashboard) not a pass/fail gate, and unreadable/malformed entries are already surfaced separately via the `unreadable`/`err` path.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/check_pins.py#1
- chunk_id: tools/audit/check_pins.py#1
- summary: Stdlib-only CI gate. `parse_requirements` reads `requirements.txt` line by line; `check` regex-validates each line is an exact `name==version` pin (`_PIN_RE`) and, if not `--no-installed-check`, compares against `importlib_metadata.version(name)`. `main` prints problems and returns exit codes 0/1/2. No writes, no subprocess, no network.
- invariants: [rounding: N/A no numeric computation; sign/off-by-one: N/A; caps: N/A; none-handling: `importlib_metadata.PackageNotFoundError` explicitly caught (L104-111) before comparing installed vs pinned; sql: N/A; never-call: N/A — only reads `requirements.txt` and interpreter package metadata, no CLN RPC; concurrency: N/A single-threaded, stateless function calls]
- most_suspicious: L113 — `if _canonical(installed) and installed != pinned:` — canonicalizes only for the truthiness guard but compares raw (non-canonicalized) `installed` against `pinned`; benign because both strings come from PEP 440 version fields which don't carry the name-separator ambiguity `_canonical` exists to fix, so this doesn't produce false negatives/positives in practice.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/gen_sbom.py#1
- chunk_id: tools/audit/gen_sbom.py#1
- summary: Enumerates `importlib_metadata.distributions()`, dedups by canonical name, builds a CycloneDX 1.5 component list sorted deterministically, computes a uuid5-derived `serialNumber` from the purl set, and writes the resulting JSON BOM to `--output` (default `docs/audit/deep/sbom.cyclonedx.json`) via `os.makedirs` + `open(...,"w")`. This is the one chunk in the set that performs an external file write (the SBOM artifact itself, by design).
- invariants: [rounding: N/A; sign/off-by-one: N/A; caps: N/A; none-handling: `if not name or not version: continue` (L60-61) skips distributions with missing metadata; license lookup wrapped in try/except (L69-72, L77-80); sql: N/A; never-call: N/A — no CLN RPC/subprocess, only package-metadata introspection and a local file write, consistent with an offline audit-tooling role; concurrency: single writer, single process, no shared state; not atomic (no temp-file+rename) but this tool is invoked standalone/CI, not by concurrent processes]
- most_suspicious: L158 — `os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)` — followed by a direct (non-atomic) `open(...,"w")` write of the SBOM JSON; acceptable because the tool is explicitly meant to (re)generate a deterministic artifact on demand and nothing reads it concurrently mid-write.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/scorecard.py#2
- chunk_id: tools/audit/scorecard.py#2
- summary: Aggregation/rendering tail of the scorecard builder. Groups per-check results into per-module rows, computing a status precedence of ERROR > WARN(new violations) > KNOWN(allowlisted) > PASS > INCONCLUSIVE (L739-748), attaches vacuous/allowlist/lossy-echo annotations, then `render()` formats a text table and `main()` optionally dumps JSON to `--json` path. Only external write is the optional `--json` report file; otherwise pure aggregation over already-collected `Check` objects from earlier (out-of-range) sweep-subprocess invocations.
- invariants: [rounding: N/A no monetary math, only counting; sign/off-by-one: N/A; caps: N/A — this is health-status classification, not budget arithmetic; none-handling: extensive `.get()`/`.setdefault()` defaulting throughout, e.g. `modules.setdefault(m, {...})` (L707,716); sql: N/A; never-call: N/A — this chunk only consumes already-parsed `Check` results, no RPC calls in range; concurrency: N/A sequential loop over `MODULE_ORDER`, single process; JSON report write (L860-861) is a single non-atomic `open(...,"w")` but is a one-shot CLI report with no concurrent reader]
- most_suspicious: L741 — `elif new_viol: status = "WARN"` — status precedence relies on `elif` chain ordering (ERROR checked first at L740, then new_viol, then known_viol, then non_vacuous); correct because each branch is mutually exclusive by construction and the ordering matches the documented severity (new-and-undocumented findings must never be masked by allowlisted/known ones).
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/sweep_data_budget.py#1
- chunk_id: tools/audit/sweep_data_budget.py#1
- summary: Pure read-only corpus sweep. Walks `$CL_MYCELIUM_HERMES_ROOT/<node>/<day>/<ts>/commands`, loading `revenue-spend-ledger.json`, `revenue-total-cost-budget.json`, `revenue-capex-status.json`, `hive-organism-status.json`, and the segment-observations datastore snapshot, tallying pass/fail counts per named invariant (`SL-*`, `TCB-*`, `CB-*`, `ML-*`, `SO-*`) into module-level `counts` plus sample-row defaultdicts. `main()` prints a text report. No writes, no RPC, no subprocess.
- invariants: [rounding: CB1-ENVELOPE explicitly allows `slack = len(channels) + 2` sats of overshoot (L108-111) to absorb independent per-channel ceil-rounding from msat, documented inline and correctly applied; sign/off-by-one: SL-NONNEG/CB-NONNEG assert all ledger values >=0 (L62-63, L129); caps: CB5-FLEET-CAP budget<=200 (L124-125), CB-TIERPPM cross-checks tier->ppm mapping (L117-119); none-handling: `int(c.get("budget_sats", 0) or 0)` pattern guards None/missing throughout (L105,114,132-134); sql: N/A — plain JSON file reads via `load()` with broad try/except; never-call: N/A no RPC calls; concurrency: N/A single-threaded sequential directory walk, module-level defaultdicts mutated only from main thread]
- most_suspicious: L138 — `and prio.get("operational", -1) == tact` — uses `-1` as a sentinel default so a genuinely-missing `operational` key fails the comparison; safe because `tact` is always derived via `int(d.get(...,0) or 0)` (L104) and thus never negative, so `-1` can never accidentally match a real value.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/sweep_fee_stack.py#1
- chunk_id: tools/audit/sweep_fee_stack.py#1
- summary: Pure read-only sweep over `revenue-status.json` and `listpeerchannels.json` across both nodes' snapshot corpus, checking fee-change bound/cap invariants (`FC-I1a/I1b/I9/I10`, alpha-guard, gossip-gate) and channel-state invariants (`FA-I1/I4/I5/I12`, vocabulary check). Dedupes fee-change records by `(node, id)`, classifies transient vs persistent channel-state "residue" against live `listpeerchannels` scids. Prints Counter-based report; returns 1 if any violation. No writes.
- invariants: [rounding: FC-I10 delta cap uses `math.ceil(0.5*old)`/`math.ceil(0.2*old)` per docstring (L159,161), consistent with contract; sign/off-by-one: gossip_refresh nudge tolerance `abs(new-old) > 1` (L147) is an explicit +/-1ppm allowance, not an unguarded off-by-one bug; caps: FC-I10 per-cycle delta cap (L162-165), alpha-guard `min_change` floor (L167-169); none-handling: `if old is None or new is None: ...; continue` (L124-126) guards before any arithmetic; sql: N/A; never-call: N/A pure JSON reads via `load_json`, no RPC; concurrency: N/A single-threaded, `seen_changes`/`residue` dicts mutated only in the main loop]
- most_suspicious: L167 — `min_change = 1 if old < 100 else max(5, -(-old * 3 // 100))` — uses the `-(-x // y)` ceiling-division idiom to compute `ceil(3% of old)`; verified correct (e.g. old=100 -> `-(-300//100)` = 3, `max(5,3)`=5), matching the docstring's `max(5, ceil(3% of old))` spec exactly.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/sweep_profitability.py#1
- chunk_id: tools/audit/sweep_profitability.py#1
- summary: Pure read-only sweep of `revenue-profitability.json` snapshots, checking `PA-I9` (`total_contribution_sats == max(fees_earned, sourced_fee_contribution)`), `PA-SUM` (class-count/list-length consistency), `PA-CLS` classification-vs-ROI-threshold boundaries, and `PA-I11w` non-negativity, plus accumulating "structural-protection upgrade" anomaly stats (D2 mask). Prints report via `Check.report()`. No writes, no RPC.
- invariants: [rounding: docstring (L9-12) justifies exact-equality `tot == max(fees, sourced)` rather than a sum, reasoning that ceil() is monotonic so max-of-ceilings equals ceiling-of-max — correctly reflected in code; sign/off-by-one: ROI threshold checks use epsilon tolerance `roi > 5.0 - 0.01` / `roi < -10.0 + 0.01` (L162,169) to avoid float-boundary false negatives; caps: N/A — thresholds here are classification bounds not spend caps; none-handling: `if None not in (fees, sourced, tot):` (L149) guards before comparison/arithmetic; sql: N/A; never-call: N/A pure JSON file reads via `load()`; concurrency: N/A single-threaded, defaultdicts mutated sequentially]
- most_suspicious: L150 — `if tot == max(fees, sourced):` — exact (non-epsilon) equality on sats-denominated integer fields per the `PA-I9` contract; appropriate here because these are documented as already-rounded sats values (not raw msat/float), unlike the ROI checks which do use epsilon tolerance.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/sweep_rebalancer.py#1
- chunk_id: tools/audit/sweep_rebalancer.py#1
- summary: Pure read-only sweep of `revenue-status.json`, `revenue-rebalance-debug.json`, `revenue-total-cost-budget.json` checking budget consistency (`RB-I1a/b/c`), diagnostic-row caps (`RB-I10`), and engine decomposition invariants (`RE-I2a/I2b/I3/I8/I9/I10`) via `check_decomposition()`. Reconstructs a deduped `rows_final` per-row-id ledger (last-write-wins by snapshot timestamp) and cross-checks it against ledger spend as anomaly material. Prints report; no writes.
- invariants: [rounding: N/A beyond the explicitly-documented D4 static ceiling note (L242-245) for diagnostic max_fee bound; sign/off-by-one: `rem == max(0, eff - spent - res)` (L214) exact-equality budget consistency check matches the capex_budget contract; caps: `amt<=50_000` and `mf<=10_000` (L245), `selected_pairs<=20`/`execution_count<=20` (L192-195); none-handling: `if None not in (eff, spent, res, rem):` (L212) guards before arithmetic; sql: N/A JSON file reads only; never-call: N/A no RPC calls; concurrency: N/A single-threaded; `rows_final` dedup at L181-184 uses `if prev is None or ts >= prev[0]:` for deterministic last-write-wins, correct for sequential single-threaded processing]
- most_suspicious: L275 — `if rebal + 2 < visible:` — adds an unexplained `+2` sats slack before flagging a ledger-vs-visible-fees mismatch (the general "visible is a lower bound" rationale is given at L273-274); acceptable because per the module docstring (L14-16) this is explicitly "anomaly material, not a strict invariant" — it only affects an informational mismatch list, not a pass/fail gate.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/sweep_routing_stack.py#1
- chunk_id: tools/audit/sweep_routing_stack.py#1
- summary: Pure read-only `Sweep` class sweeping `revenue-rebalance-debug.json`, `revenue-status.json`, segment-observations, and `revenue-spend-ledger.json` across both nodes, checking route-cost sign, hop-amount monotonicity, source pinning, cost-formula consistency, execution attempt/fee caps, and budget non-negativity (`C1`-`C9`, `S1`-`S3`, `O1`-`O2`, `L1`). `report()` only prints (JSON or text) to stdout — no file writes. Returns 1 if any violation counted.
- invariants: [rounding: C5 uses `math.ceil((amounts[0]-amounts[-1])/1000)` with `abs(implied-cost) > 1` tolerance (L156-157), correctly handling msat->sat rounding slack; sign/off-by-one: C1 flags `cost < 0` (L124-125); C2 flags `cur > prev` for route-hop amounts, correctly checking non-increasing order (L144-147); caps: `C9_attempts_gt_3` (L191-192), `C9_fee_over_pair_budget` (L198-204); none-handling: `if not isinstance(hops, list) or not hops: return` (L138) and `isinstance(...,(int,float))` guards throughout before arithmetic; sql: N/A; never-call: N/A pure file reads via `load_json`; concurrency: N/A single `Sweep` instance mutated sequentially in one process, no threads]
- most_suspicious: L198 — `if ex.get("success") and len(executions) == len(selected):` — pairs `executions[idx]` with `selected[idx]` purely by equal-length heuristic rather than an explicit id/key match, as the inline comment at L197 acknowledges; this degrades gracefully (skips the check, doesn't false-flag or mis-pair) when lengths differ, so it's a conservative under-check rather than a correctness defect.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/sweep_planner_boltz_hints.py#1
- chunk_id: tools/audit/sweep_planner_boltz_hints.py#1
- summary: Read-only corpus sweep that also imports and exercises production code: `from modules.hive_hints import HiveHintAdapter` with `pyln`/`pyln.client` stubbed as `MagicMock()` at import time (L36-41) purely to satisfy the import, no live plugin. `sweep_planner` reconstructs a deduped planner-action ledger (last-write-wins by `completed_at`) and checks `CP-I1/I2/I4/I5/I14` plus a candidate-pool-size bound; `sweep_boltz` checks `BM-H2` budget bound; `sweep_hints` replays real producer payloads through a throwaway `HiveHintAdapter(MagicMock())` instance to check `HH-I1/I2/I3/I8` bias/prior bounds. Prints report only; no files written, no live RPC.
- invariants: [rounding: N/A — no msat/sat rounding math, bias/prior values are already-normalized adapter outputs; sign/off-by-one: cooldown check `0 < lts - ets < COOLDOWN_SECONDS` (L259) correctly excludes same-timestamp (0) and boundary-exact (24h) cases as compliant; caps: `CP-I4` defib-cluster limit, `CP-I14` 24h cooldown, candidate pool `<=32` (L274-278), `BM-H2` budget bound (L301-303), `HH-I3` bias bounds `0.9-1.1`/`0.85-1.15` (L399-404), `HH-I8` fleet-fee-prior `[1,10000]` (L406-407); none-handling: `mem = members_at(...); if mem is None: continue` (L205-206) and `if snap is None: continue` (L386) guard adapter/validation misses; sql: N/A; never-call: adapter constructed with `MagicMock()` as its plugin (L384) and module-level `pyln`/`pyln.client` stubs (L36-41) mean any RPC the adapter path might attempt resolves to a `MagicMock`, not a real `lightning-cli`/datastore call — confirms the sweep stays read-only despite invoking real production adapter logic; concurrency: N/A single-threaded, module-level `results` list appended sequentially via `record()`]
- most_suspicious: L393 — `adapter._store_snapshot(snap, "datastore")` — calls a private method of `HiveHintAdapter` directly from sweep code to seed in-memory adapter state for the bias replay; a minor encapsulation reach-through but not a data-corrupting write, since the adapter instance's plugin is `MagicMock()` (L384) — `_store_snapshot`'s effect is confined to the throwaway in-process adapter object and never touches the real Core Lightning datastore.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/contrib_capital_analysis.py#1
- chunk_id: tools/audit/contrib_capital_analysis.py#1
- summary: Read-only Phase-4 capital-side contribution analysis. Loads gzip/JSON snapshot artifacts from the frozen hermes corpus (`load`, `artifacts`, L65-96), reconstructs forwards chains, channel presence, planner actions and rebalance rows (L106-210), and runs PA/CP hypothesis sections (`pa_section`, `cp_section`, L281-701) using bootstrap/Mann-Whitney statistics. All output is via `print`; no writes to corpus, DB, or RPC anywhere in this range.
- invariants: [rounding: msat()->int cents-safe parse (L89-95); capacity converted via `// 1000` floor division (L164), acceptable for descriptive sats reporting; sign/off-by-one: `window_sum` uses a consistently half-open `t0 < t <= t1` interval (L145) reused everywhere, no boundary drift; caps: N/A — no budget/spend arithmetic, PA-H2 (L426-444) only reads exported fields; none-handling: `load()` (L65-73) catches all exceptions and returns None, every caller gates on `if not d`; sql: N/A, filesystem JSON/gzip only; never-call: N/A — no RPC calls of any kind; concurrency: N/A — single-threaded script, no shared mutable state]
- most_suspicious: L94-95 — `except (TypeError, ValueError): return 0` — silently coerces any malformed/missing msat field to 0 sats instead of surfacing it; acceptable because `load()` already filters unparseable files upstream, this only affects individual malformed numeric fields, and a 0 is statistically indistinguishable from the corpus's own legitimate zero-fee/zero-forward semantics used throughout.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/contrib_capital_analysis.py#2
- chunk_id: tools/audit/contrib_capital_analysis.py#2
- summary: Continues the capital-side analysis — RB/RE rebalance-loop hypothesis tests with pre-fix twin-row dedup (`rb_re_section`, L708-809), an exploratory hold-margin/EV-gate calibration section that recomputes the engine's own EV formula on captured candidates (`hold_margin_section`, L816-925), a Boltz dormancy-proof section (`bm_section`, L931-959), and `main()` which loads forwards chains once and dispatches all sections sequentially (L964-998). Entirely read-only; only stdout output.
- invariants: [rounding: EV/ppm math uses float division guarded by `if amt else float("nan")` (L882, L883) and `if vol else 0.0` (L905) — no divide-by-zero; sign/off-by-one: twin-row dedup groups by `(from_channel,to_channel,timestamp)` and keeps diagnostic rows over normal twins pre-fix (L717-730), matching the documented 62ae545 defect; caps: N/A, RB-H2 (L753-778) only reads `rd.get("budget_blocked")` for classification; none-handling: pervasive `.get(...) or {}` / `or 0.0` guards (L760, 804, 826, 829, 839) prevent None propagation into arithmetic; sql: N/A; never-call: N/A — no RPC calls anywhere in this range; concurrency: N/A — sequential single-threaded execution]
- most_suspicious: L849 — `if key not in dedup or c["ts"] < dedup[key]["ts"]:` — the considered-candidate dedup keeps the EARLIEST-timestamped export per `(node,src,dst,amount,route_cost)` key rather than the latest; acceptable since it deliberately selects the first-observed instance of a "distinct candidate" to avoid recounting the same candidate re-exported across multiple debug snapshots — a "latest wins" policy would be equally defensible but not more correct for this purpose.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/contrib_fee_analysis.py#2
- chunk_id: tools/audit/contrib_fee_analysis.py#2
- summary: Continuation of the fee-side contribution analysis covering the tail of FC-H1 statistical printing, FC-H2 climb-governor overshoot analysis (`sec_fch2`, L772-901), FC-H3 rebalance-floor census (`sec_fch3`, L907-959), an exploratory E2 elasticity episode analysis (`sec_e2`, L965-1062), HH hive-hint hypotheses (`sec_hh`, L1068-1161), PM policy-manager census (`sec_pm`, L1167-1195), and most of FA-H1..H3 flow-analysis tests including a Kalman-filter replay that imports `modules.flow_analysis` for pure calibration functions (L1310-1400). Read-only; only prints and populates the module-level `CONFIRMATORY_P` dict for a later Holm correction.
- invariants: [rounding: msat->sat conversions consistently use `/1000.0` float division (e.g. L821) for descriptive fee sums, not floor truncation, appropriate for reporting; sign/off-by-one: `week_blocks` (L745-753) builds half-open `[w0,w1)` 7-day blocks anchored via `math.floor((lo-anchor)/(7*DAY))`, correctly anchor-aligned with no boundary duplication; caps: N/A, no budget writes; none-handling: `sec_fch3`'s `walk()` (L920-935) recursively and defensively inspects dict/list structures only touching numeric leaves matching name heuristics; sql: N/A; never-call: N/A — `from modules import flow_analysis as fam` (L1312) imports pure calculation primitives for a read-only replay, it does not construct or call the plugin's RPC-registered handlers; concurrency: N/A — single process, no threads]
- most_suspicious: L912-914 — `efv = (amt * max(c["dest_ppm"] or 0, dh) / 1e6 * EXPECTED_UTILIZATION + amt * sh_src / 1e6 * EXPECTED_UTILIZATION)` — adds a "source drain term" using the SOURCE channel's inbound-fee history into the recomputed EFV, a different formula shape than the engine's own `source_opportunity_sats`; acceptable because this whole block is explicitly self-labeled exploratory (banner L899-901) and printed with an explicit NOTE (L922-924) that it has no baseline/counterfactual — not confirmatory and nothing downstream treats it as ground truth.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/contrib_fee_analysis.py#3
- chunk_id: tools/audit/contrib_fee_analysis.py#3
- summary: Tail of the FA-H3 Kalman depletion-forecast replay (prediction evaluation against 36h LPC lookahead, L1410-1461) plus `main()` (L1472-1495): dispatches requested sections via the `SECTIONS` dict and, when running all sections, applies Holm-Bonferroni correction over `CONFIRMATORY_P` and prints it. Entirely read-only/reporting; no writes.
- invariants: [rounding: N/A in this sub-range — arithmetic is confined to the earlier chunk, here it's evaluation/aggregation of already-computed values; sign/off-by-one: lookahead-coverage guard `if not look or look[-1][0] - tt < 30*3600: continue` (L1417) requires near-full 36h coverage before counting a prediction, avoiding partial-window bias; caps: N/A; none-handling: `kf._has_nan()` guard (L1404) defensively resets filter state rather than propagating NaN into `state_at`; sql: N/A; never-call: N/A — no RPC calls, `main()` only calls the local `SECTIONS[s]()` functions; concurrency: N/A — single-threaded]
- most_suspicious: L1404-1405 — `if kf._has_nan(): kf._reset_state()` — silently resets the replay's Kalman filter state on NaN detection rather than counting/flagging the occurrence; acceptable because this mirrors the production module's own defensive `_has_nan`/`_reset_state` contract being faithfully replayed for calibration, and a silent reset only perturbs this exploratory replay's own internal state, not any persisted value.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/loop_sweep_fee.py#1
- chunk_id: tools/audit/loop_sweep_fee.py#1
- summary: Phase-3 read-only loop sweep (docstring L66 explicitly states "Read-only over corpus and repo; writes nothing"). `NodeData._load` (L132-217) ingests revenue-status/listpeerchannels/hive-export-hints/revenue-fee-debug snapshots per node; `main()` (L220-552) runs LF-1..LF-8 cross-module invariant checks (gossip diff -> recorded change, record->gossip agreement, chain continuity, pause suppression, member zero-fee pinning, bounds, state handoff, debug-surface coherence, ratchet direction) and prints pass/violation/info counters with sample rows.
- invariants: [rounding: N/A — all fee comparisons are integer ppm equality checks (e.g. L325 `adv == c["new_fee_ppm"]`), no msat/sat conversion in this file; sign/off-by-one: window-style checks use consistent slack constants (`EDGE_SLACK=120`, `2*CYCLE_SECONDS`) applied symmetrically (L265, L213, L367); caps: LF-5 bounds check (L388-423) reads `[min_fee_ppm,max_fee_ppm]` from the nearest prior snapshot and correctly exempts `hive_member_zero_fee` records (L400-401); none-handling: `state_at` (L206-217) returns None on missing/too-distant snapshot and callers explicitly branch on `before is None and after is None` (L447); sql: N/A, filesystem JSON only; never-call: N/A — no RPC calls anywhere, purely local JSON parsing; concurrency: N/A — single-threaded script]
- most_suspicious: L512 — `elif new > old:` — a guard-tagged `zero_flow_downshift`/`zero_flow_ratchet_guard` change whose fee rose is classified as informational rather than a violation; acceptable because the reasoning explicitly cites the engine's own `max(floor_ppm, min(target, cap))` design (hard floors win by construction) as the documented cause, so this is a telemetry-naming nuance the sweep reports on, not a false-negative in the check itself.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/loop_sweep_planner.py#1
- chunk_id: tools/audit/loop_sweep_planner.py#1
- summary: Phase-3 read-only capacity-planner loop sweep. Reconstructs planner actions/configs/fee-rows/rebalance-rows/forwards/fleet-member sets (L111-229), then runs L1 close-pipeline staging (L236-285), L2 fee_reduce delegation handoff (L292-372), L3 defibrillation-outcome joins (L379-484), L4 cross-module invariants including member-protection and era-aware stage-ordering (L491-560), and L5 boltz dormancy proof (L567-590); `main()` (L606-661) drives all nodes and prints a `results` summary table. No writes anywhere.
- invariants: [rounding: N/A — no msat/sat unit conversions, only ppm and sats-as-int fields read directly from corpus rows; sign/off-by-one: `at_floor = fee_before is not None and fee_before <= floor_ppm + 1` (L329) applies a deliberate +1 ppm snapping tolerance, documented by its role in verdict bucketing, not an accidental off-by-one; caps: N/A — no budget arithmetic, only reads `effective_budget_sats`/`remaining_sats` for corroboration; none-handling: `.get(...) or 0` and `or ""` guards throughout (e.g. L321, L369, L462) prevent None propagation; sql: N/A; never-call: N/A — no RPC calls, filesystem-only; concurrency: N/A — single-threaded]
- most_suspicious: L544-545 — `first_deleg = delegs_by_chan.get(scid); if first_deleg is not None and first_deleg <= t:` — the inv-e stage-ordering check treats a defib with no matching earlier fee_reduce delegation as indeterminate rather than a violation whenever the delegation-recording deploy epoch could explain the absence; acceptable because the era-aware bucketing is explicitly justified in the code comment (stage machine persists in DB, recording fires only on ENTRY transition), and the check records zero violations by design — correctly reporting "impossible to prove" rather than fabricating a verdict.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/loop_sweep_rebalance.py#1
- chunk_id: tools/audit/loop_sweep_rebalance.py#1
- summary: Phase-3 read-only rebalance+budget loop sweep. Ingests revenue-status/total-cost-budget/planner-history/segment-observation/rebalance-debug snapshots per node in one pass (`main`, L161-219), then runs LP-I1 budget reconciliation (L221-265), LP-I2 enforcement-on-writes (L283-290), LP-I4a/b defib<->diagnostic-row joins (L292-354), LP-I5 duplicate-row detection with a pre-fix carve-out (L356-370), LP-I6 segment-observation joins (L388-405), LP-I7 success-fee bound (L407-419), and LP-I8 suppressed-window checks (L421-432), printing a report plus anomaly material. No writes.
- invariants: [rounding: N/A — all comparisons are integer sats fields (`actual_fee_sats <= max_fee_sats`, L411) with no unit conversion; sign/off-by-one: `id_contiguity` (L372-386) correctly distinguishes a proven-absent row (`hi - lo == 1`) from an unprovable gap, used consistently by LP-I4a/b and LP-I6; caps: LP-I1a correctly nets out on-chain categories before comparing discretionary spend to `eff` (L235-240); none-handling: `eff is None or spent is None: continue` (L230-231) guards budget arithmetic before any comparison; sql: N/A, filesystem JSON only; never-call: N/A — no RPC calls anywhere in this sweep; concurrency: N/A — single-threaded]
- most_suspicious: L303-305 — `if cands: c4a.ok(); matched_action_row[(node, aid)] = cands[0][1]; continue` — this join takes the first candidate within the +/-900s window without marking it consumed, so the same diagnostic row could satisfy two nearby defib actions' existence check. Acceptable: this is an evidentiary "does some diagnostic row corroborate this action" existence test for a PASS/FLAG report, not a resource-allocation join requiring 1:1 exclusivity, and `matched_action_row` is not consumed elsewhere for any double-billing count.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/stress_concurrency.py#1
- chunk_id: tools/audit/stress_concurrency.py#1
- summary: Concurrency stress-harness setup and worker bodies (not the run orchestration). `StressHarness.setup()` (L200-250) wires a real `Database` on a temp SQLite file in WAL/autocommit mode and the real fee/rebalance/flow modules against a mock RPC, all on the calling (single) thread before any workers spawn. The `worker_*` reserve loops (L303-626) run on N threads each, calling the plugin's real spend-reservation RPC handlers and `Database` methods concurrently; `worker_invariant_sampler` reads both budget tables in one SELECT to avoid a torn read. Writes are entirely to the temp SQLite DB created in `setup()`, never to the hermes corpus or a live node.
- invariants: [rounding: N/A — sats amounts are integers throughout (`rng.randint(200, ...)`), no msat conversion; sign/off-by-one: N/A, no boundary/window arithmetic; caps: budget-limit values fetched fresh per-iteration via `mod._total_cost_budget_limit_provider()` with a fallback to `self.budget_sats` on exception (e.g. L363-367) so a transient provider failure degrades to a safe static ceiling; none-handling: every worker wraps its body in `try/except Exception: self.record_exception(...)` (e.g. L340-341) so a None/KeyError from real module code is captured as a diagnostic; sql: DB opened once in `setup()` with `journal_mode`/`isolation_level` explicitly asserted WAL/autocommit (L241-248) before any worker touches it; never-call: this chunk DOES call `mod.revenue_spend_reserve/release/settle` and `db.reserve_budget`/`reserve_spend` (on the action/mutation list) but exclusively against an isolated `tempfile.mkstemp` SQLite DB and a `MagicMock` RPC — the harness's documented purpose (soaking the real money-path handlers), not a read-only/Hermes task; concurrency: each worker's `live` tracker is a per-thread local (N independent instances, one per `wid`) so no cross-thread sharing; the only shared mutable state (`self.progress`/`self.op_log`) is lock-protected via `_progress_lock`/`_oplog_lock`]
- most_suspicious: L716-718 — `if stalled > timeout: self.deadlock = True; self.stop.set()` — `self.deadlock` is written by the watchdog thread with no lock; acceptable because it has exactly one writer (the watchdog) and is only read once, by the main thread, after `t.join()` completes for all workers in `_finalize` — a plain bool with single-writer/read-after-join semantics needs no lock under CPython.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/audit/stress_concurrency.py#2
- chunk_id: tools/audit/stress_concurrency.py#2
- summary: Tail of `worker_invariant_sampler` (violation capture, L681-708), `worker_watchdog` (L710-719), `StressHarness.run()` orchestration (installs `threading.excepthook`, spawns all worker/loop/sampler/watchdog threads, waits for the soak deadline, joins with a 10s timeout, restores the hook, L722-791), `_finalize()` (integrity check + temp-DB cleanup + report assembly, L793-842), and CLI plumbing (`build_arg_parser`, `_print_report`, `main`, L845-917). Writes are confined to the harness's own temp SQLite file (created and deleted within this run) and an optional `--json` report file.
- invariants: [rounding: N/A; sign/off-by-one: N/A; caps: the budget-violation capture (L681-704) compares `total = generic_reserved + rebalance_reserved` against `getattr(self.mod.config, "daily_budget_sats", ...)` — a different quantity than the per-call `effective_budget_sats` the workers reserve against, but in this fresh temp DB with no open/close spend the two coincide (fidelity note, not a coded defect); none-handling: `_finalize` wraps temp-file cleanup in `try/except OSError: pass` (L839-840); sql: `PRAGMA integrity_check` (L797-799) runs on a fresh separate connection AFTER all worker threads have been joined; never-call: N/A in this sub-range — no additional RPC calls beyond #1; concurrency: `threading.excepthook` monkey-patched (L743) before any thread is spawned and restored (L787) only after every thread is joined — correct set-before-spawn/restore-after-join bracketing]
- most_suspicious: L834-840 — the temp SQLite files are deleted (`os.unlink`) even when a worker failed to join within the 10s timeout, i.e. a straggler thread could still hold an open connection to a file being unlinked. Acceptable: this only occurs after the watchdog tripped or a join genuinely timed out (an already-failing run whose report already flags the condition), the `PRAGMA integrity_check` result is captured before this cleanup block, and the file is a disposable per-run temp DB — no already-reported result can be corrupted, only a same-process straggler's own eventual (silently dropped) I/O error.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/analyze_fee_tournament.py#1
- chunk_id: tools/analyze_fee_tournament.py#1
- summary: Read-only analysis library/CLI. `load_records` walks JSON files/dirs from disk (input only), `extract_metrics`/`rank_candidate_settings`/`segment_summaries`/`build_recommendations` are pure functions over in-memory dicts, `summarize` composes them. `main()` optionally writes `--json-out`/`--markdown-out` files via `write_text`; otherwise prints to stdout. No RPC/subprocess calls anywhere in this file.
- invariants: [rounding: all ratios/sums rounded via `round(x, 3..6)` before serialization, consistent; sign: no signed-quantity math, only counts/ppm/sats, all non-negative by construction; caps: `CANDIDATE_SETTINGS` deltas capped via `max(fee*ratio, min_ppm)` in `estimate_cycles_to_boundary`, correctly bounds step size; none-handling: every division guarded by `if total/attempts/successes else 0.0` ternaries (L341,L346-347,L466,L469-472); revenue_fee_ppm returns None safely via try/except; sql: N/A, no DB access; never-call: N/A, file performs no RPC/subprocess calls at all; concurrency: N/A, single-threaded stateless CLI, no shared/global mutable state]
- most_suspicious: L549 — `metrics = [m for m in metrics if not math.isnan(m.revenue_share)]` — `revenue_share` is always computed via a guarded ternary (`... if total_forwards else 0.0`, L169) so it can never actually be NaN; dead defensive code masking no real path to NaN, harmless.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/competitive_fee_tournament.py#1
- chunk_id: tools/competitive_fee_tournament.py#1
- summary: Polar-regtest lab driver. `run()` wraps `subprocess.run` for `docker exec`/`docker cp` against fixed Polar container names, parsing JSON stdout. Helpers wrap CLN (`lightning-cli`) and LND (`lncli`) calls. State writes: `push_hive_hints_datastore`/`clear_hive_hints_datastore` mutate CLN `datastore` key `["hive","hints"]`; `deploy_cl_hive`/`install_cl_hive_dependencies`/`start_cl_hive`/`disable_cl_hive` mutate the Polar `revenue-node` container and CLN hive-genesis state. All target names (`PAYER`, `REVENUE`, `COMPETITOR`, `SINK`) are hardcoded regtest containers, never mainnet.
- invariants: [rounding: N/A, no msat/sat arithmetic in this chunk; sign: N/A; caps: N/A; none-handling: `int_field()`/`parse_int` guard non-numeric values with try/except returning default; `node_policy`/`cln_direction_policy` return `{}` on no match rather than raising; sql: N/A; never-call: no `revenue-*` mutation RPCs called here — only `hive-status`/`hive-export-hints`/`datastore`/`deldatastore`/`listdatastore`/`plugin start|stop` and LND `updatechanpolicy`/`getchaninfo`/`listchannels`, none on the AGENTS.md forbidden action-RPC list, all targeting the regtest Polar lab by construction; concurrency: module-level global `_LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC` (L38) is mutated later with no lock — safe only because this CLI runs single-process/single-threaded]
- most_suspicious: L90 — `data.setdefault("ok", True)` — any zero-exit-code JSON response is stamped `ok=True` even if the payload represents a semantically negative CLN/LND result without an explicit `error` key; acceptable because every call site additionally checks `"error" not in result` via `rpc_result_ok()` or inspects specific fields before trusting the flag.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/competitive_fee_tournament.py#2
- chunk_id: tools/competitive_fee_tournament.py#2
- summary: `best_route` queries LND route quotes (read-only RPC). `pay_one` issues a real `addinvoice`/`payinvoice` against the Polar `SINK`/`PAYER` containers — a live pay/invoice action, but confined to the regtest lab. `force_fee_cycle` calls the CLN `revenue-fee-cycle` RPC and, on failure, stops/restarts the `cl-revenue-ops` plugin. `run_phase` orchestrates a full phase (policy set/verify, payments, forward counters) and writes `phase_<name>_<started>.json`. `main()` writes an aggregate `tournament_<started>.json`.
- invariants: [rounding: N/A, no fee-math — pure counters/JSON passthrough; sign: `max(0, revenue_forwards_after - revenue_forwards_before)` (L982-983) correctly floors possible counter resets/races at zero; caps: N/A; none-handling: `payment_succeeded` checks multiple result shapes (`status`, `stdout`, `payment_error`) before falling back to `ok`/`returncode`; sql: N/A; never-call: `force_fee_cycle` (L814) invokes `revenue-fee-cycle`, on the AGENTS.md forbidden action-RPC list "for read-only tests or Hermes tasks" — but this file is an explicitly `--execute`-gated Polar-lab tournament driver targeting only the regtest `revenue-node`, so it is in-scope deliberate tooling, not a violation; concurrency: relies on the same unlocked module-level monotonic global from #1, single-process only]
- most_suspicious: L820 — `stopped = cln(REVENUE, "plugin", "stop", plugin_path)` (followed by `plugin start`) — if `revenue-fee-cycle` merely errors transiently, `force_fee_cycle` silently falls back to restarting the entire plugin rather than surfacing the RPC failure; acceptable only because this fallback is gated behind explicit `--force-cycle-*` flags on the disposable regtest lab, but it is a heavy, opaque recovery path.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/fee_tournament_loop.py#1
- chunk_id: tools/fee_tournament_loop.py#1
- summary: Orchestrates repeated `long_fee_tournament`/`competitive_fee_tournament` runs and classifies results via `classify_summary` into `repeat_test`/`refine_tests`/`consider_algorithm_change`. `restore_payer_liquidity` issues real LND `addinvoice`/`payinvoice` between `PAYER` and `SINK` Polar containers to top up test liquidity. `run_iteration` writes `loop_plan.json`, `loop_result.json`/`loop_error.json`, `analysis.json`, `ANALYSIS.md`, `decision.json` per iteration directory; `main()` aggregates and writes `loop.json`/`REPORT.md`.
- invariants: [rounding: N/A, delegates fee math to `analyzer`/`tournament`; sign: `restore_payer_liquidity` computes `wanted = max(0, target - payer_local)`, `available = max(0, sink_local - reserve)`, `amount = min(wanted, available, max_restore)` (L258-260) — correctly floors and caps so a negative/over-reserve transfer can never be attempted; caps: `max_restore_sats` bounds the top-up via `min()`; none-handling: `classify_summary` coerces all totals via `float(totals.get(...) or 0.0)`/`int(... or 0)`; sql: N/A; never-call: no direct forbidden RPCs — mutation actions delegated to `tournament`/`long_fee_tournament`, same in-scope-tooling reasoning; concurrency: no shared global state introduced here; single-threaded sequential loop]
- most_suspicious: L668 — `if decision.get("action") in {"refine_tests", "consider_algorithm_change"}:` followed by an outer-loop re-check of `results[-1]` — the outer check is redundant with the inner `break`, a harmless double-guard rather than a behavioral bug.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/fee_tournament_loop.py#2
- chunk_id: tools/fee_tournament_loop.py#2
- summary: Tail of `main()`: writes `loop.json` (full run record) and `REPORT.md` (rendered markdown) to `out_dir`, then prints a small JSON summary to stdout. Purely a serialize-and-exit step; no RPC/subprocess calls, no computation.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: N/A, `loop`/`results` are always well-formed dicts/lists built earlier in `main()`; sql: N/A; never-call: N/A, no RPC calls in this span; concurrency: two sequential `write_text` calls to distinct files, no shared state, no race within this process]
- most_suspicious: L705 — `loop_path.write_text(json.dumps(loop, indent=2, sort_keys=True) + "\n", encoding="utf-8")` — serializes the entire accumulated `loop` structure in one call with no size guard; acceptable for a bounded local test-tooling loop where iteration counts are operator-supplied and small.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/long_fee_tournament.py#1
- chunk_id: tools/long_fee_tournament.py#1
- summary: Builds/executes long tournament matrices (`fixed_market`, `step_shock`, `adaptive_competitor`, `clboss_external`, `external` scenarios) as a shuffled `PlannedPhase` list. `run_plan` calls `tournament.ensure_cl_hive`/`disable_cl_hive`, resolves competitor ppm per phase, and drives `tournament.run_phase` (which performs live LND policy sets and CLN/LND payments against the Polar lab). `main()` writes `long_tournament_plan_*.json`/`.md` always, and `long_tournament_*.json` when `--execute` is passed. NOTE: contains a real falsy-`or` defect (see most_suspicious) escalated to the Phase-8 defect report.
- invariants: [rounding: N/A; sign: N/A; caps: `resolve_competitor_ppm` correctly clamps with `max(adaptive_min_ppm, min(adaptive_max_ppm, revenue_fee - adaptive_undercut_ppm))` (L279); none-handling: `current_revenue_fee_ppm` returns `None` on missing/non-numeric channel data and callers substitute `fallback_ppm`; sql: N/A; never-call: `run_plan` delegates to `tournament.run_phase` (can invoke `revenue-fee-cycle`) — same in-scope Polar-lab-tooling reasoning; concurrency: N/A, sequential phase loop, single-threaded]
- most_suspicious: L328 — `competitor_ppm=competitor_ppm or args.adaptive_fallback_ppm,` — a real bug: `resolve_competitor_ppm` can legitimately return `0` (e.g. `--fixed-ppms 0`, or `--adaptive-min-ppm 0`), and Python's falsy-`or` silently replaces that `0` with `args.adaptive_fallback_ppm` (default 150), driving a live LND competitor policy to the wrong ppm and invalidating the 0-ppm test scenario. NOT clean — escalated as a Phase-8 defect. Ships in-place only because it corrupts a REGTEST LAB scenario, not production funds; fix: `competitor_ppm if competitor_ppm is not None else args.adaptive_fallback_ppm`.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/hive_hints_truth_audit.py#1
- chunk_id: tools/hive_hints_truth_audit.py#1
- summary: Truth-audits exported `hive-export-hints` snapshots against producer-side ground truth and against `HiveHintAdapter` consumer effects (loaded via `importlib`, driven through an in-memory `_RpcStub`/`_PluginStub` — no live RPC in the adapter path). `collect_evidence()` performs live but exclusively read-only CLN RPCs (`hive-status`, `hive-members`, `hive-export-hints`, `hive-corridor-assignments`, `hive-yield-metrics`, `hive-traffic-intelligence`, `hive-fee-profiles`, `hive-expansion-recommendations`, `hive-network-metrics`, `hive-member-connectivity`, `listpeerchannels`, `listdatastore`) against a live Polar node via `tournament.cln`. `main()` writes `evidence.json`, `analysis.json`, `ANALYSIS.md` to a timestamped `results/` dir.
- invariants: [rounding: N/A, boolean/set comparisons only; sign: N/A; caps: N/A; none-handling: every cross-reference builds dicts defensively via `_as_list()`/`isinstance` checks before indexing; sql: N/A; never-call: `collect_evidence` calls only observability/read RPCs, none on the AGENTS.md forbidden list; the opt-in `--ensure-hive` path calls `tournament.ensure_cl_hive`, a plugin-lifecycle helper, not a forbidden action RPC; concurrency: N/A, single evidence-collection pass, no shared mutable state]
- most_suspicious: L256 — `if age_seconds > int(ttl_seconds or 900):` — `ttl_seconds` from `snapshot.get("ttl_seconds", 900)`; if a producer legitimately emitted `ttl_seconds=0` the `or 900` falsy-fallback would treat the snapshot as fresh for 900s instead of flagging it stale; accepted as non-blocking because the only observed producer TTL range is 60-7200s with no legal 0 value in the producer path, so the edge is currently unreachable — same falsy-vs-`None` class noted elsewhere this pass.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/hive_mcp_compat.py#1
- chunk_id: tools/hive_mcp_compat.py#1
- summary: Client-side regenerator for a deleted upstream cl-hive MCP server. `_read_git_file` shells out to `git -C <cl_hive_path> show <rev>:<path>` (fixed, module-level `rev`/paths, not attacker input) to restore pinned historical source files into a cache dir. `OBSERVABILITY_FUNCTION_REPLACEMENTS` holds full replacement function bodies (as source text) for a handful of handler functions, later spliced in via AST. `patch_observability_source` renames legacy RPC method-name string literals and swaps in the replacement bodies. No live-node RPC execution happens in this chunk.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: `_extract_plugin_methods`/`_read_allowlist` raise `RuntimeError` on missing files/invalid JSON rather than silently proceeding with empty state; sql: N/A; never-call: the embedded replacement handlers call only observability RPCs — `hive-status`, `getinfo`, `revenue-status`, `revenue-rebalance-debug` (summary_only), `listpeerchannels`, `listforwards`, `revenue-dashboard`, `revenue-spend-ledger`, `revenue-boltz-wallet` — none on the AGENTS.md forbidden list; concurrency: N/A, no shared state in this chunk]
- most_suspicious: L546 — `patched_source = patched_source.replace(f'"{legacy_method}"', f'"{current_method}"')` — a blind global string substitution across the entire pinned source file (not AST-scoped), which could in principle rewrite an unrelated string literal that happens to contain the same quoted substring; acceptable because the source revision is pinned (`DEFAULT_SOURCE_REV`) and the four legacy method names are namespaced RPC identifiers unlikely to collide.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/hive_mcp_compat.py#2
- chunk_id: tools/hive_mcp_compat.py#2
- summary: Builds a function call/RPC dependency graph over the compat server AST (`_function_dependency_graph`, `_resolve_function_dependencies` with memoization/cycle-guard), computes which MCP tools are supported under the current RPC allowlist (`_supported_tool_names`, fixed-point iteration), then rewrites the server source by line-span splicing (`prune_compat_server_source`) to drop unsupported tool blocks/handlers/helpers. `main()` fetches sources, patches, computes the allowlist, prunes, sets `HIVE_ALLOWED_METHODS` in the process environment, and finally `runpy.run_path(server_path, run_name="__main__")` — launches the generated server as this process.
- invariants: [rounding: N/A; sign: N/A; caps: N/A; none-handling: `_tool_handlers`/`_list_tool_blocks`/`_tool_handlers_span` all raise `RuntimeError` if the expected AST shape isn't found rather than corrupting the splice; sql: N/A; never-call: the pruning only ever *removes* tool surface based on the allowlist derived from the actual plugin method decorators — it cannot add new RPC surface, so it cannot introduce a forbidden-RPC-capable tool that wasn't already present upstream; concurrency: multiple concurrent invocations targeting the shared default `cache_dir` (`/tmp/hive-mcp-compat`) could interleave writes with another instance's `runpy.run_path` read — no file locking anywhere in this module]
- most_suspicious: L862 — `source_lines[start - 1 : end] = replacement.splitlines(keepends=True)` inside the reverse-sorted replacement loop — the pruned source is written and immediately executed via `runpy.run_path` (L929) with no `ast.parse()` sanity check on the generated output; if two computed spans ever overlapped due to an AST-offset edge case, the splice would silently produce invalid/mis-scoped Python. Acceptable today because the spans are computed from disjoint top-level AST constructs in a known pinned upstream layout, but there is no runtime guard against a future layout change breaking that disjointness.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/capex_planner_loop.py#1
- chunk_id: tools/capex_planner_loop.py#1
- summary: Fully synthetic scenario harness for `CapacityPlanner`/`CapexBudgetEngine` — every CLN RPC surface is a `MagicMock` (`plugin.rpc.feerates`, `listfunds`, `listpeerchannels`, etc.), so no live node is ever contacted. `FakeHiveHints` stands in for `modules/hive_hints.py`. Runs baseline EV/cycle scenarios plus hive A/B scenarios and asserts expected pass/fail booleans. `main()` writes `loop.json`/`ANALYSIS.md` under `results/capex-planner-loop-<timestamp>/` and best-effort maintains a `results/capex-planner-loop-latest` symlink.
- invariants: [rounding: `ev_sats` rounded to 3dp before serialization, consistent; sign: EV scenarios check `positive = ev > 0` against `expect_positive`, correct polarity; caps: delegates all real budget/ROI-hurdle enforcement to the modules under test; none-handling: `_planner()` explicitly configures mocks to `None` to exercise the "new peer fallback" path (L150-151), deliberate None-handling coverage; sql: N/A, `profitability.database.*` are all `MagicMock`; never-call: no live RPC of any kind — entire file operates on mocks, so the AGENTS.md forbidden-RPC list is structurally inapplicable; concurrency: N/A, sequential scenario list, single process]
- most_suspicious: L667-670 — `except FileNotFoundError: pass` / `except OSError: pass` around `latest.unlink()` and `latest.symlink_to(latest_target)` — both blocks swallow every `OSError`, so a real filesystem failure silently leaves `results/capex-planner-loop-latest` stale with no signal. Acceptable because this symlink is a pure convenience pointer — the timestamped `out_dir` artifacts remain the authoritative outputs regardless of symlink success.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/rebalance_capex_loop.py#1
- chunk_id: tools/rebalance_capex_loop.py#1
- summary: Defines the IterationAnalysis dataclass and CLN read helpers (_node_id, _peer_channel, _channel_local_sats/_capacity_sats/_summary). _pay_between builds an invoice, resolves a route, and executes sendpay/waitsendpay between Polar test nodes — a live payment sequence. prepare_pure_hive_corridor/prepare_intrahive_corridor call setchannel and drive payments to seed controlled channel imbalance. deploy_revenue_ops copies plugin files into a docker container. clear_hive_hints_datastore/disable_cl_hive/enable_cl_hive manage plugin lifecycle and CLN datastore. push_hive_hints_datastore/seed_hive_channel_peers write directly into cl-hive's sqlite DB inside the docker container via a parameterized inline python3 script.
- invariants: [rounding: msat->sat via `_parse_msat(...) // 1000` floor division, consistent codebase convention; sign/off-by-one: `_amount_to_target_local`/`_amount_to_reduce_local` clamp with `max(0, ...)` to prevent negative payment amounts; caps: capacity<=0 guarded to 0 in ratio/target helpers; none-handling: `_parse_msat` maps None->0, `_channel_summary` guards div-by-zero with ternary; sql: `seed_hive_channel_peers`'s embedded script uses `?`-parameterized sqlite3 execute, safe from injection; never-call: explicitly-scoped Polar validation loop (docstring "Run rebalance/capex validation loops in Polar"), not a read-only test or Hermes task, so live pay/setchannel/datastore RPC usage is within AGENTS.md's explicitly-scoped carve-out; concurrency: ad hoc sqlite3 connections against the live cl-hive DB file could race with the plugin's own connection, accepted for dev/test tooling]
- most_suspicious: L109 — `return int(value.rstrip("msat").strip() or "0")` — rstrip("msat") strips any trailing chars in the set {m,s,a,t} rather than the literal substring, but acceptable because CLN msat strings are always digit-prefixed and digits are never in that charset, so the numeric prefix is preserved.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/rebalance_capex_loop.py#2
- chunk_id: tools/rebalance_capex_loop.py#2
- summary: Continues clear_rebalance_cooldowns (deletes rows from revenue_ops.db in the container), then set_hive_hints_disabled/set_revenue_config_overrides (setconfig RPCs), restart_revenue_ops (plugin stop/start), maybe_drive_traffic (synthetic payments, writes traffic_<ts>.json), snapshot() (read-only RPC bundle), run_rebalance_cycle (calls the `revenue-rebalance-cycle` action RPC), and a block of pure-Python analysis helpers (_budget, _hive_pair_stats, _convergence_metrics, etc.) over already-captured JSON.
- invariants: [rounding: convergence fee_per_restored_sat rounded to 8dp for display only; sign/off-by-one: budget deltas computed consistently as after-before, reservation_leak = delta>0 correctly flags net reservation growth; caps: accounting_ok only enforced when successes>0 and fee_sats>0, correctly gating the spend/budget invariant; none-handling: `_budget`/`_channel_budget_total`/`_hive_count` wrap int() coercion in try/except with 0 fallback, `_as_float` returns None on failure with downstream `is not None` checks; sql: none in this chunk; never-call: `run_rebalance_cycle` invokes the listed action RPC `revenue-rebalance-cycle`, compliant via the same explicitly-scoped Polar validation-loop carve-out as chunk 1; concurrency: sequential payments with fixed sleep, no shared-state races beyond the Polar node's own handling]
- most_suspicious: L866 — `"competitor_forwards_delta": max(0, after_competitor - before_competitor),` — clamps a possibly-negative delta to zero; acceptable because this field is purely informational context in traffic_<ts>.json, not used in pass/fail accounting.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/rebalance_capex_loop.py#3
- chunk_id: tools/rebalance_capex_loop.py#3
- summary: Finishes analyze_iteration's decision tree, write_analysis renders ANALYSIS.md. main() defines the full CLI, orchestrates hive/pure-hive/intrahive setup, tuning overrides, optional plugin restart, then runs the per-iteration loop (perturb, clear cooldowns, drive traffic, snapshot before/after, run_rebalance_cycle, analyze, write per-iteration JSON). Writes setup.json/loop.json/ANALYSIS.md and a best-effort `results/rebalance-capex-loop-latest` symlink.
- invariants: [rounding: N/A, only display formatting; sign/off-by-one: `range(1, args.iterations + 1)` correctly produces 1-indexed iterations matching `iteration_{iteration:03d}` dir naming; caps: setup_errors gating (L1763-1783) aborts before driving live traffic/rebalance on a mis-configured node; none-handling: tuning override args default to None and are only added to overrides when `is not None`; sql: N/A, delegated to chunk1; never-call: main() ultimately drives the same explicitly-scoped Polar action RPCs audited in chunks 1-2, targets are hardcoded Polar node aliases; concurrency: non-atomic unlink+symlink for the "latest" pointer guarded only by a broad except, could race under concurrent invocation]
- most_suspicious: L1877 — `except OSError:` — broad exception swallow around the `results/rebalance-capex-loop-latest` symlink update; acceptable because the authoritative timestamped `out_dir` artifacts are already written before this block, so a failed symlink refresh cannot lose or corrupt real output.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/rebalance_convergence_matrix.py#1
- chunk_id: tools/rebalance_convergence_matrix.py#1
- summary: Sweeps `rebalance_capex_loop.py` across topology x hold_margin combinations via `subprocess.run` (no shell=True, check=False, returncode inspected explicitly), reads each run's `loop.json`/`cycle.json`, summarizes convergence pass/fail and score headroom (`summarize_loop`), computes a recommended hold margin with an explicit safety-guard heuristic (`recommend_hold_margin`), and writes `matrix.json`/`matrix_run.json`/`ANALYSIS.md`. This script itself performs no CLN/RPC calls — all mutation happens in the spawned child processes.
- invariants: [rounding: `round(fees / restored, 8)` and `round(min(scores), 6)` guarded by truthiness checks before division; sign/off-by-one: `guarded_cap = max(0.0, failing[0] - 0.05)` correctly biases the recommendation strictly below the lowest known-failing margin; caps: explicit 0.05 safety margin subtracted from both the failing-margin floor and the observed score floor before recommending; none-handling: `min_score`/`score_guard_cap`/headroom all explicit-checked `is not None` before arithmetic; sql: N/A, no database access; never-call: script itself calls no RPCs, only shells out to the already-audited explicitly-scoped `rebalance_capex_loop.py`; concurrency: strictly sequential run loop, no threading]
- most_suspicious: L295 — `parser.add_argument("--keep-going", action="store_true", default=True)` — a store_true flag whose default is already True, so it can never be toggled off from the CLI; a minor CLI-ergonomics wart, not a correctness defect, since the loop records and reports every run's summary regardless.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/revenue_validation_collect.py#1
- chunk_id: tools/revenue_validation_collect.py#1
- summary: Runs a fixed list of read-only JSON RPC commands (JSON_COMMANDS) plus an optional text log-extract command against each configured node over remote transport, writes each command's output to per-node JSON files and the raw log to rollback-watch.log, builds a trend_record via multi-path fallback extraction (_extract_first), appends it to a per-node JSONL trends file, and writes a daily manifest JSON.
- invariants: [rounding: `_coerce_int` divides msat by 1000 via floor `//`, and literal `"msat" in compact` substring check avoids the charset-trim ambiguity seen elsewhere; sign/off-by-one: N/A, `_days_since_t0` is a plain date subtraction; caps: N/A, pure data extraction; none-handling: `_coerce_int` returns None on every unparseable branch, `_extract_first` returns the first non-None coercion across fallback JSON paths; sql: N/A; never-call: JSON_COMMANDS contains only read/list RPCs (revenue-dashboard, revenue-report summary, revenue-profitability, revenue-status, `revenue-config get` not `set`, listforwards/listpays/listpeerchannels, hive-members, feerates) — none on the AGENTS.md forbidden list, fully compliant with the Hermes read-only mandate; concurrency: nodes processed sequentially, trends JSONL appended per call, no internal parallelism]
- most_suspicious: L323 — `default=date.today().isoformat(),` — argparse default evaluated using local system clock; acceptable since this is a one-shot CLI invoked fresh per run (not a long-lived daemon), so no staleness risk.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/revenue_validation_common.py#1
- chunk_id: tools/revenue_validation_common.py#1
- summary: Shared utility module — RunResult dataclass, load_config (YAML load), build_node_command (transport + remote command string), path helpers (dated_results_dir/node_day_dir/trends_file), and write_json_file/write_json (mkdir parents, indented sorted JSON with trailing newline). No RPC calls, no destructive operations.
- invariants: [rounding: N/A; sign/off-by-one: N/A; caps: N/A; none-handling: `load_config` returns `{}` for an empty YAML file, avoiding a None config propagating to callers; sql: N/A; never-call: N/A, no RPC calls in this module; concurrency: `write_json_file` performs a plain non-atomic write_text (no temp+rename), acceptable for this single-operator batch tooling]
- most_suspicious: L23 — `return yaml.safe_load(f) or {}` — uses `yaml.safe_load` (not `yaml.load`), correctly avoiding arbitrary object deserialization from a config file; the most security-relevant line in the file, implemented safely.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/revenue_validation_daily.py#1
- chunk_id: tools/revenue_validation_daily.py#1
- summary: Thin orchestrator invoking revenue_validation_collect.main, revenue_validation_watch.main, revenue_validation_report.main in sequence with the same --config/--date args, collecting each return code and returning 1 if any failed. No direct RPC/file I/O beyond delegating to the three sub-modules.
- invariants: [rounding: N/A; sign/off-by-one: N/A; caps: N/A; none-handling: N/A, each sub-main handles its own missing-file defaults; sql: N/A; never-call: delegates only to collect (read-only RPCs, audited) and watch/report (pure file-based analyzers, no RPCs) — fully compliant; concurrency: N/A, single-process sequential execution]
- most_suspicious: L46 — `codes = [` — begins the unconditional sequential execution of all three phases (L46-50) with no short-circuit on an earlier phase's failure; acceptable because both watch.py and report.py tolerate missing input files via `{}`/`[]` defaults rather than crashing, and the aggregate non-zero exit code still correctly signals failure.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/revenue_validation_report.py#1
- chunk_id: tools/revenue_validation_report.py#1
- summary: Pure read-and-render module — reads saved daily JSON/JSONL evidence written by earlier phases, computes T+14/T+28 checkpoint summaries (trend deltas, activation-pattern log counts, hive-fee-status, pre/post fee-window comparisons), and writes two Markdown reports only when a node crosses the relevant day threshold and no report with that label already exists (`_report_exists` glob guard). No RPC calls anywhere.
- invariants: [rounding: window fee/cost sums delegate to watch._coerce_int's floor msat//1000; pre_avg/post_avg are plain floats for display only; sign/off-by-one: half-open `[start, end)` window checks used consistently, `post_days = max(..., 1)` guards zero-day divisor; caps: N/A, this module only reports on values governed elsewhere; none-handling: `_read_json`/`_read_jsonl` return `{}`/`[]` for missing files, `_format_int` maps None->"n/a", `_find_rule` result always defaulted with `or {}`; sql: N/A; never-call: no RPC calls anywhere, purely reads evidence and writes Markdown; concurrency: check-then-write TOCTOU on `_report_exists`+write, acceptable for a once-daily cron-style tool]
- most_suspicious: L504 — `if _due_for_checkpoint(latest_rows, 14) and not _report_exists(reports_root, "t14"):` — `_due_for_checkpoint` triggers on **any** node reaching the threshold, generating one fleet-wide T14 report even if a sibling node is still early; acceptable because each per-node table row still reports that node's own accurate days_since_t0, and the report text calls out the small-fleet caveat explicitly.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/revenue_validation_watch.py#1
- chunk_id: tools/revenue_validation_watch.py#1
- summary: Pure read-and-evaluate rollback-watch module — reads saved daily JSON snapshots and log extracts, runs a fixed battery of red/yellow/green threshold checks purely over already-collected data, aggregates highest_severity per node/fleet, and writes findings JSON via common.write_json. No RPC calls at all.
- invariants: [rounding: `round((success_count/attempt_count)*100, 1)` guarded by an explicit zero-attempt check before division; `_coerce_int` uses floor `// 1000` for msat, consistent; sign/off-by-one: half-open `[start, end)` windows reused consistently, `max(..., 1)` divisor guard mirrors report.py; caps: `severity` only evaluated against `floor_pct` when `attempt_count` is truthy, avoiding false-red on zero-attempt days; none-handling: `_coerce_int` returns None on every unparseable branch, all call sites check or default; `_config_value` returns None on missing keys; sql: N/A; never-call: no RPC calls anywhere, strictly reads saved evidence, fully compliant; concurrency: N/A, single-process single-pass evaluation]
- most_suspicious: L301 — `drop_observed = post_window_complete and pre_avg > 0 and post_avg < pre_avg * (1 - drop_pct / 100)` — `drop_pct` has no upper-bound validation so a misconfigured value >100 would make the threshold negative; acceptable because `drop_pct` is operator-controlled config (not RPC/attacker input) and `post_window_complete` additionally gates the check from firing on an incomplete window.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/router_v3_safety_monitor.py#1
- chunk_id: tools/router_v3_safety_monitor.py#1
- summary: Phase-3 rollout A/B safety monitor — parses a CLN log file for REBAL_PICK/REBAL_SKIP lines tagged router=v2|v3 (regex, read-only file access), computes a picks/(picks+skips) success rate per router version, and if v3 regresses beyond --threshold below v2's baseline, calls `lightning-cli -k setconfig config=revenue-ops-rebalance-router val=v2` via subprocess (skipped under --dry-run). The only file in this batch that performs a genuine live production config mutation by design.
- invariants: [rounding: N/A, plain float ratios for threshold comparison and percentage display only; sign/off-by-one: `v3_success_rate < v2_success_rate - threshold` correctly triggers rollback only when v3 is worse by more than threshold; caps: N/A, threshold is a comparison margin not a budget cap; none-handling: `.get("v2", {"picks":0,"skips":0})` zero-defaults missing router versions, explicit `v2_total==0 or v3_total==0` guard prevents ZeroDivisionError; sql: N/A; never-call: calls CLN core `setconfig` (not an AGENTS.md-listed `revenue-*` action RPC or Boltz/pay/open/close RPC) as its documented, intended operational function — AGENTS.md's action-RPC restriction is scoped to read-only tests or Hermes tasks, which this monitor is neither, so this is not a violation; concurrency: N/A, single invocation single log-parse pass]
- most_suspicious: L87 — `subprocess.run(cmd, check=True)` — a failed lightning-cli call raises an uncaught CalledProcessError out of main(), crashing with a traceback rather than a clean error; acceptable, and arguably the safer failure mode, because a loud crash on a failed safety-critical rollback is preferable to a silently-swallowed failure that leaves the operator believing the rollback succeeded.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/perf/cycle_driver.py#1
- chunk_id: tools/perf/cycle_driver.py#1
- summary: Read-only synthetic test-fixture module — builds a 36-channel fake listpeerchannels/gossip payload, wires a MagicMock CLN RPC whose `.call()` side effect recognizes only a small enumerated set of read-only diagnostic RPC methods, and build_stack(database) constructs the real PolicyManager/ChannelProfitabilityAnalyzer/FeeController/EVRebalancer/RebalanceEngine objects against this mock RPC and a caller-supplied database. No file writes, no live RPC calls, no network I/O.
- invariants: [rounding: N/A, no sat/msat accounting in this fixture; sign/off-by-one: synthetic test-data generation only; caps: N/A; none-handling: N/A, all payloads internally generated; sql: N/A, delegates to the real Database/module constructors audited elsewhere; never-call: `.call.side_effect` only returns canned data for a hardcoded allowlist of read-only method names, anything else returns a generic MagicMock/empty dict rather than a real RPC — no live or action RPC reachable by construction; concurrency: N/A, single-threaded object construction]
- most_suspicious: L117 — `return {}` — fallback for any `.call()` method name not in the explicit allowlist returns an empty dict rather than raising, slightly understating the docstring's "any path the synthetic RPC cannot satisfy raises" claim; acceptable because the enumerated methods are exactly the diagnostic endpoints the constructors probe, and genuinely unhandled RPC calls still surface as an uninitialized MagicMock that build_stack's own try/except converts into a None stack entry.
- auditor: fable-phase8  date: 2026-07-02

---

### tools/perf/profile_cycles.py#1
- chunk_id: tools/perf/profile_cycles.py#1
- summary: Performance-profiling CLI. When --db is supplied it copies the real DB to a private tempdir before opening (never touches the original), honoring the documented read-only contract; otherwise seeds a fresh synthetic SQLite DB to production-scale row counts via the real Database schema, using batched executemany inserts wrapped in explicit BEGIN/COMMIT per table. Times and cProfiles DB read-query benchmarks plus EXPLAIN QUERY PLAN, best-effort profiles cycle methods via cycle_driver.build_stack, and optionally writes a Markdown report.
- invariants: [rounding: N/A for DB writes, synthetic sat/msat values are raw random integers for load-testing only; sign/off-by-one: plain range() loops control fixture volume only; caps: N/A, benchmark/profiling tool not a budget-enforcement path; none-handling: every benchmarked call wrapped in try/except Exception recording an error row rather than crashing the whole run; sql: seed_synthetic_db wraps each table's bulk insert in explicit BEGIN/COMMIT, with a nested try/except sqlite3.OperationalError guarding the best-effort peer_connection_history ROLLBACK — correct transaction boundaries; never-call: no CLN/revenue RPC invoked anywhere, only real Database SQL plus MagicMock-backed module code (read-only canned responses); concurrency: N/A, single-process sequential benchmarking, process-private tempdir]
- most_suspicious: L461 — `shutil.copy(args.db, dst)` — a plain filesystem copy (not a SQLite-level backup API) of the source DB before opening; if the source is being actively written at the exact moment of copy the snapshot could be captured mid-write, but acceptable because SQLite's default journal mode leaves the main DB file always valid between transactions, so worst case is stale committed data, not corruption of the real DB, which this script never opens for writing.
- auditor: fable-phase8  date: 2026-07-02

---

### scripts/clean-local.sh#1
- chunk_id: scripts/clean-local.sh#1
- summary: Bash cleanup utility with `set -euo pipefail`. Parses --apply/--artifacts/--heavy/--help flags (default dry-run, safe by default), cd's to the git repo root, builds a whitelist array of relative paths (hardcoded core list plus a find-discovered set of __pycache__ dirs excluding .git/.venv/.worktrees/vendor, optionally extended with specific results/<prefix>-* artifact dirs and .venv/.worktrees/vendor under --heavy), deduplicates, then either prints "would remove" or executes `rm -rf -- "$path"` per existing path.
- invariants: [rounding: N/A; sign/off-by-one: N/A; caps: N/A; none-handling: `[[ -e "$path" ]] || continue` guards every deletion candidate, silently skipping stale whitelist entries despite `set -e`; sql: N/A; never-call: N/A, no CLN/RPC calls in this script; concurrency: no locking around the rm -rf loop, a concurrent writer into a targeted directory could race, accepted for single-operator/CI tooling]
- most_suspicious: L89 — `rm -rf -- "$path"` — the sole destructive line, acceptable because it only fires under explicit `--apply` (default is dry-run), every `$path` is either a fixed literal or comes from a `find` scoped to `__pycache__` names or known `results/<prefix>-*` dirs with `.git`/`.venv`/`.worktrees`/`vendor` pruned, the script `cd`s to repo root first so relative paths resolve inside the repository, and `--` blocks flag injection — no free-form user-supplied path reaches this line.
- auditor: fable-phase8  date: 2026-07-02

### modules/rebalance_engine_v2.py#10
- chunk_id: modules/rebalance_engine_v2.py#10
- blob: 140f886e678a
- summary: Tail of `_run_cycle_locked` as_completed timeout handling (3596-3615): on cycle-timeout, a still-cancellable future is cancelled and recorded as an `executor_timeout_cancelled` ExecutionResult (fee_sats/msat=0, amount from the pair); an un-cancellable still-running worker is logged "will finish bookkeeping asynchronously" and left to complete; then `_cache_cycle_result(result)` + return. State writes: appends to result.executions; caches the cycle result.
- checklist: [rounding: OK — fee 0 on cancelled, amount int-coerced; sign: OK; money-path: the orphan (un-cancellable) worker is the P4-008 double-pay window — covered by the in-flight-destination guard (_register_inflight_dest in _execute_pair -> finally-unregister -> find_candidates filter), so the next cycle cannot re-select the same dest; caps: N/A; none-handling: getattr(pair,'amount_sats',0) guarded; concurrency: runs under the single-flight _cycle_lock; no new lock]
- most_suspicious: L3607-3611 "worker will finish bookkeeping asynchronously" — the abandoned-orphan path; acceptable because P4-008's in-flight-destination guard prevents a second payment to that dest while the orphan settles, and P4-007 holds its reservation on pending. Verified clean at HEAD by the Phase-4 re-refutation passes.
- auditor: closure-reconcile  date: 2026-07-02
