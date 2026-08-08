# HISTORICAL CONTRACT - RETIRED IN VERSION 3.0.0

This document records the removed implementation at its cited commit. The module, RPCs, options, threads, and execution authority described below are absent from the current plugin. It is not an operator guide.

# Intent Contract: modules/lnplus_swaps.py

Authored 2026-07-08 from code at commit 7eadaf9. Structure follows the capacity_planner.md
sibling; anchors are function names (not line numbers) throughout, per the docs-remediation
brief. This is the only plugin module family that previously had no contract doc in this
directory.

## 1. Purpose

`modules/lnplus_swaps.py` automates joining [lightningnetwork.plus](https://lightningnetwork.plus)
(LN+) liquidity swap rings — a ring of nodes where each participant opens one channel to the
next and receives one in return, so an outbound open buys an equal-capacity inbound channel.
This is **join-only**: the plugin applies to swaps other operators posted; it never creates one.
Three collaborators are wired together in `cl-revenue-ops.py`:

- `LNPlusClient` — a stdlib-`urllib` HTTPS client for the LN+ REST API, authenticated via CLN
  `signmessage` (challenge/response). Read methods: `get_applicable_swaps()`, `get_swap()`,
  `get_my_swaps()`, `get_notifications()`. Mutating methods: `create_application()`,
  `delete_application()`, `complete_application()`, `mark_read_notifications()`,
  `create_rating()`.
- `SwapEvaluator` — the pre-application gate chain (spec gates 0-9), run once per cycle from
  `run_cycle()` (invoked by the capacity planner — see `docs/audit/contracts/capacity_planner.md`
  CP-I16). At most one application per cycle.
- `SwapLifecycle` — the obligations watcher / state machine (spec gates 10-14), run once per
  `revenue-ops-lnplus-watcher-interval` seconds from `run_watcher_once()` →
  `_run_watcher_once_locked()`. Drives every step after application: connect, `fundchannel`,
  `complete_application`, activation (`no_close` protection on both sides of the contract),
  mid-contract vanish detection, and finally rating + release once the contract's `ends_at`
  passes.

Applying to a filled swap slot is an **irreversible commitment**: once the last slot fills, a
48-hour clock starts to open the assigned channel (a missed deadline trips the circuit breaker —
see LN-I2). Every gate lives before `create_application()`; everything after only exists to
execute an already-committed obligation safely.

## 2. Inputs / Outputs

**RPC exposed (via main plugin, cl-revenue-ops.py):** `revenue-lnplus-status` →
`SwapLifecycle.get_status()`, `revenue-lnplus-breaker-clear` → `SwapLifecycle.clear_breaker()`,
`revenue-lnplus-abandon <swap_id>` (marks the local row failed and trips the breaker — a
deliberate defection, last resort only), `revenue-lnplus-backfill` →
`SwapLifecycle.backfill_from_lnplus()` (operator remedy for swaps applied/opened/settled manually
on the LN+ website; safe to run repeatedly, see LN-I6).

**External HTTPS API (LN+, `https://lightningnetwork.plus/api/2`):** consumed entirely through
`LNPlusClient`. Every response is a bounded (`_MAX_RESPONSE_BYTES` = 1MB) untrusted payload —
pubkeys are re-validated against `_valid_pubkey()`, connect addresses against
`_valid_connect_addr()`, and timestamps parsed defensively via `_parse_ts()` (rejects a
TZ-less ISO string being interpreted in the node's local timezone — LN+ deadlines are UTC)
before any value reaches an RPC call or gets written to the database.

**CLN RPC consumed:** `signmessage` (auth challenge/response, inside `LNPlusClient._auth_params()`),
`getinfo` (own-node check in `SwapEvaluator._check_participants()`), `feerates` (apply-time
ceiling gate `SwapEvaluator._feerate_ok()`; open-time feerate escalation in
`SwapLifecycle._execute_swap_open()`), `listfunds` (confirmed-funds gate in
`SwapEvaluator._select_and_apply()`; existing-channel/idempotency checks), `listpeerchannels`
(duplicate-peer gate `SwapEvaluator._check_existing_channel()`; open idempotency and mid-contract
vanish detection in `SwapLifecycle`), `connect` + `fundchannel`
(`SwapLifecycle._execute_swap_open()`).

**Database (modules/database.py, `lnplus_swaps` / `lnplus_peers` tables):**
`lnplus_record_swap`, `lnplus_update_swap`, `lnplus_get_swap`, `lnplus_get_swaps_by_status`,
`lnplus_inflight_swaps`, `lnplus_reserved_sats` (consumed by the capacity planner and the Boltz
auto-cycle's on-chain-sats calculation — see the sibling contracts), `lnplus_bump_peer`,
`lnplus_get_peer`, `lnplus_prune_terminal`; plus the generic spend-reservation rail
(`reserve_spend` / `mark_spend_reservation_spent` / `release_spend_reservation`) and
`record_planner_action` / `update_planner_action` for the shared planner-action audit trail.
`config_overrides` doubles as ad-hoc persistent flags for this module: `_BREAKER_KEY`
(`_lnplus_breaker`) and `_BACKFILL_FLAG` (`_lnplus_backfill_done`).

**Modules consumed:** the capacity planner (`self._planner`, injected into `SwapEvaluator`) for
`_calculate_open_ev()` (both the outbound EV and, reused as a conservative corridor-value proxy,
the inbound credit), `_estimate_open_cost()`, `_unified_reserve_budget_params()`, and
`_score_candidate()` (planner reputation veto, gate 5); `PolicyManager` (injected into
`SwapLifecycle`) for the `no_close` tag (`add_tag`/`remove_tag`/`get_policy`, see LN-I3); the
`hive_hints` adapter (optional) for `get_lnplus_swap_hints()` (EV bias / duplicate-peer-veto
bypass — see `docs/contracts/HIVE_HINTS_CONTRACT.md` "LN+ Swap Hints") and hive-membership
detection (fleet/hive participants are fully trusted — see LN-I7); an `ignore_peer_fn` callback
for filing a negative-outcome ignore on a defecting counterparty.

## 3. Invariants

- **LN-I1 (serialization)** — At most one swap is ever in flight per node. `run_cycle()` checks
  `SwapLifecycle.has_inflight()` (backed by `database.lnplus_inflight_swaps()`) before every
  cycle, ahead of even fetching applicable swaps from the LN+ API, so an outage or a tripped
  breaker can never queue up a second commitment. There is no code path that submits a second
  `create_application()` while a prior application is applied, opening, or active.
- **LN-I2 (breaker one-strike)** — `SwapLifecycle.trip_breaker(reason)` persists a `config_overrides`
  row keyed `_lnplus_breaker`; `breaker_tripped()` / `run_cycle()` check it before every new
  application. The breaker preserves the FIRST trip reason: a second `trip_breaker()` call while
  already tripped only logs (at error level) and returns — it never overwrites the original
  diagnostic. Trip triggers: a missed 48-hour open deadline
  (`_maybe_trip_deadline_miss()`, called from every `_execute_swap_open()` failure path) and a
  reconcile divergence between the local ledger and what LN+ reports for an in-flight swap
  (`_reconcile()`). The breaker only blocks *new* applications — obligations already in flight
  (an open in progress, an active contract) are still driven to completion by the watcher
  regardless of breaker state (`run_watcher_once()` does not check it). Clearing is always an
  explicit operator action (`clear_breaker()` via `revenue-lnplus-breaker-clear`); it never
  clears itself.
- **LN-I3 (both-side no_close protection)** — `_activate()` calls `_protect_peer_no_close()`
  twice: once for `outbound_peer` (flag column `tag_added`) and once for `incoming_peer` (flag
  column `incoming_tag_added`) — the LN+ agreement binds both the channel we opened and the
  channel opened to us, and closing either mid-contract is a defection. `_protect_peer_no_close()`
  records whether *this* call actually added the `no_close` tag (vs. finding it already present
  from an operator or another contract) so release never clobbers a tag it didn't add.
  `_release_no_close_if_ours()` (called from `_finalize()`) only removes the tag when this row's
  flag is 1 AND no other `active` row still references the same peer in either contract role —
  a peer shared across two overlapping swap contracts (or a fleet-dedup edge case) keeps its
  protection until the last contract referencing it ends.
- **LN-I4 (intent-first writes)** — Every irreversible external call is preceded by a local
  ledger write recording the *intent*, so a crash between the write and the call (or between the
  call and its result) leaves a locally-visible, retriable state rather than a silent gap:
  `_select_and_apply()` writes `lnplus_record_swap(..., "applied", ...)` before
  `create_application()`; `_execute_swap_open()` writes `status="opening"` before `connect`/
  `fundchannel`; `_activate()` persists the authoritative contract terms (`ends_at`,
  `incoming_peer`) before applying `no_close` protection, and only marks `status="active"` after
  protection is applied (comments in the code label these steps "Intent:" / "Outcome:"
  explicitly).
- **LN-I5 (reserve/settle rail)** — `_execute_swap_open()` reserves the estimated open cost via
  `database.reserve_spend()` (category `channel_open`, subcategory `lnplus_swap`) immediately
  before `fundchannel`, using a reservation id unique per attempt
  (`lnplus-open-{sid}-{timestamp}`) so a released reservation from a prior failed attempt can
  never block a retry. On any pre-fundchannel failure the reservation is released
  (`_release_swap_open_reservation()`, best-effort, logged-and-swallowed since it was never a
  committed spend). On a successful `fundchannel`, `_settle_swap_open_reservation()` mirrors
  `capacity_planner._settle_capex_reservation()`'s loud/bounded-retry semantics: it retries
  `mark_spend_reservation_spent()` up to 3 times and, on persistent failure, logs LOUDLY at error
  level and leaves the reservation **active** rather than releasing it — the on-chain fee is
  already committed, so it must stay counted against the unified budget even if the settle write
  keeps failing.
- **LN-I6 (backfill idempotency)** — `backfill_from_lnplus()` adopts LN+ account state accumulated
  before this automation existed (or from deliberate manual operator action afterward) into the
  local ledger. The common rule across `_backfill_pending()` / `_backfill_opening()` /
  `_backfill_completed()` is an unconditional skip if a local row for that `swap_id` already
  exists (`lnplus_record_swap` is `INSERT OR REPLACE` and would otherwise clobber
  automation-owned state or resurrect a terminal row) — this makes the whole method idempotent
  and safe to call any number of times, whether triggered automatically (once, gated by the
  `_BACKFILL_FLAG` config-override flag checked under double-checked locking in `_reconcile()`)
  or manually via `revenue-lnplus-backfill` (which relies purely on the per-swap skip, not the
  flag).
- **LN-I7 (fleet trust, fail-closed rank floor)** — Fleet/hive participants (identity = the
  `lnplus_fleet_pubkeys` config CSV union a live hive-membership check,
  `_is_fleet_participant()` / `_is_hive_fleet_member()`) are fully trusted in
  `_check_participants()`: they skip the positive-ratings floor, negative-ratio ceiling, and rank
  floor, and count as reliability 1.0 (no Tor discount) in `_swap_ev()`'s EV computation. Every
  non-fleet participant must independently clear `lnplus_min_peer_positive_ratings`, a negative
  ratio ceiling, and `lnplus_min_peer_rank` (`_check_participants()`) — a **missing or zero rank
  fails closed** (treated as below the floor, not as a pass). Dual (2-participant) swaps are
  rejected outright (`_filter_swap()`, `lnplus_min_participants` default 3); among multiple
  qualifying swaps of equal EV, the smaller ring wins the tie-break
  (`_select_and_apply()`'s sort key).
- **LN-I8 (EV double-count fix)** — `_swap_ev()`'s `outbound_ev` comes from
  `capacity_planner._calculate_open_ev()`, which already nets out open+close on-chain costs
  internally. `_select_and_apply()`'s funds/capex gates use `open_cost` unmodified for their own
  purpose (confirming the swap's upfront on-chain commitment is affordable), but the EV value
  itself is never additionally reduced by `open_cost` a second time — matching how regular
  planner candidates are ranked (their `_planned_ev` is `_calculate_open_ev()`'s result used
  as-is), so LN+ swaps are not unfairly penalized against regular opens in the unified EV
  ranking.
- **LN-I9 (hints bias, never bypass, except one named exception)** — An optional
  `lnplus_swap_hints` section (see `docs/contracts/HIVE_HINTS_CONTRACT.md`) can multiplicatively
  bias the EV of the assigned outbound peer (`_lnplus_hint_multiplier()`, clamped
  `[0.8, 1.5]`, applied in `_swap_ev()`) and, for an `allow_duplicate` action only, skip the
  duplicate-peer veto in `_check_existing_channel()` — the single named exception where a hint is
  allowed to bypass a gate rather than merely bias a score. No other gate (peer quality, rank
  floor, feerate ceiling, funds, capex budget, serialization, breaker) is hint-influenced.

## 4. Config surface

Thirteen runtime controls, all under `PUBLIC_RUNTIME_KEYS` in `modules/config.py` (settable via
`revenue-config`, and refreshed live from `setconfig`/config-file on the dynamic-config loop
subject to the same DB-override-precedence rule as every other runtime key — see README
"revenue-config: actions and override precedence"):
`lnplus_swaps_enabled` (master switch), `lnplus_execute_applications` (false = recommendation-only;
gates still evaluated and logged), `lnplus_swap_preference_margin` (regular-open EV must beat swap
EV by this fraction to win the slot), `lnplus_max_duration_months` / `lnplus_min_participants` /
`lnplus_max_participants`, `lnplus_min_peer_positive_ratings` / `lnplus_min_peer_rank`,
`lnplus_apply_feerate_ceiling`, `lnplus_pending_timeout_days`, `lnplus_inbound_credit_factor`,
`lnplus_fleet_pubkeys`, `lnplus_watcher_interval`. See `config/cl-revenue-ops.conf.full` for
defaults and one-line descriptions of each.

## 5. RPCs

| Command | Backing method | Notes |
|---|---|---|
| `revenue-lnplus-status` | `SwapLifecycle.get_status()` | Breaker state, in-flight swap, active/recently-ended/recently-failed contracts, backfill-done flag, last watcher pass, recent notifications. |
| `revenue-lnplus-breaker-clear` | `SwapLifecycle.clear_breaker()` | Explicit operator acknowledgment; the only way the breaker clears (LN-I2). |
| `revenue-lnplus-abandon <swap_id>` | marks the local row `failed`, trips the breaker, best-effort `delete_application()` | Emergency-only: a deliberate defection on our side, will draw a negative rating. |
| `revenue-lnplus-backfill` | `SwapLifecycle.backfill_from_lnplus()` | Operator remedy for pre-existing/manually-managed LN+ swaps; idempotent (LN-I6), safe to run repeatedly. |

## 6. Uncertainties

- No dedicated hypotheses/outcome-tracking exists yet for this module (unlike the Tier-1
  capacity_planner.md / fee_controller.md siblings) — swap-EV realized-vs-forecast accuracy is
  untested against production data.
- `_swap_ev()`'s `inbound_credit` reuses `_calculate_open_ev()` as a corridor-value proxy for the
  incoming channel (conservative by design, per the code comment), but this has not been
  cross-checked against how the inbound channel's value is actually realized once the contract
  ends and it reverts to normal planner management.
- The interaction between LN-I7's fleet trust and a fleet member that later defects (leaves the
  ring open on our side but not theirs) is not separately tested from the general
  `_finalize()` negative-rating path — whether a fleet participant can defect without
  consequence (since fleet trust also means no local-defection-history check in
  `_check_participants()`) is worth a closer look.

### Close-vs-defibrillation policy (operator, 2026-07-09)

LN+ contract channels are **excluded from auto-closes for the full life of
their agreement** — `no_close` tags are placed on both the outbound and
incoming peers at contract activation and released only when the contract
concludes; `_check_close_allowed` vetoes every close at execution time.
They **remain eligible for defibrillation**: diagnostic rebalances carry no
policy gate, and the dead-capital staging pipeline holds protected channels
at DEFIBRILLATE rather than advancing them to CLOSE. Pinned by
`TestLnplusChannelsDefibrillatableButNotCloseable` in
tests/test_capacity_planner.py.
