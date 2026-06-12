# Intent Contract: modules/hive_router.py

Tier 2 — medium treatment. Audited 2026-06-12.

## Purpose

`HiveRouter` is the plugin-wide, *older* fleet-routing utility (distinct from
`RebalanceHiveRouter`, which prices rebalance pairs). It maintains two askrene
layers — `hive-fleet` (zero-fee overrides + node bias +5 for fleet member
channels, either detected as cl-hive-managed or self-created in standalone
mode, `:119-258`) and `revenue-local` (profitability-classification biases,
`:604-666`) — caches fleet membership and gossip-pushed member balances, and
offers `discover_route()` to find a cheap path to a peer through the fleet.
It is instantiated once at plugin init when hive hints exist
(`cl-revenue-ops.py:2098-2110`). Crucially, it is **not** the rebalance pricing
router: the engine accepts it only as a hive-membership check because it lacks
`price_pair` (`rebalance_engine_v2.py:104-107`), and builds its own
`RebalanceHiveRouter` instead (`:172-179`). Its live consumers are layer/cache
refresh (`hive_runtime.py:38-52`), inbound-fee estimation in the EV rebalancer
(`rebalancer.py:1638-1650`), Boltz loop-out first-hop selection
(`cl-revenue-ops.py:8020-8030`), and Boltz swap topology scoring
(`cl-revenue-ops.py:7615-7618`).

## Inputs / Outputs

- Constructor: `(plugin, hive_hints)` (`hive_router.py:41`); `data_service` and
  `profitability_analyzer` injected afterwards (`cl-revenue-ops.py:2107-2110`).
- Callers: `refresh_hive_runtime` calls `refresh_layer()`,
  `refresh_fleet_balances()`, `clear_route_cache()` each hive-refresh tick
  (`modules/hive_runtime.py:22-52`; invoked from `cl-revenue-ops.py:2234`,
  `:2516`, `:2626`, `:3283`). `RebalanceEngine` receives it (`hive_router=`
  param, `cl-revenue-ops.py:2124`) and duck-types it to a membership-only
  check (`_membership_router`, `rebalance_engine_v2.py:103-105`) — it never
  becomes the pricing router (the dead `RebalanceExecutor` is the only code
  that ever took a `hive_router=` constructor argument directly). The EV
  rebalancer also consults `is_hive_member` directly (`rebalancer.py:580`).
- RPC surface: `askrene-listlayers` (`:84-92`), `askrene-create-layer`
  (`:94-103`), `askrene-age` (`:105-113`), `askrene-update-channel` zero-fee
  overrides (`:197-217`), `askrene-bias-node` (`:219-240`),
  `askrene-bias-channel` (`:639-654`), `askrene-reserve`/`askrene-unreserve`
  (`:702-732`), `getroutes` (`:316-330`), `listpeerchannels` (`:61-65`).
- Inputs from hive: `hive_hints.is_hive_member`, `hive_hints.get_fleet_balance`
  (balances pushed via hive-export-hints, `:410-425`).
- Output: `HiveRoute{fee_ppm, hops, source_scid, path, probability_ppm}`
  (`:20-28`). Datastore keys: none.

## Invariants

- **HR-1** `discover_route` returns `None` unless `self.available`, and caches
  per-peer results (including `None` failures) for 60 s within a cycle
  (`:276-287`, `:381-386`); `clear_route_cache()` resets at cycle start.
- **HR-2** `auto.sourcefree` is never used; the layer list is built from live
  `askrene-listlayers`, and a failed `getroutes` is retried once with
  auto-only layers to survive the unknown-layer TOCTOU crash (`:296-330`).
- **HR-3** Discovery fee is capped at 1% of the amount (`max_fee_msat =
  amount_msat // 100`, `:294`).
- **HR-4** Layer ownership defers to cl-hive: if `hive-fleet` already exists,
  only membership is cached; standalone creation happens only when absent
  (`refresh_layer`, `:119-148`), setting both directions of member channels to
  0 fee and biasing member nodes +5 (`:185-245`).
- **HR-5** `max_rebalance_through_member` = min(25% of member capacity,
  liquidity above a 40% healthy floor); 0 when balance unknown (`:435-461`).
- **HR-6** `reserve_path`/`unreserve_path` only submit normalized
  `{short_channel_id_dir, amount_msat>0}` entries and return False rather than
  raising (`:582-594`, `:702-732`); `reserve_for_job` accepts only explicit
  askrene directions 0/1, never intent strings (`:668-700`).
- **HR-7** `revenue-local` biases channels strictly by the profitability map
  {profitable:+3, break_even:0, underwater:-3, stagnant_candidate:-5,
  zombie:-8}; zero-bias classes are skipped (`:596-657`).

## Revenue role

Indirect, two-sided: the `hive-fleet`/`revenue-local` layers it maintains are
consumed by *other* routers (v3, RebalanceHiveRouter list them by name), so its
biases shape which routes the whole stack prefers; its fee estimates feed the
rebalancer's inbound-cost EV math; and its topology scoring tilts Boltz swap
selection. Bad layer state here silently distorts every routing decision.

## Observable surface

Layer contents are visible via `askrene-listlayers` (not a hermes artifact).
Behavior surfaces indirectly: inbound fee estimates appear in
`revenue-rebalance-debug` EV fields and `INBOUND FEE EST` log lines
(`rebalancer.py:1645`); Boltz route substitutions log `BOLTZ HIVE ROUTE`
(`cl-revenue-ops.py:8024`) and their fees land in `revenue-spend-ledger.json`
(structural category); `revenue-status.json` hive sections reflect
availability. No artifact records the layer biases themselves.

## Uncertainties

- Dead surface: `reserve_path`, `unreserve_path`, `reserve_for_job`,
  `unreserve_for_job`, `max_rebalance_through_member`,
  `suggest_fleet_rebalance_chunks`, `get_fleet_member_balance`, and
  `fleet_member_can_route` have no live callers — only the dead
  `modules/rebalance_executor.py` and tests use them. `rebalancer.py:339`'s
  comment ("so _handle_job_* methods can call unreserve") describes calls that
  do not exist.
- `score_channel_for_hive`'s non-fleet branch is an acknowledged stub that
  always returns 1.0 (`:573-576`).
- `discover_route` uses `final_cltv=18` hardcoded (`:322`) rather than the
  node's `cltv-final` — harmless for estimation, wrong if a route were executed.
- Does standalone layer creation ever fight cl-hive's managed layer if cl-hive
  starts *after* this plugin created `hive-fleet`?
