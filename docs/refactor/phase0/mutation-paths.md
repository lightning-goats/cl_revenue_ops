# Mutation-path inventory (baseline 5e8f747)

Pin test: `tests/test_mutation_path_inventory.py` (file×verb allowlist).
Prior art: `docs/audit/deep/concurrency-map.md` (thread/lock graph),
`docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md` (RPC classification).

## Finding relevant to Workstream G

`modules/data_service.py` is ALREADY a partial central CLN execution
adapter ("NEVER cached" tier): `set_channel` (:275), `fund_channel`
(:281), `close_channel` (:288), `create_invoice` (:316), `send_pay`
(:320), `wait_send_pay` (:324), `delete_pay` (:328), `delete_invoice`
(:332), `pay` (:337), askrene mutations (:367–:425), `datastore_push`
(:461). The refactor's CLN adapter should grow from this seam.

Bypass sites (call plugin.rpc directly, must be routed through the
adapter during Phase 3):

- `modules/rebalance_native_executor_v2.py` — `_rpc_call` (:36) raw
  invoke: invoice build (:450), sendpay (:461), waitsendpay (:463),
  delpay (:386), delinvoice (:391)
- `modules/rebalance_engine_v2.py` — delpay (:2792), datastore fallback
  (:2869), askrene-remove-layer (:316)
- `modules/rebalance_router_v3.py` — askrene-create-layer (:615),
  askrene-update-channel (:631, :649), askrene-remove-layer (:668);
  each has a data_service-preferred branch
- `modules/lnplus_swaps.py` — connect (:1417), fundchannel (:1462),
  signmessage (:130, LN+ auth)
- `modules/capacity_planner.py` — fundchannel/close fallback paths when
  no data_service is wired (:3138 open; close execution path)

## Wrapper callers (economic writers)

- Fee broadcasts: `modules/fee_controller.py:7624` → data_service.set_channel
  (sole setchannel caller)
- Channel opens: `modules/capacity_planner.py:3138` → fund_channel
- Boltz loop-out invoice pay: `modules/boltz_manager.py:844` → pay
- Datastore telemetry: `cl-revenue-ops.py:3510,3515,3557,7242`,
  `modules/profitability_analyzer.py:758`,
  `modules/rebalance_engine_v2.py:2862` → datastore_push

## External write APIs (non-CLN)

- Boltz (`modules/boltz_manager.py`) — via `boltzcli` SUBPROCESS, not
  HTTP: `_run` (:444) / `_run_json` (:469). Writes: createreverseswap
  external-pay (:2017/:2031), createreverseswap (:2133, exec
  :2152–:2214), createswap loop-in (:1823/:1824), createchainswap
  (:2395/:2408), claimswaps (:2338), refund (:2312), withdraw/wallet
  send (:2429/:2471/:2477; also `cl-revenue-ops.py:7517`)
- LN+ (`modules/lnplus_swaps.py`) — HTTP POST via urllib
  (`LNPlusClient._request` :82, base https://lightningnetwork.plus/api/2):
  create_application (:200), delete_application (:205),
  complete_application (:210), create_rating (:229),
  mark_read_notifications (:220)

## Autonomous initiators (background threads)

All `threading.Thread` daemons started at `cl-revenue-ops.py:3428–3435`
(+ RPC-drain :597); each sleeps its interval with ±10–20% random jitter.

| Thread | Def | Interval | Can execute |
|---|---|---|---|
| flow-analysis | :3010 | flow_interval ≥60s | analytics, datastore writes |
| fee-adjustment | :3067 | fee_interval ≥60s | setchannel fee broadcasts |
| rebalance-check | :3108 | rebalance_interval ≥60s | circular sendpay, askrene layers, budget reservations |
| boltz-auto-cycle | :3148 | boltz_auto_cycle_interval_minutes (15m default) | Boltz loop-in/out/withdraw |
| capacity-planner | :3239 | planner_interval ≥600s (default 6h) | fundchannel opens, closes, reservations |
| lnplus-watcher | :3207 | lnplus_watcher_interval (1h default) | LN+ apply/complete/fundchannel/ratings |
| financial-snapshot | :3331 | 24h | DB snapshot writes only |
| startup-snapshot | :3431 | one-shot | peer snapshot to DB |

## Budget/spend enforcement points (Workstream D input)

Four distinct implementations gate spending today:

1. Generic spend ledger, `modules/database.py`: `reserve_spend` (:3895,
   atomic BEGIN IMMEDIATE; `_reserve_budget_atomic` :94),
   `mark_spend_reservation_spent` (:4019) + `record_spend_event` (:4072),
   `release_spend_reservation` (:4010), `cleanup_stale_spend_reservations`
   (:4168), `get_budget_status` (:4512)
2. Rebalance-specific, `modules/database.py`: `reserve_budget` (:3693),
   `release_budget_reservation` (:3734), `mark_budget_spent` (:3752)
3. Capex, `modules/capex_budget.py`: `budget_sats` (:76),
   `tactical_budget_sats` (:107), `get_channel_budget` (:332),
   `reserve/settle/release_boltz_swap_budget` (:429/:459/:504)
4. Growth/efficiency, `modules/growth_budget.py`
   `compute_growth_budget_status` (:90); `modules/capital_efficiency.py`
   `analyze` (:59)

Gate call sites before spend: `rebalancer.py:1451`,
`rebalance_engine_v2.py:1938`, `capacity_planner.py:3199/:3667`,
`boltz_manager.py:1642`, `lnplus_swaps.py:1439`,
`cl-revenue-ops.py:7321/:7351/:7396`.
