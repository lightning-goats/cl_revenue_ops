# Mutation-path inventory (updated for liquidity-executor decommission)

Pin test: `tests/test_mutation_path_inventory.py` (file×verb allowlist).
Prior art: `docs/audit/deep/concurrency-map.md` (thread/lock graph),
`docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md` (RPC classification).

## Finding relevant to Workstream G

`modules/data_service.py` is ALREADY a partial central CLN execution
adapter ("NEVER cached" tier): `set_channel`, invoice and payment helpers, askrene mutations, and `datastore_push`. The refactor's CLN adapter should grow from this seam.

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

## Wrapper callers (economic writers)

- Fee broadcasts: `modules/fee_controller.py:7624` → data_service.set_channel
  (sole setchannel caller)
- Datastore telemetry: `cl-revenue-ops.py:3510,3515,3557,7242`,
  `modules/profitability_analyzer.py:758`,
  `modules/rebalance_engine_v2.py:2862` → datastore_push

## External write APIs (non-CLN)

No external swap-provider write API remains in the plugin.

## Autonomous initiators (background threads)

All `threading.Thread` daemons started at `cl-revenue-ops.py:3428–3435`
(+ RPC-drain :597); each sleeps its interval with ±10–20% random jitter.

| Thread | Def | Interval | Can execute |
|---|---|---|---|
| flow-analysis | :3010 | flow_interval ≥60s | analytics, datastore writes |
| fee-adjustment | :3067 | fee_interval ≥60s | setchannel fee broadcasts |
| rebalance-check | :3108 | rebalance_interval ≥60s | circular sendpay, askrene layers, budget reservations || financial-snapshot | :3331 | 24h | DB snapshot writes only |
| startup-snapshot | :3431 | one-shot | peer snapshot to DB |

## Budget/spend enforcement points (Workstream D input)

The retained implementations gate spending today:

1. Generic spend ledger, `modules/database.py`: `reserve_spend` (:3895,
   atomic BEGIN IMMEDIATE; `_reserve_budget_atomic` :94),
   `mark_spend_reservation_spent` (:4019) + `record_spend_event` (:4072),
   `release_spend_reservation` (:4010), `cleanup_stale_spend_reservations`
   (:4168), `get_budget_status` (:4512)
2. Rebalance compatibility API over the generic ledger, `modules/database.py`: `reserve_budget` (:3693),
   `release_budget_reservation` (:3734), `mark_budget_spent` (:3752)
3. Capex, `modules/capex_budget.py`: channel budgets and envelopes
4. Growth/efficiency, `modules/growth_budget.py`
   `compute_growth_budget_status` (:90); `modules/capital_efficiency.py`
   `analyze` (:59)

Gate call sites before spend: `rebalancer.py:1451`,
`rebalance_engine_v2.py:1938`,
`cl-revenue-ops.py:7321/:7351/:7396`.
