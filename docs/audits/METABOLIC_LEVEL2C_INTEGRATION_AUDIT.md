# Metabolic Level 2c Integration Audit

Date: 2026-05-28

## Scope

This audit covers the cross-repo Level 2c path:

```text
cl-mycelium metabolic_arbitration
  -> metabolic_influence/v1 in hive hints
  -> cl_revenue_ops HiveHintAdapter
  -> bounded fee/rebalance/planner scoring diagnostics
```

The audit does not enable execution, change budgets, broaden M2 scope, or claim Level 3 value.

## Producer Result

cl-mycelium produces `metabolic_influence/v1` only when `hive-organism-metabolism-m2-influence` is explicitly enabled. Default config remains off. The producer emits a top-level hint extension rather than mutating legacy per-peer hint fields.

Producer-side `channel_and_fleet_peers` scoping keeps direct channel peers and fleet members eligible and leaves graph-only peers out of peer effects. The integration fixture also injects a deliberately out-of-scope peer effect after producer construction to prove consumer-side scope enforcement does not rely only on producer filtering.

## Consumer Result

`cl_revenue_ops` consumes the payload only through `HiveHintAdapter`. Missing, stale, malformed, unsupported, low-confidence, or insufficient-coverage metabolic payloads return neutral accessors:

- metabolic fee bias: `1.0`;
- metabolic rebalance bias: `1.0`;
- metabolic open bias: `1.0`;
- closure-watch bias: `1.0`;
- no additional action permission.

## Scope Result

The integrated fixture confirms:

```json
{
  "metabolic_influence": {
    "present": true,
    "fresh": true,
    "usable": true,
    "m2_scope": "channel_and_fleet_peers",
    "out_of_scope_peer_effect_count": 1
  }
}
```

The direct channel peer receives bounded advisory influence. The out-of-scope graph peer is neutralized by the consumer.

## Freshness Result

Fresh payloads are usable. Stale section-level `generated_at` / `ttl_seconds` neutralizes metabolic influence even when the outer hint snapshot is otherwise fresh. Malformed `peer_effects` neutralizes without crashing.

## Debug Result

The integration fixture checks these read-only surfaces:

- `HiveHintAdapter.get_status(live_refresh=False)` includes metabolic diagnostics.
- `FeeController.get_hive_fee_hint_debug()` reports `metabolic_fee_influence.seen=true` and `usable=true`.
- `RebalanceEngine` last-cycle metabolic diagnostics report seen/usable candidate bias and non-authorizing constraints.
- `CapacityPlanner` open scoring receives bounded positive influence in growth-ready fixtures.
- `revenue-status` reports zero-budget / dry-run canary controls with no recent fee changes or rebalances in the fixture.

Current cl-mycelium producer posture rules keep `fee_bias_delta` neutral; fee debug therefore confirms section consumption rather than a non-neutral fee delta. Rebalance and planner/open scoring show bounded non-neutral effects.

## Zero-Budget Result

The integration fixture verifies metabolic action constraints report:

```json
{
  "additional_permission": false,
  "execution_authority": "cl_revenue_ops",
  "budget_authority": "cl_revenue_ops"
}
```

Growth-ready metabolic influence does not make a zero unified budget pass. Rebalance test paths apply score modifiers only and do not call action RPCs.

## Tests

Added in `cl_revenue_ops`:

- `tests/test_metabolic_level2c_integration.py`

Relevant producer/consumer suites:

- cl-mycelium `tests/test_organism_metabolic_influence.py`
- cl-mycelium `tests/test_export_hints.py`
- cl_revenue_ops `tests/test_metabolic_influence_hints.py`
- cl_revenue_ops `tests/test_fee_hive_bias.py`
- cl_revenue_ops `tests/test_rebalance_engine_v2.py`
- cl_revenue_ops `tests/test_capacity_planner.py`

## Residual Risks

Level 2c remains advisory scoring only. It is not proof of long-horizon value. Level 3 still requires 7d/30d evidence. The producer currently keeps fee deltas neutral; non-neutral fee influence would require a future explicit producer rule and tests.

## Verdict

PASS - Level 2c integrated safely.
