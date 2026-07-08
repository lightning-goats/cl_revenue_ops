# Metabolic Influence Contract

This contract defines the optional top-level `metabolic_influence` section inside the `["hive", "hints"]` snapshot. It is a Level 2c advisory scoring input from cl-mycelium to `cl_revenue_ops`; it is not an execution command.

## Producer

cl-mycelium produces `metabolic_influence/v1` from its metabolic arbitration layer when `hive-organism-metabolism-m2-influence=true`, which is the production default (shipped default ON; default scope `channel_and_fleet_peers`). Producers should pre-scope peer effects, but consumers must still enforce scope locally.

## Consumer

`cl_revenue_ops` consumes this payload only through `modules/hive_hints.py` (`HiveHintAdapter`). The adapter exposes neutral-safe accessors such as `get_metabolic_status`, `get_metabolic_peer_effect`, `get_metabolic_fee_bias`, `get_metabolic_rebalance_bias`, `get_metabolic_open_bias`, `get_metabolic_closure_watch_bias`, and `get_metabolic_action_constraints`.

`cl_revenue_ops` remains independent and must run safely when cl-mycelium is absent.

Terminology: Level 2b is the cl-mycelium producer side. Level 2c is this consumer side: fresh, scope-valid metabolic influence may change bounded scores, but cannot execute, set fees, spend, open or close channels, override budgets, or prove Level 3 value.

## Generated At

`metabolic_influence.generated_at` is required for behavior use and is a Unix epoch timestamp in seconds. It is evaluated separately from the outer hint snapshot `generated_at`.

## TTL And Freshness

`metabolic_influence.ttl_seconds` declares the section freshness window. The consumer requires both the outer hint snapshot and the metabolic section to be fresh. Stale fallback never materializes metabolic influence.

## Units

Fields named `*_sats` are satoshis. Peer deltas are unitless additive score deltas that the consumer converts to bounded multiplicative biases. Coverage values are categorical strings: `sufficient`, `partial`, `insufficient`, or `unknown`.

## Required Fields

- `schema_version`: must be `metabolic-influence/v1`.
- `generated_at`: Unix epoch seconds.
- `ttl_seconds`: freshness window in seconds.
- `enabled`: boolean.
- `m2_scope`: one of `legacy_seed_only`, `channel_peers`, `channel_and_fleet_peers`, or `all_hints`.
- `confidence`: `high`, `medium`, `low`, or `unknown`.
- `coverage`: object containing at least the windows used by the consumer.
- `global_effects`: object.
- `peer_effects`: object keyed by peer id.
- `safety`: object declaring advisory/executor boundaries.

## Optional Fields

Optional fields include `source`, `metabolic_posture`, `metabolic_peer_posture`, `reason_codes`, `fee_bias_delta`, `rebalance_priority_delta`, `open_confidence_delta`, `closure_watch_priority_delta`, and `max_rebalance_burn_sats`.

Unknown optional fields must be ignored by the consumer.

## Stale Behavior

Stale metabolic influence returns neutral behavior values: fee bias `1.0`, rebalance bias `1.0`, open bias `1.0`, closure-watch bias `1.0`, and no additional action permission.

## Malformed Behavior

Malformed roots, unsupported schema versions, invalid timestamps, invalid TTLs, non-object `peer_effects`, or non-object `global_effects` must not crash `cl_revenue_ops`. The consumer reports diagnostics and returns neutral values.

## Neutral Fallback Behavior

Missing, stale, malformed, unsupported, low-confidence, insufficient-coverage, or out-of-scope metabolic influence neutralizes safely. `all_hints` is rejected unless the local operator explicitly enables lab-mode all-hints M2 consumption.

## Versioning

The current version is `metabolic-influence/v1`. Future versions must be neutral until explicitly supported by the consumer.

## Consumer Caps

The consumer clamps each additive peer delta and converts it to a bounded multiplicative bias (`modules/hive_hints.py`):

- Fee metabolic bias: `fee_bias_delta` clamped to `±0.05` → multiplier `[0.95, 1.05]` (`METABOLIC_FEE_BIAS_CAP`).
- Rebalance metabolic bias: `rebalance_priority_delta` clamped to `±0.15` → multiplier `[0.85, 1.15]` (`METABOLIC_REBALANCE_BIAS_CAP`).
- Planner/open metabolic bias: `open_confidence_delta` clamped to `[-0.15, +0.10]` → multiplier `[0.85, 1.10]` (`METABOLIC_OPEN_NEGATIVE_CAP` / `METABOLIC_OPEN_POSITIVE_CAP`).
- Closure-watch bias: `closure_watch_priority_delta` clamped to `±0.15` → multiplier `[0.85, 1.15]` (`METABOLIC_CLOSURE_WATCH_CAP`); diagnostic/advisory only and cannot call close.

The metabolic fee-bias channel is reserved / currently-neutral by producer choice: cl-hive emits `fee_bias_delta: 0.0` unconditionally, so `get_metabolic_fee_bias` always returns `1.0` in practice. The consumer machinery is wired up only for forward compatibility and will activate if the producer gains fee-advisory capability.

The producer additionally bounds every peer effect by a configurable `max_peer_effect` (cl-hive `modules/organism/metabolic_influence.py`, `DEFAULT_MAX_PEER_EFFECT = 0.15`, plugin option `hive-organism-metabolism-max-peer-effect`). This is defense-in-depth: because the producer value is configurable, the per-field consumer caps above are what actually keep values in-contract on the `cl_revenue_ops` side.

Metabolic influence never overrides min/max fee rails, budget gates, route-cost gates, ROI floors, dry-run mode, planner enablement, close/open authorization, or local executor policy.

## Action Constraints Are Advisory Only

`global_effects` fields — `max_rebalance_burn_sats`, `growth_allowed`, `rebalance_allowed`, `exploration_allowed` — are surfaced by `get_metabolic_action_constraints` as non-authorizing diagnostics only. They are never enforced as a spend, growth, or execution gate. `execution_authority` and `budget_authority` remain `cl_revenue_ops`: rebalance amounts and budgets are computed locally and independently of these constraints.

## Example Payload

```json
{
  "schema_version": "metabolic-influence/v1",
  "generated_at": 1760000000,
  "ttl_seconds": 300,
  "source": "metabolic_arbitration",
  "enabled": true,
  "m2_scope": "channel_and_fleet_peers",
  "metabolic_posture": "repair_only",
  "confidence": "medium",
  "coverage": {
    "1h": "sufficient",
    "6h": "sufficient",
    "24h": "sufficient",
    "7d": "partial",
    "30d": "insufficient"
  },
  "global_effects": {
    "growth_allowed": false,
    "rebalance_allowed": "repair_only",
    "exploration_allowed": false,
    "max_rebalance_burn_sats": null
  },
  "peer_effects": {
    "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
      "metabolic_peer_posture": "repair_target",
      "rebalance_priority_delta": 0.08,
      "fee_bias_delta": 0.02,
      "open_confidence_delta": -0.05,
      "closure_watch_priority_delta": 0.04,
      "reason_codes": ["repair_only"]
    }
  },
  "safety": {
    "executor_required": true,
    "executor_authority": "cl_revenue_ops",
    "direct_execution": false,
    "budget_authority": "cl_revenue_ops",
    "m2_scope_mutated": false,
    "budgets_mutated": false,
    "hints_are_advisory": true
  }
}
```
