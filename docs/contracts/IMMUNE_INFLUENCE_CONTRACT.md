# Immune Influence Contract

This contract defines the optional top-level `immune_influence` section inside the `["hive","hints"]` snapshot. It is a Level 2c advisory scoring input from cl-mycelium to `cl_revenue_ops`; it is not an execution command.

## Producer

cl-mycelium produces `immune_influence/v1` from `goal_state.diagnostics.immune_advisory` when `hive-organism-immune-m2-influence=true`, which is the production default (shipped default ON). Producers pre-scope peer effects, but consumers must still enforce scope locally.

## Consumer

`cl_revenue_ops` consumes this payload only through `modules/hive_hints.py` (`HiveHintAdapter`). The adapter exposes neutral-safe accessors such as `get_immune_status`, `get_immune_peer_effect`, `get_immune_fee_bias`, `get_immune_rebalance_bias`, `get_immune_open_bias`, `get_immune_closure_watch_bias`, and `get_immune_action_constraints`.

## Safety

Immune influence is optional, fresh-only, scope-checked, bounded, and neutral on missing/stale/malformed data. It cannot execute, set fees, spend, open or close channels, override budgets, suppress peers directly, mutate M2 scope, or prove Level 3 value.

## Required Fields

- `schema_version`: must be `immune-influence/v1`.
- `generated_at`: Unix epoch timestamp in seconds.
- `ttl_seconds`: freshness window in seconds.
- `enabled`: boolean.
- `m2_scope`: one of `legacy_seed_only`, `channel_peers`, `channel_and_fleet_peers`, or `all_hints`.
- `immune_posture`: `clear`, `watch`, `guardrail`, `rehabilitating`, or `unknown`.
- `confidence`: `high`, `medium`, `low`, or `unknown`.
- `global_effects`: object with non-authorizing advisory constraints.
- `peer_effects`: object keyed by peer id.
- `safety`: object confirming executor and budget authority remain local to `cl_revenue_ops`.

## Peer Effects

Peer effects may include bounded `fee_bias_delta`, `rebalance_priority_delta`, `open_confidence_delta`, and `closure_watch_priority_delta`. These are scoring modifiers only. Under `channel_and_fleet_peers`, effects apply only to peers marked `direct_channel_peer=true` or `member=true` in the same hint snapshot.

By design, immune influence carries no fee authority: the producer always emits `fee_bias_delta: 0.0`. The immune fee-bias channel is therefore reserved / currently-neutral by producer choice — `get_immune_fee_bias` always returns `1.0` in practice, and the consumer machinery is wired up only for forward compatibility. Fee biasing belongs to metabolic influence and the local fee controller; immune effects are limited to rebalance priority, open confidence, and closure watch. The consumer must still clamp and bound any nonzero value it receives so a future schema revision cannot exceed documented bounds.

## Consumer Caps

The consumer applies these per-field caps to each peer delta (`modules/hive_hints.py`), converting additive deltas to bounded multiplicative biases:

- Fee immune bias: `fee_bias_delta` clamped to `±0.05` → multiplier `[0.95, 1.05]` (`IMMUNE_FEE_BIAS_CAP`; currently always neutral, see above).
- Rebalance immune bias: `rebalance_priority_delta` clamped to `±0.15` → multiplier `[0.85, 1.15]` (`IMMUNE_REBALANCE_BIAS_CAP`).
- Planner/open immune bias: `open_confidence_delta` clamped to `[-0.15, +0.10]` → multiplier `[0.85, 1.10]` (`IMMUNE_OPEN_NEGATIVE_CAP` / `IMMUNE_OPEN_POSITIVE_CAP`).
- Closure-watch immune bias: `closure_watch_priority_delta` clamped to `±0.15` → multiplier `[0.85, 1.15]` (`IMMUNE_CLOSURE_WATCH_CAP`); diagnostic/advisory only and cannot call close.

The producer additionally bounds every peer effect by a configurable `max_peer_effect` (cl-hive `modules/organism/immune_influence.py`, `DEFAULT_MAX_PEER_EFFECT = 0.12`). This is defense-in-depth: because the producer value is configurable up to `1.0`, the per-field consumer caps above are what keep values in-contract.

## Action Constraints Are Advisory Only

`global_effects` fields — `max_rebalance_burn_sats`, `growth_allowed`, `rebalance_allowed`, `exploration_allowed` — are surfaced by `get_immune_action_constraints` as non-authorizing diagnostics only. They are never enforced as a spend, growth, or execution gate. `execution_authority` and `budget_authority` remain `cl_revenue_ops`: the consumer computes rebalance amounts and budgets locally and independently of these constraints.

## Neutralization

Missing, stale, malformed, unsupported, low-confidence, disabled, or out-of-scope immune influence returns neutral behavior values: fee bias `1.0`, rebalance bias `1.0`, open bias `1.0`, closure-watch bias `1.0`, and no additional action permission. `all_hints` is rejected unless local operator config explicitly enables lab-mode all-hints M2 consumption.
