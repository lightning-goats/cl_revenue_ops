# Revenue Profitability Summary Contract

Datastore key: `["revenue", "profitability-summary"]`

## Producer

`cl_revenue_ops` produces this payload from `ChannelProfitabilityAnalyzer._push_profitability_summary()` after profitability analysis.

## Consumer

cl-hive / cl-mycelium and other read-only consumers may consume this datastore payload as economic telemetry. Consumers must not call cl_revenue_ops action RPCs to obtain it.

## Generated At

The current payload uses `timestamp` as the generated-at field. It is a Unix epoch timestamp in seconds and should be normalized by consumers as `generated_at`. Producers may add `generated_at` later, but `timestamp` remains the compatibility field.

## TTL And Freshness

Recommended freshness TTL is 1800 seconds. Stale profitability data may be used for diagnostics with lowered confidence, but must not force hint changes or executor actions.

## Units

Fields ending in `_msat` are millisatoshis. Fields ending in `_sats` are satoshis. Fee and contribution sats are ceiling-rounded from msat so non-zero sub-sat fees are not lost. Volume sats, where exposed elsewhere, may floor-round; this compact contract keeps volume in msat. `roi_pct` is percent rounded to two decimals.

## Required Fields

- `timestamp`: Unix epoch seconds.
- `channels`: object keyed by channel id.

Required channel fields:
- `channel_id`
- `peer_id`
- `class`
- `net_profit_sats`
- `roi_pct`
- `days_open`
- `role`
- `fee_multiplier`
- `forward_count`
- `sourced_forward_count`
- `total_forward_count`
- `fees_earned_msat`
- `fees_earned_sats`
- `volume_routed_msat`
- `sourced_volume_msat`
- `sourced_fee_contribution_msat`
- `sourced_fee_contribution_sats`
- `total_contribution_msat`
- `total_contribution_sats`
- `open_cost_msat`
- `rebalance_cost_msat`
- `net_pnl_msat`

## Optional Fields

Producers may add channel diagnostics such as age windows, margin fields, confidence, or classification reasons. Consumers must ignore unknown fields.

## Stale Behavior

Consumers should mark samples stale when `now - timestamp` exceeds the configured freshness window. Stale samples may inform reports, but must not command cl_revenue_ops, bypass executor budgets, or broaden M2 scope.

## Malformed Behavior

Malformed JSON, a non-object root, missing `timestamp`, or missing/non-object `channels` means the payload is unusable. Consumers should record the error and fall back to unknown profitability.

## Neutral Fallback Behavior

Missing, stale, or malformed payloads produce unknown profitability, zero confidence, and no direct hint mutation. cl_revenue_ops continues operating from its local cache/RPC surfaces.

## Versioning

This is the current compact profitability summary contract. `timestamp` is retained for compatibility. Optional fields may be added without a version bump; breaking changes require a new schema/version field and a transition period.

## Example Payload

```json
{
  "timestamp": 1760000000,
  "channels": {
    "100x1x0": {
      "channel_id": "100x1x0",
      "peer_id": "02cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "class": "profitable",
      "net_profit_sats": 7,
      "roi_pct": 12.35,
      "days_open": 30,
      "role": "unknown",
      "fee_multiplier": 1.0,
      "forward_count": 3,
      "sourced_forward_count": 2,
      "total_forward_count": 5,
      "fees_earned_msat": 1,
      "fees_earned_sats": 1,
      "volume_routed_msat": 1234567,
      "sourced_volume_msat": 9876543,
      "sourced_fee_contribution_msat": 1001,
      "sourced_fee_contribution_sats": 2,
      "total_contribution_msat": 1001,
      "total_contribution_sats": 2,
      "open_cost_msat": 123000,
      "rebalance_cost_msat": 2000,
      "net_pnl_msat": -124999
    }
  }
}
```
