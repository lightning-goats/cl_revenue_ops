# Design: cl_hive Hint Integration

## Goal

Allow `cl_revenue_ops` to incorporate trusted-fleet recommendations from `cl_hive` as small bounded soft biases, while preserving local execution authority over all fee and rebalance decisions.

## Product Rules

- `cl_revenue_ops` remains the local execution layer
- `cl_hive` remains a coordination / recommendation source
- `cl_hive` must not directly set fees or trigger rebalances
- All final decisions remain local
- Integration must be small, safe, explainable, and easy to disable

## Architecture

One new module -- `modules/hive_hints.py` -- is the sole integration boundary with `cl_hive`. Fee controller and rebalancer each get one thin call site that asks for a bounded multiplicative bias factor.

```
cl_hive (hive-export-hints RPC)
    |
    v
HiveHintAdapter (poll, validate, cache, TTL)
    |
    +---> FeeController._get_hive_fee_bias(peer_id) -> [0.9, 1.1]
    |
    +---> EVRebalancer._get_hive_rebalance_bias(peer_id) -> [0.85, 1.15]
```

## RPC Schema

Assumed local read-only RPC: `hive-export-hints`

```json
{
  "generated_at": 1760000000,
  "ttl_seconds": 900,
  "hints": {
    "02abc...": {
      "member": true,
      "corridor_role": "owner",
      "competition_bias": 1,
      "peer_quality_score": 0.82,
      "traffic_confidence": 0.74,
      "rebalance_preference": "sink"
    }
  }
}
```

All per-peer fields are optional. Missing or invalid values degrade to neutral (1.0 bias).

## Fee Controller Insertion Point

**Line 3450 in `fee_controller.py`** -- between `dts_fee * pid_multiplier` and the hard clamp:

```python
# Before:
post_pid_target_ppm = int(dts_fee * pid_multiplier)
bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, post_pid_target_ppm))

# After:
post_pid_target_ppm = int(dts_fee * pid_multiplier)
hive_bias = self._get_hive_fee_bias(peer_id)
post_pid_target_ppm = int(post_pid_target_ppm * hive_bias)
bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, post_pid_target_ppm))
```

Bias is applied before clamp, blend, and damping. All existing safety rails remain intact.

## Rebalancer Insertion Point

**Line 2337 in `rebalancer.py`** -- the `sort_key` function for final candidate ranking:

```python
# Before:
def sort_key(c):
    dest_state = self.database.get_channel_state(c.to_channel)
    flow_state = dest_state.get("state", "balanced") if dest_state else "balanced"
    priority = 2 if flow_state == "source" else 1
    return (priority, c.expected_profit_sats)

# After:
def sort_key(c):
    dest_state = self.database.get_channel_state(c.to_channel)
    flow_state = dest_state.get("state", "balanced") if dest_state else "balanced"
    priority = 2 if flow_state == "source" else 1
    hive_bias = self._get_hive_rebalance_bias(c.to_peer)
    biased_profit = c.expected_profit_sats * hive_bias
    return (priority, biased_profit)
```

Only affects sort order of already-EV-positive candidates. Cannot trigger rebalances, bypass EV gating, or change budget controls.

## Bias Interpretation

| Hint Field | Fee Effect | Rebalance Effect |
|---|---|---|
| `corridor_role=owner` | +3% upward | -- |
| `corridor_role=secondary` | -3% downward | -- |
| `competition_bias` (0-2, neutral=1) | +/-2% scaled | -- |
| `peer_quality_score` (0-1) | -- | +/-5% on sort score |
| `rebalance_preference=sink` | -- | +5% (favor as destination) |
| `rebalance_preference=source` | -- | -5% (deprioritize) |
| `traffic_confidence` (0-1) | Scales all fee effects | Scales all rebalance effects |

Hard caps: fee bias [0.9, 1.1], rebalance bias [0.85, 1.15].

## HiveHintAdapter Module Shape

```python
class HiveHintAdapter:
    def __init__(self, plugin, config):
        self._plugin = plugin
        self._config = config
        self._snapshot = None
        self._snapshot_fetched_at = 0
        self._ttl_seconds = 900

    def poll(self):
        """Fetch fresh hints via hive-export-hints RPC. Fail-open."""

    def get_fee_bias(self, peer_id: str) -> float:
        """Return multiplicative fee bias [0.9, 1.1]. 1.0 if unavailable."""

    def get_rebalance_bias(self, peer_id: str) -> float:
        """Return multiplicative score bias [0.85, 1.15]. 1.0 if unavailable."""

    def get_status(self) -> dict:
        """Diagnostics for revenue-status / debug surfaces."""
```

## Failure Behavior

- RPC fails: keep last snapshot until TTL expires, then 1.0 everywhere
- Invalid schema: log once, ignore bad fields, use valid ones
- Missing peer: 1.0
- Disabled: 1.0, never poll

No pause. No operator intervention required.

## Configuration

Two options only:
- `revenue-ops-hive-hints-enabled` (bool, default: false)
- `revenue-ops-hive-hints-ttl` (int seconds, optional override)

Bias caps are hardcoded constants, not configurable.

## Diagnostics

Extend `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug` with:

```json
"hive_hints": {
  "enabled": true,
  "snapshot_age_seconds": 312,
  "snapshot_fresh": true,
  "hints_count": 14
}
```

Per-channel fee debug includes `hive_fee_bias: 1.03` when a bias was applied.

## What This Design Does NOT Do

- Does not let hive set fees or trigger rebalances
- Does not rewrite policies
- Does not bypass any existing safety rail
- Does not add a large config surface
- Does not require operator intervention on failure
- Does not spread hive-specific code beyond hive_hints.py + two thin call sites
