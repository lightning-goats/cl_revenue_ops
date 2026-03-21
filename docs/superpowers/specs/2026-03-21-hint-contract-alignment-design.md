# Hint Contract Alignment Design

## Goal

Make cl-hive and cl_revenue_ops a coherent hint producer/consumer pair across all hint categories by fixing the `competition_bias` encoding mismatch, adding membership hint consumption, adding a cross-plugin contract test, and documenting the integration.

## Architecture

cl-hive exports compact read-only hints via `hive-export-hints`. cl_revenue_ops consumes them through `HiveHintAdapter` (modules/hive_hints.py), which polls, validates, caches, and exposes bounded bias lookups. All changes are in cl_revenue_ops — cl-hive's export surface is correct as-is.

## Work Items

### 1. Fix `competition_bias` interpretation

**Problem:** cl-hive exports `competition_bias` as integer `-1 / 0 / 1` (centered at 0). cl_revenue_ops interprets it as float `0.0–2.0` (centered at 1.0). The consumer's math `(comp - 1.0) * weight` produces wrong results: `0` (neutral) yields `-0.02` instead of `0.0`; `1` (lean-in) yields `0.0` instead of `+0.02`.

**Fix:** In `modules/hive_hints.py`, `get_fee_bias()`:
- Change clamp from `max(0.0, min(2.0, comp))` to `max(-1.0, min(1.0, comp))`
- Change bias calculation from `(comp - 1.0) * FEE_COMPETITION_WEIGHT` to `comp * FEE_COMPETITION_WEIGHT`

**Test updates:** Change all test fixtures from `1.2/0.8/1.5/100.0` to `-1/0/1` and equivalent edge cases like `-50/50`.

### 2. Add `is_hive_member()` and consume in fee controller

**Problem:** cl-hive exports `member: true/false` per peer. The old `set_hive_policy(strategy="hive")` was removed from cl-hive, but cl_revenue_ops has no mechanism to discover membership from hints and apply 0-PPM for fleet peers.

**Fix in `modules/hive_hints.py`:** Add method:
```python
def is_hive_member(self, peer_id: str) -> bool:
    hint = self._get_peer_hint(peer_id)
    return bool(hint.get("member", False))
```

**Fix in `modules/fee_controller.py`:** Insert the member check in the fee pipeline after STATIC policy handling and before DYNAMIC processing begins (i.e., after the static-strategy return and before DTS+PID runs). This ensures operator-set policies (STATIC, PASSIVE) always take precedence over automatic hive member detection.

When `self.hive_hints.is_hive_member(peer_id)` returns True:
- Return 0 PPM immediately as the final fee
- Do NOT update DTS posterior (the member override is external context, not market signal)
- Do NOT trigger hysteresis sleep (so when membership ends, the next cycle runs DTS+PID normally)
- Log as `reason: "hive_member"` for explainability

Also add the same check to `_set_initial_fee()` so newly opened channels to hive members get 0 PPM from the start rather than waiting for the first fee cycle.

**Behavior when hints unavailable:** `is_hive_member()` returns `False` (fail-open). No 0-PPM override. Operator can achieve the same effect with `revenue-policy set <peer_id> strategy=static fee_ppm=0`.

**Gossip oscillation protection:** If hints go stale after a peer was assigned 0 PPM via the member hint, the fee controller should hold 0 PPM for one additional TTL period before reverting to DTS+PID. This prevents gossip churn from intermittent cl-hive availability.

**Rebalancing to 0-PPM members:** No special handling needed — the EV gating in the rebalancer naturally suppresses rebalancing to 0-PPM peers (revenue = 0 → EV < cost).

### 3. Cross-plugin contract test

**Problem:** Each side has unit tests with mock data, but no test validates that cl-hive's actual output parses correctly in cl_revenue_ops. The `competition_bias` mismatch went undetected because consumer tests used `1.2/0.8` (the consumer's assumed encoding) rather than `-1/0/1` (what the producer actually sends).

**Fix:** Add a contract test in `tests/test_hive_contract.py` with a golden fixture matching cl-hive's exact output format:

```python
GOLDEN_HIVE_SNAPSHOT = {
    "generated_at": <recent_timestamp>,
    "ttl_seconds": 900,
    "peer_count": 3,
    "hints": {
        "02member_peer": {
            "member": True,
            "corridor_role": "owner",
            "competition_bias": 1,
            "peer_quality_score": 0.82,
            "traffic_confidence": 0.74,
            "rebalance_preference": "sink",
            "channel_open_hint": {
                "open_preference": "open",
                "topology_confidence": 0.71,
                "suggested_size_bucket": "medium",
                "reason": "underserved_corridor"
            }
        },
        "03nonmember_peer": {
            "member": False,
            "corridor_role": "secondary",
            "competition_bias": -1,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source"
        },
        "02neutral_peer": {
            "member": True,
            "corridor_role": "none",
            "competition_bias": 0,
            "rebalance_preference": "neutral"
        }
    }
}
```

Tests to run against this fixture:
- Fee bias direction: `competition_bias: 1` → positive bias, `-1` → negative, `0` → neutral
- Fee bias gating: hint with competition_bias but no traffic_confidence → returns 1.0 (neutral)
- Rebalance bias direction: "sink" → positive, "source" → negative, "neutral" → 1.0
- `is_hive_member()`: True for member peers, False for non-member peers, False for absent peers
- `get_channel_open_hint()`: returns validated dict for peer with hint, empty dict for peer without
- `get_open_candidates()`: includes only peers with `open_preference: "open"`
- Missing `member` field in hint → `is_hive_member()` returns False

### 4. Documentation

**Problem:** cl_revenue_ops CLAUDE.md and README have zero mention of the hive hints integration.

**Fix:** Add a section to `CLAUDE.md` describing:
- `HiveHintAdapter` in `modules/hive_hints.py` as the sole integration boundary
- Opt-in via `revenue-ops-hive-hints-enabled` config
- Bounded bias semantics: ±10% fee, ±15% rebalance
- `member: true` → 0-PPM short-circuit in fee pipeline (after STATIC, before DTS+PID)
- Fail-open behavior: missing/stale/invalid hints → neutral (1.0) biases
- Channel-open hints consumed by capacity planner for scoring
- Gossip oscillation protection: 0 PPM held for one additional TTL after hints go stale

## Files Changed

| File | Change |
|------|--------|
| `modules/hive_hints.py` | Fix competition_bias math, add `is_hive_member()` |
| `modules/fee_controller.py` | Add member 0-PPM short-circuit in fee pipeline + `_set_initial_fee()` |
| `tests/test_hive_hints.py` | Fix competition_bias fixtures |
| `tests/test_fee_hive_bias.py` | Fix competition_bias fixture values |
| `tests/test_hive_contract.py` | New: golden fixture contract test |
| `CLAUDE.md` | Add hive integration section |

## Safety

- All bias hard caps remain unchanged (±10% fee, ±15% rebalance)
- `is_hive_member()` fails open (returns False when hints unavailable)
- 0-PPM override only activates when hints are fresh AND member flag is true
- 0-PPM override does NOT update DTS posterior or trigger hysteresis sleep
- 0-PPM held for one additional TTL period after hints go stale (gossip oscillation protection)
- Operator-set policies (STATIC, PASSIVE) always override automatic member detection
- All existing safety rails (fee clamps, damping, EV gating, budgets, futility breakers) preserved
- Feature remains opt-in and disabled by default
