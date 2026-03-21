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

**Fix in `modules/fee_controller.py`:** At the top of the fee pipeline (alongside static/passive policy checks), before DTS+PID runs, check `self.hive_hints.is_hive_member(peer_id)`. If true, return 0 PPM immediately. This is a categorical trust-based policy decision, not a continuous bias — it belongs at policy-check level, not in the bias multiplier.

**Behavior when hints unavailable:** `is_hive_member()` returns `False` (fail-open). No 0-PPM override. Operator can still manually set `revenue-policy set peer_id strategy=hive` if needed.

### 3. Cross-plugin contract test

**Problem:** Each side has unit tests with mock data, but no test validates that cl-hive's actual output parses correctly in cl_revenue_ops. The `competition_bias` mismatch went undetected because consumer tests used `1.2/0.8` (the consumer's assumed encoding) rather than `-1/0/1` (what the producer actually sends).

**Fix:** Add a contract test (in `tests/test_hive_contract.py` or appended to `tests/test_hive_hints.py`) that:
- Hardcodes a fixture matching cl-hive's exact output format: integer `competition_bias` (-1/0/1), boolean `member`, optional `peer_quality_score`/`traffic_confidence`, nested `channel_open_hint`
- Feeds it through HiveHintAdapter
- Verifies fee bias direction is correct for each competition_bias value
- Verifies rebalance bias direction is correct for each rebalance_preference value
- Verifies `is_hive_member()` returns True for members, False for non-members
- Verifies channel_open_hint parsing produces valid results

### 4. Documentation

**Problem:** cl_revenue_ops CLAUDE.md and README have zero mention of the hive hints integration.

**Fix:** Add a section to `CLAUDE.md` describing:
- `HiveHintAdapter` in `modules/hive_hints.py` as the sole integration boundary
- Opt-in via `revenue-ops-hive-hints-enabled` config
- Bounded bias semantics: ±10% fee, ±15% rebalance
- `member: true` → 0-PPM short-circuit in fee pipeline
- Fail-open behavior: missing/stale/invalid hints → neutral (1.0) biases
- Channel-open hints consumed by capacity planner for scoring

## Files Changed

| File | Change |
|------|--------|
| `modules/hive_hints.py` | Fix competition_bias math, add `is_hive_member()` |
| `modules/fee_controller.py` | Add member 0-PPM short-circuit at pipeline top |
| `tests/test_hive_hints.py` | Fix competition_bias fixtures, add contract test |
| `tests/test_fee_hive_bias.py` | Fix competition_bias fixture values |
| `CLAUDE.md` | Add hive integration section |

## Safety

- All bias hard caps remain unchanged (±10% fee, ±15% rebalance)
- `is_hive_member()` fails open (returns False when hints unavailable)
- 0-PPM override only activates when hints are fresh AND member flag is true
- All existing safety rails (fee clamps, damping, EV gating, budgets, futility breakers) are preserved
- Feature remains opt-in and disabled by default
