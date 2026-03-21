# Hint Contract Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the competition_bias encoding mismatch, add membership hint consumption for 0-PPM fleet peer policy, add a cross-plugin contract test, and document the integration.

**Architecture:** All changes are in cl_revenue_ops. The `HiveHintAdapter` in `modules/hive_hints.py` is the sole integration boundary. Fee controller gets a member check inserted after STATIC policy and before DTS+PID. A golden fixture contract test validates the exact schema cl-hive produces.

**Tech Stack:** Python 3.10+, pytest, cl-revenue-ops plugin framework

---

## File Structure

| File | Responsibility |
|------|---------------|
| `modules/hive_hints.py` | Fix competition_bias math (lines 113-116), add `is_hive_member()` |
| `modules/fee_controller.py` | Add hive member 0-PPM check in fee pipeline (after line 2493) and initial fee path (after line 4144) |
| `tests/test_hive_hints.py` | Fix VALID_SNAPSHOT fixture, fix competition_bias test values |
| `tests/test_fee_hive_bias.py` | Fix competition_bias fixture value |
| `tests/test_hive_contract.py` | New: golden fixture contract test |
| `CLAUDE.md` | Add hive integration section |

---

### Task 1: Fix competition_bias math in adapter

**Files:**
- Modify: `modules/hive_hints.py:113-116`
- Modify: `tests/test_hive_hints.py:25,33,163,181,299-302`
- Modify: `tests/test_fee_hive_bias.py:43`

- [ ] **Step 1: Write failing test for correct competition_bias interpretation**

In `tests/test_hive_hints.py`, add a test to `TestFeeBias` class:

```python
def test_competition_bias_integer_encoding(self, mock_plugin):
    """cl-hive exports competition_bias as -1/0/1, not 0.0-2.0."""
    for comp_val, expected_direction in [(-1, "negative"), (0, "neutral"), (1, "positive")]:
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02test": {
                    "corridor_role": "none",
                    "competition_bias": comp_val,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02test")
        if expected_direction == "negative":
            assert bias < 1.0, f"comp={comp_val} should give negative bias, got {bias}"
        elif expected_direction == "neutral":
            assert bias == 1.0, f"comp={comp_val} should give neutral bias, got {bias}"
        elif expected_direction == "positive":
            assert bias > 1.0, f"comp={comp_val} should give positive bias, got {bias}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hive_hints.py::TestFeeBias::test_competition_bias_integer_encoding -v`
Expected: FAIL (comp=0 gives -0.02 instead of 0.0, comp=1 gives 0.0 instead of positive)

- [ ] **Step 3: Fix the adapter math**

In `modules/hive_hints.py`, change lines 113-116:

```python
        comp = hint.get("competition_bias")
        if isinstance(comp, (int, float)):
            comp = max(-1.0, min(1.0, comp))
            bias += comp * FEE_COMPETITION_WEIGHT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hive_hints.py::TestFeeBias::test_competition_bias_integer_encoding -v`
Expected: PASS

- [ ] **Step 5: Fix VALID_SNAPSHOT fixture and existing tests**

In `tests/test_hive_hints.py`, update the module-level `VALID_SNAPSHOT`:

```python
VALID_SNAPSHOT = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "hints": {
        "02aabbcc": {
            "member": True,
            "corridor_role": "owner",
            "competition_bias": 1,
            "peer_quality_score": 0.82,
            "traffic_confidence": 0.74,
            "rebalance_preference": "sink",
        },
        "02ddeeff": {
            "member": True,
            "corridor_role": "secondary",
            "competition_bias": -1,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source",
        },
    },
}
```

Update `test_fee_bias_hard_cap` (line 163): change `"competition_bias": 100.0` to `"competition_bias": 50`

Update `test_zero_traffic_confidence_neutralizes` (line 181): change `"competition_bias": 1.5` to `"competition_bias": 1`

Update the extreme-values loop (lines 299-302): change `for comp in [0.0, 1.0, 2.0, 100.0, -50.0]` to `for comp in [-1, 0, 1, 50, -50]`

In `tests/test_fee_hive_bias.py`, update line 43: change `"competition_bias": 1.0` to `"competition_bias": 1`

- [ ] **Step 6: Run full test suite to verify all pass**

Run: `python3 -m pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add modules/hive_hints.py tests/test_hive_hints.py tests/test_fee_hive_bias.py
git commit -m "fix: correct competition_bias encoding to match cl-hive's -1/0/1 schema"
```

---

### Task 2: Add is_hive_member() to adapter

**Files:**
- Modify: `modules/hive_hints.py`
- Modify: `tests/test_hive_hints.py`

- [ ] **Step 1: Write failing test**

In `tests/test_hive_hints.py`, add a new test class:

```python
class TestMemberLookup:
    def test_is_hive_member_true(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02aabbcc") is True

    def test_is_hive_member_false_for_nonmember(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02nonmember": {"member": False, "corridor_role": "none", "competition_bias": 0},
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02nonmember") is False

    def test_is_hive_member_false_for_unknown(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02unknown") is False

    def test_is_hive_member_false_when_stale(self, mock_plugin):
        stale = dict(VALID_SNAPSHOT)
        stale["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = stale
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02aabbcc") is False

    def test_is_hive_member_false_when_field_missing(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02noflag": {"corridor_role": "none", "competition_bias": 0},
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02noflag") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hive_hints.py::TestMemberLookup -v`
Expected: FAIL with AttributeError (is_hive_member not defined)

- [ ] **Step 3: Implement is_hive_member()**

In `modules/hive_hints.py`, add after `_get_peer_hint` method (after line 88):

```python
    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def is_hive_member(self, peer_id: str) -> bool:
        """Return True if peer is a hive fleet member. False if unavailable/stale."""
        hint = self._get_peer_hint(peer_id)
        return bool(hint.get("member", False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hive_hints.py::TestMemberLookup -v`
Expected: All 5 pass

- [ ] **Step 5: Commit**

```bash
git add modules/hive_hints.py tests/test_hive_hints.py
git commit -m "feat: add is_hive_member() to HiveHintAdapter"
```

---

### Task 3: Consume member hint in fee controller

**Files:**
- Modify: `modules/fee_controller.py:1676,2493,4144`
- Modify: `tests/test_fee_hive_bias.py`

- [ ] **Step 1: Write failing tests for member 0-PPM**

In `tests/test_fee_hive_bias.py`, add:

```python
class TestMemberZeroFee:
    def test_hive_member_gets_zero_ppm(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02member")
        assert result == 0

    def test_non_member_returns_none(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = False
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02nonmember")
        assert result is None

    def test_no_adapter_returns_none(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = None
        result = fc._check_hive_member_fee("02peer")
        assert result is None

    def test_grace_period_holds_zero_after_stale(self, mock_plugin, mock_config, mock_database):
        """0-PPM held for one TTL after hints go stale (gossip oscillation protection)."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()

        # First: peer is a member, mark as 0-PPM
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        assert fc._check_hive_member_fee("02peer") == 0

        # Now hints go stale, but grace period holds
        adapter.is_hive_member.return_value = False
        adapter.is_fresh.return_value = False
        # _hive_member_set_at was just set, so within grace period
        assert fc._check_hive_member_fee("02peer") == 0

    def test_grace_period_expires(self, mock_plugin, mock_config, mock_database):
        """After grace period expires, revert to DTS+PID (return None)."""
        import time as _time
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter

        # Mark as member
        fc._check_hive_member_fee("02peer")

        # Simulate grace period expiry by backdating
        fc._hive_member_set_at["02peer"] = int(_time.time()) - 1801

        adapter.is_hive_member.return_value = False
        adapter.is_fresh.return_value = False
        assert fc._check_hive_member_fee("02peer") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fee_hive_bias.py::TestMemberZeroFee -v`
Expected: FAIL (method not defined)

- [ ] **Step 3: Implement _check_hive_member_fee()**

In `modules/fee_controller.py`, add `self._hive_member_set_at = {}` to `__init__` (near line 1641 where `self.hive_hints = None` is set).

Then add after `_get_hive_fee_bias` method (after line 1676):

```python
    def _check_hive_member_fee(self, peer_id: str) -> int | None:
        """Return 0 if peer is a hive member (0-PPM fleet policy), else None.

        This is a categorical trust-based decision, not a continuous bias.
        It does NOT update DTS posterior or trigger hysteresis sleep.

        Gossip oscillation protection: if a peer was recently set to 0-PPM
        via the member hint and hints go stale, hold 0-PPM for one
        additional TTL period before reverting.
        """
        if self.hive_hints is None:
            return None

        try:
            if self.hive_hints.is_hive_member(peer_id):
                self._hive_member_set_at[peer_id] = int(time.time())
                return 0
        except Exception:
            pass

        # Grace period: hold 0-PPM for one TTL after hints go stale
        last_set = self._hive_member_set_at.get(peer_id)
        if last_set is not None:
            try:
                ttl = self.hive_hints._effective_ttl()
            except Exception:
                ttl = 900
            if int(time.time()) - last_set <= ttl * 2:
                return 0
            else:
                # Grace period expired, clean up
                del self._hive_member_set_at[peer_id]

        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fee_hive_bias.py::TestMemberZeroFee -v`
Expected: All 5 pass

- [ ] **Step 5: Insert member check into fee pipeline**

In `modules/fee_controller.py`, after the STATIC strategy `continue` (line 2493), before the DYNAMIC comment (line 2495), add:

```python
                # HIVE MEMBER: 0-PPM fleet policy (hint-driven, no DTS/hysteresis)
                hive_member_fee = self._check_hive_member_fee(peer_id)
                if hive_member_fee is not None:
                    channel_info = channels.get(channel_id)
                    if channel_info:
                        current_fee = channel_info.get("fee_proportional_millionths", 0)
                        if current_fee != 0:
                            try:
                                self.set_channel_fee(
                                    channel_id, 0,
                                    reason="Hive member: 0-PPM fleet policy",
                                    reason_code=FeeReasonCode.POLICY_STATIC.value
                                )
                            except Exception as e:
                                self.plugin.log(f"Error setting hive member fee for {channel_id}: {e}", level='error')
                    continue
```

- [ ] **Step 6: Insert member check into _set_initial_fee()**

In `modules/fee_controller.py`, after the STATIC policy return (line 4144), before the DYNAMIC DTS prior sample (line 4146), add:

```python
            # HIVE MEMBER: 0-PPM fleet policy (hint-driven)
            if self.hive_hints is not None:
                try:
                    if self.hive_hints.is_hive_member(peer_id):
                        self._hive_member_set_at[peer_id] = int(time.time())
                        self.plugin.log(
                            f"INITIAL_FEE: {scid[:16]}... -> 0 PPM (hive member)"
                        )
                        return self.set_channel_fee(
                            scid, 0,
                            reason="Initial fee: hive member",
                            reason_code=FeeReasonCode.POLICY_STATIC.value,
                            channel_info=channel_info
                        )
                except Exception:
                    pass
```

- [ ] **Step 7: Run all fee tests**

Run: `python3 -m pytest tests/test_fee_hive_bias.py tests/test_fee_controller.py -v --tb=short`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add modules/fee_controller.py tests/test_fee_hive_bias.py
git commit -m "feat: consume member hint for 0-PPM fleet policy with gossip oscillation protection"
```

---

### Task 4: Cross-plugin contract test

**Files:**
- Create: `tests/test_hive_contract.py`

- [ ] **Step 1: Write the contract test**

```python
"""
Cross-plugin contract test: validates cl_revenue_ops correctly
parses the exact hint schema that cl-hive produces.

The GOLDEN_HIVE_SNAPSHOT fixture below matches the output of
cl-hive's hive-export-hints RPC (modules/rpc_commands.py:export_hints).
"""

import time
import pytest
from unittest.mock import MagicMock

from modules.hive_hints import HiveHintAdapter


# Exact schema cl-hive produces: integer competition_bias, boolean member,
# optional peer_quality_score/traffic_confidence, nested channel_open_hint.
GOLDEN_HIVE_SNAPSHOT = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "peer_count": 3,
    "hints": {
        "02member_owner": {
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
                "reason": "underserved_corridor",
            },
        },
        "03nonmember_secondary": {
            "member": False,
            "corridor_role": "secondary",
            "competition_bias": -1,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source",
        },
        "02member_neutral": {
            "member": True,
            "corridor_role": "none",
            "competition_bias": 0,
            "rebalance_preference": "neutral",
        },
        "02no_member_field": {
            "corridor_role": "none",
            "competition_bias": 0,
            "rebalance_preference": "neutral",
        },
    },
}


@pytest.fixture
def adapter():
    plugin = MagicMock()
    plugin.rpc.call.return_value = GOLDEN_HIVE_SNAPSHOT
    a = HiveHintAdapter(plugin, ttl_override=0)
    a.poll()
    return a


class TestContractFeeBias:
    """Fee bias must produce correct direction for cl-hive's integer competition_bias."""

    def test_positive_competition_bias_raises_fee(self, adapter):
        bias = adapter.get_fee_bias("02member_owner")
        assert bias > 1.0, "competition_bias=1 (lean in) + owner should raise fee"

    def test_negative_competition_bias_lowers_fee(self, adapter):
        bias = adapter.get_fee_bias("03nonmember_secondary")
        assert bias < 1.0, "competition_bias=-1 (back off) + secondary should lower fee"

    def test_zero_competition_bias_no_competition_effect(self, adapter):
        bias = adapter.get_fee_bias("02member_neutral")
        # No traffic_confidence in this hint -> returns 1.0
        assert bias == 1.0

    def test_missing_traffic_confidence_returns_neutral(self, adapter):
        """Hint with competition_bias but no traffic_confidence -> 1.0."""
        assert adapter.get_fee_bias("02member_neutral") == 1.0

    def test_all_biases_within_hard_caps(self, adapter):
        for peer_id in GOLDEN_HIVE_SNAPSHOT["hints"]:
            bias = adapter.get_fee_bias(peer_id)
            assert 0.9 <= bias <= 1.1, f"{peer_id}: fee bias {bias} out of range"


class TestContractRebalanceBias:
    """Rebalance bias must produce correct direction for cl-hive's preference enum."""

    def test_sink_preference_raises_score(self, adapter):
        bias = adapter.get_rebalance_bias("02member_owner")
        assert bias > 1.0, "sink preference should raise rebalance priority"

    def test_source_preference_lowers_score(self, adapter):
        bias = adapter.get_rebalance_bias("03nonmember_secondary")
        assert bias < 1.0, "source preference should lower rebalance priority"

    def test_neutral_preference_is_neutral(self, adapter):
        bias = adapter.get_rebalance_bias("02member_neutral")
        assert bias == 1.0

    def test_all_biases_within_hard_caps(self, adapter):
        for peer_id in GOLDEN_HIVE_SNAPSHOT["hints"]:
            bias = adapter.get_rebalance_bias(peer_id)
            assert 0.85 <= bias <= 1.15, f"{peer_id}: rebal bias {bias} out of range"


class TestContractMembership:
    """is_hive_member must correctly read boolean member field."""

    def test_member_true(self, adapter):
        assert adapter.is_hive_member("02member_owner") is True

    def test_member_false(self, adapter):
        assert adapter.is_hive_member("03nonmember_secondary") is False

    def test_unknown_peer(self, adapter):
        assert adapter.is_hive_member("02unknown") is False

    def test_missing_member_field(self, adapter):
        assert adapter.is_hive_member("02no_member_field") is False


class TestContractChannelOpen:
    """Channel-open hints must parse correctly from cl-hive schema."""

    def test_open_hint_parsed(self, adapter):
        hint = adapter.get_channel_open_hint("02member_owner")
        assert hint["open_preference"] == "open"
        assert hint["suggested_size_bucket"] == "medium"
        assert hint["reason"] == "underserved_corridor"
        assert 0.0 <= hint["topology_confidence"] <= 1.0

    def test_no_open_hint_returns_empty(self, adapter):
        assert adapter.get_channel_open_hint("03nonmember_secondary") == {}

    def test_open_candidates_list(self, adapter):
        candidates = adapter.get_open_candidates()
        peer_ids = [pid for pid, _ in candidates]
        assert "02member_owner" in peer_ids
        assert "03nonmember_secondary" not in peer_ids
```

- [ ] **Step 2: Run contract tests**

Run: `python3 -m pytest tests/test_hive_contract.py -v`
Expected: All pass (after Task 1 and Task 2 fixes)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hive_contract.py
git commit -m "test: add cross-plugin contract test with cl-hive golden fixture"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add hive integration section to CLAUDE.md**

After the "Configuration Categories" section, add:

```markdown
### Hive Fleet Hint Integration

cl-revenue-ops optionally consumes fleet coordination hints from cl-hive via a single adapter:

**Module:** `modules/hive_hints.py` (`HiveHintAdapter`)

**Enable:** `revenue-ops-hive-hints-enabled = true` (disabled by default)

**How it works:**
- Polls `hive-export-hints` RPC once per fee cycle
- Caches snapshot with TTL (default 900s, override with `revenue-ops-hive-hints-ttl`)
- Exposes bounded bias lookups consumed by fee controller, rebalancer, and capacity planner

**Bias bounds (hard-coded, not configurable):**
- Fee: ±10% max (`get_fee_bias()`)
- Rebalance: ±15% max (`get_rebalance_bias()`)
- Member: 0 PPM categorical override (`is_hive_member()`)

**Fail-open:** If cl-hive is unavailable, hints are stale, or the feature is disabled, all lookups return neutral (1.0) and `is_hive_member()` returns False. Local safety rails are never bypassed.

**Gossip oscillation protection:** When a peer was assigned 0-PPM via the member hint, that fee is held for one additional TTL period after hints go stale. This prevents gossip churn from intermittent cl-hive availability.

**Hint fields consumed:**
- `member` → 0-PPM fleet policy (short-circuits fee pipeline before DTS+PID)
- `corridor_role` → fee bias (owner +3%, secondary -3%)
- `competition_bias` → fee bias (integer -1/0/1, ±2%)
- `traffic_confidence` → weights all biases (0.0-1.0)
- `peer_quality_score` → rebalance bias (±5%)
- `rebalance_preference` → rebalance bias (sink +5%, source -5%)
- `channel_open_hint` → capacity planner scoring (±30%)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add hive fleet hint integration section to CLAUDE.md"
```

---

### Task 6: Final validation

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: All pass (including 876+ existing + new tests)

- [ ] **Step 2: Verify no stale competition_bias values remain**

Run: `grep -rn 'competition_bias.*[0-9]\.' tests/ | grep -v '__pycache__'`
Expected: No matches (all float values like `1.2`, `0.8` should be replaced with integers)

- [ ] **Step 3: Commit if any fixups needed**

```bash
git add -A && git commit -m "fix: resolve any remaining test fixture issues"
```
