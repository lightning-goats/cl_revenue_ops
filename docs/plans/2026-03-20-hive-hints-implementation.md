# Hive Hints Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow cl_revenue_ops to incorporate cl_hive fleet recommendations as small bounded soft biases on fee targets and rebalance candidate ranking.

**Architecture:** A single new module (`modules/hive_hints.py`) polls a local `hive-export-hints` RPC, validates and caches the snapshot with TTL, and exposes two lookup helpers that return bounded multiplicative bias factors. Fee controller and rebalancer each get one thin call site — bias is applied before existing safety rails.

**Tech Stack:** Python 3.10+, pyln-client RPC, pytest

---

### Task 1: Add hive_hints config fields

**Files:**
- Modify: `modules/config.py:36-143` (CONFIG_FIELD_TYPES, CONFIG_FIELD_RANGES)
- Modify: `modules/config.py:245-420` (Config dataclass)
- Modify: `modules/config.py:700-790` (ConfigSnapshot frozen dataclass)

**Step 1: Add type registrations**

In `CONFIG_FIELD_TYPES` (around line 131, after `'planner_max_fee_rate_sat_vb': float`), add:

```python
    # Hive Hints
    'hive_hints_enabled': bool,
    'hive_hints_ttl_seconds': int,
```

**Step 2: Add range constraint**

In `CONFIG_FIELD_RANGES` (around line 234, after the planner block), add:

```python
    'hive_hints_ttl_seconds': (60, 7200),
```

**Step 3: Add Config dataclass fields**

After the planner fields (around line 412), add:

```python
    # Hive Hints integration
    hive_hints_enabled: bool = False
    hive_hints_ttl_seconds: int = 0  # 0 = use snapshot's ttl_seconds
```

**Step 4: Add ConfigSnapshot fields**

After the planner fields (around line 787), add:

```python
    # Hive Hints
    hive_hints_enabled: bool = False
    hive_hints_ttl_seconds: int = 0
```

**Step 5: Run tests**

Run: `python3 -m pytest tests/test_config.py -v -x`
Expected: PASS (existing tests should still pass with new defaults)

**Step 6: Commit**

```bash
git add modules/config.py
git commit -m "feat(config): add hive_hints_enabled and hive_hints_ttl_seconds fields"
```

---

### Task 2: Register plugin options and wire config loading

**Files:**
- Modify: `cl-revenue-ops.py:757-760` (after planner-enabled option block)
- Modify: `cl-revenue-ops.py:1144-1148` (config_kwargs construction)

**Step 1: Register plugin options**

After the planner option block (around line 790, find the last `plugin.add_option` call for planner), add:

```python
plugin.add_option(
    name='revenue-ops-hive-hints-enabled',
    default='false',
    description='Enable bounded fee/rebalance bias from cl_hive fleet hints (default: false)'
)
plugin.add_option(
    name='revenue-ops-hive-hints-ttl',
    default='0',
    description='Override hint snapshot TTL in seconds; 0 = use snapshot value (default: 0)'
)
```

**Step 2: Wire config loading**

In the config_kwargs dict (around line 1148, after the planner lines), add:

```python
        hive_hints_enabled=options.get('revenue-ops-hive-hints-enabled', 'false').lower() in ('true', '1', 'yes'),
        hive_hints_ttl_seconds=_safe_int('revenue-ops-hive-hints-ttl'),
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat(options): register hive-hints-enabled and hive-hints-ttl plugin options"
```

---

### Task 3: Build hive_hints adapter module — tests first

**Files:**
- Create: `tests/test_hive_hints.py`
- Create: `modules/hive_hints.py`

**Step 1: Write failing tests for HiveHintAdapter**

Create `tests/test_hive_hints.py`:

```python
"""Tests for hive_hints adapter module."""

import time
import pytest
from unittest.mock import MagicMock, patch

from modules.hive_hints import HiveHintAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


VALID_SNAPSHOT = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "hints": {
        "02aabbcc": {
            "member": True,
            "corridor_role": "owner",
            "competition_bias": 1.2,
            "peer_quality_score": 0.82,
            "traffic_confidence": 0.74,
            "rebalance_preference": "sink",
        },
        "02ddeeff": {
            "member": True,
            "corridor_role": "secondary",
            "competition_bias": 0.8,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source",
        },
    },
}


# ---------------------------------------------------------------------------
# Polling and caching
# ---------------------------------------------------------------------------

class TestPolling:
    def test_poll_success_caches_snapshot(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is not None
        assert adapter._snapshot["hints"]["02aabbcc"]["corridor_role"] == "owner"

    def test_poll_rpc_failure_keeps_last_good(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        first_snapshot = adapter._snapshot

        mock_plugin.rpc.call.side_effect = Exception("connection refused")
        adapter.poll()
        assert adapter._snapshot is first_snapshot  # kept last good

    def test_poll_rpc_failure_no_prior_snapshot(self, mock_plugin):
        mock_plugin.rpc.call.side_effect = Exception("connection refused")
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_invalid_schema_no_generated_at(self, mock_plugin):
        mock_plugin.rpc.call.return_value = {"hints": {}}
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_invalid_schema_no_hints_dict(self, mock_plugin):
        mock_plugin.rpc.call.return_value = {"generated_at": 123, "ttl_seconds": 900}
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_invalid_hints_not_dict(self, mock_plugin):
        mock_plugin.rpc.call.return_value = {"generated_at": 123, "ttl_seconds": 900, "hints": "bad"}
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None


# ---------------------------------------------------------------------------
# TTL / freshness
# ---------------------------------------------------------------------------

class TestTTL:
    def test_fresh_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_fresh()

    def test_stale_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000  # older than 900s TTL
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert not adapter.is_fresh()

    def test_ttl_override(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 500  # 500s old
        snapshot["ttl_seconds"] = 300  # would be stale with snapshot TTL
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=900)  # but override says 900
        adapter.poll()
        assert adapter.is_fresh()

    def test_no_snapshot_is_not_fresh(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert not adapter.is_fresh()


# ---------------------------------------------------------------------------
# Fee bias
# ---------------------------------------------------------------------------

class TestFeeBias:
    def test_owner_corridor_biases_up(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02aabbcc")
        assert bias > 1.0
        assert bias <= 1.1  # hard cap

    def test_secondary_corridor_biases_down(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02ddeeff")
        assert bias < 1.0
        assert bias >= 0.9  # hard cap

    def test_unknown_peer_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02unknown") == 1.0

    def test_stale_snapshot_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02aabbcc") == 1.0

    def test_no_snapshot_returns_neutral(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert adapter.get_fee_bias("02aabbcc") == 1.0

    def test_fee_bias_hard_cap(self, mock_plugin):
        """Even extreme hint values cannot exceed ±10%."""
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02extreme": {
                    "corridor_role": "owner",
                    "competition_bias": 100.0,  # absurd value
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02extreme")
        assert 0.9 <= bias <= 1.1

    def test_zero_traffic_confidence_neutralizes(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02lowconf": {
                    "corridor_role": "owner",
                    "competition_bias": 1.5,
                    "traffic_confidence": 0.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02lowconf") == 1.0

    def test_missing_optional_fields_degrade_gracefully(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02minimal": {
                    "member": True,
                    # no corridor_role, competition_bias, or traffic_confidence
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02minimal") == 1.0


# ---------------------------------------------------------------------------
# Rebalance bias
# ---------------------------------------------------------------------------

class TestRebalanceBias:
    def test_sink_preference_biases_up(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_rebalance_bias("02aabbcc")
        assert bias > 1.0
        assert bias <= 1.15

    def test_source_preference_biases_down(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_rebalance_bias("02ddeeff")
        assert bias < 1.0
        assert bias >= 0.85

    def test_unknown_peer_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_rebalance_bias("02unknown") == 1.0

    def test_stale_snapshot_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_rebalance_bias("02aabbcc") == 1.0

    def test_no_snapshot_returns_neutral(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert adapter.get_rebalance_bias("02aabbcc") == 1.0

    def test_rebalance_bias_hard_cap(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02extreme": {
                    "rebalance_preference": "sink",
                    "peer_quality_score": 100.0,  # absurd
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_rebalance_bias("02extreme")
        assert 0.85 <= bias <= 1.15


# ---------------------------------------------------------------------------
# Status / diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_status_when_no_snapshot(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        status = adapter.get_status()
        assert status["snapshot_fresh"] is False
        assert status["hints_count"] == 0

    def test_status_with_fresh_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        status = adapter.get_status()
        assert status["snapshot_fresh"] is True
        assert status["hints_count"] == 2
        assert "snapshot_age_seconds" in status
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_hive_hints.py -v -x`
Expected: FAIL (module doesn't exist yet)

**Step 3: Implement HiveHintAdapter**

Create `modules/hive_hints.py`:

```python
"""
Hive Hints adapter — sole integration boundary with cl_hive.

Polls a local cl_hive RPC for a compact hint snapshot, validates and
caches it with TTL, and exposes bounded multiplicative bias factors
for the fee controller and rebalancer.

If hints are missing, stale, invalid, or the RPC fails, all lookups
silently return 1.0 (neutral / no effect).
"""

import time

# ---------------------------------------------------------------------------
# Hard-coded bias caps — not configurable by design
# ---------------------------------------------------------------------------
MAX_FEE_BIAS = 0.10          # ±10% max fee effect
MAX_REBALANCE_BIAS = 0.15    # ±15% max rebalance score effect

# Per-field contribution weights (sum to roughly MAX when all maxed)
FEE_CORRIDOR_WEIGHT = 0.03   # corridor_role: ±3%
FEE_COMPETITION_WEIGHT = 0.02  # competition_bias: ±2%
REBAL_PREFERENCE_WEIGHT = 0.05  # rebalance_preference: ±5%
REBAL_QUALITY_WEIGHT = 0.05    # peer_quality_score: ±5%

VALID_CORRIDOR_ROLES = {"owner", "secondary"}
VALID_REBALANCE_PREFS = {"sink", "source"}


class HiveHintAdapter:
    """Adapter that polls cl_hive for fleet hints and exposes bounded bias lookups."""

    def __init__(self, plugin, ttl_override: int = 0):
        """
        Args:
            plugin: pyln Plugin reference (for RPC and logging).
            ttl_override: If >0, override the snapshot's ttl_seconds.
        """
        self._plugin = plugin
        self._ttl_override = ttl_override
        self._snapshot = None
        self._snapshot_fetched_at = 0

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self):
        """Fetch a fresh hint snapshot from cl_hive. Fail-open on any error."""
        try:
            raw = self._plugin.rpc.call("hive-export-hints")
        except Exception as e:
            self._plugin.log(
                f"HIVE_HINTS: poll failed: {e}", level='debug'
            )
            return  # keep last good snapshot

        if not self._validate_snapshot(raw):
            self._plugin.log(
                "HIVE_HINTS: invalid snapshot schema, ignoring", level='debug'
            )
            return  # keep last good snapshot

        self._snapshot = raw
        self._snapshot_fetched_at = int(time.time())

    @staticmethod
    def _validate_snapshot(raw) -> bool:
        """Validate top-level schema. Per-peer fields are validated at read time."""
        if not isinstance(raw, dict):
            return False
        if not isinstance(raw.get("generated_at"), (int, float)):
            return False
        if not isinstance(raw.get("hints"), dict):
            return False
        return True

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def _effective_ttl(self) -> int:
        if self._ttl_override > 0:
            return self._ttl_override
        if self._snapshot and isinstance(self._snapshot.get("ttl_seconds"), (int, float)):
            return int(self._snapshot["ttl_seconds"])
        return 900  # default

    def is_fresh(self) -> bool:
        if self._snapshot is None:
            return False
        age = int(time.time()) - int(self._snapshot.get("generated_at", 0))
        return age <= self._effective_ttl()

    # ------------------------------------------------------------------
    # Peer hint lookup (internal)
    # ------------------------------------------------------------------

    def _get_peer_hint(self, peer_id: str) -> dict:
        """Return hint dict for peer, or empty dict if unavailable/stale."""
        if not self.is_fresh():
            return {}
        hints = self._snapshot.get("hints", {})
        return hints.get(peer_id, {})

    # ------------------------------------------------------------------
    # Fee bias
    # ------------------------------------------------------------------

    def get_fee_bias(self, peer_id: str) -> float:
        """
        Return a multiplicative fee bias in [1 - MAX_FEE_BIAS, 1 + MAX_FEE_BIAS].

        Returns 1.0 (neutral) if hints are unavailable, stale, or the peer
        has no actionable hint data.
        """
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return 1.0

        confidence = hint.get("traffic_confidence")
        if not isinstance(confidence, (int, float)) or confidence <= 0:
            return 1.0

        confidence = min(confidence, 1.0)
        bias = 0.0

        # corridor_role contribution
        role = hint.get("corridor_role")
        if role == "owner":
            bias += FEE_CORRIDOR_WEIGHT
        elif role == "secondary":
            bias -= FEE_CORRIDOR_WEIGHT

        # competition_bias contribution (neutral = 1.0, range assumed 0-2)
        comp = hint.get("competition_bias")
        if isinstance(comp, (int, float)):
            comp = max(0.0, min(2.0, comp))
            bias += (comp - 1.0) * FEE_COMPETITION_WEIGHT

        # Scale by confidence
        bias *= confidence

        # Hard clamp
        bias = max(-MAX_FEE_BIAS, min(MAX_FEE_BIAS, bias))

        return 1.0 + bias

    # ------------------------------------------------------------------
    # Rebalance bias
    # ------------------------------------------------------------------

    def get_rebalance_bias(self, peer_id: str) -> float:
        """
        Return a multiplicative rebalance score bias in
        [1 - MAX_REBALANCE_BIAS, 1 + MAX_REBALANCE_BIAS].

        Returns 1.0 (neutral) if hints are unavailable, stale, or the peer
        has no actionable hint data.
        """
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return 1.0

        confidence = hint.get("traffic_confidence")
        if not isinstance(confidence, (int, float)) or confidence <= 0:
            return 1.0

        confidence = min(confidence, 1.0)
        bias = 0.0

        # rebalance_preference contribution
        pref = hint.get("rebalance_preference")
        if pref == "sink":
            bias += REBAL_PREFERENCE_WEIGHT
        elif pref == "source":
            bias -= REBAL_PREFERENCE_WEIGHT

        # peer_quality_score contribution (0-1, neutral = 0.5)
        quality = hint.get("peer_quality_score")
        if isinstance(quality, (int, float)):
            quality = max(0.0, min(1.0, quality))
            bias += (quality - 0.5) * 2.0 * REBAL_QUALITY_WEIGHT

        # Scale by confidence
        bias *= confidence

        # Hard clamp
        bias = max(-MAX_REBALANCE_BIAS, min(MAX_REBALANCE_BIAS, bias))

        return 1.0 + bias

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return concise status dict for revenue-status / debug surfaces."""
        if self._snapshot is None:
            return {
                "snapshot_fresh": False,
                "snapshot_age_seconds": None,
                "hints_count": 0,
            }
        age = int(time.time()) - int(self._snapshot.get("generated_at", 0))
        hints = self._snapshot.get("hints", {})
        return {
            "snapshot_fresh": self.is_fresh(),
            "snapshot_age_seconds": age,
            "hints_count": len(hints),
        }
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_hive_hints.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/hive_hints.py tests/test_hive_hints.py
git commit -m "feat: add hive_hints adapter with polling, caching, TTL, and bounded bias"
```

---

### Task 4: Instantiate HiveHintAdapter and wire polling in main plugin

**Files:**
- Modify: `cl-revenue-ops.py:1374-1407` (module instantiation in init())
- Modify: `cl-revenue-ops.py` (import block and fee_controller_loop / rebalance_loop)

**Step 1: Add import**

Near the top of `cl-revenue-ops.py` where other modules are imported, add:

```python
from modules.hive_hints import HiveHintAdapter
```

**Step 2: Add module-level variable**

Near other module-level variables (`fee_controller = None`, `rebalancer = None`, etc.), add:

```python
hive_hints: Optional[HiveHintAdapter] = None
```

**Step 3: Instantiate in init()**

After the rebalancer wiring (around line 1406), add:

```python
    # Hive Hints adapter (sole integration boundary with cl_hive)
    global hive_hints
    if config.hive_hints_enabled:
        hive_hints = HiveHintAdapter(
            safe_plugin,
            ttl_override=config.hive_hints_ttl_seconds,
        )
        plugin.log("HiveHintAdapter initialized — fleet hint bias enabled")
    else:
        hive_hints = None
```

**Step 4: Inject adapter into fee_controller and rebalancer**

After the HiveHintAdapter instantiation:

```python
    if fee_controller is not None:
        fee_controller.hive_hints = hive_hints
    if rebalancer is not None:
        rebalancer.hive_hints = hive_hints
```

**Step 5: Add polling to fee_controller_loop**

In the `fee_controller_loop` function, at the start of each cycle (before `fee_controller.run_fee_cycle(cfg)`), add:

```python
            # Poll hive hints once per fee cycle
            if hive_hints is not None:
                try:
                    hive_hints.poll()
                except Exception:
                    pass  # fail-open
```

**Step 6: Run tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 7: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat: instantiate HiveHintAdapter and poll in fee cycle"
```

---

### Task 5: Integrate fee bias — tests first

**Files:**
- Create: `tests/test_fee_hive_bias.py`
- Modify: `modules/fee_controller.py:1589-1608` (__init__)
- Modify: `modules/fee_controller.py:3449-3453` (DTS+PID target)
- Modify: `modules/fee_controller.py:3472-3474` (decision_reason)

**Step 1: Write failing integration test**

Create `tests/test_fee_hive_bias.py`:

```python
"""Tests for hive hint bias integration in fee controller."""

import time
import pytest
from unittest.mock import MagicMock, PropertyMock

from modules.fee_controller import FeeController
from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_fee_ppm = 10
    c.max_fee_ppm = 5000
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


class TestFeeHiveBias:
    def test_get_hive_fee_bias_with_adapter(self, mock_plugin, mock_config, mock_database):
        """Fee controller returns bias from adapter when available."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {
                    "corridor_role": "owner",
                    "competition_bias": 1.0,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        fc.hive_hints = adapter
        bias = fc._get_hive_fee_bias("02aabb")
        assert bias > 1.0
        assert bias <= 1.1

    def test_get_hive_fee_bias_no_adapter(self, mock_plugin, mock_config, mock_database):
        """Fee controller returns 1.0 when no adapter is set."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = None
        assert fc._get_hive_fee_bias("02aabb") == 1.0

    def test_get_hive_fee_bias_exception_returns_neutral(self, mock_plugin, mock_config, mock_database):
        """Fee controller returns 1.0 if adapter raises."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.side_effect = Exception("boom")
        fc.hive_hints = adapter
        assert fc._get_hive_fee_bias("02aabb") == 1.0

    def test_bias_within_hard_cap(self, mock_plugin, mock_config, mock_database):
        """Bias returned by _get_hive_fee_bias is always in [0.9, 1.1]."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 1.5  # adapter somehow returns out of range
        fc.hive_hints = adapter
        bias = fc._get_hive_fee_bias("02aabb")
        assert 0.9 <= bias <= 1.1
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_fee_hive_bias.py -v -x`
Expected: FAIL (method doesn't exist)

**Step 3: Add hive_hints attribute and _get_hive_fee_bias method to FeeController**

In `__init__` (around line 1608), add:

```python
        # Hive hints adapter (injected by main plugin; None = disabled)
        self.hive_hints = None
```

Add method (after `__init__`, before the first existing method):

```python
    def _get_hive_fee_bias(self, peer_id: str) -> float:
        """Return bounded multiplicative fee bias from hive hints. 1.0 if unavailable."""
        if self.hive_hints is None:
            return 1.0
        try:
            bias = self.hive_hints.get_fee_bias(peer_id)
            return max(0.9, min(1.1, bias))
        except Exception:
            return 1.0
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fee_hive_bias.py -v`
Expected: All PASS

**Step 5: Inject bias at the DTS+PID target computation**

At line 3450 in `fee_controller.py`, change:

```python
            # Before:
            raw_dts_target_ppm = int(dts_fee)
            post_pid_target_ppm = int(dts_fee * pid_multiplier)
            bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, post_pid_target_ppm))
```

To:

```python
            raw_dts_target_ppm = int(dts_fee)
            post_pid_target_ppm = int(dts_fee * pid_multiplier)
            # Hive hint bias: small bounded nudge before hard clamp
            hive_fee_bias = self._get_hive_fee_bias(peer_id)
            if hive_fee_bias != 1.0:
                post_pid_target_ppm = int(post_pid_target_ppm * hive_fee_bias)
            bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, post_pid_target_ppm))
```

**Step 6: Add hive bias to decision_reason string**

At line 3472, change:

```python
            # Before:
            decision_reason = (
                f"dts_pid (dts={dts_fee}, pid={pid_multiplier:.2f}, "
                f"flow={flow_state_str})"
            )
```

To:

```python
            hive_tag = f", hive={hive_fee_bias:.2f}" if hive_fee_bias != 1.0 else ""
            decision_reason = (
                f"dts_pid (dts={dts_fee}, pid={pid_multiplier:.2f}, "
                f"flow={flow_state_str}{hive_tag})"
            )
```

**Step 7: Run full tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 8: Commit**

```bash
git add modules/fee_controller.py tests/test_fee_hive_bias.py
git commit -m "feat: integrate hive hint fee bias into DTS+PID pipeline"
```

---

### Task 6: Integrate rebalance bias — tests first

**Files:**
- Create: `tests/test_rebalance_hive_bias.py`
- Modify: `modules/rebalancer.py:1875-1902` (__init__)
- Modify: `modules/rebalancer.py:2333-2337` (sort_key)

**Step 1: Write failing integration test**

Create `tests/test_rebalance_hive_bias.py`:

```python
"""Tests for hive hint bias integration in rebalancer."""

import time
import pytest
from unittest.mock import MagicMock

from modules.rebalancer import EVRebalancer
from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.max_concurrent_jobs = 3
    c.sling_job_timeout_seconds = 300
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


class TestRebalanceHiveBias:
    def test_get_hive_rebalance_bias_with_adapter(self, mock_plugin, mock_config, mock_database):
        """Rebalancer returns bias from adapter when available."""
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {
                    "rebalance_preference": "sink",
                    "peer_quality_score": 0.9,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        reb.hive_hints = adapter
        bias = reb._get_hive_rebalance_bias("02aabb")
        assert bias > 1.0
        assert bias <= 1.15

    def test_get_hive_rebalance_bias_no_adapter(self, mock_plugin, mock_config, mock_database):
        """Rebalancer returns 1.0 when no adapter is set."""
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        reb.hive_hints = None
        assert reb._get_hive_rebalance_bias("02aabb") == 1.0

    def test_get_hive_rebalance_bias_exception_returns_neutral(self, mock_plugin, mock_config, mock_database):
        """Rebalancer returns 1.0 if adapter raises."""
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_rebalance_bias.side_effect = Exception("boom")
        reb.hive_hints = adapter
        assert reb._get_hive_rebalance_bias("02aabb") == 1.0

    def test_bias_within_hard_cap(self, mock_plugin, mock_config, mock_database):
        """Bias is always clamped to [0.85, 1.15]."""
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_rebalance_bias.return_value = 2.0  # out of range
        reb.hive_hints = adapter
        bias = reb._get_hive_rebalance_bias("02aabb")
        assert 0.85 <= bias <= 1.15
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_hive_bias.py -v -x`
Expected: FAIL

**Step 3: Add hive_hints attribute and _get_hive_rebalance_bias to EVRebalancer**

In `__init__` (around line 1902, after `self.job_manager`), add:

```python
        # Hive hints adapter (injected by main plugin; None = disabled)
        self.hive_hints = None
```

Add method (after `__init__`):

```python
    def _get_hive_rebalance_bias(self, peer_id: str) -> float:
        """Return bounded multiplicative rebalance score bias from hive hints. 1.0 if unavailable."""
        if self.hive_hints is None:
            return 1.0
        try:
            bias = self.hive_hints.get_rebalance_bias(peer_id)
            return max(0.85, min(1.15, bias))
        except Exception:
            return 1.0
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_hive_bias.py -v`
Expected: All PASS

**Step 5: Inject bias into sort_key**

At lines 2333-2337 in `rebalancer.py`, change:

```python
            # Before:
            def sort_key(c):
                dest_state = self.database.get_channel_state(c.to_channel)
                flow_state = dest_state.get("state", "balanced") if dest_state else "balanced"
                priority = 2 if flow_state == "source" else 1
                return (priority, c.expected_profit_sats)
```

To:

```python
            def sort_key(c):
                dest_state = self.database.get_channel_state(c.to_channel)
                flow_state = dest_state.get("state", "balanced") if dest_state else "balanced"
                priority = 2 if flow_state == "source" else 1
                hive_bias = self._get_hive_rebalance_bias(c.to_peer_id)
                biased_profit = c.expected_profit_sats * hive_bias
                return (priority, biased_profit)
```

**Step 6: Run full tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 7: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalance_hive_bias.py
git commit -m "feat: integrate hive hint rebalance bias into candidate ranking"
```

---

### Task 7: Add diagnostics to status/debug RPCs

**Files:**
- Modify: `cl-revenue-ops.py:1866-1897` (revenue-status return dict)
- Modify: `cl-revenue-ops.py:2224-2317` (revenue-fee-debug return dict)
- Modify: `cl-revenue-ops.py:1900-2050` (revenue-rebalance-debug return dict)

**Step 1: Add hive_hints to revenue-status**

In the `revenue_status` return dict (around line 1896, before the closing `}`), add:

```python
        "hive_hints": hive_hints.get_status() if hive_hints else {"snapshot_fresh": False, "hints_count": 0},
```

**Step 2: Add hive_hints to revenue-fee-debug**

In the `revenue_fee_debug` function, find the `result` dict construction and add before `return result`:

```python
    result["hive_hints"] = hive_hints.get_status() if hive_hints else {"snapshot_fresh": False, "hints_count": 0}
```

**Step 3: Add hive_hints to revenue-rebalance-debug**

In the `revenue_rebalance_debug` function, add before the final return:

```python
    result["hive_hints"] = hive_hints.get_status() if hive_hints else {"snapshot_fresh": False, "hints_count": 0}
```

**Step 4: Run full tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 5: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat: add hive hints diagnostics to status and debug RPCs"
```

---

### Task 8: Add safety rail preservation tests

**Files:**
- Modify: `tests/test_hive_hints.py` (add safety rail tests)

**Step 1: Add tests proving hive cannot bypass safety rails**

Append to `tests/test_hive_hints.py`:

```python
# ---------------------------------------------------------------------------
# Safety rail preservation
# ---------------------------------------------------------------------------

class TestSafetyRails:
    """Prove that hive hints cannot override local safety logic."""

    def test_fee_bias_cannot_exceed_ten_percent(self, mock_plugin):
        """No combination of hint values can produce bias outside [0.9, 1.1]."""
        extreme_hints = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {},
        }
        # Test all extreme combinations
        for role in ["owner", "secondary", "unknown", None]:
            for comp in [0.0, 1.0, 2.0, 100.0, -50.0]:
                for conf in [0.0, 0.5, 1.0, 100.0]:
                    peer_id = f"02test_{role}_{comp}_{conf}"
                    hint = {"traffic_confidence": conf, "competition_bias": comp}
                    if role:
                        hint["corridor_role"] = role
                    extreme_hints["hints"][peer_id] = hint

        mock_plugin.rpc.call.return_value = extreme_hints
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        for peer_id in extreme_hints["hints"]:
            bias = adapter.get_fee_bias(peer_id)
            assert 0.9 <= bias <= 1.1, f"Fee bias {bias} out of range for {peer_id}"

    def test_rebalance_bias_cannot_exceed_fifteen_percent(self, mock_plugin):
        """No combination of hint values can produce bias outside [0.85, 1.15]."""
        extreme_hints = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {},
        }
        for pref in ["sink", "source", "unknown", None]:
            for quality in [0.0, 0.5, 1.0, 100.0, -50.0]:
                for conf in [0.0, 0.5, 1.0, 100.0]:
                    peer_id = f"02test_{pref}_{quality}_{conf}"
                    hint = {"traffic_confidence": conf, "peer_quality_score": quality}
                    if pref:
                        hint["rebalance_preference"] = pref
                    extreme_hints["hints"][peer_id] = hint

        mock_plugin.rpc.call.return_value = extreme_hints
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        for peer_id in extreme_hints["hints"]:
            bias = adapter.get_rebalance_bias(peer_id)
            assert 0.85 <= bias <= 1.15, f"Rebalance bias {bias} out of range for {peer_id}"

    def test_local_only_behavior_preserved_when_disabled(self, mock_plugin):
        """When no adapter is set, all biases are neutral."""
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        # Don't poll — simulates disabled state
        for peer_id in ["02aabb", "02ccdd", "02eeff"]:
            assert adapter.get_fee_bias(peer_id) == 1.0
            assert adapter.get_rebalance_bias(peer_id) == 1.0
```

**Step 2: Run all tests**

Run: `python3 -m pytest tests/test_hive_hints.py -v`
Expected: All PASS

**Step 3: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/test_hive_hints.py
git commit -m "test: add safety rail preservation tests for hive hints"
```

---

### Task 9: Final validation and push

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All pass

**Step 2: Verify no hive-specific code leaked outside boundary**

Run: `grep -r "hive-export-hints\|hive_hints\|HiveHint" modules/ --include="*.py" | grep -v hive_hints.py | grep -v __pycache__`

Expected: Only `fee_controller.py` and `rebalancer.py` should reference `hive_hints` (as an attribute), nothing else in modules/.

**Step 3: Push**

```bash
git push origin refactor/standalone-dts-pid
```
