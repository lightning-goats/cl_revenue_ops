# Planner & Rebalancer RPC Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~130 redundant gossip RPCs per planner cycle and ~12 redundant `listfunds`/`getinfo` RPCs per rebalancer cycle by adding per-cycle caching.

**Architecture:** Two independent fixes. Task 1 adds per-cycle gossip caching to `CapacityPlanner` (dict cleared each cycle, no TTL needed). Task 2 migrates uncached `listfunds`/`getinfo` calls in the rebalancer to use the existing `rpc_cache`.

**Tech Stack:** Python 3.10+, pyln-client, pytest

**Spec:** `docs/plans/2026-04-05-planner-rebalancer-rpc-cache-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/capacity_planner.py:30-45` | Modify | Add cycle cache dicts to `__init__` |
| `modules/capacity_planner.py:118` | Modify | Call `_init_cycle_cache()` at top of `execute_cycle` |
| `modules/capacity_planner.py` (new methods) | Modify | Add `_init_cycle_cache`, `_get_cached_channels`, `_get_cached_node` |
| `modules/capacity_planner.py:768,816,932,1025,1067,1279` | Modify | Replace direct RPC with cache helpers |
| `modules/rebalancer.py:354,842,4786,2256` | Modify | Migrate to `rpc_cache` pattern |
| `tests/test_planner_rpc_cache.py` | Create | Tests for planner cycle cache |
| `tests/test_rebalancer_rpc_cache.py` | Create | Tests for rebalancer cache migration |

---

### Task 1: Planner per-cycle gossip cache

**Files:**
- Modify: `modules/capacity_planner.py:30-45, 118, 768, 816, 932, 1025, 1067, 1279`
- Create: `tests/test_planner_rpc_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_planner_rpc_cache.py
"""Tests for capacity planner per-cycle gossip cache."""

import pytest
from unittest.mock import MagicMock

from modules.capacity_planner import CapacityPlanner


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    p.rpc.getinfo.return_value = {"id": "02our_node"}
    return p


@pytest.fixture
def planner(mock_plugin):
    return CapacityPlanner(mock_plugin, MagicMock(), MagicMock())


class TestInitCycleCache:
    """_init_cycle_cache fetches listnodes once and indexes by ID."""

    def test_populates_nodes_by_id(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": [
            {"nodeid": "02aaa", "alias": "Alice", "addresses": []},
            {"nodeid": "02bbb", "alias": "Bob", "addresses": [{"type": "ipv4"}]},
        ]}
        planner._init_cycle_cache()
        assert "02aaa" in planner._cycle_nodes_by_id
        assert "02bbb" in planner._cycle_nodes_by_id
        assert planner._cycle_nodes_by_id["02bbb"]["alias"] == "Bob"

    def test_clears_previous_cycle(self, planner, mock_plugin):
        planner._cycle_channels_dest["old_peer"] = [{"stale": True}]
        planner._cycle_channels_source["old_peer"] = [{"stale": True}]
        planner._cycle_nodes_by_id["old_node"] = {"stale": True}

        mock_plugin.rpc.listnodes.return_value = {"nodes": []}
        planner._init_cycle_cache()

        assert "old_peer" not in planner._cycle_channels_dest
        assert "old_peer" not in planner._cycle_channels_source
        assert "old_node" not in planner._cycle_nodes_by_id

    def test_handles_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.side_effect = Exception("RPC timeout")
        planner._init_cycle_cache()  # Should not raise
        assert planner._cycle_nodes_by_id == {}


class TestGetCachedChannels:
    """_get_cached_channels caches listchannels per peer per direction."""

    def test_caches_destination_channels(self, planner, mock_plugin):
        channels = [{"source": "02a", "fee_per_millionth": 100}]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}

        result1 = planner._get_cached_channels("02peer", "destination")
        result2 = planner._get_cached_channels("02peer", "destination")

        assert result1 == channels
        assert result2 == channels
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_caches_source_channels(self, planner, mock_plugin):
        channels = [{"destination": "02b", "fee_per_millionth": 200}]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}

        result = planner._get_cached_channels("02peer", "source")
        assert result == channels
        mock_plugin.rpc.listchannels.assert_called_once_with(source="02peer")

    def test_different_peers_separate_cache(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        planner._get_cached_channels("02peer_a", "destination")
        planner._get_cached_channels("02peer_b", "destination")
        assert mock_plugin.rpc.listchannels.call_count == 2

    def test_returns_empty_on_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.side_effect = Exception("timeout")
        result = planner._get_cached_channels("02peer", "destination")
        assert result == []

    def test_caches_empty_on_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.side_effect = Exception("timeout")
        planner._get_cached_channels("02peer", "destination")
        planner._get_cached_channels("02peer", "destination")
        assert mock_plugin.rpc.listchannels.call_count == 1


class TestGetCachedNode:
    """_get_cached_node returns from indexed dict, falls back to RPC."""

    def test_returns_from_preloaded_dict(self, planner, mock_plugin):
        planner._cycle_nodes_by_id["02peer"] = {"nodeid": "02peer", "alias": "Test"}
        result = planner._get_cached_node("02peer")
        assert result["alias"] == "Test"
        mock_plugin.rpc.listnodes.assert_not_called()

    def test_falls_back_to_rpc_if_missing(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": [
            {"nodeid": "02peer", "alias": "Found"}
        ]}
        result = planner._get_cached_node("02peer")
        assert result["alias"] == "Found"
        mock_plugin.rpc.listnodes.assert_called_once_with(id="02peer")

    def test_caches_rpc_fallback(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": [
            {"nodeid": "02peer", "alias": "Found"}
        ]}
        planner._get_cached_node("02peer")
        planner._get_cached_node("02peer")
        # Second call hits cache, not RPC
        assert mock_plugin.rpc.listnodes.call_count == 1

    def test_returns_none_on_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.side_effect = Exception("timeout")
        result = planner._get_cached_node("02peer")
        assert result is None

    def test_returns_none_for_empty_result(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": []}
        result = planner._get_cached_node("02peer")
        assert result is None


class TestScoringUsesCache:
    """_score_candidate uses cached channels/nodes, not direct RPC."""

    def test_line_1067_uses_cached_channels(self, planner, mock_plugin):
        """listchannels(destination=) in _score_candidate uses cache."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"satoshis": 10_000_000, "active": True},
            {"satoshis": 8_000_000, "active": True},
        ]}
        planner._cycle_nodes_by_id["02peer"] = {"nodeid": "02peer", "addresses": []}

        planner._score_candidate("02peer", 1.0)
        planner._score_candidate("02peer", 1.0)

        # listchannels called once (cached), listnodes not called (preloaded)
        assert mock_plugin.rpc.listchannels.call_count == 1
        mock_plugin.rpc.listnodes.assert_not_called()

    def test_line_1279_reuses_scoring_cache(self, planner, mock_plugin):
        """_size_channel's listchannels(destination=) hits cache from _score_candidate."""
        channels = [
            {"satoshis": 5_000_000, "active": True},
            {"satoshis": 3_000_000, "active": True},
        ]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}
        planner._cycle_nodes_by_id["02peer"] = {"nodeid": "02peer", "addresses": []}

        # Score first (populates cache)
        planner._score_candidate("02peer", 1.0)
        # Size second (should hit cache)
        cfg = MagicMock()
        cfg.planner_min_channel_sats = 500_000
        cfg.planner_max_channel_sats = 16_000_000
        candidates = [{"peer_id": "02peer", "score": 1.0}]
        planner._size_channel(candidates[0], candidates, 10_000_000, cfg)

        # Only 1 listchannels call total
        assert mock_plugin.rpc.listchannels.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_planner_rpc_cache.py -v`
Expected: FAIL — `_init_cycle_cache`, `_get_cached_channels`, `_get_cached_node` don't exist

- [ ] **Step 3: Add cycle cache fields to `__init__`**

In `modules/capacity_planner.py`, after line 45 (`self._last_cycle_ts: int = 0`), add:

```python
        # Per-cycle gossip cache (cleared at start of each execute_cycle)
        self._cycle_nodes_by_id: Dict[str, dict] = {}
        self._cycle_channels_dest: Dict[str, list] = {}
        self._cycle_channels_source: Dict[str, list] = {}
```

- [ ] **Step 4: Add cache helper methods**

In `modules/capacity_planner.py`, insert before `execute_cycle` (before line 118):

```python
    def _init_cycle_cache(self):
        """Fetch listnodes once and index by ID. Clear per-peer caches."""
        self._cycle_channels_dest.clear()
        self._cycle_channels_source.clear()
        self._cycle_nodes_by_id.clear()
        try:
            nodes = self.plugin.rpc.listnodes().get("nodes", [])
            self._cycle_nodes_by_id = {n["nodeid"]: n for n in nodes if "nodeid" in n}
        except Exception:
            pass

    def _get_cached_channels(self, peer_id: str, direction: str = "destination") -> list:
        """Get listchannels result, cached for this cycle."""
        cache = self._cycle_channels_dest if direction == "destination" else self._cycle_channels_source
        if peer_id in cache:
            return cache[peer_id]
        try:
            if direction == "destination":
                result = self.plugin.rpc.listchannels(destination=peer_id).get("channels", [])
            else:
                result = self.plugin.rpc.listchannels(source=peer_id).get("channels", [])
        except Exception:
            result = []
        cache[peer_id] = result
        return result

    def _get_cached_node(self, peer_id: str) -> dict | None:
        """Get node info, preferring preloaded dict, falling back to RPC."""
        if peer_id in self._cycle_nodes_by_id:
            return self._cycle_nodes_by_id[peer_id]
        try:
            nodes = self.plugin.rpc.listnodes(id=peer_id).get("nodes", [])
            if nodes:
                self._cycle_nodes_by_id[peer_id] = nodes[0]
                return nodes[0]
        except Exception:
            pass
        return None

```

- [ ] **Step 5: Call `_init_cycle_cache()` at top of `execute_cycle`**

In `modules/capacity_planner.py`, at the start of `execute_cycle`, after the `cfg` setup (after line 121), add:

```python
        self._init_cycle_cache()
```

- [ ] **Step 6: Refactor line 768 — `_discover_from_neighbors`**

Replace:
```python
                channels = self.plugin.rpc.listchannels(source=patron_peer_id)
```
With:
```python
                channels = {"channels": self._get_cached_channels(patron_peer_id, "source")}
```

- [ ] **Step 7: Refactor line 816 — `_discover_from_graph`**

Replace:
```python
        try:
            nodes = self.plugin.rpc.listnodes().get("nodes", [])
        except Exception:
            return []
```
With:
```python
        nodes = list(self._cycle_nodes_by_id.values())
        if not nodes:
            return []
```

(The graph was already fetched by `_init_cycle_cache`. If it failed, `_cycle_nodes_by_id` is empty, which maps to the old empty-return behavior.)

- [ ] **Step 8: Refactor line 932 — `_discover_from_route_pairs`**

Replace:
```python
                channels = self.plugin.rpc.listchannels(source=route_peer)
```
With:
```python
                channels = {"channels": self._get_cached_channels(route_peer, "source")}
```

- [ ] **Step 9: Refactor line 1025 — `_score_candidate` clearnet check**

Replace:
```python
        try:
            node_info = self.plugin.rpc.listnodes(id=peer_id)
            nodes = node_info.get("nodes", [])
            if nodes:
                addresses = nodes[0].get("addresses", [])
```
With:
```python
        try:
            node = self._get_cached_node(peer_id)
            if node:
                addresses = node.get("addresses", [])
```

And remove the old `nodes` variable — the indented block inside becomes:
```python
        try:
            node = self._get_cached_node(peer_id)
            if node:
                addresses = node.get("addresses", [])
                has_clearnet = any(
                    a.get("type") in ("ipv4", "ipv6")
                    for a in addresses
                )
                if has_clearnet:
                    score *= 1.25
        except Exception:
            pass
```

- [ ] **Step 10: Refactor line 1067 — `_score_candidate` large-channel check**

Replace:
```python
        try:
            peer_channels = self.plugin.rpc.listchannels(destination=peer_id)
            capacities = [
                ch.get("satoshis", 0)
                for ch in peer_channels.get("channels", [])
                if ch.get("active", False) and ch.get("satoshis", 0) > 0
            ]
```
With:
```python
        try:
            peer_channels = self._get_cached_channels(peer_id, "destination")
            capacities = [
                ch.get("satoshis", 0)
                for ch in peer_channels
                if ch.get("active", False) and ch.get("satoshis", 0) > 0
            ]
```

- [ ] **Step 11: Refactor line 1279 — `_size_channel` competitive sizing**

Replace:
```python
                peer_channels = self.plugin.rpc.listchannels(destination=peer_id)
                capacities = [
                    ch.get("satoshis", 0)
                    for ch in peer_channels.get("channels", [])
                    if ch.get("active", False) and ch.get("satoshis", 0) > 0
                ]
```
With:
```python
                peer_channels = self._get_cached_channels(peer_id, "destination")
                capacities = [
                    ch.get("satoshis", 0)
                    for ch in peer_channels
                    if ch.get("active", False) and ch.get("satoshis", 0) > 0
                ]
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_planner_rpc_cache.py -v`
Expected: All PASS

- [ ] **Step 13: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 14: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/capacity_planner.py tests/test_planner_rpc_cache.py
git commit -m "perf: add per-cycle gossip cache to capacity planner

Fetches listnodes() once per cycle and indexes by node ID. Caches
listchannels(source/destination) per-peer within the cycle. Eliminates
~100 redundant RPCs per 10-minute planner cycle. Line 1279 (channel
sizing) now hits cache populated by line 1067 (scoring)."
```

---

### Task 2: Rebalancer `rpc_cache` migration

**Files:**
- Modify: `modules/rebalancer.py:354, 842, 2256, 4786`
- Create: `tests/test_rebalancer_rpc_cache.py`

`JobManager` (lines 354, 842) is deprecated but still called from `monitor_jobs`. It has no `rpc_cache`. Rather than injecting one, we use the simplest fix: `JobManager` gets access to the `EVRebalancer`'s cache via a reference set after construction.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rebalancer_rpc_cache.py
"""Tests for rebalancer rpc_cache migration — listfunds and getinfo."""

import pytest
from unittest.mock import MagicMock

from modules.rebalancer import EVRebalancer, JobManager


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_wallet_reserve = 500_000
    c.snapshot.return_value = c
    return c


@pytest.fixture
def mock_database():
    db = MagicMock()
    db.get_rebalance_budget_used_24h.return_value = 0
    return db


@pytest.fixture
def mock_rpc_cache():
    cache = MagicMock()
    cache.listfunds.return_value = {
        "outputs": [{"amount_msat": 1_000_000_000, "status": "confirmed"}],
        "channels": [
            {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
             "state": "CHANNELD_NORMAL"},
        ]
    }
    cache.getinfo.return_value = {"blockheight": 900000}
    return cache


class TestCapitalControlsUsesCache:
    """_check_capital_controls uses rpc_cache.listfunds when available."""

    def test_uses_rpc_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = mock_rpc_cache

        result = ev._check_capital_controls(mock_config)
        mock_rpc_cache.listfunds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()

    def test_falls_back_without_cache(self, mock_plugin, mock_config, mock_database):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = None

        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 1_000_000_000, "status": "confirmed"}],
            "channels": [
                {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
                 "state": "CHANNELD_NORMAL"},
            ]
        }

        result = ev._check_capital_controls(mock_config)
        mock_plugin.rpc.listfunds.assert_called()


class TestChannelAgeDaysUsesCache:
    """_get_channel_age_days uses rpc_cache.getinfo when available."""

    def test_uses_rpc_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = mock_rpc_cache

        age = ev._get_channel_age_days("800000x1x0")
        mock_rpc_cache.getinfo.assert_called()
        mock_plugin.rpc.getinfo.assert_not_called()
        assert age > 0

    def test_falls_back_without_cache(self, mock_plugin, mock_config, mock_database):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = None
        mock_plugin.rpc.getinfo.return_value = {"blockheight": 900000}

        age = ev._get_channel_age_days("800000x1x0")
        mock_plugin.rpc.getinfo.assert_called()


class TestJobManagerUsesCache:
    """JobManager listfunds calls use rpc_cache when available."""

    def test_get_channel_local_balance_uses_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        jm = JobManager(mock_plugin, mock_config, mock_database)
        jm.rpc_cache = mock_rpc_cache

        balance = jm._get_channel_local_balance("800000x1x0")
        mock_rpc_cache.listfunds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()
        assert balance == 500_000  # 500_000_000 msat / 1000

    def test_get_local_balances_map_uses_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        jm = JobManager(mock_plugin, mock_config, mock_database)
        jm.rpc_cache = mock_rpc_cache

        balances = jm._get_local_balances_map()
        mock_rpc_cache.listfunds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()
        assert "800000x1x0" in balances

    def test_falls_back_without_cache(self, mock_plugin, mock_config, mock_database):
        jm = JobManager(mock_plugin, mock_config, mock_database)
        jm.rpc_cache = None
        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [],
            "channels": [
                {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
                 "state": "CHANNELD_NORMAL"},
            ]
        }

        balance = jm._get_channel_local_balance("800000x1x0")
        mock_plugin.rpc.listfunds.assert_called()
        assert balance == 500_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_rebalancer_rpc_cache.py -v`
Expected: FAIL — `rpc_cache` attribute not on `JobManager`, `_check_capital_controls` and `_get_channel_age_days` don't use cache

- [ ] **Step 3: Add `rpc_cache` to `JobManager.__init__`**

In `modules/rebalancer.py`, at line 224 (end of `JobManager.__init__`), add:

```python
        self.rpc_cache = None  # Shared RPC cache (injected after construction)
```

- [ ] **Step 4: Wire `rpc_cache` from `EVRebalancer` to `JobManager`**

In `modules/rebalancer.py`, in `EVRebalancer.__init__`, after line 1926 (`self.job_manager = JobManager(...)`), the `rpc_cache` is set later by the main plugin. Find where `EVRebalancer.rpc_cache` is set and ensure `job_manager.rpc_cache` is also set. The cleanest way: add a property setter.

After line 1932 (`self.rpc_cache = None`), replace the simple attribute with a property pattern. Actually, simpler: just set it in the existing setter pattern. Search for where `ev_rebalancer.rpc_cache = rpc_cache` is called in `cl-revenue-ops.py` and add `ev_rebalancer.job_manager.rpc_cache = rpc_cache` on the next line. 

Alternatively, just propagate in `_get_channel_local_balance` and `_get_local_balances_map` by checking `self.rpc_cache`:

In `JobManager._get_channel_local_balance` (line 354), replace:
```python
            listfunds = self.plugin.rpc.listfunds()
```
With:
```python
            listfunds = self.rpc_cache.listfunds() if self.rpc_cache else self.plugin.rpc.listfunds()
```

In `JobManager._get_local_balances_map` (line 842), replace:
```python
            listfunds = self.plugin.rpc.listfunds()
```
With:
```python
            listfunds = self.rpc_cache.listfunds() if self.rpc_cache else self.plugin.rpc.listfunds()
```

- [ ] **Step 5: Migrate `_check_capital_controls` (line 4786)**

Replace:
```python
                listfunds = self.plugin.rpc.listfunds()
```
With:
```python
                listfunds = self.rpc_cache.listfunds() if self.rpc_cache else self.plugin.rpc.listfunds()
```

- [ ] **Step 6: Migrate `_get_channel_age_days` (line 2256)**

Replace:
```python
            getinfo = self.plugin.rpc.getinfo()
```
With:
```python
            getinfo = self.rpc_cache.getinfo() if self.rpc_cache else self.plugin.rpc.getinfo()
```

- [ ] **Step 7: Wire `rpc_cache` to `JobManager` in main plugin**

In `cl-revenue-ops.py`, find where `ev_rebalancer.rpc_cache` is set. Add `ev_rebalancer.job_manager.rpc_cache = rpc_cache` on the next line. Search for `rpc_cache` assignments in `cl-revenue-ops.py` to find the exact location.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_rebalancer_rpc_cache.py -v`
Expected: All PASS

- [ ] **Step 9: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/rebalancer.py tests/test_rebalancer_rpc_cache.py cl-revenue-ops.py
git commit -m "perf: migrate rebalancer listfunds/getinfo to rpc_cache

Moves 3 uncached listfunds() calls and 1 uncached getinfo() call to
use the existing rpc_cache (30s TTL). JobManager gets rpc_cache
injected via EVRebalancer. Eliminates ~12 redundant RPCs per 15-min
rebalance cycle."
```

---

## Verification Checklist

- [ ] `python3 -m pytest tests/test_planner_rpc_cache.py -v` — all tests pass
- [ ] `python3 -m pytest tests/test_rebalancer_rpc_cache.py -v` — all tests pass
- [ ] `python3 -m pytest tests/test_capacity_planner.py -v` — existing tests still pass
- [ ] `python3 -m pytest tests/ -q` — full suite passes
- [ ] `grep "self.plugin.rpc.listchannels" modules/capacity_planner.py` — no direct calls remain in discovery/scoring
- [ ] `grep "self.plugin.rpc.listnodes" modules/capacity_planner.py` — only in `_get_cached_node` fallback
- [ ] `grep "self.plugin.rpc.listfunds" modules/rebalancer.py` — only in non-hot paths or with cache fallback pattern
- [ ] Deploy and monitor: planner cycles should complete faster, no `listchannels`/`listnodes` timeouts
