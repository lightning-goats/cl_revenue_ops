# Gossip Channel Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~252 redundant `listchannels` and `getinfo` RPCs per fee cycle by caching gossip data and node identity.

**Architecture:** Add `_get_our_id()` (cached forever) and `_get_peer_inbound_channels(peer_id)` (cached 30 min via existing `_neighbor_fee_cache` dict). Refactor `_get_neighbor_fee_median` and `_get_competitive_undercut_pct` to use these instead of direct RPC calls. Pass `neighbor_median` to undercut function from caller to eliminate redundant internal call.

**Tech Stack:** Python 3.10+, pyln-client, pytest

**Spec:** `docs/plans/2026-04-05-listchannels-gossip-cache-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/fee_controller.py:1675` | Modify | Add `_our_node_id` field to `__init__` |
| `modules/fee_controller.py:1819-1894` | Modify | Add `_get_our_id()`, `_get_peer_inbound_channels()`, refactor `_get_neighbor_fee_median` |
| `modules/fee_controller.py:1896-1951` | Modify | Refactor `_get_competitive_undercut_pct` to use cache and accept `neighbor_median` param |
| `modules/fee_controller.py:3919` | Modify | Pass `neighbor_median` to `_get_competitive_undercut_pct` |
| `tests/test_gossip_cache.py` | Create | Tests for cached node ID, cached gossip channels, refactored functions |

---

### Task 1: Add `_get_our_id()` and `_get_peer_inbound_channels()`, refactor both consumer functions

**Files:**
- Modify: `modules/fee_controller.py:1675, 1819-1951, 3919`
- Create: `tests/test_gossip_cache.py`

All changes are tightly coupled (the refactored functions depend on the new helpers), so they're one task.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gossip_cache.py
"""Tests for gossip channel cache — eliminates redundant listchannels/getinfo RPCs."""

import time
import pytest
from unittest.mock import MagicMock, call

from modules.fee_controller import FeeController


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
    c.thompson_prior_std_fee = 100
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


@pytest.fixture
def fc(mock_plugin, mock_config, mock_database):
    mock_plugin.rpc.getinfo.return_value = {"id": "02our_node_id"}
    return FeeController(mock_plugin, mock_config, mock_database)


class TestGetOurId:
    """_get_our_id() caches node identity forever."""

    def test_returns_node_id(self, fc, mock_plugin):
        assert fc._get_our_id() == "02our_node_id"

    def test_caches_after_first_call(self, fc, mock_plugin):
        fc._get_our_id()
        fc._get_our_id()
        fc._get_our_id()
        # getinfo called only once (during _get_our_id, not during __init__)
        # Filter to only getinfo calls made after construction
        mock_plugin.rpc.getinfo.reset_mock()
        fc._get_our_id()
        mock_plugin.rpc.getinfo.assert_not_called()

    def test_handles_empty_id(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {}
        fc = FeeController(mock_plugin, mock_config, mock_database)
        assert fc._get_our_id() == ""


class TestGetPeerInboundChannels:
    """_get_peer_inbound_channels() caches listchannels(destination=) for 30 min."""

    def test_returns_channel_list(self, fc, mock_plugin):
        channels = [
            {"source": "02node1", "fee_per_millionth": 100, "active": True},
            {"source": "02node2", "fee_per_millionth": 200, "active": True},
        ]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}
        result = fc._get_peer_inbound_channels("02peer")
        assert result == channels

    def test_caches_for_30_minutes(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [{"source": "02a"}]}
        fc._get_peer_inbound_channels("02peer")
        fc._get_peer_inbound_channels("02peer")
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_cache_expires_after_30_minutes(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [{"source": "02a"}]}
        fc._get_peer_inbound_channels("02peer")
        # Expire the cache entry
        cache_key = "gossip_channels_02peer"
        fc._neighbor_fee_cache[cache_key]["ts"] = time.time() - 1801
        fc._get_peer_inbound_channels("02peer")
        assert mock_plugin.rpc.listchannels.call_count == 2

    def test_different_peers_cached_separately(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        fc._get_peer_inbound_channels("02peer_a")
        fc._get_peer_inbound_channels("02peer_b")
        assert mock_plugin.rpc.listchannels.call_count == 2

    def test_returns_empty_on_rpc_error(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.side_effect = Exception("RPC timeout")
        result = fc._get_peer_inbound_channels("02peer")
        assert result == []

    def test_caches_empty_on_rpc_error(self, fc, mock_plugin):
        """RPC error caches [] to avoid hammering a failing RPC."""
        mock_plugin.rpc.listchannels.side_effect = Exception("RPC timeout")
        fc._get_peer_inbound_channels("02peer")
        fc._get_peer_inbound_channels("02peer")
        assert mock_plugin.rpc.listchannels.call_count == 1


class TestNeighborMedianUsesCache:
    """_get_neighbor_fee_median uses _get_peer_inbound_channels, not direct RPC."""

    def test_no_direct_listchannels_call(self, fc, mock_plugin):
        """After calling _get_neighbor_fee_median, listchannels should be called
        via _get_peer_inbound_channels (destination= kwarg), not directly."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": f"02node{i}", "fee_per_millionth": 100 + i * 50, "active": True}
            for i in range(5)
        ]}
        fc._get_neighbor_fee_median("02peer")
        # Should have called listchannels(destination="02peer")
        mock_plugin.rpc.listchannels.assert_called_once_with(destination="02peer")

    def test_no_direct_getinfo_call(self, fc, mock_plugin):
        """_get_neighbor_fee_median should use _get_our_id, not direct getinfo."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": f"02node{i}", "fee_per_millionth": 100 + i * 50, "active": True}
            for i in range(5)
        ]}
        mock_plugin.rpc.getinfo.reset_mock()
        fc._get_neighbor_fee_median("02peer")
        # getinfo not called again (already cached from first _get_our_id call)
        mock_plugin.rpc.getinfo.assert_not_called()


class TestUndercutUsesCache:
    """_get_competitive_undercut_pct uses cached channels and accepts neighbor_median."""

    def test_uses_cached_channels(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
            {"source": "02comp2", "satoshis": 2000000, "active": True},
        ]}
        fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=200)
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_no_direct_getinfo_call(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
        ]}
        mock_plugin.rpc.getinfo.reset_mock()
        fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=200)
        mock_plugin.rpc.getinfo.assert_not_called()

    def test_uses_passed_neighbor_median(self, fc, mock_plugin):
        """When neighbor_median is passed, doesn't call _get_neighbor_fee_median."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
        ]}
        # High-fee corridor (>300) should add 0.05 to base undercut
        pct = fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=400)
        assert pct >= 0.10  # Base + high-fee corridor bonus
        # listchannels called once (for channels), NOT twice (no internal _get_neighbor_fee_median)
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_none_median_skips_corridor_adjustment(self, fc, mock_plugin):
        """When neighbor_median is None, corridor adjustment is skipped."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
        ]}
        pct = fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=None)
        # Should still return a valid undercut (rank-based only, no corridor adj)
        assert 0.03 <= pct <= 0.20


class TestSharedCacheIntegration:
    """Both functions share the same gossip cache — second call is free."""

    def test_median_then_undercut_one_rpc(self, fc, mock_plugin):
        """Calling median then undercut for same peer uses only 1 listchannels RPC."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "fee_per_millionth": 200, "active": True},
            {"source": "02comp1", "satoshis": 500000, "fee_per_millionth": 100, "active": True},
            {"source": "02comp2", "satoshis": 800000, "fee_per_millionth": 150, "active": True},
            {"source": "02comp3", "satoshis": 1200000, "fee_per_millionth": 200, "active": True},
        ]}
        median = fc._get_neighbor_fee_median("02peer")
        pct = fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=median)
        # Only 1 listchannels call total (shared cache)
        assert mock_plugin.rpc.listchannels.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_gossip_cache.py -v`
Expected: FAIL — `_get_our_id` and `_get_peer_inbound_channels` don't exist yet

- [ ] **Step 3: Add `_our_node_id` field to `__init__`**

In `modules/fee_controller.py`, after line 1675 (`self._neighbor_fee_cache: Dict[str, Dict] = {}`), add:

```python
        self._our_node_id: str = ""
```

- [ ] **Step 4: Add `_get_our_id()` method**

In `modules/fee_controller.py`, insert before `_get_neighbor_fee_median` (before line 1819):

```python
    def _get_our_id(self) -> str:
        """Return our node ID, cached forever (never changes at runtime)."""
        if not self._our_node_id:
            self._our_node_id = self.plugin.rpc.getinfo().get("id", "")
        return self._our_node_id

    def _get_peer_inbound_channels(self, peer_id: str) -> list:
        """Get channels pointing at peer_id, cached for 30 minutes.

        Uses the same cache dict as _get_neighbor_fee_median but with
        a different key prefix. Returns [] on RPC failure.
        """
        cache_key = f"gossip_channels_{peer_id}"
        cached = self._neighbor_fee_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < 1800:
            return cached["value"]

        try:
            channels = self.plugin.rpc.listchannels(destination=peer_id)
            result = channels.get("channels", [])
        except Exception:
            result = []

        self._neighbor_fee_cache[cache_key] = {"value": result, "ts": time.time()}
        return result

```

- [ ] **Step 5: Refactor `_get_neighbor_fee_median`**

In `modules/fee_controller.py`, in `_get_neighbor_fee_median`, replace lines 1854-1857:

```python
        try:
            now = time.time()
            our_id = self.plugin.rpc.getinfo().get("id", "")
            channels = self.plugin.rpc.listchannels(destination=peer_id)
```

With:

```python
        try:
            now = time.time()
            our_id = self._get_our_id()
            peer_channels = self._get_peer_inbound_channels(peer_id)
```

And replace line 1861 (the loop over channels):

```python
            for ch in channels.get("channels", []):
```

With:

```python
            for ch in peer_channels:
```

- [ ] **Step 6: Refactor `_get_competitive_undercut_pct`**

In `modules/fee_controller.py`, change the method signature at line 1896:

```python
    def _get_competitive_undercut_pct(self, peer_id: str, channel_id: str) -> float:
```

To:

```python
    def _get_competitive_undercut_pct(self, peer_id: str, channel_id: str, neighbor_median: int | None = None) -> float:
```

Replace lines 1909-1912:

```python
        try:
            our_id = self.plugin.rpc.getinfo().get("id", "")
            channels = self.plugin.rpc.listchannels(destination=peer_id)
            all_channels = channels.get("channels", [])
```

With:

```python
        try:
            our_id = self._get_our_id()
            all_channels = self._get_peer_inbound_channels(peer_id)
```

Replace line 1940:

```python
            neighbor_median = self._get_neighbor_fee_median(peer_id)
```

With:

```python
            if neighbor_median is None:
                neighbor_median = self._get_neighbor_fee_median(peer_id)
```

- [ ] **Step 7: Update caller to pass `neighbor_median`**

In `modules/fee_controller.py`, at line 3919:

```python
                undercut_pct = self._get_competitive_undercut_pct(peer_id, channel_id)
```

Change to:

```python
                undercut_pct = self._get_competitive_undercut_pct(peer_id, channel_id, neighbor_median)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_gossip_cache.py -v`
Expected: All PASS

- [ ] **Step 9: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass (existing `test_fee_intelligence_improvements.py` tests for neighbor median and undercut should still pass — they mock `listchannels` which now flows through `_get_peer_inbound_channels`)

- [ ] **Step 10: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/fee_controller.py tests/test_gossip_cache.py
git commit -m "perf: cache gossip channels and node ID to eliminate ~252 RPCs per fee cycle

Adds _get_our_id() (cached forever) and _get_peer_inbound_channels()
(cached 30 min) shared by _get_neighbor_fee_median and
_get_competitive_undercut_pct. Passes neighbor_median from caller
to avoid redundant internal lookup. Fixes listchannels RPC timeout
on nodes with large gossip tables."
```

---

## Verification Checklist

- [ ] `python3 -m pytest tests/test_gossip_cache.py -v` — all tests pass
- [ ] `python3 -m pytest tests/test_fee_intelligence_improvements.py -v` — existing tests still pass
- [ ] `python3 -m pytest tests/ -q` — full suite passes
- [ ] `grep "_get_our_id" modules/fee_controller.py` — method exists
- [ ] `grep "_get_peer_inbound_channels" modules/fee_controller.py` — method exists
- [ ] `grep "getinfo" modules/fee_controller.py | grep -v "_get_our_id" | grep -v "#" | grep -v '"""'` — no direct getinfo calls remain in median/undercut functions
- [ ] Deploy and check logs: `RPC timeout after 15s on listchannels` should no longer appear during fee cycles
