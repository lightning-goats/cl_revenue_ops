# Hive Router Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract askrene hive routing into a shared module and integrate it into both the rebalancer and Boltz swap system for fleet-aware routing and channel selection.

**Architecture:** New `modules/hive_router.py` provides fleet route discovery via CLN askrene layers. The rebalancer's inline askrene code is replaced with HiveRouter calls. Boltz balance planning gains a topology score and loop-out execution uses fleet-preferred first hops. All consumers get fleet routing through a single shared module.

**Tech Stack:** Python 3.12+, CLN askrene RPC (v24.11+), pyln-client

**Spec:** `docs/superpowers/specs/2026-04-03-hive-router-module-design.md`

---

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `modules/hive_router.py` | Create | HiveRoute dataclass, HiveRouter class (layer mgmt, route discovery, topology scoring) |
| `modules/rebalancer.py` | Modify | Replace inline askrene with HiveRouter calls |
| `cl-revenue-ops.py` | Modify | Create and wire HiveRouter during init |
| `tests/test_hive_router.py` | Create | Unit tests for HiveRouter |

---

### Task 1: Create `modules/hive_router.py`

**Files:**
- Create: `modules/hive_router.py`

- [ ] **Step 1: Create the module with HiveRoute dataclass and HiveRouter skeleton**

```python
"""
hive_router — Shared askrene fleet route discovery for cl-revenue-ops.

Manages a transient askrene layer with zero-fee overrides for hive fleet
member channels.  Consumers (rebalancer, Boltz) call discover_route() to
find cheap circular paths through the fleet.

Degrades gracefully when askrene is unavailable (CLN < 24.11).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class HiveRoute:
    """A route discovered through the hive fleet via askrene."""
    fee_ppm: int
    hops: int
    source_scid: str
    path: List[Dict[str, Any]] = field(default_factory=list)
    probability_ppm: int = 0


class HiveRouter:
    """
    Fleet-aware route discovery using CLN askrene layers.

    Creates a transient 'hive-fleet' layer where fleet member channels
    are zero-fee, enabling getroutes to discover cheap circular paths.
    """

    LAYER_NAME = "hive-fleet"

    def __init__(self, plugin, hive_hints):
        """
        Args:
            plugin: CLN plugin reference for RPC calls
            hive_hints: HiveHintAdapter for fleet membership queries
        """
        self.plugin = plugin
        self.hive_hints = hive_hints
        self.available: bool = False
        self._member_ids: Set[str] = set()
        self._our_id: Optional[str] = None
        self._last_refresh: float = 0

    def _get_our_id(self) -> Optional[str]:
        if self._our_id:
            return self._our_id
        try:
            info = self.plugin.rpc.getinfo()
            self._our_id = info.get("id")
        except Exception:
            pass
        return self._our_id

    def _log(self, msg: str, level: str = "debug") -> None:
        if self.plugin:
            self.plugin.log(f"[HiveRouter] {msg}", level=level)

    # ------------------------------------------------------------------
    # Layer Management
    # ------------------------------------------------------------------

    def refresh_layer(self) -> bool:
        """Recreate the hive-fleet askrene layer with zero-fee fleet channels.

        Also applies positive node bias (+5) on fleet members to gently
        prefer fleet paths for multi-hop routes.

        Returns:
            True if layer was created successfully.
        """
        if not self.hive_hints or not self.plugin:
            return False

        try:
            # Remove stale layer
            try:
                self.plugin.rpc.call("askrene-remove-layer", {"layer": self.LAYER_NAME})
            except Exception:
                pass

            self.plugin.rpc.call("askrene-create-layer", {"layer": self.LAYER_NAME})

            our_id = self._get_our_id()
            if not our_id:
                return False

            channels = self.plugin.rpc.listpeerchannels()
            member_ids: Set[str] = set()
            updated = 0

            for ch in channels.get("channels", []):
                if ch.get("state") != "CHANNELD_NORMAL":
                    continue
                peer_id = ch.get("peer_id", "")
                scid = ch.get("short_channel_id", "")
                if not peer_id or not scid:
                    continue
                if not self.hive_hints.is_hive_member(peer_id):
                    continue

                member_ids.add(peer_id)

                for direction in (0, 1):
                    try:
                        self.plugin.rpc.call("askrene-update-channel", {
                            "layer": self.LAYER_NAME,
                            "short_channel_id_dir": f"{scid}/{direction}",
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        })
                        updated += 1
                    except Exception:
                        pass

            # Apply node-level bias so getroutes prefers fleet paths
            for mid in member_ids:
                for direction in ("in", "out"):
                    try:
                        self.plugin.rpc.call("askrene-bias-node", {
                            "layer": self.LAYER_NAME,
                            "node": mid,
                            "direction": direction,
                            "bias": 5,
                            "description": "hive fleet preference",
                        })
                    except Exception:
                        pass

            self._member_ids = member_ids
            self._last_refresh = time.time()
            self.available = updated > 0

            if updated > 0:
                self._log(
                    f"Refreshed layer ({updated} channel dirs at 0 fee, "
                    f"{len(member_ids)} fleet nodes biased)",
                )
            return self.available

        except Exception as e:
            self._log(f"Layer refresh failed (askrene likely unavailable): {e}")
            self.available = False
            return False

    # ------------------------------------------------------------------
    # Route Discovery
    # ------------------------------------------------------------------

    def discover_route(
        self, dest_peer_id: str, amount_sats: int
    ) -> Optional[HiveRoute]:
        """Find cheapest route through fleet to a destination peer.

        Args:
            dest_peer_id: Destination node pubkey
            amount_sats: Amount to route

        Returns:
            HiveRoute if found, None otherwise.
        """
        if not self.available or not self.plugin:
            return None

        our_id = self._get_our_id()
        if not our_id:
            return None

        amount_msat = amount_sats * 1000
        max_fee_msat = amount_msat // 100  # 1% discovery cap

        try:
            result = self.plugin.rpc.call("getroutes", {
                "source": our_id,
                "destination": dest_peer_id,
                "amount_msat": amount_msat,
                "layers": ["auto.localchans", "auto.sourcefree", self.LAYER_NAME],
                "maxfee_msat": max_fee_msat,
                "final_cltv": 18,
            })

            routes = result.get("routes", [])
            if not routes:
                return None

            route = routes[0]
            path = route.get("path", [])
            if not path:
                return None

            first_hop_amount = path[0].get("amount_msat", amount_msat)
            total_fee_msat = first_hop_amount - amount_msat
            fee_ppm = (total_fee_msat * 1_000_000) // amount_msat if amount_msat > 0 else 0
            source_scid = path[0].get("short_channel_id", "")
            probability = result.get("probability_ppm", 0)

            self._log(
                f"Route to {dest_peer_id[:12]}...: "
                f"{len(path)} hops, {fee_ppm} ppm, via {source_scid}, "
                f"prob={probability/10000:.1f}%",
                level="info",
            )

            return HiveRoute(
                fee_ppm=fee_ppm,
                hops=len(path),
                source_scid=source_scid,
                path=path,
                probability_ppm=probability,
            )

        except Exception as e:
            self._log(f"Route discovery to {dest_peer_id[:12]}... failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Membership Helpers
    # ------------------------------------------------------------------

    def get_hive_members(self) -> Set[str]:
        """Return cached set of fleet member pubkeys."""
        return set(self._member_ids)

    def is_hive_member(self, peer_id: str) -> bool:
        """Check if peer is a fleet member (uses cached set)."""
        return peer_id in self._member_ids

    # ------------------------------------------------------------------
    # Topology Scoring
    # ------------------------------------------------------------------

    def score_channel_for_hive(
        self,
        peer_id: str,
        direction: str,
        liquidity_ratio: float = 0.5,
    ) -> float:
        """Score how beneficial a Boltz swap on this channel is for fleet topology.

        Returns a multiplier: 1.0 = neutral, >1.0 = beneficial.

        Args:
            peer_id: Channel peer's pubkey
            direction: 'out' for loop-out, 'in' for loop-in
            liquidity_ratio: Current local/capacity ratio (0-1)
        """
        if not self._member_ids:
            return 1.0

        if peer_id in self._member_ids:
            if direction == "out":
                # Loop-out through fleet peer: creates inbound on fleet channel.
                # More beneficial when our side is heavy (high ratio).
                boost = 1.2 + 0.3 * max(0, liquidity_ratio - 0.5)
                return min(1.5, boost)
            elif direction == "in":
                # Loop-in on fleet peer channel: builds outbound toward fleet.
                # More beneficial when our side is light (low ratio).
                boost = 1.1 + 0.2 * max(0, 0.5 - liquidity_ratio)
                return min(1.3, boost)

        # Non-fleet peer — check if any fleet member is adjacent
        # (hive_hints may know from gossip state, but we don't have that
        # data here without RPC; return neutral for now)
        return 1.0
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('modules/hive_router.py').read()); print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add modules/hive_router.py
git commit -m "feat: add HiveRouter module for shared askrene fleet route discovery"
```

---

### Task 2: Create tests for HiveRouter

**Files:**
- Create: `tests/test_hive_router.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for HiveRouter module."""

import time
from unittest.mock import MagicMock, patch, call
import pytest

from modules.hive_router import HiveRouter, HiveRoute


class MockHiveHints:
    def __init__(self, members=None):
        self._members = set(members or [])

    def is_hive_member(self, peer_id):
        return peer_id in self._members


class TestHiveRouterInit:
    def test_defaults(self):
        router = HiveRouter(plugin=MagicMock(), hive_hints=MockHiveHints())
        assert router.available is False
        assert router.get_hive_members() == set()
        assert router.is_hive_member("abc") is False

    def test_no_hints(self):
        router = HiveRouter(plugin=MagicMock(), hive_hints=None)
        assert router.refresh_layer() is False
        assert router.available is False

    def test_no_plugin(self):
        router = HiveRouter(plugin=None, hive_hints=MockHiveHints())
        assert router.refresh_layer() is False


class TestHiveRouterRefresh:
    def test_refresh_creates_layer_and_updates_channels(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "peer_id": "fleet_a", "short_channel_id": "100x1x0"},
                {"state": "CHANNELD_NORMAL", "peer_id": "external", "short_channel_id": "200x1x0"},
                {"state": "CHANNELD_NORMAL", "peer_id": "fleet_b", "short_channel_id": "300x1x0"},
            ]
        }
        plugin.rpc.call.return_value = {}

        hints = MockHiveHints(members=["fleet_a", "fleet_b"])
        router = HiveRouter(plugin, hints)
        result = router.refresh_layer()

        assert result is True
        assert router.available is True
        assert router.get_hive_members() == {"fleet_a", "fleet_b"}
        assert router.is_hive_member("fleet_a") is True
        assert router.is_hive_member("external") is False

    def test_refresh_fails_gracefully_when_askrene_unavailable(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        plugin.rpc.call.side_effect = Exception("Unknown method askrene-create-layer")

        router = HiveRouter(plugin, MockHiveHints(["fleet_a"]))
        result = router.refresh_layer()

        assert result is False
        assert router.available is False


class TestHiveRouterDiscover:
    def test_discover_returns_route(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.call.return_value = {
            "probability_ppm": 850000,
            "routes": [{
                "path": [
                    {"short_channel_id": "100x1x0", "amount_msat": 501000},
                    {"short_channel_id": "200x1x0", "amount_msat": 500000},
                ]
            }]
        }

        router = HiveRouter(plugin, MockHiveHints())
        router.available = True
        router._our_id = "our_id"

        route = router.discover_route("dest_peer", 500)
        assert route is not None
        assert route.source_scid == "100x1x0"
        assert route.hops == 2
        assert route.fee_ppm == 2000  # (501000-500000)*1e6/500000 = 2000
        assert route.probability_ppm == 850000

    def test_discover_returns_none_when_unavailable(self):
        router = HiveRouter(MagicMock(), MockHiveHints())
        router.available = False
        assert router.discover_route("dest", 1000) is None

    def test_discover_returns_none_on_no_routes(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {"routes": []}

        router = HiveRouter(plugin, MockHiveHints())
        router.available = True
        router._our_id = "our_id"

        assert router.discover_route("dest", 1000) is None


class TestHiveRouterTopologyScore:
    def test_fleet_peer_loop_out_high_ratio(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("fleet_a", "out", liquidity_ratio=0.9)
        assert score > 1.3  # High ratio = very beneficial

    def test_fleet_peer_loop_out_balanced(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("fleet_a", "out", liquidity_ratio=0.5)
        assert 1.15 < score < 1.25

    def test_fleet_peer_loop_in_low_ratio(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("fleet_a", "in", liquidity_ratio=0.2)
        assert score > 1.15

    def test_non_fleet_peer_neutral(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("random_peer", "out", liquidity_ratio=0.9)
        assert score == 1.0

    def test_no_members_neutral(self):
        router = HiveRouter(MagicMock(), MockHiveHints())
        router._member_ids = set()
        assert router.score_channel_for_hive("any", "out") == 1.0
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/test_hive_router.py -v
```

Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hive_router.py
git commit -m "test: add HiveRouter unit tests"
```

---

### Task 3: Refactor rebalancer to use HiveRouter

**Files:**
- Modify: `modules/rebalancer.py:1911` (add hive_router field)
- Modify: `modules/rebalancer.py:2069-2200` (remove inline askrene methods)
- Modify: `modules/rebalancer.py:2437` (replace _askrene_available)
- Modify: `modules/rebalancer.py:~3048` (replace _discover_hive_route call)
- Modify: `modules/rebalancer.py:~3745` (replace is_hive_member call in source selection)

- [ ] **Step 1: Add hive_router field to EVRebalancer.__init__**

At `rebalancer.py:1911`, after `self.hive_hints = None`, add:

```python
        self.hive_router = None  # HiveRouter for fleet route discovery
```

- [ ] **Step 2: Remove inline _refresh_hive_askrene_layer and _discover_hive_route**

Delete the two methods at lines 2069-2200 (the entire `# ASKRENE HIVE ROUTE DISCOVERY` section). These are being replaced by HiveRouter.

- [ ] **Step 3: Replace layer refresh in find_rebalance_candidates**

At `rebalancer.py:2437`, change:

```python
            self._askrene_available = self._refresh_hive_askrene_layer()
```

To:

```python
            if self.hive_router:
                self.hive_router.refresh_layer()
```

- [ ] **Step 4: Replace route discovery in _analyze_rebalance_ev**

Find the hive route discovery block (around line 3048 after the refactor, search for `HIVE ROUTE DISCOVERY`). Replace:

```python
        hive_route = None
        if getattr(self, '_askrene_available', False) and dest_peer_id:
            hive_route = self._discover_hive_route(dest_peer_id, rebalance_amount)
```

With:

```python
        hive_route = None
        if self.hive_router and self.hive_router.available and dest_peer_id:
            hr = self.hive_router.discover_route(dest_peer_id, rebalance_amount)
            if hr:
                hive_route = {
                    "fee_ppm": hr.fee_ppm,
                    "hops": hr.hops,
                    "source_scid": hr.source_scid,
                }
```

- [ ] **Step 5: Replace is_hive_member in source selection**

In `_select_source_candidates`, find the hive source check (search for `is_hive_source`). Change:

```python
            is_hive_source = False
            if self.hive_hints and pid:
                is_hive_source = self.hive_hints.is_hive_member(pid)
```

To:

```python
            is_hive_source = False
            if pid:
                if self.hive_router:
                    is_hive_source = self.hive_router.is_hive_member(pid)
                elif self.hive_hints:
                    is_hive_source = self.hive_hints.is_hive_member(pid)
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 906 passed.

- [ ] **Step 7: Commit**

```bash
git add modules/rebalancer.py
git commit -m "refactor: replace inline askrene with HiveRouter in rebalancer"
```

---

### Task 4: Wire HiveRouter in cl-revenue-ops.py init

**Files:**
- Modify: `cl-revenue-ops.py:1508-1526` (init section)

- [ ] **Step 1: Import HiveRouter**

At the top of `cl-revenue-ops.py`, in the imports section, add:

```python
from modules.hive_router import HiveRouter
```

- [ ] **Step 2: Create HiveRouter and inject into rebalancer**

After the hive_hints initialization block (after line 1526), add:

```python
    # Hive Router (shared askrene fleet route discovery)
    global hive_router
    hive_router = None
    if hive_hints is not None:
        hive_router = HiveRouter(safe_plugin, hive_hints)
        plugin.log("HiveRouter initialized - fleet route discovery enabled")

    if rebalancer is not None and hive_router is not None:
        rebalancer.hive_router = hive_router
```

- [ ] **Step 3: Declare global at module level**

Near the other module-level globals (search for `hive_hints = None`), add:

```python
hive_router = None
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 906 passed.

- [ ] **Step 5: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat: wire HiveRouter into plugin init, inject into rebalancer"
```

---

### Task 5: Boltz balance plan — hive topology scoring

**Files:**
- Modify: `cl-revenue-ops.py:~5770` (multi-goal scoring in _build_boltz_balance_plan)

- [ ] **Step 1: Add hive topology score to loop-out scoring**

Find the multi-goal scoring block in `_build_boltz_balance_plan()` (search for `multi_goal_value`). After the existing `hive_bonus` line, add the topology score:

Change:

```python
            hive_bonus = hive_rebal_bias  # ±15% from fleet hints, compounds with route signal
            multi_goal_value = excess_ratio * (0.35 * roi_signal + 0.35 * fee_signal + 0.30) * flow_bonus * sling_bonus * planner_bonus * route_bonus * hive_bonus
```

To:

```python
            hive_bonus = hive_rebal_bias  # ±15% from fleet hints, compounds with route signal
            # Hive topology: prefer swaps that benefit fleet structure
            hive_topo = 1.0
            if hive_router and hive_router.available:
                hive_topo = hive_router.score_channel_for_hive(
                    peer_id, direction, liquidity_ratio=local_pct / 100.0
                )
            multi_goal_value = excess_ratio * (0.35 * roi_signal + 0.35 * fee_signal + 0.30) * flow_bonus * sling_bonus * planner_bonus * route_bonus * hive_bonus * hive_topo
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 906 passed.

- [ ] **Step 3: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat: add hive topology score to Boltz balance plan scoring"
```

---

### Task 6: Boltz loop-out — fleet-preferred first-hop routing

**Files:**
- Modify: `cl-revenue-ops.py:~6094` (loop-out execution in revenue_boltz_balance_cycle)

- [ ] **Step 1: Add hive route discovery before loop-out execution**

In `revenue_boltz_balance_cycle()`, find the loop-out execution block (search for `elif direction == "loop_out":`). Replace:

```python
            elif direction == "loop_out":
                currency = loop_out_currency
                if currency == "auto":
                    try:
                        currency = _select_boltz_currency("loop_out", amount_sats)
                    except Exception:
                        currency = "LBTC"
                res = bm.loop_out(amount_sats=amount_sats, channel_id=ch_id, peer_id=peer_id, currency=currency)
```

With:

```python
            elif direction == "loop_out":
                currency = loop_out_currency
                if currency == "auto":
                    try:
                        currency = _select_boltz_currency("loop_out", amount_sats)
                    except Exception:
                        currency = "LBTC"
                # Hive route discovery: find cheaper first-hop through fleet
                exec_ch_id = ch_id
                exec_peer_id = peer_id
                if hive_router and hive_router.available and peer_id:
                    try:
                        hr = hive_router.discover_route(peer_id, amount_sats)
                        if hr and hr.source_scid and hr.fee_ppm < 200:
                            exec_ch_id = hr.source_scid
                            plugin.log(
                                f"BOLTZ HIVE ROUTE: Using fleet path for loop-out "
                                f"({hr.hops} hops, {hr.fee_ppm} ppm, via {hr.source_scid})",
                            )
                    except Exception:
                        pass  # Fall back to original channel selection
                res = bm.loop_out(amount_sats=amount_sats, channel_id=exec_ch_id, peer_id=exec_peer_id, currency=currency)
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 906 passed.

- [ ] **Step 3: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat: use hive fleet routes for Boltz loop-out first-hop selection"
```

---

### Task 7: Final integration test

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/ -x -q --tb=short
```

Expected: 906 passed.

- [ ] **Step 2: Verify module imports work**

```bash
python3 -c "
from modules.hive_router import HiveRouter, HiveRoute
print('HiveRouter:', HiveRouter)
print('HiveRoute:', HiveRoute)
r = HiveRoute(fee_ppm=10, hops=2, source_scid='100x1x0')
print('Route:', r)
print('OK')
"
```

Expected: Prints classes and OK.

- [ ] **Step 3: Push**

```bash
git push
```
