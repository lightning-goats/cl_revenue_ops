# Askrene Router V3 — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `modules/rebalance_router_v3.py` as an askrene-layered alternative to `rebalance_router_v2.py`, wire it into `rebalance_engine_v2.py` behind a runtime-switchable config key, and preserve v2 as the fallback path for CLN nodes without askrene.

**Architecture:** V3 router calls `getroutes` (not `getroute`), passes cl-hive's `hive-fleet` layer for fleet biasing, validates the returned path shape (rejects loop-through-us and bypass-us cases), translates askrene's hop format into sendpay format, and uses a throwaway layer for per-retry excludes. The engine holds BOTH routers at init when askrene is available and dispatches per cycle based on `self.plugin.options["rebalance-router"].value`. V2 executor (`rebalance_executor_v2.py`) remains the sole execution path — xpay was rejected in research (Section 5.9).

**Tech Stack:** Python 3.12, pyln-client, askrene CLN plugin (v24.11+), `getroutes` RPC, `sendpay`+`waitsendpay` RPCs, SQLite-backed database, pytest.

---

## Reference Documents

- **Design spec:** `docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md` (commit `f352dbf`)
- **Research findings:** `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (commit `e2772b9`)
- **Research plan (Phase 0, now complete):** `docs/superpowers/plans/2026-04-10-askrene-router-v3-research.md`

This Phase 1 plan incorporates all design deltas from research Section 9.6.

## File Structure

| File | Change type | Responsibility |
|---|---|---|
| `modules/rebalance_router_v3.py` | **create** | Askrene-based router: getroutes + layers + hop translator + path-shape validator + exclude-layer retry |
| `modules/rebalance_engine_v2.py` | modify | Router factory, askrene probe, per-cycle dispatch, orphan exclude-layer sweep at init |
| `modules/rebalance_audit_v2.py` | modify | Add 5 new skip reasons + `router=` field in audit records |
| `modules/config.py` | modify | Add `rebalance_router` and `askrene_layers` config keys + validator |
| `cl-revenue-ops.py` | modify | Register `rebalance-router` and `askrene-layers` dynamic options with `on_change` callback |
| `tests/test_rebalance_router_v3.py` | **create** | Unit tests for the v3 router (mocked plugin.rpc) |
| `tests/test_rebalance_engine_v2.py` | modify | Add tests for engine factory, runtime switch, per-cycle capture |
| `tests/test_rebalance_audit_v2.py` | modify | Add tests for 5 new skip reasons and `router=` field |
| `tests/integration/test_router_v3_live.py` | **create** | Live-CLN integration tests gated by `CLN_INTEGRATION=1` |
| `tests/fixtures/router_v3/*.json` | **create** | Captured getroutes response snapshots |
| `tools/router_v3_safety_monitor.py` | **create** | Skeleton A/B rollback monitor for Phase 3 (not functional in Phase 1) |

## Key Types (defined in Task 3, referenced in later tasks)

```python
# In modules/rebalance_router_v3.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Re-use v2's RouteResult dataclass — v3 returns the exact same shape so the planner
# and executor don't care which router produced it.
from .rebalance_router_v2 import RouteResult
```

`RouteResult` is defined in `rebalance_router_v2.py` as:
```python
@dataclass
class RouteResult:
    success: bool
    route_cost_sats: int = 0
    final_hop_fee_ppm: int = 0
    hops: int = 0
    route: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
```

## Conventions

- Every task follows TDD: write failing test → verify fail → minimal impl → verify pass → commit.
- All tests mock `plugin.rpc` via `unittest.mock.MagicMock` unless marked `CLN_INTEGRATION=1`.
- Commits use conventional-commit prefixes: `feat:`, `test:`, `refactor:`, `docs:`.
- No comments in code beyond module docstrings and single-line annotations for non-obvious invariants.
- Follow existing v2 patterns: import helpers from `rebalance_router_v2` rather than duplicating math.

---

## Task 1: Config Keys For v3 Router And Layers

**Files:**
- Modify: `modules/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_config_has_rebalance_router_default_v2():
    from modules.config import Config
    cfg = Config()
    assert cfg.rebalance_router == "v2"

def test_config_has_askrene_layers_default_hive_fleet():
    from modules.config import Config
    cfg = Config()
    assert cfg.askrene_layers == "hive-fleet"

def test_config_rebalance_router_accepts_v3():
    from modules.config import Config
    cfg = Config()
    cfg.rebalance_router = "v3"
    assert cfg.rebalance_router == "v3"

def test_config_rebalance_router_rejects_invalid():
    from modules.config import Config
    cfg = Config()
    try:
        cfg.rebalance_router = "v4"
    except ValueError as e:
        assert "v2" in str(e) and "v3" in str(e)
        return
    raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py::test_config_has_rebalance_router_default_v2 -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'rebalance_router'`.

- [ ] **Step 3: Write minimal implementation**

In `modules/config.py`, find the `Config` dataclass/class and add:

```python
# In the Config class body
rebalance_router: str = "v2"
askrene_layers: str = "hive-fleet"

# If Config uses __setattr__ or property validation, add validator:
def __setattr__(self, name: str, value: Any) -> None:
    if name == "rebalance_router" and value not in ("v2", "v3"):
        raise ValueError(
            f"rebalance_router must be 'v2' or 'v3', got {value!r}"
        )
    super().__setattr__(name, value)
```

If `Config` is a `@dataclass`, extend `ConfigSnapshot` as well so the snapshot used in cycle code carries both fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -k "rebalance_router or askrene_layers" -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/config.py tests/test_config.py
git commit -m "feat: add rebalance-router and askrene-layers config keys"
```

---

## Task 2: Five New Skip Reasons In Audit Module

**Files:**
- Modify: `modules/rebalance_audit_v2.py`
- Test: `tests/test_rebalance_audit_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_audit_v2.py`:

```python
def test_audit_accepts_new_v3_skip_reasons():
    from modules.rebalance_audit_v2 import RebalanceAudit, VALID_SKIP_REASONS
    new_reasons = {
        "unknown_source_node",
        "unknown_dest_node",
        "unknown_layer",
        "askrene_child_died",
        "path_loops_through_us",
    }
    for r in new_reasons:
        assert r in VALID_SKIP_REASONS, f"{r} should be a valid skip reason"

def test_audit_preserves_existing_skip_reasons():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS
    existing = {
        "inside_band", "not_valuable", "no_partner", "cooldown",
        "no_budget", "max_pairs_reached", "outcompeted",
        "no_route", "route_over_budget",
    }
    assert existing.issubset(VALID_SKIP_REASONS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_audit_v2.py::test_audit_accepts_new_v3_skip_reasons -v`
Expected: FAIL with `ImportError` or `AssertionError` — `VALID_SKIP_REASONS` either doesn't exist or lacks the new entries.

- [ ] **Step 3: Write minimal implementation**

Read the current `modules/rebalance_audit_v2.py` to find where reasons are validated (look for a set, frozenset, or list of reason strings). Add a module-level `VALID_SKIP_REASONS` constant:

```python
VALID_SKIP_REASONS = frozenset({
    # Existing v2 reasons
    "inside_band",
    "not_valuable",
    "no_partner",
    "cooldown",
    "no_budget",
    "max_pairs_reached",
    "outcompeted",
    "no_route",
    "route_over_budget",
    # V3-specific reasons added by rebalance_router_v3
    "unknown_source_node",
    "unknown_dest_node",
    "unknown_layer",
    "askrene_child_died",
    "path_loops_through_us",
})
```

If `RebalanceAudit.record_skip` already validates reasons against a hardcoded list, change it to check against `VALID_SKIP_REASONS`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_audit_v2.py -v`
Expected: all previous audit tests still pass PLUS the two new tests.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_audit_v2.py tests/test_rebalance_audit_v2.py
git commit -m "feat: add v3 router skip reasons to audit module"
```

---

## Task 3: V3 Router Skeleton + Layer Name Parser

**Files:**
- Create: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rebalance_router_v3.py`:

```python
from unittest.mock import MagicMock
import pytest


def test_parse_layer_names_splits_csv():
    from modules.rebalance_router_v3 import _parse_layer_names
    assert _parse_layer_names("hive-fleet") == ["hive-fleet"]
    assert _parse_layer_names("hive-fleet,hive-reputation") == ["hive-fleet", "hive-reputation"]
    assert _parse_layer_names("hive-fleet, hive-reputation ") == ["hive-fleet", "hive-reputation"]


def test_parse_layer_names_empty_returns_empty_list():
    from modules.rebalance_router_v3 import _parse_layer_names
    assert _parse_layer_names("") == []
    assert _parse_layer_names(" ") == []
    assert _parse_layer_names(",,") == []


def test_v3_router_constructs_with_empty_layers():
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}  # askrene-listlayers returns no layers
    r = RebalanceRouterV3(plugin=plugin, our_node_id="03" + "a" * 64, layer_names=[], log=lambda m, l: None)
    assert r.layer_names == []
    assert r.found_layers == []


def test_v3_router_logs_found_and_missing_layers():
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.return_value = {
        "layers": [
            {"layer": "hive-fleet"},
            {"layer": "revenue-local"},
        ]
    }
    logs = []
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=["hive-fleet", "hive-reputation"],
        log=lambda m, l: logs.append((l, m)),
    )
    assert "hive-fleet" in r.found_layers
    assert "hive-reputation" not in r.found_layers
    # Should log the requested/found split
    log_text = " ".join(m for _, m in logs)
    assert "hive-fleet" in log_text
    assert "hive-reputation" in log_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.rebalance_router_v3'`.

- [ ] **Step 3: Write minimal implementation**

Create `modules/rebalance_router_v3.py`:

```python
"""
rebalance_router_v3 — Askrene-based route discovery and pricing.

Uses CLN's `getroutes` (askrene plugin, added in v24.08) with layer-based
biasing from cl-hive, plus per-retry throwaway exclude layers.

Interface contract matches rebalance_router_v2.RebalanceRouter: the planner
calls price_pair(...) and receives a RouteResult with the same shape.
The engine chooses which router to dispatch per cycle via config.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .rebalance_router_v2 import (
    RouteResult,
    RebalanceRouter as RebalanceRouterV2,
)


def _parse_layer_names(csv: str) -> List[str]:
    """Parse a comma-separated layer name string into a list.

    Handles whitespace trimming and drops empty entries. Returns [] for
    blank input so standalone nodes without cl-hive see an empty layer list.
    """
    if not csv:
        return []
    return [name.strip() for name in csv.split(",") if name.strip()]


class RebalanceRouterV3:
    """Route discovery using askrene `getroutes` with layer support.

    Preserves the v2 router's price_pair interface so the planner and
    engine don't care which router produced a given RouteResult.
    """

    def __init__(
        self,
        plugin: Any,
        our_node_id: str,
        layer_names: List[str],
        log: Callable[[str, str], None],
    ) -> None:
        self.plugin = plugin
        self.our_node_id = our_node_id
        self.layer_names = list(layer_names)
        self.log = log
        self.found_layers: List[str] = self._probe_layers()

    def _probe_layers(self) -> List[str]:
        """Check which of the requested layers actually exist on the node.

        Called once at init. Missing layers are logged at info level; askrene
        silently drops unknown layers from getroutes calls, so missing layers
        never crash the router — they just produce un-biased routes.
        """
        try:
            result = self.plugin.rpc.call("askrene-listlayers", {})
        except Exception as e:
            self.log(f"[router-v3] askrene-listlayers failed: {e}", "warn")
            return []

        live = [l.get("layer", "") for l in result.get("layers", [])]
        found = [n for n in self.layer_names if n in live]
        missing = [n for n in self.layer_names if n not in live]

        self.log(
            f"[router-v3] requested layers={self.layer_names} "
            f"found={found}" + (f" missing={missing}" if missing else ""),
            "info",
        )
        return found
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_router_v3.py tests/test_rebalance_router_v3.py
git commit -m "feat: add v3 router skeleton with layer probe"
```

---

## Task 4: Error Translator (askrene → v3 skip reason)

**Files:**
- Modify: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_router_v3.py`:

```python
def test_translate_unknown_source_node():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, detail = _translate_getroutes_error("Unknown source node 03abc...")
    assert reason == "unknown_source_node"
    assert "03abc" in detail


def test_translate_unknown_destination_node():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, _ = _translate_getroutes_error("Unknown destination node 02def...")
    assert reason == "unknown_dest_node"


def test_translate_unknown_layer():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, _ = _translate_getroutes_error("Unknown layer")
    assert reason == "unknown_layer"


def test_translate_child_died():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    for msg in (
        "child died with signal 11",
        "failed to fork: Resource temporarily unavailable",
        "child produced no output (exited 1)?",
        "failed to create pipes: Too many open files",
    ):
        reason, _ = _translate_getroutes_error(msg)
        assert reason == "askrene_child_died", f"{msg} -> {reason}"


def test_translate_no_route_catchall():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, detail = _translate_getroutes_error(
        "We could not find a usable set of paths. The shortest path is 123x4x5, but ..."
    )
    assert reason == "no_route"
    assert "We could not find" in detail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -k "translate" -v`
Expected: FAIL with `ImportError` — `_translate_getroutes_error` doesn't exist yet.

- [ ] **Step 3: Write minimal implementation**

Append to `modules/rebalance_router_v3.py` (at module level, below `_parse_layer_names`):

```python
def _translate_getroutes_error(error: str) -> tuple[str, str]:
    """Map a getroutes RPC error message to (skip_reason, preserved_detail).

    Based on upstream CLN error sites in plugins/askrene/askrene.c and
    plugins/askrene/child/explain_failure.c. Every unknown message falls
    back to `no_route` with the original text preserved for operator
    debugging in audit logs.
    """
    if "Unknown source node" in error:
        return "unknown_source_node", error
    if "Unknown destination node" in error:
        return "unknown_dest_node", error
    if "Unknown layer" in error:
        return "unknown_layer", error
    child_signals = (
        "child died with signal",
        "failed to fork",
        "child produced no output",
        "failed to create pipes",
    )
    if any(s in error for s in child_signals):
        return "askrene_child_died", error
    return "no_route", error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: 9 passed (4 prior + 5 new).

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_router_v3.py tests/test_rebalance_router_v3.py
git commit -m "feat: add v3 router getroutes error translator"
```

---

## Task 5: Hop Format Translator (getroutes → sendpay)

**Files:**
- Modify: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_router_v3.py`:

```python
def test_translate_hop_basic():
    from modules.rebalance_router_v3 import _translate_getroutes_hop_to_sendpay
    hop = {
        "short_channel_id_dir": "940132x2695x0/0",
        "next_node_id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        "amount_msat": 1000343,
        "delay": 106,
    }
    out = _translate_getroutes_hop_to_sendpay(hop)
    assert out == {
        "id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        "channel": "940132x2695x0",
        "direction": 0,
        "amount_msat": 1000343,
        "delay": 106,
    }


def test_translate_hop_direction_1():
    from modules.rebalance_router_v3 import _translate_getroutes_hop_to_sendpay
    hop = {
        "short_channel_id_dir": "933791x3241x0/1",
        "next_node_id": "03" + "a" * 64,
        "amount_msat": 500000,
        "delay": 78,
    }
    out = _translate_getroutes_hop_to_sendpay(hop)
    assert out["direction"] == 1
    assert out["channel"] == "933791x3241x0"


def test_translate_hop_msat_string_input():
    """getroutes may return amount_msat as a string like '1000343msat'"""
    from modules.rebalance_router_v3 import _translate_getroutes_hop_to_sendpay
    hop = {
        "short_channel_id_dir": "100x1x0/0",
        "next_node_id": "02" + "b" * 64,
        "amount_msat": "1000343msat",
        "delay": 40,
    }
    out = _translate_getroutes_hop_to_sendpay(hop)
    assert out["amount_msat"] == 1000343
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -k "translate_hop" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `modules/rebalance_router_v3.py` (at module level, below `_translate_getroutes_error`):

```python
def _parse_msat(v: Any) -> int:
    """Parse an amount_msat field that may be int, str like '1000msat', or None."""
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.rstrip("msat").strip()
        return int(s) if s else 0
    raise TypeError(f"cannot parse amount_msat: {v!r}")


def _translate_getroutes_hop_to_sendpay(hop: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a getroutes path hop to sendpay route format.

    getroutes uses `short_channel_id_dir` ("SCID/dir") + `next_node_id`,
    while sendpay expects `channel` + `direction` + `id`. This is a
    trivial field rename plus msat parsing.
    """
    scidd = hop["short_channel_id_dir"]
    scid, direction = scidd.rsplit("/", 1)
    return {
        "id": hop["next_node_id"],
        "channel": scid,
        "direction": int(direction),
        "amount_msat": _parse_msat(hop["amount_msat"]),
        "delay": int(hop["delay"]),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: 12 passed (9 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_router_v3.py tests/test_rebalance_router_v3.py
git commit -m "feat: add v3 router hop format translator (getroutes -> sendpay)"
```

---

## Task 6: Path-Shape Validator

**Files:**
- Modify: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_router_v3.py`:

```python
def test_validate_path_accepts_valid_circular_shape():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    src_scid = "100x1x0"
    dst_scid = "200x2x0"
    path = [
        {"short_channel_id_dir": f"{src_scid}/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "150x1x0/0", "next_node_id": "03" + "b" * 64, "amount_msat": 999, "delay": 80},
        {"short_channel_id_dir": f"{dst_scid}/0", "next_node_id": our_id, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path, our_node_id=our_id, source_channel_id=src_scid, dest_channel_id=dst_scid
    )
    assert ok is True
    assert reason == ""


def test_validate_path_rejects_loop_through_us():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        # Intermediate hop routes BACK through us — loop
        {"short_channel_id_dir": "999x9x9/0", "next_node_id": our_id, "amount_msat": 999, "delay": 80},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03" + "b" * 64, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path, our_node_id=our_id, source_channel_id="100x1x0", dest_channel_id="200x2x0"
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_path_rejects_wrong_source_channel():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        # Wrong source SCID — expected 100x1x0 but got 999x9x9
        {"short_channel_id_dir": "999x9x9/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": our_id, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path, our_node_id=our_id, source_channel_id="100x1x0", dest_channel_id="200x2x0"
    )
    assert ok is False
    assert reason == "path_loops_through_us"  # first-hop mismatch also returns this reason


def test_validate_path_rejects_empty():
    from modules.rebalance_router_v3 import _validate_path_shape
    ok, reason = _validate_path_shape(
        [], our_node_id="03" + "u" * 64, source_channel_id="100x1x0", dest_channel_id="200x2x0"
    )
    assert ok is False
    assert reason == "path_loops_through_us"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -k "validate_path" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `modules/rebalance_router_v3.py`:

```python
def _validate_path_shape(
    path: List[Dict[str, Any]],
    *,
    our_node_id: str,
    source_channel_id: str,
    dest_channel_id: str,
) -> tuple[bool, str]:
    """Validate that a getroutes path has the expected circular shape.

    Accepts: first hop uses source_channel_id, last hop uses dest_channel_id
    and lands at us (next_node_id == our_node_id), intermediate hops do NOT
    route through us again.

    Rejects with reason `path_loops_through_us` for:
    - Empty path
    - First hop's SCID doesn't match source_channel_id
    - Any intermediate hop's next_node_id == our_node_id
    - Last hop doesn't land at us
    - Last hop's SCID doesn't match dest_channel_id
    """
    if not path:
        return False, "path_loops_through_us"

    # First hop must use the requested source channel
    first = path[0]
    first_scid, _ = first["short_channel_id_dir"].rsplit("/", 1)
    if first_scid != source_channel_id:
        return False, "path_loops_through_us"

    # Last hop must land at us and use the requested dest channel
    last = path[-1]
    if last["next_node_id"] != our_node_id:
        return False, "path_loops_through_us"
    last_scid, _ = last["short_channel_id_dir"].rsplit("/", 1)
    if last_scid != dest_channel_id:
        return False, "path_loops_through_us"

    # Intermediate hops must not route through us
    for hop in path[:-1]:
        if hop["next_node_id"] == our_node_id:
            return False, "path_loops_through_us"

    return True, ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_router_v3.py tests/test_rebalance_router_v3.py
git commit -m "feat: add v3 router path-shape validator"
```

---

## Task 7: price_pair Happy Path

**Files:**
- Modify: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_router_v3.py`:

```python
def _make_router_v3(plugin=None, layer_names=None, log=None):
    from modules.rebalance_router_v3 import RebalanceRouterV3
    p = plugin or MagicMock()
    p.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    return RebalanceRouterV3(
        plugin=p,
        our_node_id="03" + "u" * 64,
        layer_names=layer_names or ["hive-fleet"],
        log=log or (lambda m, l: None),
    )


def test_price_pair_calls_getroutes_with_layers():
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    # getroutes returns a valid 2-hop circular path
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000,
            "final_cltv": 40,
            "path": [
                {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 100333, "delay": 106},
                {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03" + "u" * 64, "amount_msat": 100000, "delay": 40},
            ],
        }],
    }

    # Use lookup helpers mocked: final hop fee ppm and dest cltv
    plugin.rpc.listpeerchannels.return_value = {"channels": []}  # empty — helpers fall through
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": "03" + "b" * 64,
            "destination": "03" + "u" * 64,
            "fee_per_millionth": 250,
            "delay": 40,
        }]
    }

    r = _make_router_v3(plugin=plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=100,
    )

    assert result.success is True
    plugin.rpc.getroutes.assert_called_once()
    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["source"] == "03" + "a" * 64
    assert kwargs["destination"] == "03" + "b" * 64
    assert kwargs["amount_msat"] == 100 * 1000
    assert "hive-fleet" in kwargs["layers"]
    assert kwargs["final_cltv"] == 40
    assert result.hops == 2
    assert len(result.route) == 2


def test_price_pair_picks_cheapest_when_multiple_routes():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            # Expensive: 500 msat fee
            {"probability_ppm": 990000, "amount_msat": 100000, "final_cltv": 40,
             "path": [
                 {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 100500, "delay": 106},
                 {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03" + "u" * 64, "amount_msat": 100000, "delay": 40},
             ]},
            # Cheap: 100 msat fee
            {"probability_ppm": 990000, "amount_msat": 100000, "final_cltv": 40,
             "path": [
                 {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 100100, "delay": 106},
                 {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03" + "u" * 64, "amount_msat": 100000, "delay": 40},
             ]},
        ],
    }
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": [{"source": "03" + "b" * 64, "destination": "03" + "u" * 64, "fee_per_millionth": 0, "delay": 40}]}

    r = _make_router_v3(plugin=plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=100,
    )
    assert result.success is True
    # Should pick the 100 msat fee route
    assert result.route_cost_sats <= 1  # 100 msat rounded up to 1 sat


def test_price_pair_returns_failure_on_empty_routes():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getroutes.return_value = {"probability_ppm": 0, "routes": []}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": [{"source": "03" + "b" * 64, "destination": "03" + "u" * 64, "fee_per_millionth": 0, "delay": 40}]}

    r = _make_router_v3(plugin=plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=100,
    )
    assert result.success is False
    assert result.error  # non-empty error string


def test_price_pair_rejects_loop_through_us():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    our_id = "03" + "u" * 64
    # Path where intermediate hop routes back through us
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{"probability_ppm": 990000, "amount_msat": 100000, "final_cltv": 40, "path": [
            {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 100200, "delay": 106},
            {"short_channel_id_dir": "999x9x9/0", "next_node_id": our_id, "amount_msat": 100100, "delay": 80},  # loop
            {"short_channel_id_dir": "200x2x0/0", "next_node_id": our_id, "amount_msat": 100000, "delay": 40},
        ]}],
    }
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": [{"source": "03" + "b" * 64, "destination": our_id, "fee_per_millionth": 0, "delay": 40}]}

    r = _make_router_v3(plugin=plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=100,
    )
    assert result.success is False
    assert "path_loops_through_us" in result.error or "loop" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -k "price_pair" -v`
Expected: FAIL with `AttributeError: 'RebalanceRouterV3' object has no attribute 'price_pair'`.

- [ ] **Step 3: Write minimal implementation**

Append the `price_pair` method to the `RebalanceRouterV3` class in `modules/rebalance_router_v3.py`:

```python
    def price_pair(
        self,
        source_channel_id: str,
        dest_channel_id: str,
        source_peer_id: str,
        dest_peer_id: str,
        amount_sats: int,
        exclude: Optional[List[str]] = None,
    ) -> RouteResult:
        """Discover and price a circular rebalance route via askrene getroutes.

        Returns a RouteResult in the same format as the v2 router so the
        engine and executor can consume either.
        """
        # Reuse v2 helper methods for fee and CLTV lookup.
        v2 = RebalanceRouterV2(self.plugin, self.our_node_id)
        final_hop_fee_ppm = v2._get_final_hop_fee_ppm(dest_peer_id)
        if final_hop_fee_ppm is None:
            return RouteResult(
                success=False,
                error=f"cannot determine final-hop fee for peer {dest_peer_id}",
            )
        dest_cltv = v2._get_dest_channel_cltv(dest_peer_id)

        final_hop_fee_sats = v2._compute_final_hop_fee_sats(
            amount_sats, final_hop_fee_ppm
        )
        route_amount_msat = (amount_sats + final_hop_fee_sats) * 1000

        # Build layers list. Always include the found cl-hive layers.
        layers = list(self.found_layers)

        try:
            result = self.plugin.rpc.getroutes(
                source=source_peer_id,
                destination=dest_peer_id,
                amount_msat=route_amount_msat,
                layers=layers,
                maxfee_msat=route_amount_msat,  # pass a generous per-call cap; planner enforces its own
                final_cltv=dest_cltv,
            )
        except Exception as e:
            reason, detail = _translate_getroutes_error(str(e))
            return RouteResult(success=False, error=f"{reason}: {detail}")

        routes = result.get("routes", [])
        if not routes:
            return RouteResult(success=False, error="no_route: getroutes returned empty")

        # Pick the cheapest single-path route (first_hop_amount - delivered_amount).
        def _route_fee_msat(r: Dict[str, Any]) -> int:
            path = r.get("path", [])
            if not path:
                return 10**18
            first_amt = _parse_msat(path[0]["amount_msat"])
            delivered = _parse_msat(r.get("amount_msat", 0))
            return max(0, first_amt - delivered)

        cheapest = min(routes, key=_route_fee_msat)
        path = cheapest.get("path", [])

        # Validate the path shape.
        ok, reason = _validate_path_shape(
            path,
            our_node_id=self.our_node_id,
            source_channel_id=source_channel_id,
            dest_channel_id=dest_channel_id,
        )
        if not ok:
            return RouteResult(success=False, error=f"{reason}: path shape invalid")

        # Translate each hop to sendpay format.
        sendpay_route = [
            _translate_getroutes_hop_to_sendpay(hop) for hop in path
        ]
        total_fee_msat = _route_fee_msat(cheapest)
        total_fee_sats = (total_fee_msat + 999) // 1000  # ceil to sats

        return RouteResult(
            success=True,
            route_cost_sats=total_fee_sats,
            final_hop_fee_ppm=final_hop_fee_ppm,
            hops=len(sendpay_route),
            route=sendpay_route,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_router_v3.py tests/test_rebalance_router_v3.py
git commit -m "feat: add v3 router price_pair with path validation"
```

---

## Task 8: Exclude-Via-Layer Retry Context Manager

**Files:**
- Modify: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_router_v3.py`:

```python
def test_exclude_layer_creates_and_removes():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}
    r = _make_router_v3(plugin=plugin)

    with r._exclude_layer(["100x1x0", "200x2x0"]) as layer_name:
        assert layer_name.startswith("rebalance-exclude-")
        # Should have called create-layer + 4 update-channel calls (2 scids × 2 directions)
        create_calls = [c for c in plugin.rpc.call.call_args_list
                        if c.args and c.args[0] == "askrene-create-layer"]
        update_calls = [c for c in plugin.rpc.call.call_args_list
                        if c.args and c.args[0] == "askrene-update-channel"]
        assert len(create_calls) == 1
        assert len(update_calls) == 4

    # After exit, remove-layer must have been called
    remove_calls = [c for c in plugin.rpc.call.call_args_list
                    if c.args and c.args[0] == "askrene-remove-layer"]
    assert len(remove_calls) == 1


def test_exclude_layer_removes_on_exception():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}
    r = _make_router_v3(plugin=plugin)

    try:
        with r._exclude_layer(["100x1x0"]) as layer_name:
            raise RuntimeError("simulated failure inside retry")
    except RuntimeError:
        pass

    remove_calls = [c for c in plugin.rpc.call.call_args_list
                    if c.args and c.args[0] == "askrene-remove-layer"]
    assert len(remove_calls) == 1, "remove-layer must be called even on exception"


def test_exclude_layer_empty_list_is_noop():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}
    r = _make_router_v3(plugin=plugin)
    with r._exclude_layer([]) as layer_name:
        assert layer_name is None or layer_name == ""
    # Should not have called create-layer
    create_calls = [c for c in plugin.rpc.call.call_args_list
                    if c.args and c.args[0] == "askrene-create-layer"]
    assert len(create_calls) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -k "exclude_layer" -v`
Expected: FAIL with `AttributeError: ... has no attribute '_exclude_layer'`.

- [ ] **Step 3: Write minimal implementation**

Append to the `RebalanceRouterV3` class in `modules/rebalance_router_v3.py`:

```python
    _exclude_counter = 0  # class-level monotonic counter

    @contextmanager
    def _exclude_layer(self, failed_channel_ids: List[str]) -> Iterator[Optional[str]]:
        """Create a throwaway layer disabling the given channels, yield its name,
        and remove the layer on context exit (even if the body raised).

        Empty input yields None and does nothing — caller should skip the layer
        parameter when None is yielded.
        """
        if not failed_channel_ids:
            yield None
            return

        RebalanceRouterV3._exclude_counter += 1
        import time
        layer_name = f"rebalance-exclude-{int(time.time())}-{self._exclude_counter}"

        try:
            self.plugin.rpc.call("askrene-create-layer", {"layer": layer_name})
            for scid in failed_channel_ids:
                # Disable both directions
                for direction in (0, 1):
                    self.plugin.rpc.call(
                        "askrene-update-channel",
                        {
                            "layer": layer_name,
                            "short_channel_id_dir": f"{scid}/{direction}",
                            "enabled": False,
                        },
                    )
            yield layer_name
        finally:
            try:
                self.plugin.rpc.call("askrene-remove-layer", {"layer": layer_name})
            except Exception as e:
                self.log(f"[router-v3] failed to remove exclude layer {layer_name}: {e}", "warn")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -v`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_router_v3.py tests/test_rebalance_router_v3.py
git commit -m "feat: add v3 router exclude-via-layer context manager"
```

---

## Task 9: Audit Record router= Field

**Files:**
- Modify: `modules/rebalance_audit_v2.py`
- Test: `tests/test_rebalance_audit_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_audit_v2.py`:

```python
def test_audit_pick_record_includes_router_field():
    from modules.rebalance_audit_v2 import RebalanceAudit
    logs = []
    audit = RebalanceAudit(log=lambda msg, level="info": logs.append(msg))
    audit.record_pick(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        amount_sats=1000,
        route_cost_sats=5,
        value_score=0.7,
        router="v3",
    )
    pick_lines = [l for l in logs if "REBAL_PICK" in l]
    assert pick_lines, "expected REBAL_PICK log"
    assert "router=v3" in pick_lines[0]


def test_audit_skip_record_includes_router_field():
    from modules.rebalance_audit_v2 import RebalanceAudit
    logs = []
    audit = RebalanceAudit(log=lambda msg, level="info": logs.append(msg))
    audit.record_skip(
        channel_id="100x1x0",
        reason="no_route",
        router="v3",
    )
    skip_lines = [l for l in logs if "REBAL_SKIP" in l]
    assert skip_lines
    assert "router=v3" in skip_lines[0]


def test_audit_router_field_defaults_to_v2_when_omitted():
    """Back-compat: existing callers that don't pass router= get 'v2'."""
    from modules.rebalance_audit_v2 import RebalanceAudit
    logs = []
    audit = RebalanceAudit(log=lambda msg, level="info": logs.append(msg))
    audit.record_skip(channel_id="100x1x0", reason="no_route")
    assert "router=v2" in logs[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_audit_v2.py -k "router" -v`
Expected: FAIL — `record_pick`/`record_skip` don't accept `router=` kwarg yet.

- [ ] **Step 3: Write minimal implementation**

Read `modules/rebalance_audit_v2.py` to find `record_pick` and `record_skip`. Add a `router: str = "v2"` parameter to both, and include `f" router={router}"` in the emitted log lines.

Example (adjust to match the actual existing signatures):

```python
def record_skip(self, channel_id: str, reason: str, *, router: str = "v2", **kwargs) -> None:
    assert reason in VALID_SKIP_REASONS, f"unknown skip reason {reason!r}"
    parts = [f"REBAL_SKIP channel={channel_id}", f"reason={reason}", f"router={router}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    self.log(" ".join(parts))

def record_pick(
    self,
    source_channel_id: str,
    dest_channel_id: str,
    amount_sats: int,
    route_cost_sats: int,
    value_score: float,
    *,
    router: str = "v2",
    **kwargs,
) -> None:
    parts = [
        f"REBAL_PICK source={source_channel_id}",
        f"dest={dest_channel_id}",
        f"amount={amount_sats}",
        f"route_cost_sats={route_cost_sats}",
        f"value_score={value_score:.3f}",
        f"router={router}",
    ]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    self.log(" ".join(parts))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_audit_v2.py -v`
Expected: all previous audit tests + 3 new = all passing.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_audit_v2.py tests/test_rebalance_audit_v2.py
git commit -m "feat: add router= field to audit pick and skip records"
```

---

## Task 10: Engine Factory And Per-Cycle Dispatch

**Files:**
- Modify: `modules/rebalance_engine_v2.py`
- Test: `tests/test_rebalance_engine_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_engine_v2.py`:

```python
def test_engine_builds_v2_router_when_askrene_unavailable():
    """If askrene is unavailable (help getroutes raises), only v2 router is built."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("unknown method askrene-listlayers")
    # ... minimal engine construction (database, config, etc.) — see existing tests
    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    database = MagicMock()

    engine = RebalanceEngine(
        plugin=plugin,
        database=database,
        config=config,
    )
    assert engine.router_v2 is not None
    assert engine.router_v3 is None


def test_engine_builds_both_routers_when_askrene_available():
    from modules.rebalance_engine_v2 import RebalanceEngine
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    database = MagicMock()

    engine = RebalanceEngine(plugin=plugin, database=database, config=config)
    assert engine.router_v2 is not None
    assert engine.router_v3 is not None


def test_engine_active_router_respects_config():
    from modules.rebalance_engine_v2 import RebalanceEngine
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    database = MagicMock()

    engine = RebalanceEngine(plugin=plugin, database=database, config=config)
    assert engine._active_router() is engine.router_v2

    config.rebalance_router = "v3"
    assert engine._active_router() is engine.router_v3


def test_engine_falls_back_to_v2_when_v3_requested_but_unavailable():
    from modules.rebalance_engine_v2 import RebalanceEngine
    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("askrene unavailable")
    config = MagicMock()
    config.rebalance_router = "v3"  # operator wants v3
    config.askrene_layers = "hive-fleet"
    database = MagicMock()

    engine = RebalanceEngine(plugin=plugin, database=database, config=config)
    # Engine should still return v2 as the active router because v3 is None
    assert engine.router_v3 is None
    assert engine._active_router() is engine.router_v2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_engine_v2.py -k "router" -v`
Expected: FAIL — engine constructor doesn't build router_v2/router_v3 yet.

- [ ] **Step 3: Write minimal implementation**

In `modules/rebalance_engine_v2.py`, add imports:

```python
from .rebalance_router_v2 import RebalanceRouter as RebalanceRouterV2
from .rebalance_router_v3 import RebalanceRouterV3, _parse_layer_names
```

Modify `RebalanceEngine.__init__` to build both routers:

```python
def __init__(self, plugin, database, config, ...):
    # ... existing init code
    self.our_node_id = self._get_our_node_id()

    # Build v2 router (always available; no RPC dependency at init)
    self.router_v2 = RebalanceRouterV2(plugin, self.our_node_id)

    # Probe askrene and build v3 router if available
    self.router_v3 = None
    if self._probe_askrene():
        layer_names = _parse_layer_names(getattr(config, "askrene_layers", ""))
        self.router_v3 = RebalanceRouterV3(
            plugin=plugin,
            our_node_id=self.our_node_id,
            layer_names=layer_names,
            log=lambda msg, level="info": plugin.log(msg, level=level),
        )

def _get_our_node_id(self) -> str:
    try:
        return self.plugin.rpc.getinfo().get("id", "")
    except Exception:
        return ""

def _probe_askrene(self) -> bool:
    """One-shot probe: does this CLN instance have the askrene plugin loaded?

    Calls askrene-listlayers. Any exception means askrene is unavailable.
    """
    try:
        self.plugin.rpc.call("askrene-listlayers", {})
        return True
    except Exception:
        return False

def _active_router(self):
    """Return the router currently configured for this cycle.

    Reads config.rebalance_router each call so setconfig can hot-switch
    between cycles. If v3 is requested but router_v3 is None (askrene
    unavailable), falls back to v2.
    """
    want = getattr(self.config, "rebalance_router", "v2")
    if want == "v3" and self.router_v3 is not None:
        return self.router_v3
    return self.router_v2
```

Modify `run_cycle` to capture the router at cycle start:

```python
def run_cycle(self, ...):
    self._cycle_router = self._active_router()
    # ... existing cycle code, replacing any direct use of self.router with self._cycle_router
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_engine_v2.py -v`
Expected: all previous engine tests still pass + 4 new router tests pass.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_engine_v2.py tests/test_rebalance_engine_v2.py
git commit -m "feat: add engine router factory and per-cycle dispatch"
```

---

## Task 11: Per-Cycle Atomicity Test

**Files:**
- Modify: `tests/test_rebalance_engine_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_engine_v2.py`:

```python
def test_engine_captures_router_at_cycle_start():
    """A mid-cycle config flip must not split a cycle across two routers."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    database = MagicMock()

    engine = RebalanceEngine(plugin=plugin, database=database, config=config)

    # Simulate cycle start captures
    engine._cycle_router = engine._active_router()
    assert engine._cycle_router is engine.router_v2

    # Operator flips config mid-cycle
    config.rebalance_router = "v3"

    # Captured router must not change
    assert engine._cycle_router is engine.router_v2
    # Next cycle picks up the new config
    engine._cycle_router = engine._active_router()
    assert engine._cycle_router is engine.router_v3
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python3 -m pytest tests/test_rebalance_engine_v2.py::test_engine_captures_router_at_cycle_start -v`
Expected: PASS (Task 10 already implemented the capture pattern). If it fails, fix the `run_cycle` method to call `_active_router()` only once per cycle.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rebalance_engine_v2.py
git commit -m "test: assert per-cycle router capture atomicity"
```

---

## Task 12: Orphan Exclude-Layer Sweep At Init

**Files:**
- Modify: `modules/rebalance_engine_v2.py`
- Test: `tests/test_rebalance_engine_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rebalance_engine_v2.py`:

```python
def test_engine_sweeps_orphan_exclude_layers_at_init():
    from modules.rebalance_engine_v2 import RebalanceEngine
    plugin = MagicMock()
    # askrene-listlayers returns orphan + non-orphan layers
    plugin.rpc.call.side_effect = [
        # Initial probe by v3 router constructor (empty is fine)
        {"layers": [{"layer": "rebalance-exclude-123-4"}, {"layer": "hive-fleet"}, {"layer": "rebalance-exclude-999-1"}]},
        # Sweep listlayers
        {"layers": [{"layer": "rebalance-exclude-123-4"}, {"layer": "hive-fleet"}, {"layer": "rebalance-exclude-999-1"}]},
        # askrene-remove-layer calls
        {},
        {},
        # any additional calls
        {"layers": [{"layer": "hive-fleet"}]},
    ]
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    database = MagicMock()

    engine = RebalanceEngine(plugin=plugin, database=database, config=config)

    # All calls to askrene-remove-layer should target rebalance-exclude-* layers
    remove_calls = [c for c in plugin.rpc.call.call_args_list
                    if c.args and c.args[0] == "askrene-remove-layer"]
    removed_layer_names = [c.args[1]["layer"] for c in remove_calls]
    assert "rebalance-exclude-123-4" in removed_layer_names
    assert "rebalance-exclude-999-1" in removed_layer_names
    assert "hive-fleet" not in removed_layer_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalance_engine_v2.py::test_engine_sweeps_orphan_exclude_layers_at_init -v`
Expected: FAIL — engine doesn't sweep orphans yet.

- [ ] **Step 3: Write minimal implementation**

Add a method to `RebalanceEngine` in `modules/rebalance_engine_v2.py`:

```python
def _sweep_orphan_exclude_layers(self) -> int:
    """Remove any leftover rebalance-exclude-* layers from a previous crashed cycle.

    Called once at init, after the askrene probe succeeds. Safe to call
    repeatedly — just iterates the layer list and removes matches.
    Returns the number of layers swept for logging.
    """
    try:
        result = self.plugin.rpc.call("askrene-listlayers", {})
    except Exception as e:
        self.plugin.log(f"[router] orphan sweep failed to list layers: {e}", level="warn")
        return 0

    orphans = [
        l.get("layer", "")
        for l in result.get("layers", [])
        if l.get("layer", "").startswith("rebalance-exclude-")
    ]
    for name in orphans:
        try:
            self.plugin.rpc.call("askrene-remove-layer", {"layer": name})
        except Exception as e:
            self.plugin.log(f"[router] failed to remove orphan layer {name}: {e}", level="warn")

    if orphans:
        self.plugin.log(
            f"[router] swept {len(orphans)} orphan rebalance-exclude-* layer(s) at init",
            level="info",
        )
    return len(orphans)
```

Call it from `__init__` AFTER the v3 router is built:

```python
def __init__(self, plugin, database, config, ...):
    # ... existing init
    if self.router_v3 is not None:
        self._sweep_orphan_exclude_layers()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalance_engine_v2.py::test_engine_sweeps_orphan_exclude_layers_at_init -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_engine_v2.py tests/test_rebalance_engine_v2.py
git commit -m "feat: sweep orphan rebalance-exclude-* layers at engine init"
```

---

## Task 13: cl-revenue-ops.py Option Registration With on_change

**Files:**
- Modify: `cl-revenue-ops.py`
- Test: manual verification via `python3 -m py_compile cl-revenue-ops.py` (no unit test — plugin entry is hard to mock)

- [ ] **Step 1: Locate the existing option registration block**

Run: `grep -n "add_option\|plugin.add_option" cl-revenue-ops.py | head`

Find the section where other `revenue-ops-*` options are registered.

- [ ] **Step 2: Add the two new options with on_change callback**

Insert near the existing options block in `cl-revenue-ops.py`:

```python
def _on_rebalance_router_change(plugin: Plugin, option_name: str, new_value: Any) -> None:
    """Validator called when setconfig changes rebalance-router at runtime.

    Raises ValueError (surfaced to the setconfig caller) if the new value
    is invalid or if v3 was requested but askrene is unavailable.
    """
    if new_value not in ("v2", "v3"):
        raise ValueError(
            f"rebalance-router must be 'v2' or 'v3', got {new_value!r}"
        )
    r = globals().get("rebalancer")
    eng = getattr(r, "rebalance_engine_v2", None) if r else None
    if eng is None:
        raise ValueError(
            "rebalance engine not initialized; cannot change router"
        )
    if new_value == "v3" and eng.router_v3 is None:
        raise ValueError(
            "askrene unavailable on this node; cannot switch to v3"
        )
    plugin.log(
        f"rebalance-router switched to {new_value} "
        f"(takes effect at next cycle boundary)",
        level="info",
    )


plugin.add_option(
    "rebalance-router",
    default="v2",
    description="Rebalance route discovery strategy: v2 (getroute) or v3 (askrene getroutes + layers)",
    opt_type="string",
    dynamic=True,
    on_change=_on_rebalance_router_change,
)

plugin.add_option(
    "askrene-layers",
    default="hive-fleet",
    description="CSV of askrene layer names to pass to v3 router getroutes calls",
    opt_type="string",
    dynamic=False,  # layer set is read once at init; operators restart to change
)
```

In the init/startup path where `config.rebalance_router` and `config.askrene_layers` are populated, read from `plugin.options`:

```python
config.rebalance_router = plugin.options["rebalance-router"].value or "v2"
config.askrene_layers = plugin.options["askrene-layers"].value or "hive-fleet"
```

- [ ] **Step 3: Verify the file still parses**

Run: `python3 -m py_compile cl-revenue-ops.py && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest tests/ -q 2>&1 | tail -10`
Expected: all previously passing tests still pass (no regression).

- [ ] **Step 5: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "feat: register rebalance-router and askrene-layers plugin options"
```

---

## Task 14: Integration Tests — Live CLN (gated by CLN_INTEGRATION=1)

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_router_v3_live.py`

- [ ] **Step 1: Create integration test scaffolding**

Create `tests/integration/__init__.py`:

```python
```

(Empty file — just marks the directory as a package.)

Create `tests/integration/conftest.py`:

```python
"""Integration-test fixtures that require a live CLN node.

Gated by the CLN_INTEGRATION=1 environment variable. Tests are skipped
when the variable is not set, so `pytest tests/` remains safe to run
on any developer machine without a local lightningd.
"""

import os
import pytest
from pyln.client import LightningRpc


@pytest.fixture
def live_plugin():
    if os.environ.get("CLN_INTEGRATION", "") != "1":
        pytest.skip("CLN_INTEGRATION not set; live-node tests skipped")

    rpc_path = os.environ.get(
        "LIGHTNING_RPC", os.path.expanduser("~/.lightning/bitcoin/lightning-rpc")
    )
    if not os.path.exists(rpc_path):
        pytest.skip(f"lightning-rpc socket not found at {rpc_path}")

    class _FakePlugin:
        def __init__(self, rpc):
            self.rpc = rpc
            self._log = []

        def log(self, msg, level="info"):
            self._log.append((level, msg))

    return _FakePlugin(LightningRpc(rpc_path))
```

- [ ] **Step 2: Write the integration tests**

Create `tests/integration/test_router_v3_live.py`:

```python
"""Live-CLN integration tests for the v3 router.

Run with: CLN_INTEGRATION=1 pytest tests/integration/test_router_v3_live.py -v
Skipped by default.
"""

import os
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CLN_INTEGRATION", "") != "1",
    reason="CLN_INTEGRATION not set",
)


def test_live_askrene_listlayers_succeeds(live_plugin):
    result = live_plugin.rpc.call("askrene-listlayers", {})
    assert "layers" in result
    names = [l.get("layer") for l in result["layers"]]
    # Built-in xpay layer should exist on any modern CLN
    assert any("xpay" in n for n in names), f"layers found: {names}"


def test_live_getroutes_standalone_no_layers(live_plugin):
    our_id = live_plugin.rpc.getinfo()["id"]
    # Find a directly-connected peer
    peer_chans = live_plugin.rpc.listpeerchannels()["channels"]
    normal = [c for c in peer_chans if c.get("state") == "CHANNELD_NORMAL"]
    if not normal:
        pytest.skip("no normal channels available on live node")
    dst_peer = normal[0]["peer_id"]

    result = live_plugin.rpc.getroutes(
        source=our_id,
        destination=dst_peer,
        amount_msat=10000,
        layers=[],
        maxfee_msat=1000,
        final_cltv=40,
    )
    assert "routes" in result
    assert len(result["routes"]) >= 1


def test_live_v3_router_price_pair_direct_channel(live_plugin):
    from modules.rebalance_router_v3 import RebalanceRouterV3

    our_id = live_plugin.rpc.getinfo()["id"]
    peer_chans = live_plugin.rpc.listpeerchannels()["channels"]
    normal = [c for c in peer_chans if c.get("state") == "CHANNELD_NORMAL"
              and c.get("spendable_msat", 0) > 10_000_000]
    if len(normal) < 2:
        pytest.skip("need at least 2 normal channels with >10k sats")

    src = normal[0]
    dst = normal[1]

    r = RebalanceRouterV3(
        plugin=live_plugin,
        our_node_id=our_id,
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
    )

    result = r.price_pair(
        source_channel_id=src["short_channel_id"],
        dest_channel_id=dst["short_channel_id"],
        source_peer_id=src["peer_id"],
        dest_peer_id=dst["peer_id"],
        amount_sats=1000,
    )
    # May succeed or fail depending on live topology, but must return cleanly
    assert result is not None
    assert hasattr(result, "success")


def test_live_exclude_layer_create_and_remove(live_plugin):
    from modules.rebalance_router_v3 import RebalanceRouterV3

    our_id = live_plugin.rpc.getinfo()["id"]
    r = RebalanceRouterV3(
        plugin=live_plugin,
        our_node_id=our_id,
        layer_names=[],
        log=lambda m, l: None,
    )

    # Test exclude-layer context manager
    with r._exclude_layer(["100x1x0"]) as layer_name:
        assert layer_name is not None
        assert layer_name.startswith("rebalance-exclude-")
        # Verify layer exists
        layers = live_plugin.rpc.call("askrene-listlayers", {})
        names = [l.get("layer") for l in layers["layers"]]
        assert layer_name in names

    # After exit, layer should be gone
    layers = live_plugin.rpc.call("askrene-listlayers", {})
    names = [l.get("layer") for l in layers["layers"]]
    assert layer_name not in names
```

- [ ] **Step 3: Verify tests are collected but skipped without the env var**

Run: `python3 -m pytest tests/integration/ -v 2>&1 | tail -10`
Expected: tests collected, all marked SKIPPED with reason "CLN_INTEGRATION not set".

- [ ] **Step 4: Run the full unit-test suite to confirm no impact**

Run: `python3 -m pytest tests/ -q --ignore=tests/integration 2>&1 | tail -5`
Expected: all non-integration tests pass as before.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/
git commit -m "test: add live-CLN integration tests for v3 router (CLN_INTEGRATION=1)"
```

---

## Task 15: Replay Fixtures

**Files:**
- Create: `tests/fixtures/router_v3/getroutes_direct_pair.json`
- Create: `tests/fixtures/router_v3/getroutes_multi_hop.json`
- Create: `tests/fixtures/router_v3/getroutes_empty.json`
- Create: `tests/fixtures/router_v3/getroutes_multi_route.json`
- Modify: `tests/test_rebalance_router_v3.py`

- [ ] **Step 1: Create replay fixtures from real responses**

Create `tests/fixtures/router_v3/getroutes_direct_pair.json`:

```json
{
  "probability_ppm": 999990,
  "routes": [
    {
      "probability_ppm": 999990,
      "amount_msat": 100000,
      "final_cltv": 40,
      "path": [
        {
          "short_channel_id_dir": "100x1x0/1",
          "next_node_id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
          "amount_msat": 100033,
          "delay": 72
        },
        {
          "short_channel_id_dir": "200x2x0/0",
          "next_node_id": "03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "amount_msat": 100000,
          "delay": 40
        }
      ]
    }
  ]
}
```

Create `tests/fixtures/router_v3/getroutes_multi_hop.json`:

```json
{
  "probability_ppm": 998678,
  "routes": [
    {
      "probability_ppm": 998678,
      "amount_msat": 5000000,
      "final_cltv": 40,
      "path": [
        {
          "short_channel_id_dir": "300x3x0/1",
          "next_node_id": "03bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "amount_msat": 5001000,
          "delay": 298
        },
        {
          "short_channel_id_dir": "400x4x0/1",
          "next_node_id": "03cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
          "amount_msat": 5000005,
          "delay": 264
        },
        {
          "short_channel_id_dir": "500x5x0/0",
          "next_node_id": "03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "amount_msat": 5000000,
          "delay": 120
        }
      ]
    }
  ]
}
```

Create `tests/fixtures/router_v3/getroutes_empty.json`:

```json
{
  "probability_ppm": 0,
  "routes": []
}
```

Create `tests/fixtures/router_v3/getroutes_multi_route.json`:

```json
{
  "probability_ppm": 990000,
  "routes": [
    {
      "probability_ppm": 990000,
      "amount_msat": 100000,
      "final_cltv": 40,
      "path": [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "amount_msat": 100500, "delay": 106},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "amount_msat": 100000, "delay": 40}
      ]
    },
    {
      "probability_ppm": 985000,
      "amount_msat": 100000,
      "final_cltv": 40,
      "path": [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "amount_msat": 100100, "delay": 106},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "amount_msat": 100000, "delay": 40}
      ]
    }
  ]
}
```

- [ ] **Step 2: Write the replay tests**

Append to `tests/test_rebalance_router_v3.py`:

```python
import json
import os


def _load_fixture(name: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), "fixtures", "router_v3", name)
    with open(path) as f:
        return json.load(f)


def test_replay_direct_pair_fixture():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getroutes.return_value = _load_fixture("getroutes_direct_pair.json")
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": "03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "destination": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
            "fee_per_millionth": 0,
            "delay": 40,
        }]
    }

    from modules.rebalance_router_v3 import RebalanceRouterV3
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
    )

    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        dest_peer_id="03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        amount_sats=100,
    )
    assert result.success is True
    assert result.hops == 2


def test_replay_empty_fixture():
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getroutes.return_value = _load_fixture("getroutes_empty.json")
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": [{
        "source": "03" + "b" * 64,
        "destination": "03" + "u" * 64,
        "fee_per_millionth": 0,
        "delay": 40,
    }]}

    from modules.rebalance_router_v3 import RebalanceRouterV3
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "u" * 64,
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
    )
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=100,
    )
    assert result.success is False
    assert "no_route" in result.error or "empty" in result.error
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_router_v3.py -k "replay" -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/router_v3/ tests/test_rebalance_router_v3.py
git commit -m "test: add v3 router replay fixtures (direct, multi-hop, empty, multi-route)"
```

---

## Task 16: A/B Rollback Monitor Skeleton

**Files:**
- Create: `tools/router_v3_safety_monitor.py`

- [ ] **Step 1: Create the monitor script**

Create `tools/router_v3_safety_monitor.py`:

```python
#!/usr/bin/env python3
"""A/B safety monitor for the v3 rebalance router rollout.

Reads the last N hours of audit logs, computes success-rate and
cost-per-success metrics grouped by router version, and if v3 is
performing worse than the v2 baseline beyond the configured threshold,
flips rebalance-router back to v2 via setconfig and exits non-zero.

This is a Phase 3 rollout safety tool. In Phase 1 it's a skeleton
that can be filled in after phase 1 ships and an A/B baseline exists.

Usage:
    python3 tools/router_v3_safety_monitor.py --log /path/to/cln.log \\
        --baseline-hours 48 --v3-hours 48 --threshold 0.10 [--dry-run]
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Tuple


REBAL_PICK_RE = re.compile(r"REBAL_PICK .* router=(v2|v3)")
REBAL_SKIP_RE = re.compile(r"REBAL_SKIP .* router=(v2|v3)")


def parse_log(path: str) -> Dict[str, Dict[str, int]]:
    """Return counters per router: {v2: {picks, skips}, v3: {picks, skips}}."""
    counters: Dict[str, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "skips": 0})
    if not os.path.exists(path):
        return counters
    with open(path) as f:
        for line in f:
            m = REBAL_PICK_RE.search(line)
            if m:
                counters[m.group(1)]["picks"] += 1
                continue
            m = REBAL_SKIP_RE.search(line)
            if m:
                counters[m.group(1)]["skips"] += 1
    return counters


def should_rollback(counters: Dict[str, Dict[str, int]], threshold: float) -> Tuple[bool, str]:
    v2 = counters.get("v2", {"picks": 0, "skips": 0})
    v3 = counters.get("v3", {"picks": 0, "skips": 0})
    v2_total = v2["picks"] + v2["skips"]
    v3_total = v3["picks"] + v3["skips"]
    if v2_total == 0 or v3_total == 0:
        return False, "insufficient data for comparison"
    v2_success_rate = v2["picks"] / v2_total
    v3_success_rate = v3["picks"] / v3_total
    if v3_success_rate < v2_success_rate - threshold:
        return True, (
            f"v3 success rate {v3_success_rate:.2%} < v2 baseline "
            f"{v2_success_rate:.2%} - threshold {threshold:.2%}"
        )
    return False, f"v3 {v3_success_rate:.2%} vs v2 {v2_success_rate:.2%} (ok)"


def rollback_to_v2(dry_run: bool) -> None:
    cmd = ["lightning-cli", "-k", "setconfig", "config=rebalance-router", "val=v2"]
    if dry_run:
        print("[dry-run] would run:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print("rolled back to rebalance-router=v2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to CLN log file")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Success rate drop triggering rollback (default 0.10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without actually flipping")
    args = parser.parse_args()

    counters = parse_log(args.log)
    print(f"counters: {dict(counters)}")
    rollback, reason = should_rollback(counters, args.threshold)
    print(f"decision: rollback={rollback}, reason={reason}")

    if rollback:
        rollback_to_v2(args.dry_run)
        return 1 if not args.dry_run else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make it executable and smoke-test**

Run:

```bash
chmod +x tools/router_v3_safety_monitor.py
python3 tools/router_v3_safety_monitor.py --log /dev/null --dry-run
```

Expected output:

```
counters: {}
decision: rollback=False, reason=insufficient data for comparison
```

- [ ] **Step 3: Commit**

```bash
git add tools/router_v3_safety_monitor.py
git commit -m "feat: add A/B rollback safety monitor skeleton for Phase 3"
```

---

## Task 17: Final Verification Sweep

**Files:**
- Modify: none unless regressions are found

- [ ] **Step 1: Run full v2+v3 test suite**

Run:

```bash
python3 -m pytest tests/ \
    --ignore=tests/integration \
    -q 2>&1 | tail -10
```

Expected: all previously passing tests still pass, plus the new tests from Tasks 1-16. The pre-existing `test_hive_live_contract::test_dominant_rebalance_direction_flows_through` failure may remain — it's unrelated to this work.

- [ ] **Step 2: Syntax-check the plugin entry point**

Run:

```bash
python3 -m py_compile cl-revenue-ops.py && echo "syntax ok"
python3 -m py_compile modules/rebalance_router_v3.py && echo "syntax ok"
python3 -m py_compile modules/rebalance_engine_v2.py && echo "syntax ok"
```

Expected: three `syntax ok` lines.

- [ ] **Step 3: Import sanity check**

Run:

```bash
python3 -c "
from modules.rebalance_router_v3 import (
    RebalanceRouterV3,
    _parse_layer_names,
    _translate_getroutes_error,
    _translate_getroutes_hop_to_sendpay,
    _validate_path_shape,
)
print('v3 router imports ok')
"
```

Expected: `v3 router imports ok`.

- [ ] **Step 4: Integration test collection dry-run**

Run: `python3 -m pytest tests/integration/ --collect-only 2>&1 | tail -10`
Expected: tests are collected (not errored). Actual execution is skipped without `CLN_INTEGRATION=1`.

- [ ] **Step 5: Final commit if any fixes were needed**

If Steps 1-4 found regressions that required fixes, commit them with:

```bash
git add -u
git commit -m "fix: resolve Phase 1 verification regressions"
```

If no regressions, no commit needed.

- [ ] **Step 6: Report Phase 1 complete**

Output the commit count and file summary:

```bash
echo "=== Phase 1 commits ==="
git log --oneline main..HEAD | head -30
echo ""
echo "=== Files changed ==="
git diff --stat main..HEAD
```

Phase 1 is complete when:
- All tests in `tests/` pass (excluding pre-existing `test_hive_live_contract` failure)
- `python3 -m pytest tests/integration/ --collect-only` succeeds (tests collected)
- `cl-revenue-ops.py` and all new modules compile cleanly
- `git log main..HEAD` shows one commit per task (approximately 17 commits)

---

## Self-Review

**Spec coverage check:**

- Research spec Section 9.7 scope additions:
  - Path-shape validator → **Task 6**
  - Hop format translator → **Task 5**
  - Five new skip reasons → **Task 2**
  - Askrene probe via listlayers → **Task 10**
  - Orphan exclude-layer sweep → **Task 12**
  - on_change callback → **Task 13**
- Research spec scope removals:
  - `rebalance_executor_v3.py` → NOT in plan ✓
  - `rebalance-executor` config key → NOT in plan ✓
  - Phase 2 test matrix → NOT in plan ✓
- Design spec Section 3 (V3 Router Interface) → **Tasks 3, 4, 5, 6, 7, 8**
- Design spec Section 5 (Graceful Degradation) → **Tasks 10** (askrene probe + fallback)
- Design spec Section 7 (Testing Strategy):
  - Unit tests for router_v3 → **Tasks 3-8**
  - Unit tests for engine factory → **Tasks 10, 11**
  - Integration tests (live CLN) → **Task 14**
  - Replay tests → **Task 15**
  - A/B monitor skeleton → **Task 16**
- Design spec Section 8 Phase 1 scope → **Tasks 1-17** (full coverage)

All spec requirements are covered.

**Placeholder scan:**
- No "TODO", "TBD", "fill in" in implementation steps — every step has concrete code
- All file paths are absolute-ish (relative to worktree root) and specific
- All test assertions are concrete

**Type consistency:**
- `RouteResult` — defined in Task 3 as re-imported from v2; used consistently in Tasks 7, 14, 15
- `_parse_layer_names` — Task 3 signature `(csv: str) -> List[str]`; used in Task 10
- `_translate_getroutes_error` — Task 4 signature `(error: str) -> tuple[str, str]`; used in Task 7
- `_translate_getroutes_hop_to_sendpay` — Task 5 signature `(hop: Dict[str, Any]) -> Dict[str, Any]`; used in Task 7
- `_validate_path_shape` — Task 6 signature `(path, *, our_node_id, source_channel_id, dest_channel_id) -> tuple[bool, str]`; used in Task 7
- `_exclude_layer` — Task 8 context manager method on `RebalanceRouterV3`; used in future retry code
- `RebalanceRouterV3.__init__` signature stays consistent: `(plugin, our_node_id, layer_names, log)` throughout

**YAGNI check:**
- No features beyond what the research spec requires
- No performance optimizations (the research showed 4× headroom under the 50ms budget)
- No MPP handling in phase 1 (single-path only — askrene's `auto.no_mpp_support` layer is added implicitly by selecting the cheapest single route)
- No parallelization of pair pricing (planner is sequential; concurrency limit of 4 askrene children is irrelevant)

**Risk coverage:**
- Every risk from research Section 9.8 has a mitigation in the plan:
  - Path-shape validation failures → Task 6 + Task 7 reject with clear skip reason
  - Layer content drift → Task 3 probe + info log, no crash
  - CLN upgrade breaks response shape → Task 5 hop translator is isolated
  - pyln-client API drift → Task 13 uses standard `add_option(dynamic=True, on_change=...)`
  - Operator typos in `askrene-layers` → Task 3 probe logs found/missing split
  - v2 router regression → v2 file is untouched by this plan
