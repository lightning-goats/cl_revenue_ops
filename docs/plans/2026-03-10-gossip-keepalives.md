# Gossip Keepalives Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in gossip keepalive subsystem that preserves a minimum total peer connectivity level by dialing pure P2P peers when overall connected peers drop too low.

**Architecture:** A new `modules/gossip_keeper.py` module owns candidate discovery, graph scoring, filtering, and backoff state. `cl-revenue-ops.py` adds config wiring and a background `gossip_maintenance_loop` that reads total connected peers and invokes the manager conservatively. Hive members are used as priority targets through a small helper in `hive_bridge.py`, but the feature degrades cleanly to public-graph discovery.

**Tech Stack:** Python 3.10+, pyln RPC (`listpeers`, `listnodes`, `listchannels`, `connect`), existing config/dataclass system, pytest

**Design doc:** `docs/plans/2026-03-10-gossip-keepalives-design.md`

---

### Task 1: Add Hive Helper And Core Keepalive Tests

**Files:**
- Create: `tests/test_gossip_keeper.py`
- Modify: `modules/hive_bridge.py`
- Modify: `tests/test_hive_integrations.py`

**Step 1: Write the failing tests**

In `tests/test_hive_integrations.py`, add tests for a new helper:

```python
def test_get_priority_gossip_targets_returns_member_pubkeys(mock_hive_bridge):
    mock_hive_bridge._init_complete = True
    mock_hive_bridge._hive_available = True
    mock_hive_bridge.plugin.rpc.call.return_value = {
        "members": [
            {"peer_id": "02" + "a" * 64, "tier": "member"},
            {"peer_id": "03" + "b" * 64, "tier": "neophyte"},
        ]
    }

    result = mock_hive_bridge.get_priority_gossip_targets()

    assert result == ["02" + "a" * 64, "03" + "b" * 64]


def test_get_priority_gossip_targets_returns_empty_when_unavailable(mock_hive_bridge):
    mock_hive_bridge._hive_available = False
    assert mock_hive_bridge.get_priority_gossip_targets() == []
```

Create `tests/test_gossip_keeper.py` with focused tests for:
- counting total connected peers from `listpeers`
- filtering out self, connected peers, and peers with channels
- preferring hive targets before public graph targets

```python
def test_connected_peer_count_uses_all_connected_peers():
    manager = GossipKeepaliveManager(plugin=mock_plugin, config=config, hive_bridge=None)
    assert manager.count_connected_peers({
        "peers": [
            {"id": "02" + "a" * 64, "connected": True},
            {"id": "02" + "b" * 64, "connected": False},
            {"id": "02" + "c" * 64, "connected": True},
        ]
    }) == 2
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_integrations.py -k gossip_targets -v
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_gossip_keeper.py -v
```

Expected:
- `AttributeError` for missing `get_priority_gossip_targets`
- `ImportError` for missing `modules.gossip_keeper`

**Step 3: Write the minimal implementation**

In `modules/hive_bridge.py`, add:
- `get_priority_gossip_targets(self) -> List[str]`
- `hive-members` RPC usage with optional-read policy
- deduped `peer_id` extraction, empty-list fallback

Create `modules/gossip_keeper.py` with:
- `GossipKeepaliveManager`
- `count_connected_peers()`
- `extract_channel_peer_ids()`
- `filter_candidates()`
- `get_ranked_targets()`

Do not add `connect` execution or backoff yet. Only implement the discovery and
ordering behavior needed by the tests.

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_integrations.py -k gossip_targets -v
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_gossip_keeper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_gossip_keeper.py tests/test_hive_integrations.py modules/hive_bridge.py modules/gossip_keeper.py
git commit -m "feat: add gossip keepalive target discovery"
```

---

### Task 2: Add Graph Ranking, Dialing, And Backoff

**Files:**
- Modify: `modules/gossip_keeper.py`
- Modify: `tests/test_gossip_keeper.py`

**Step 1: Write the failing tests**

Extend `tests/test_gossip_keeper.py` with:
- graph ranking from `listnodes` + `listchannels`
- connect only enough peers to fill the deficit
- skip peers under backoff
- failed `connect` applies backoff

```python
def test_public_targets_rank_by_active_edges_then_capacity():
    ranked = manager.rank_public_graph_targets(nodes_payload, channels_payload, excluded_peer_ids=set())
    assert ranked[:2] == ["02" + "a" * 64, "02" + "b" * 64]


def test_maintain_connections_connects_only_deficit(mock_plugin):
    mock_plugin.rpc.listpeers.return_value = {"peers": [{"id": "02" + "a" * 64, "connected": True}]}
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "f" * 64}
    manager = GossipKeepaliveManager(plugin=mock_plugin, config=config, hive_bridge=None)

    manager.maintain_connections()

    assert mock_plugin.rpc.connect.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_gossip_keeper.py -v
```

Expected: FAIL because ranking / connect / backoff methods are incomplete.

**Step 3: Write the minimal implementation**

In `modules/gossip_keeper.py`, add:
- public graph aggregation from `listchannels`
- deterministic sorting by active edge count then aggregate capacity
- `maintain_connections()`
- capped exponential backoff state keyed by peer id
- success path that clears peer backoff

Keep behavior conservative:
- no disconnects
- no channel opens
- skip peers already connected or already channel-linked

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_gossip_keeper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_gossip_keeper.py modules/gossip_keeper.py
git commit -m "feat: add gossip keepalive connection maintenance"
```

---

### Task 3: Wire Config And Background Loop

**Files:**
- Modify: `modules/config.py`
- Modify: `cl-revenue-ops.py`
- Modify: `config/cl-revenue-ops.conf.full`
- Modify: `config/cl-revenue-ops.conf.minimal`
- Modify: `tests/test_plugin_audit_regressions.py`

**Step 1: Write the failing tests**

Add tests for:
- config dataclass fields and type registration
- plugin init parsing for new options
- loop startup gating when `enable_gossip_keepalives` is false

```python
def test_config_supports_gossip_keepalive_fields():
    cfg = Config(enable_gossip_keepalives=True, target_gossip_peers=7)
    assert cfg.enable_gossip_keepalives is True
    assert cfg.target_gossip_peers == 7
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -k gossip -v
```

Expected: FAIL due to missing config fields / option wiring / loop startup.

**Step 3: Write the minimal implementation**

In `modules/config.py`:
- add `enable_gossip_keepalives` and `target_gossip_peers`
- register types and numeric ranges
- include fields in `ConfigSnapshot`

In `cl-revenue-ops.py`:
- add plugin options
- parse them into `config_kwargs`
- initialize a `GossipKeepaliveManager`
- add `gossip_maintenance_loop`
- start the daemon thread alongside the existing loops

In config examples:
- document the new options with default values

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -k gossip -v
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_gossip_keeper.py tests/test_hive_integrations.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add modules/config.py cl-revenue-ops.py config/cl-revenue-ops.conf.full config/cl-revenue-ops.conf.minimal tests/test_plugin_audit_regressions.py
git commit -m "feat: wire gossip keepalive loop into plugin"
```

---

### Task 4: Full Verification

**Files:**
- Verify only

**Step 1: Run targeted verification**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_gossip_keeper.py tests/test_hive_integrations.py tests/test_plugin_audit_regressions.py -v
```

Expected: PASS

**Step 2: Run full suite**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/
```

Expected: PASS, no regressions

**Step 3: Review config/docs diffs**

Check:
- option names match issue acceptance criteria
- defaults are `false` and `5`
- docs/examples mention total-peer semantics

**Step 4: Commit verification or fixups if needed**

```bash
git status --short
```

Expected: clean working tree or only intentional uncommitted changes
