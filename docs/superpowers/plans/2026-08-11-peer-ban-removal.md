# Peer Ban Surface Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete peer-ban RPC, dispatcher, and PolicyManager surface so a diagnostic `revenue-policy` caller cannot mutate policy through `ban` or `unban`.

**Architecture:** Delete the ban-specific public boundary and its now-unconsumed PolicyManager abstraction. Preserve generic peer policies and stored rows unchanged; historical `banned` tag strings remain ordinary inert tag data, while `passive` and `disabled` continue to drive retained fee and rebalance gates.

**Tech Stack:** Python 3.10+, pyln-client plugin RPC registration, pytest, Markdown contract documentation.

## Global Constraints

- Base all implementation on `origin/main` commit `17d131eb69d22018da2932890ff101d44f0b9f51` plus the approved design-spec commit.
- Do not migrate, delete, or rewrite existing database policy rows.
- Do not invoke live Core Lightning action RPCs, policy mutation RPCs, fee/rebalance cycles, payments, opens, closes, withdrawals, or swaps during validation.
- Preserve `revenue-policy list|get|find|changes` for diagnostics and the existing internal/admin gate for generic tactical writes.
- Preserve generic `passive` fee strategy, `disabled` rebalance mode, and generic tags.
- Preserve standalone operation and the no-Sling, no-Hive/Mycelium, no-planner/Boltz/LN+ invariants.
- Historical planning and audit records that accurately describe past behavior remain unchanged.
- Tier-1 completion requires independent review by an agent other than the owner.

---

## File Map

- `tests/test_phase_c_dispatchers.py`: realistic dispatcher-boundary regression proving removed action names are unknown and do not touch PolicyManager.
- `tests/test_rpc_surface_inventory.py`: normative registered-RPC contract; remove the three standalone ban names and change the expected count from 39 to 36.
- `tests/test_policy_manager.py`: remove ban behavior tests/import and pin the absence of the ban-specific API while retaining generic tag/policy behavior.
- `cl-revenue-ops.py`: remove the three helpers, three registered standalone methods, three dispatcher branches, import, usage text, and valid-action advertisement.
- `modules/policy_manager.py`: remove `BANNED_TAG` and the three ban-specific methods only.
- `README.md`: remove ban commands from current operator guidance and compatibility text.
- `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md`: remove the two write/action RPCs and one read RPC from the canonical current inventory.
- `docs/refactor/phase0/compatibility-catalog.md`: change the normative method count and remove ban actions from current primary-name language.

### Task 1: Encode the Removed Security Boundary

**Files:**
- Modify: `tests/test_phase_c_dispatchers.py:206-263`
- Modify: `tests/test_rpc_surface_inventory.py:12-51`
- Modify: `tests/test_policy_manager.py:20-33,539-592`

**Interfaces:**
- Consumes: `revenue_policy(plugin, action, peer_id=None, **kwargs) -> dict`, `PolicyManager`, and static `@plugin.method` registration text.
- Produces: regression expectations that the three dispatcher actions are unknown, the three standalone RPCs are absent, and PolicyManager has no ban-specific API.

- [ ] **Step 1: Replace vulnerable dispatcher behavior tests with a failing no-mutation regression**

Replace `TestPolicyBanActions` with:

```python
class TestRemovedPolicyBanActions:
    @pytest.mark.parametrize("action", ["ban", "unban", "list-banned"])
    def test_removed_action_is_unknown_and_does_not_touch_policy_manager(
        self, mod, plugin, monkeypatch, action
    ):
        pm = MagicMock()
        monkeypatch.setattr(mod, "policy_manager", pm)

        result = mod.revenue_policy(plugin, action=action, peer_id=PEER)

        assert result["error"].startswith(f"Unknown action: {action}.")
        assert "'ban'" not in result["error"]
        assert "'unban'" not in result["error"]
        assert "'list-banned'" not in result["error"]
        assert pm.mock_calls == []
```

Remove the `revenue_list_banned` deprecation assertion from `TestDeprecationNotices.test_family_notices_point_at_their_dispatcher`, because that Python function will no longer exist.

- [ ] **Step 2: Change the normative RPC inventory to the desired absent surface**

Delete these entries from `EXPECTED_RPC_METHODS`:

```python
"revenue-ban", "revenue-unban", "revenue-list-banned",
```

Change the count assertion to:

```python
def test_expected_count():
    # Post-peer-ban-removal retained RPC surface.
    assert len(EXPECTED_RPC_METHODS) == 36
```

- [ ] **Step 3: Replace ban implementation tests with API-removal and inert-data controls**

Import the module for namespace inspection without importing `BANNED_TAG`:

```python
import modules.policy_manager as policy_manager_module
from modules.policy_manager import (
    PolicyManager,
    PeerPolicy,
    FeeStrategy,
    RebalanceMode,
    MAX_POLICY_CHANGES_PER_MINUTE,
)
```

Replace `TestPeerBanning` with:

```python
class TestRemovedPeerBanAPI:
    PK = "02" + "e" * 64

    def test_ban_specific_api_is_absent(self):
        assert not hasattr(policy_manager_module, "BANNED_TAG")
        assert not hasattr(PolicyManager, "ban_peer")
        assert not hasattr(PolicyManager, "unban_peer")
        assert not hasattr(PolicyManager, "is_peer_banned")

    def test_historical_banned_tag_is_generic_inert_metadata(
        self, mock_database, mock_plugin
    ):
        pm = PolicyManager(mock_database, mock_plugin)
        policy = pm.set_policy(
            self.PK,
            strategy="passive",
            rebalance_mode="disabled",
            tags=["banned"],
        )

        assert policy.tags == ["banned"]
        assert policy.strategy == FeeStrategy.PASSIVE
        assert policy.rebalance_mode == RebalanceMode.DISABLED
        assert pm.should_manage_fees(self.PK) is False
        assert pm.should_rebalance(self.PK) is False
```

- [ ] **Step 4: Run the regression tests and verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_phase_c_dispatchers.py::TestRemovedPolicyBanActions \
  tests/test_rpc_surface_inventory.py \
  tests/test_policy_manager.py::TestRemovedPeerBanAPI
```

Expected: failures show `ban`/`unban`/`list-banned` still dispatch or appear in valid actions, three standalone RPCs are unexpectedly registered, and the ban-specific module/class API still exists. The inert-metadata control should pass.

### Task 2: Remove the Production Ban Surface

**Files:**
- Modify: `cl-revenue-ops.py:45-55,4093-4177,4193-4210,4375-4392,4451-4457`
- Modify: `modules/policy_manager.py:52-60,829-885`
- Test: `tests/test_phase_c_dispatchers.py`
- Test: `tests/test_rpc_surface_inventory.py`
- Test: `tests/test_policy_manager.py`

**Interfaces:**
- Consumes: the RED tests from Task 1.
- Produces: a `revenue-policy` dispatcher whose allowed set is exactly `READ_ONLY_POLICY_ACTIONS | TACTICAL_POLICY_ACTIONS`, a 36-method plugin RPC surface, and a generic-only PolicyManager API.

- [ ] **Step 1: Remove the ban-specific PolicyManager API**

Delete the constant and its stale planner comment:

```python
BANNED_TAG = "banned"
```

Delete the entire `Operator Bans` block containing:

```python
def ban_peer(...)
def unban_peer(...)
def is_peer_banned(...)
```

Do not alter `set_policy`, `add_tag`, `remove_tag`, `get_peers_by_tag`, `should_manage_fees`, or `should_rebalance`.

- [ ] **Step 2: Remove standalone RPCs and shared helpers**

Remove `BANNED_TAG` from the `modules.policy_manager` import. Delete `_rpc_ban`, `revenue_ban`, `_rpc_unban`, `revenue_unban`, `_rpc_list_banned`, and `revenue_list_banned`, including their decorators and deprecation aliases.

- [ ] **Step 3: Remove dispatcher action branches and advertisement**

Delete the `ban`, `unban`, and `list-banned` usage lines from the `revenue_policy` docstring. Delete all three `elif` branches. Replace the unknown-action set construction with:

```python
allowed = sorted(READ_ONLY_POLICY_ACTIONS | TACTICAL_POLICY_ACTIONS)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_phase_c_dispatchers.py \
  tests/test_rpc_surface_inventory.py \
  tests/test_policy_manager.py \
  tests/test_operator_surface.py \
  tests/test_rebalance_policy_gate.py \
  tests/test_architecture_guard.py
```

Expected: all tests pass; no test invokes a live RPC.

- [ ] **Step 5: Commit the test-first production removal**

```bash
git add cl-revenue-ops.py modules/policy_manager.py \
  tests/test_phase_c_dispatchers.py tests/test_rpc_surface_inventory.py \
  tests/test_policy_manager.py
git commit -m "fix(policy): remove obsolete peer-ban surface"
```

### Task 3: Synchronize Current Operator Contracts

**Files:**
- Modify: `README.md:208-229`
- Modify: `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md:20-70`
- Modify: `docs/refactor/phase0/compatibility-catalog.md:15-24`

**Interfaces:**
- Consumes: the 36-method production and test inventory from Task 2.
- Produces: current documentation with no claim that ban RPCs or actions exist.

- [ ] **Step 1: Update README operator guidance**

Delete the Primary RPC table row for `revenue-policy ban|unban|list-banned`. Replace the retained-alias paragraph with:

```markdown
The primary dispatchers are `revenue-cycle`, `revenue-budget`, and
`revenue-policy`. Retained standalone aliases cover fee/rebalance cycles,
analysis/wake, and total-cost/capex/spend-ledger reads. Removed peer-ban,
planner, Boltz, and LN+ names are not compatibility aliases and must return
method-not-found.
```

- [ ] **Step 2: Update the canonical action-RPC inventory**

Delete `revenue-ban` and `revenue-unban` from the action/mutation list. Delete `revenue-list-banned` from the read-only list. Preserve the surrounding classification and explanatory text.

- [ ] **Step 3: Update the compatibility catalog**

Change `Full 39-method list` to `Full 36-method list`. Replace the current Phase C primary-name statement with:

```markdown
Phase C retained operator-surface dispatchers: `revenue-cycle <subsystem>`,
`revenue-budget [section]`, and the diagnostic `revenue-policy` actions are the
primary operator names.
```

- [ ] **Step 4: Scan current contracts for stale live-surface claims**

Run:

```bash
rg -n -i 'revenue-(ban|unban|list-banned)|revenue-policy (ban|unban|list-banned)|BANNED_TAG|ban_peer|unban_peer|is_peer_banned' \
  cl-revenue-ops.py modules tests README.md \
  docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md \
  docs/refactor/phase0/compatibility-catalog.md
```

Expected: no matches. Historical documents outside this command are intentionally excluded.

- [ ] **Step 5: Commit contract synchronization**

```bash
git add README.md docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md \
  docs/refactor/phase0/compatibility-catalog.md
git commit -m "docs(policy): retire peer-ban contract"
```

### Task 4: Ordered Security Verification

**Files:**
- Inspect: all files changed since `17d131e`
- Test: repository verification only; no production files change in this task.

**Interfaces:**
- Consumes: Tasks 1-3 commits.
- Produces: exact evidence for applicability/buildability, security closure, bypass review, preserved behavior, repository checks, and independent review.

- [ ] **Step 1: Applicability and buildability gate**

Run:

```bash
git diff --check 17d131e..HEAD
git diff --stat 17d131e..HEAD
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  cl-revenue-ops.py modules/policy_manager.py
```

Expected: no whitespace errors, only approved files changed, and compilation succeeds.

- [ ] **Step 2: Security-closure gate**

Run the focused dispatcher regression:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_phase_c_dispatchers.py::TestRemovedPolicyBanActions \
  tests/test_rpc_surface_inventory.py \
  tests/test_policy_manager.py::TestRemovedPeerBanAPI
```

Expected: all pass. Re-run the current-contract `rg` scan from Task 3 and verify zero matches.

- [ ] **Step 3: Change-aware bypass review gate**

Inspect every retained `revenue_policy` branch and all direct callers of `PolicyManager.set_policy`, `delete_policy`, `add_tag`, `remove_tag`, and `set_policies_batch`:

```bash
rg -n 'revenue_policy\(|set_policy\(|delete_policy\(|add_tag\(|remove_tag\(|set_policies_batch\(' \
  cl-revenue-ops.py modules tests
```

Confirm the only generic dispatcher mutations remain in `TACTICAL_POLICY_ACTIONS` and require `internal` or `admin`. Exercise `BAN`, whitespace-padded `unban`, and `list-banned` through the parametrized regression so normalization cannot restore a bypass.

- [ ] **Step 4: Preserved-behavior gate**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_operator_surface.py \
  tests/test_rebalance_policy_gate.py \
  tests/test_fee_controller.py \
  tests/test_fee_setting_execution.py \
  tests/test_retained_revenue_core.py \
  tests/test_architecture_guard.py
```

Expected: diagnostics, authorized generic policy behavior, passive fee behavior, disabled/passive rebalance behavior, and standalone guards all pass.

- [ ] **Step 5: Repository-check gate**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q
```

Expected: full repository suite passes with no failures. Do not proceed to a `fixed` outcome if this relevant gate is unavailable or fails.

- [ ] **Step 6: Verify regression sensitivity**

Temporarily compare the focused regression against base commit `17d131e` or inspect the recorded RED output from Task 1. Confirm failures were specifically caused by the present ban surface, not test setup or syntax errors. Do not modify or reset the completed branch to perform this check.

- [ ] **Step 7: Independent tier-1 review**

Provide the reviewer task 92, the approved design, the final diff, RED/GREEN evidence, and ordered gate results. The reviewer independently retraces the source-to-sink path and passes or fails only the `review` criterion through `hexmem_task_verify`; the owner must not pass it.

- [ ] **Step 8: Record routing outcome and final status**

After independent review, record a routing observation for task 92. Mark the owner-controlled `tests` criterion only from fresh command output. The task completes only when the independent reviewer passes `review`.
