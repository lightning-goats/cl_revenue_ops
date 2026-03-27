# Remaining Audit Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the remaining 2026-03-27 audit findings by hardening planner close safety in `cl-revenue-ops`, removing the unsupported predictive rebalance RPC from `cl-hive`, and re-baselining the audit docs and live MCP contract.

**Architecture:** Keep destructive paths fail-closed and keep the public RPC surface aligned with real backends. The `cl-revenue-ops` change should make close-policy lookup errors block close eligibility instead of permitting it. The `cl-hive` change should remove the stubbed `hive-rebalance-recommendations` RPC entirely so callers cannot mistake an always-empty success response for a real analysis result.

**Tech Stack:** Python, pytest, pyln-client plugin decorators, MCP compatibility wrapper (local-only), two local repos: `/home/sat/bin/cl_revenue_ops` and `/home/sat/bin/cl-hive`

---

### Task 1: Fail Closed on Planner Close Policy Errors

**Files:**
- Modify: `tests/test_capacity_planner.py`
- Modify: `modules/capacity_planner.py`
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`

**Step 1: Write the failing test**

Update the existing close-policy regression in `tests/test_capacity_planner.py` so policy lookup errors block close eligibility instead of allowing it.

```python
def test_close_allowed_on_policy_exception(self):
    """Policy lookup failures must block auto-close decisions."""
    planner, db, pm = _make_close_planner()
    pm.get_policy.side_effect = Exception("DB error")

    allowed, reason = planner._check_close_allowed("peer_abc")

    assert allowed is False
    assert "Policy unavailable" in reason
```

**Step 2: Run test to verify it fails**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py::TestDirectClose::test_close_allowed_on_policy_exception -q
```

Expected: `FAIL` because the current code returns `allowed is True`.

**Step 3: Write minimal implementation**

Change the exception path in `modules/capacity_planner.py` so policy lookup failures return a blocking result.

```python
except Exception as e:
    self.plugin.log(f"Policy check failed for {peer_id[:12]}...: {e}", level='warn')
    return False, "Policy unavailable"
```

Keep the existing hard blocks for `static`, `passive`, `protect`, and `no_close`.

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py::TestDirectClose::test_close_allowed_on_policy_exception tests/test_capacity_planner.py::TestDirectClose::test_close_respects_static_policy tests/test_capacity_planner.py::TestDirectClose::test_close_allowed_for_dynamic_policy -q
```

Expected: `PASS`

Then run the touched file:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py -q
```

Expected: all tests in `tests/test_capacity_planner.py` pass.

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl_revenue_ops add tests/test_capacity_planner.py modules/capacity_planner.py docs/audits/2026-03-27-full-plugin-audit-report.md
git -C /home/sat/bin/cl_revenue_ops commit -m "fix: fail closed on planner close policy errors"
```


### Task 2: Remove the Stubbed `hive-rebalance-recommendations` RPC

**Files:**
- Modify: `../cl-hive/tests/test_rpc.py`
- Modify: `../cl-hive/cl-hive.py`
- Modify: `../cl-hive/modules/rpc_commands.py`
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`

**Step 1: Write the failing test**

Teach the test loader in `../cl-hive/tests/test_rpc.py` to record registered plugin method names, then add a contract test asserting the stubbed RPC is no longer exported.

```python
class _TestPlugin:
    def __init__(self):
        self.methods = {}

    def method(self, name, *_args, **_kwargs):
        def decorator(fn):
            self.methods[name] = fn.__name__
            return fn
        return decorator
```

```python
def test_stubbed_rebalance_rpc_not_registered():
    mod = _load_cl_hive_main()
    assert "hive-rebalance-recommendations" not in mod.plugin.methods
```

**Step 2: Run test to verify it fails**

Run:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest /home/sat/bin/cl-hive/tests/test_rpc.py::test_stubbed_rebalance_rpc_not_registered -q
```

Expected: `FAIL` because the current main plugin still registers `hive-rebalance-recommendations`.

**Step 3: Write minimal implementation**

Remove the dead surface from `../cl-hive/cl-hive.py` and `../cl-hive/modules/rpc_commands.py`.

- Delete the import alias for `rebalance_recommendations as rpc_rebalance_recommendations`
- Delete the `@plugin.method("hive-rebalance-recommendations")` wrapper
- Delete the dead `rebalance_recommendations()` handler that only returns an empty payload

Minimal target state:

```python
# No rpc_rebalance_recommendations import
# No @plugin.method("hive-rebalance-recommendations") wrapper
# No dead rebalance_recommendations() handler in modules/rpc_commands.py
```

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest /home/sat/bin/cl-hive/tests/test_rpc.py::test_stubbed_rebalance_rpc_not_registered -q
```

Expected: `PASS`

Then run the touched contract files:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest /home/sat/bin/cl-hive/tests/test_rpc.py /home/sat/bin/cl-hive/tests/test_rpc_commands_audit.py -q
```

Expected: all tests in those files pass.

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add /home/sat/bin/cl-hive/tests/test_rpc.py /home/sat/bin/cl-hive/cl-hive.py /home/sat/bin/cl-hive/modules/rpc_commands.py
git -C /home/sat/bin/cl-hive commit -m "fix: remove stubbed rebalance recommendation rpc"
```


### Task 3: Re-Baseline Audit Docs and Verify the Live Contract

**Files:**
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`

**Step 1: Update the audit artifacts**

Mark the two remaining findings as resolved and update the matrix to note that `hive-rebalance-recommendations` has been removed from the supported `cl-hive` RPC inventory.

Suggested doc changes:

```markdown
- Resolved on 2026-03-27: close-policy errors now fail closed in `modules/capacity_planner.py`
- Resolved on 2026-03-27: `hive-rebalance-recommendations` removed from `cl-hive` export surface
```

**Step 2: Run repo-level verification**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py tests/test_operator_surface.py -q
/home/sat/bin/cl-hive/.venv/bin/python -m pytest /home/sat/bin/cl-hive/tests/test_rpc.py /home/sat/bin/cl-hive/tests/test_rpc_commands_audit.py -q
```

Expected: all tests pass.

**Step 3: Verify the live MCP contract after service reload**

Reload the production MCP server using your normal service restart path, then verify:

- `hive-status` still returns successfully for `hive-nexus-01`
- `hive-rebalance-recommendations` is no longer advertised or callable
- The local-only compatibility wrapper rebuilds its allowlist from the updated live decorators

If the service restart path is available in-session, re-run the corresponding MCP checks after reload.

**Step 4: Commit the audit-doc updates**

```bash
git -C /home/sat/bin/cl_revenue_ops add docs/audits/2026-03-27-full-plugin-audit-report.md docs/audits/2026-03-27-plugin-rpc-matrix.md docs/plans/2026-03-27-remaining-audit-remediation.md
git -C /home/sat/bin/cl_revenue_ops commit -m "docs: close remaining audit findings"
```

**Step 5: Final verification note**

Record in the closing notes:

- the exact pytest commands that passed
- whether the MCP server was reloaded
- whether the RPC disappeared from the live MCP surface

