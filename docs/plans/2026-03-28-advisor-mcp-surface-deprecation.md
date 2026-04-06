# Advisor MCP Surface Deprecation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop mixing dead legacy advisor endpoints with live operational MCP surfaces by explicitly classifying historical DB readers, hard-failing dead advisor-runtime tools with replacement guidance, and removing stale advisor service artifacts.

**Architecture:** The current system has two separate paths under the "advisor" name: live operational state from `cl-hive` and `cl-revenue-ops`, and a dead legacy proactive advisor runtime that stopped producing fresh snapshots on March 6, 2026. The cleanup should make that split explicit in the MCP layer, preserve read-only historical access to `advisor.db`, and retire the inactive `hive-advisor` service/timer so operators are not misled by stale surfaces.

**Tech Stack:** Python MCP server, `cl-hive` production systemd units, SQLite `advisor.db`, pytest

---

### Task 1: Re-establish the Advisor MCP Source of Truth

**Files:**
- Create or restore: `/home/sat/bin/cl-hive/tools/mcp-hive-server.py`
- Create or restore: `/home/sat/bin/cl-hive/tools/mcp_hive_server_helpers.py`
- Create or restore: `/home/sat/bin/cl-hive/tools/advisor_db.py`
- Create: `/home/sat/bin/cl-hive/tests/test_mcp_hive_server.py`
- Inspect reference only: `/home/sat/bin/cl-hive/.worktrees/hive-directional-bias-20260314/tools/mcp-hive-server.py`
- Inspect reference only: `/home/sat/bin/cl-hive/.worktrees/hive-directional-bias-20260314/tools/advisor_db.py`
- Inspect reference only: `/home/sat/bin/cl-hive/.worktrees/hive-directional-bias-20260314/tools/mcp_hive_server_helpers.py`

**Step 1: Write the failing source-presence test**

```python
from pathlib import Path


def test_advisor_mcp_server_sources_exist():
    assert Path("/home/sat/bin/cl-hive/tools/mcp-hive-server.py").exists()
    assert Path("/home/sat/bin/cl-hive/tools/mcp_hive_server_helpers.py").exists()
    assert Path("/home/sat/bin/cl-hive/tools/advisor_db.py").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py::test_advisor_mcp_server_sources_exist -v`
Expected: FAIL because the current `cl-hive/main` checkout only has compiled advisor artifacts and no checked-in MCP server source files.

**Step 3: Restore the authoritative MCP server sources**

```python
# Restore the missing source files to /home/sat/bin/cl-hive/tools/
# from the last known good source location or git history.
# Do not continue deprecation work against pyc-only artifacts.
```

**Step 4: Run test to verify it passes**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py::test_advisor_mcp_server_sources_exist -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add tools/mcp-hive-server.py tools/mcp_hive_server_helpers.py tools/advisor_db.py tests/test_mcp_hive_server.py
git -C /home/sat/bin/cl-hive commit -m "chore: restore advisor mcp server sources"
```

### Task 2: Freeze the Advisor Surface Classification Contract

**Files:**
- Modify: `/home/sat/bin/cl-hive/tools/mcp-hive-server.py`
- Modify: `/home/sat/bin/cl-hive/tools/mcp_hive_server_helpers.py`
- Modify: `/home/sat/bin/cl-hive/tests/test_mcp_hive_server.py`

**Step 1: Write the failing classification tests**

```python
import pytest


@pytest.mark.parametrize(
    ("tool_name", "expected_class"),
    [
        ("advisor_get_context_brief", "historical"),
        ("advisor_get_trends", "historical"),
        ("advisor_get_cycle_history", "historical"),
        ("advisor_get_recent_decisions", "historical"),
        ("advisor_db_stats", "historical"),
        ("advisor_get_goals", "historical"),
        ("advisor_dedup_status", "historical"),
        ("advisor_measure_outcomes", "historical"),
        ("advisor_get_status", "deprecated_runtime"),
        ("advisor_scan_opportunities", "deprecated_runtime"),
        ("advisor_get_learning", "deprecated_runtime"),
    ],
)
def test_advisor_tool_classification(tool_name, expected_class, advisor_tool_registry):
    meta = advisor_tool_registry[tool_name]
    assert meta["surface_class"] == expected_class
    assert meta["replacement_surfaces"]
```

**Step 2: Run test to verify it fails**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py -k advisor_tool_classification -v`
Expected: FAIL because the current tool layer does not expose an explicit live-vs-historical-vs-deprecated contract.

**Step 3: Implement the metadata table**

```python
ADVISOR_SURFACE_CLASSIFICATION = {
    "advisor_get_context_brief": {
        "surface_class": "historical",
        "replacement_surfaces": [
            "hive_health",
            "hive_node_diagnostic",
            "revenue_status",
            "revenue_ops_health",
        ],
    },
    "advisor_get_status": {
        "surface_class": "deprecated_runtime",
        "replacement_surfaces": [
            "hive_health",
            "hive_status",
            "revenue_status",
        ],
    },
}
```

**Step 4: Run test to verify it passes**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py -k advisor_tool_classification -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add tools/mcp-hive-server.py tools/mcp_hive_server_helpers.py tests/test_mcp_hive_server.py
git -C /home/sat/bin/cl-hive commit -m "feat: classify legacy advisor mcp surfaces"
```

### Task 3: Make Dead Advisor Runtime Tools Fail Explicitly

**Files:**
- Modify: `/home/sat/bin/cl-hive/tools/mcp-hive-server.py`
- Modify: `/home/sat/bin/cl-hive/tests/test_mcp_hive_server.py`

**Step 1: Write the failing deprecated-runtime tests**

```python
@pytest.mark.parametrize(
    "tool_name",
    [
        "advisor_get_status",
        "advisor_scan_opportunities",
        "advisor_get_learning",
        "learning_engine_insights",
    ],
)
def test_dead_advisor_runtime_tools_return_deprecation_payload(call_mcp_tool, tool_name):
    result = call_mcp_tool(tool_name)
    assert result["ok"] is False
    assert result["error_code"] == "legacy_advisor_runtime_unavailable"
    assert "replacement_surfaces" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py -k dead_advisor_runtime_tools -v`
Expected: FAIL because the current runtime leaks import/runtime errors like `No module named 'learning_engine'`.

**Step 3: Replace runtime/import leakage with a stable deprecation error**

```python
def legacy_advisor_runtime_unavailable(*replacement_surfaces):
    return {
        "ok": False,
        "error": "Legacy proactive advisor runtime is not part of current operations",
        "error_code": "legacy_advisor_runtime_unavailable",
        "replacement_surfaces": list(replacement_surfaces),
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py -k dead_advisor_runtime_tools -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add tools/mcp-hive-server.py tests/test_mcp_hive_server.py
git -C /home/sat/bin/cl-hive commit -m "fix: hard-fail dead advisor runtime tools with replacements"
```

### Task 4: Mark Historical Advisor Readers as Historical, Not Live

**Files:**
- Modify: `/home/sat/bin/cl-hive/tools/mcp-hive-server.py`
- Modify: `/home/sat/bin/cl-hive/tools/advisor_db.py`
- Modify: `/home/sat/bin/cl-hive/tests/test_mcp_hive_server.py`

**Step 1: Write the failing historical-surface tests**

```python
@pytest.mark.parametrize(
    "tool_name",
    [
        "advisor_get_context_brief",
        "advisor_get_trends",
        "advisor_get_cycle_history",
        "advisor_get_recent_decisions",
        "advisor_db_stats",
        "advisor_get_goals",
        "advisor_dedup_status",
        "advisor_measure_outcomes",
    ],
)
def test_historical_advisor_tools_include_freshness_metadata(call_mcp_tool, tool_name):
    result = call_mcp_tool(tool_name)
    assert result["ok"] is True
    assert result["meta"]["surface_class"] == "historical"
    assert result["meta"]["live_authoritative"] is False
    assert result["meta"]["replacement_surfaces"]
```

**Step 2: Run test to verify it fails**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py -k historical_advisor_tools -v`
Expected: FAIL because these tools currently return raw DB payloads without explicit freshness classification.

**Step 3: Wrap DB-backed outputs with historical metadata**

```python
def with_historical_meta(payload, *replacement_surfaces):
    payload["meta"] = {
        "surface_class": "historical",
        "live_authoritative": False,
        "replacement_surfaces": list(replacement_surfaces),
    }
    return payload
```

**Step 4: Run test to verify it passes**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py -k historical_advisor_tools -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add tools/mcp-hive-server.py tools/advisor_db.py tests/test_mcp_hive_server.py
git -C /home/sat/bin/cl-hive commit -m "feat: annotate historical advisor surfaces"
```

### Task 5: Retire the Stale Advisor Service and Timer Artifacts

**Files:**
- Modify: `/home/sat/bin/cl-hive/production/systemd/hive-advisor.service`
- Modify: `/home/sat/bin/cl-hive/production/systemd/hive-advisor.timer`
- Create: `/home/sat/bin/cl-hive/docs/advisor-surface-matrix.md`

**Step 1: Write the failing artifact-consistency test**

```python
from pathlib import Path


def test_hive_advisor_service_does_not_point_to_missing_script():
    service_text = Path("/home/sat/bin/cl-hive/production/systemd/hive-advisor.service").read_text()
    assert "run-advisor.sh" not in service_text
```

**Step 2: Run test to verify it fails**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py::test_hive_advisor_service_does_not_point_to_missing_script -v`
Expected: FAIL because the current service still points to `production/scripts/run-advisor.sh`, which is absent from `cl-hive/main`.

**Step 3: Replace the legacy unit with an explicit deprecation stub and doc link**

```ini
[Unit]
Description=Deprecated Hive Proactive AI Advisor

[Service]
Type=oneshot
ExecStart=/usr/bin/printf 'Deprecated: use live MCP operational surfaces instead of hive-advisor.service\n'
```

**Step 4: Run test to verify it passes**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py::test_hive_advisor_service_does_not_point_to_missing_script -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add production/systemd/hive-advisor.service production/systemd/hive-advisor.timer docs/advisor-surface-matrix.md tests/test_mcp_hive_server.py
git -C /home/sat/bin/cl-hive commit -m "chore: retire stale hive advisor service artifacts"
```

### Task 6: Publish the Replacement Matrix for Operators

**Files:**
- Modify: `/home/sat/bin/cl-hive/docs/advisor-surface-matrix.md`
- Modify: `/home/sat/bin/cl-hive/README.md`

**Step 1: Write the failing doc-check test**

```python
from pathlib import Path


def test_advisor_surface_matrix_documents_replacements():
    text = Path("/home/sat/bin/cl-hive/docs/advisor-surface-matrix.md").read_text()
    assert "advisor_get_context_brief -> historical" in text
    assert "advisor_get_status -> deprecated_runtime" in text
    assert "revenue_status" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py::test_advisor_surface_matrix_documents_replacements -v`
Expected: FAIL because the matrix doc does not exist yet.

**Step 3: Write the operator-facing matrix**

```markdown
| Legacy surface | Class | Use now |
| --- | --- | --- |
| advisor_get_context_brief | historical | No - use hive_node_diagnostic + revenue_status |
| advisor_get_status | deprecated_runtime | No - use hive_health + hive_status + revenue_status |
```

**Step 4: Run test to verify it passes**

Run: `pytest /home/sat/bin/cl-hive/tests/test_mcp_hive_server.py::test_advisor_surface_matrix_documents_replacements -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add docs/advisor-surface-matrix.md README.md tests/test_mcp_hive_server.py
git -C /home/sat/bin/cl-hive commit -m "docs: publish advisor mcp deprecation matrix"
```

