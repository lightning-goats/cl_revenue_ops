# Boltz Standing Reserve Auto-Cycle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the in-plugin Boltz auto-cycle automatically choose between standing-reserve treasury swaps and balance-driven Boltz swaps, while keeping channel rebalancing separate.

**Architecture:** CLBOSS does not have one global "rebalance or swap" chooser. It runs rebalancing and swapping as separate subsystems with separate triggers, and `cl-revenue_ops` should keep that same boundary: Sling/rebalancer continues to move liquidity between channels, while Boltz automation decides only which Boltz mode to run in a given cycle. The auto-cycle should prefer `treasury` mode whenever confirmed on-chain funds are below the configured reserve target, even if there are no planned channel opens, and otherwise fall back to `balance` mode when a profitable loop-in or loop-out candidate exists.

**Tech Stack:** Python 3.10+, `pyln-client`, existing plugin RPC methods in `cl-revenue-ops.py`, `BoltzCliManager`, SQLite-backed plugin state, `pytest`

---

## Design Constraints

- Preserve the current product boundary: rebalancing remains `revenue_rebalance` / Sling territory; Boltz automation does not replace channel-to-channel liquidity movement.
- Maintain a standing on-chain reserve even when no channel opens are queued.
- Reuse the existing Boltz RPCs and execution paths: `revenue_boltz_expansion_treasury_cycle()` and `revenue_boltz_balance_cycle()`.
- Change decisioning and orchestration, not the low-level Boltz execution flow, except where budget and result bookkeeping need hooks.
- Execute at most one Boltz mode per scheduler tick.

## Reference Behavior From CLBOSS

- CLBOSS swaps for two reasons: on-chain funding need (`NeedsOnchainFundsSwapper`) and node-wide inbound shortage (`NodeBalanceSwapper`).
- CLBOSS rebalances through separate modules (`JitRebalancer`, `EarningsRebalancer`).
- The practical takeaway for this repo: do not build a "swap instead of rebalance" arbiter. Build a Boltz-mode selector that operates inside the Boltz subsystem only.

## Non-Goals

- Do not redesign `modules/boltz_manager.py` swap creation APIs.
- Do not replace or bypass the current rebalancer.
- Do not make Boltz decisions depend on queued planner opens; the reserve target is intentionally independent.

## Phase 1: Treasury-First Mode Selection

### Task 1: Add a pure Boltz auto-cycle mode selector

**Files:**
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_boltz_integration.py`

**Step 1: Write the failing tests**

In `tests/test_boltz_integration.py`, add a small plugin-module loader and a new test class for pure selection logic:

```python
from tests.plugin_test_utils import load_plugin_module


def _load_boltz_plugin_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    return mod


class TestBoltzAutoCycleModeSelection:
    def test_prefers_treasury_when_reserve_below_target(self):
        mod = _load_boltz_plugin_module()

        selection = mod._select_boltz_auto_cycle_mode(
            treasury_plan={
                "status": "ok",
                "recommendations": [{"channel_id": "100x1x0"}],
                "treasury": {"deficit_sats": 700000, "min_deficit_sats": 250000},
            },
            balance_plan={
                "recommendations": [{"channel_id": "200x1x0"}],
            },
        )

        assert selection["mode"] == "treasury"
        assert selection["reason"] == "standing_onchain_reserve_below_target"
        assert selection["reserve_deficit_sats"] == 700000

    def test_falls_back_to_balance_when_treasury_is_at_target(self):
        mod = _load_boltz_plugin_module()

        selection = mod._select_boltz_auto_cycle_mode(
            treasury_plan={
                "status": "at_target",
                "recommendations": [],
                "treasury": {"deficit_sats": 0, "min_deficit_sats": 250000},
            },
            balance_plan={
                "recommendations": [{"channel_id": "200x1x0"}],
            },
        )

        assert selection["mode"] == "balance"
        assert selection["reason"] == "onchain_reserve_healthy_use_balance_mode"

    def test_returns_idle_when_no_mode_has_candidates(self):
        mod = _load_boltz_plugin_module()

        selection = mod._select_boltz_auto_cycle_mode(
            treasury_plan={"status": "at_target", "recommendations": [], "treasury": {"deficit_sats": 0}},
            balance_plan={"recommendations": []},
        )

        assert selection["mode"] == "idle"
        assert selection["reason"] == "no_eligible_boltz_actions"
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_boltz_integration.py -k "BoltzAutoCycleModeSelection" -v
```

Expected: FAIL with `AttributeError` for missing `_select_boltz_auto_cycle_mode`.

**Step 3: Write the minimal implementation**

In `cl-revenue-ops.py`, add a small pure helper near `_run_boltz_auto_cycle_once()`:

```python
def _select_boltz_auto_cycle_mode(*, treasury_plan: Optional[Dict[str, Any]], balance_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    treasury_plan = treasury_plan if isinstance(treasury_plan, dict) else {}
    balance_plan = balance_plan if isinstance(balance_plan, dict) else {}

    treasury = treasury_plan.get("treasury", {}) if isinstance(treasury_plan.get("treasury"), dict) else {}
    treasury_recs = list(treasury_plan.get("recommendations", []))
    balance_recs = list(balance_plan.get("recommendations", []))
    deficit = int(treasury.get("deficit_sats", 0) or 0)

    if str(treasury_plan.get("status") or "") == "ok" and treasury_recs:
        return {
            "mode": "treasury",
            "reason": "standing_onchain_reserve_below_target",
            "reserve_deficit_sats": deficit,
            "treasury_candidate_count": len(treasury_recs),
            "balance_candidate_count": len(balance_recs),
        }

    if balance_recs:
        return {
            "mode": "balance",
            "reason": "onchain_reserve_healthy_use_balance_mode",
            "reserve_deficit_sats": deficit,
            "treasury_candidate_count": len(treasury_recs),
            "balance_candidate_count": len(balance_recs),
        }

    return {
        "mode": "idle",
        "reason": "no_eligible_boltz_actions",
        "reserve_deficit_sats": deficit,
        "treasury_candidate_count": len(treasury_recs),
        "balance_candidate_count": len(balance_recs),
    }
```

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_boltz_integration.py -k "BoltzAutoCycleModeSelection" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py tests/test_boltz_integration.py
git commit -m "feat: add boltz auto-cycle mode selector"
```

---

### Task 2: Route the scheduler through treasury-first selection

**Files:**
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_boltz_integration.py`

**Step 1: Write the failing tests**

Extend `tests/test_boltz_integration.py` with integration tests for `_run_boltz_auto_cycle_once()`:

```python
def test_auto_cycle_executes_treasury_mode_first():
    mod = _load_boltz_plugin_module()
    mod.boltz_manager = MagicMock(enabled=True)
    mod.config = MagicMock()
    mod.config.snapshot.return_value = MagicMock(
        boltz_auto_cycle_enabled=True,
        boltz_auto_cycle_max_actions=1,
        expansion_treasury_enabled=True,
        expansion_treasury_onchain_target_sats=5_000_000,
        expansion_treasury_min_deficit_sats=250_000,
        expansion_treasury_preferred_currency="BTC",
        expansion_treasury_max_actions=1,
        expansion_treasury_min_source_local_pct=80.0,
        expansion_treasury_exclude_protected=True,
    )
    mod._build_boltz_expansion_treasury_plan = MagicMock(return_value={
        "status": "ok",
        "recommendations": [{"channel_id": "100x1x0"}],
        "treasury": {"deficit_sats": 900000, "min_deficit_sats": 250000},
    })
    mod._build_boltz_balance_plan = MagicMock(return_value={
        "recommendations": [{"channel_id": "200x1x0"}],
    })
    mod.revenue_boltz_expansion_treasury_cycle = MagicMock(return_value={"status": "executed", "executed_count": 1})
    mod.revenue_boltz_balance_cycle = MagicMock(return_value={"status": "executed", "executed_count": 1})

    result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

    assert result["mode"] == "treasury"
    mod.revenue_boltz_expansion_treasury_cycle.assert_called_once()
    mod.revenue_boltz_balance_cycle.assert_not_called()


def test_auto_cycle_falls_back_to_balance_mode():
    mod = _load_boltz_plugin_module()
    mod.boltz_manager = MagicMock(enabled=True)
    mod.config = MagicMock()
    mod.config.snapshot.return_value = MagicMock(
        boltz_auto_cycle_enabled=True,
        boltz_auto_cycle_max_actions=1,
        expansion_treasury_enabled=True,
        expansion_treasury_onchain_target_sats=5_000_000,
        expansion_treasury_min_deficit_sats=250_000,
        expansion_treasury_preferred_currency="BTC",
        expansion_treasury_max_actions=1,
        expansion_treasury_min_source_local_pct=80.0,
        expansion_treasury_exclude_protected=True,
    )
    mod._build_boltz_expansion_treasury_plan = MagicMock(return_value={
        "status": "at_target",
        "recommendations": [],
        "treasury": {"deficit_sats": 0, "min_deficit_sats": 250000},
    })
    mod._build_boltz_balance_plan = MagicMock(return_value={
        "recommendations": [{"channel_id": "200x1x0"}],
    })
    mod.revenue_boltz_expansion_treasury_cycle = MagicMock(return_value={"status": "executed", "executed_count": 1})
    mod.revenue_boltz_balance_cycle = MagicMock(return_value={"status": "executed", "executed_count": 1})

    result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

    assert result["mode"] == "balance"
    mod.revenue_boltz_balance_cycle.assert_called_once()
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_boltz_integration.py -k "auto_cycle_executes_treasury_mode_first or auto_cycle_falls_back_to_balance_mode" -v
```

Expected: FAIL because `_run_boltz_auto_cycle_once()` always calls `revenue_boltz_balance_cycle()`.

**Step 3: Update `_run_boltz_auto_cycle_once()` and status reporting**

In `cl-revenue-ops.py`:

- Build the treasury plan first when `expansion_treasury_enabled` is on.
- Only build the balance plan when treasury mode is not selected.
- Execute exactly one cycle method based on the selection.
- Preserve `status`, `executed_count`, and underlying plan output, but also stamp the result with:
  - `mode`
  - `selection_reason`
  - `reserve_deficit_sats`
  - `trigger`
- Expand `revenue_boltz_auto_cycle_status()` config output to include:
  - `expansion_treasury_enabled`
  - `expansion_treasury_onchain_target_sats`
  - `expansion_treasury_min_deficit_sats`

Use this structure:

```python
treasury_plan = None
if bool(getattr(cfg, "expansion_treasury_enabled", False)):
    treasury_plan = _build_boltz_expansion_treasury_plan(...)

selection = _select_boltz_auto_cycle_mode(treasury_plan=treasury_plan, balance_plan=None)

balance_plan = None
if selection["mode"] != "treasury":
    balance_plan = _build_boltz_balance_plan(...)
    selection = _select_boltz_auto_cycle_mode(
        treasury_plan=treasury_plan,
        balance_plan=balance_plan,
    )

if selection["mode"] == "treasury":
    result = revenue_boltz_expansion_treasury_cycle(plugin=plugin, dry_run=False, max_actions=max_actions, allow_concurrent_swaps=False)
elif selection["mode"] == "balance":
    result = revenue_boltz_balance_cycle(plugin=plugin, dry_run=False, max_actions=max_actions, allow_concurrent_swaps=False, loop_in_currency="LBTC", loop_out_currency="LBTC")
else:
    result = {"status": "idle", "executed_count": 0, "skipped_count": 0}

result["mode"] = selection["mode"]
result["selection_reason"] = selection["reason"]
result["reserve_deficit_sats"] = selection["reserve_deficit_sats"]
result["trigger"] = trigger
```

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_boltz_integration.py -k "auto_cycle_executes_treasury_mode_first or auto_cycle_falls_back_to_balance_mode or BoltzAutoCycleModeSelection" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py tests/test_boltz_integration.py
git commit -m "feat: make boltz auto-cycle treasury-first"
```

---

## Phase 2: Budget Correctness

### Task 3: Make the Boltz budget guard count pending swap reserves

**Files:**
- Modify: `modules/boltz_manager.py`
- Test: `tests/test_boltz_manager.py`

**Step 1: Write the failing test**

In `tests/test_boltz_manager.py`, add a test for `get_budget_status()`:

```python
def test_get_budget_status_counts_local_pending_swaps_as_reserved():
    mgr = _make_manager(daily_budget_sats=200, enforce_budget=True)
    mgr._get_global_budget_limit = MagicMock(return_value={"budget_sats": 200, "source": "fixed"})
    mgr.get_boltz_cost_components = MagicMock(return_value={
        "spent_24h_sats": 40,
        "reserved_24h_sats": 60,
        "counted_details": [],
        "skipped_without_timestamp": 0,
    })
    mgr._get_external_liquidity_costs = MagicMock(return_value={
        "spent_24h_sats": 20,
        "reserved_24h_sats": 10,
    })

    result = mgr.get_budget_status()

    assert result["reserved_24h_sats_estimate"] == 70
    assert result["remaining_24h_sats_estimate"] == 70
    assert result["boltz_reserved_24h_sats_estimate"] == 60
```

**Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_boltz_manager.py -k "counts_local_pending_swaps_as_reserved" -v
```

Expected: FAIL because `get_budget_status()` currently ignores local Boltz `reserved_24h_sats`.

**Step 3: Implement the budget fix**

In `modules/boltz_manager.py:get_budget_status()`:

- Read `local_reserved = local.get("reserved_24h_sats", 0)`.
- Compute `total_reserved = local_reserved + external_reserved`.
- Add `boltz_reserved_24h_sats_estimate` to the returned dict.
- Keep the external-liquidity breakdown as-is for visibility.

Use this exact shape:

```python
local_reserved = max(0, self._parse_int(local.get("reserved_24h_sats"), 0))
external_reserved = max(0, self._parse_int(external.get("reserved_24h_sats"), 0))
total_reserved = local_reserved + external_reserved
remaining = max(0, budget - total_spent - total_reserved)

return {
    ...
    "reserved_24h_sats_estimate": total_reserved,
    "boltz_reserved_24h_sats_estimate": local_reserved,
    ...
}
```

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_boltz_manager.py -k "counts_local_pending_swaps_as_reserved or pending_swap_counted_as_reserved" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/boltz_manager.py tests/test_boltz_manager.py
git commit -m "fix: count boltz pending reserves in budget guard"
```

---

## Phase 3: Operator Clarity

### Task 4: Document the treasury-first automation boundary

**Files:**
- Modify: `README.md`
- Modify: `tests/test_operator_surface.py`

**Step 1: Write the failing README test**

In `tests/test_operator_surface.py`, add a README assertion that makes the product boundary explicit:

```python
def test_readme_describes_boltz_treasury_first_boundary():
    readme = Path("README.md").read_text()

    assert "standing on-chain reserve" in readme
    assert "treasury mode first" in readme
    assert "does not replace channel rebalancing" in readme
```

**Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_operator_surface.py -k "boltz_treasury_first_boundary" -v
```

Expected: FAIL because the README does not yet describe the new auto-cycle behavior.

**Step 3: Update the README**

In `README.md`, add or update the Boltz section with:

- One paragraph explaining that the scheduler chooses `treasury` mode first when confirmed on-chain reserve is below `expansion_treasury_onchain_target_sats`.
- One paragraph explaining that when reserve is healthy, the scheduler falls back to the existing balance cycle.
- One sentence stating that Boltz automation does not replace channel rebalancing; Sling still handles channel-to-channel liquidity movement.
- One sentence stating that reserve maintenance is independent of pending planner opens.

Suggested wording:

```markdown
The in-plugin Boltz auto-cycle is treasury-first. When confirmed on-chain funds are below the configured reserve target, it runs expansion-treasury reverse swaps to rebuild a standing on-chain reserve. When the reserve is healthy, it falls back to the existing balance cycle and only considers profitable loop-in or loop-out candidates.

This automation does not replace channel rebalancing. Sling remains responsible for channel-to-channel liquidity movement; Boltz is only used for Lightning-to-on-chain or on-chain-to-Lightning conversion decisions.
```

**Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_operator_surface.py -k "boltz_treasury_first_boundary" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md tests/test_operator_surface.py
git commit -m "docs: document treasury-first boltz automation"
```

---

## Final Verification

Run the focused suite:

```bash
python3 -m pytest tests/test_boltz_integration.py tests/test_boltz_manager.py tests/test_operator_surface.py -v
```

Expected: PASS.

Run one manual plugin smoke check after deployment:

```bash
lightning-cli revenue-boltz-auto-cycle-status
lightning-cli revenue-boltz-expansion-treasury-status
```

Expected:

- `revenue-boltz-auto-cycle-status` shows treasury config and the most recent `mode`.
- `revenue-boltz-expansion-treasury-status` shows `needs_harvest=true` when reserve is below target and `false` when reserve is healthy.

## Follow-On Work (Not In This Plan)

- Feed real depletion prediction into `_boltz_dynamic_channel_tuning()` instead of leaving `predicted_depletion_hours = None`.
- If later needed, persist Boltz cooldown state across restarts. Keep that out of this first pass unless restart churn becomes a real operational issue.
