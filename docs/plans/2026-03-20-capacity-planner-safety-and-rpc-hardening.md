# Capacity Planner Safety And RPC Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the capacity planner's live execution bugs by removing wrapper-dependent CLN RPC calls, making channel closes recommendation-only by default, and hardening planner safety/accounting behavior.

**Architecture:** Keep the planner as the decisioning layer, but move all mutating CLN calls behind tiny planner-local RPC helpers that use generic JSON-RPC invocation instead of pyln convenience wrappers. Split close handling into two explicit behaviors: recommendation logging by default, and actual close execution only when an opt-in config flag is enabled. Tighten the cycle so live execution is conservative: fee gates apply before mutations, cooldown/database failures fail closed, and failed opens do not consume the rest of the cycle.

**Tech Stack:** Python 3.10+, `pyln-client` / generic CLN JSON-RPC, SQLite planner action log in `modules/database.py`, plugin option parsing in `cl-revenue-ops.py`, `pytest`

---

## Review Findings This Plan Fixes

- `fundchannel(id=...)` is broken in production because the deployed pyln wrapper rejects the `id` keyword.
- `close(id=...)` uses the same wrapper-dependent pattern and should be treated as suspect until fixed the same way.
- Close execution is live by default once the planner is enabled; that is unsafe and does not match the desired product behavior.
- The top-level fee gate is only advisory for closes.
- Failed opens still consume per-cycle open budget and local available-funds accounting.
- Cooldown/database errors currently fail open.
- `planner_max_opens_per_cycle` and `planner_max_closes_per_cycle` exist in config, but are not exposed as plugin options.

## Non-Goals

- Do not redesign winner/loser selection heuristics in this pass.
- Do not add multifundchannel batching in this pass.
- Do not change the planner candidate schema.
- Do not merge planner open/close execution with Boltz or Sling orchestration.

## Task 1: Lock Planner Mutations To Generic CLN RPC Calls

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

In `tests/test_capacity_planner.py`, replace the wrapper-shaped expectations with generic RPC call expectations:

```python
def test_execute_open_calls_generic_rpc_fundchannel(self):
    planner, db = _make_open_planner()
    cfg = _make_open_cfg()
    planner.plugin.rpc.call.return_value = {"channel_id": "123x1x0"}

    result = planner._execute_open("peer1", 2_000_000, cfg, "test reason")

    planner.plugin.rpc.call.assert_any_call(
        "fundchannel",
        {"id": "peer1", "amount": 2_000_000, "announce": True},
    )
    assert result["status"] == "completed"


def test_execute_close_calls_generic_rpc_close(self):
    planner, db = _make_close_planner()
    cfg = _make_close_cfg(planner_execute_closes=True)
    planner.plugin.rpc.call.return_value = {"type": "mutual"}

    result = planner._execute_close("100x1x0", "peer1", cfg, "test close")

    planner.plugin.rpc.call.assert_any_call("close", {"id": "100x1x0"})
    assert result["status"] == "completed"
```

Update the test fixture helpers so the default success path stubs `plugin.rpc.call.return_value` instead of `plugin.rpc.fundchannel.return_value` / `plugin.rpc.close.return_value`.

**Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_capacity_planner.py -k "generic_rpc_fundchannel or generic_rpc_close" -v
```

Expected: FAIL because `modules/capacity_planner.py` still calls wrapper methods directly.

**Step 3: Write the minimal implementation**

In `modules/capacity_planner.py`, add small helpers near `_execute_open()` / `_execute_close()`:

```python
    def _rpc_fundchannel(self, peer_id: str, amount_sats: int) -> Dict[str, Any]:
        return self.plugin.rpc.call(
            "fundchannel",
            {"id": peer_id, "amount": amount_sats, "announce": True},
        )

    def _rpc_close(self, channel_id: str) -> Dict[str, Any]:
        return self.plugin.rpc.call("close", {"id": channel_id})
```

Then switch `_execute_open()` and `_execute_close()` to use those helpers instead of `self.plugin.rpc.fundchannel(...)` / `self.plugin.rpc.close(...)`.

**Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_capacity_planner.py -k "generic_rpc_fundchannel or generic_rpc_close or TestChannelOpen or TestChannelClose" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "fix: use generic cln rpc calls in capacity planner"
```

---

## Task 2: Make Close Handling Recommendation-Only By Default

**Files:**
- Modify: `modules/config.py`
- Modify: `cl-revenue-ops.py`
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

Add a dedicated close-execution flag to the planner config test helpers and write regression tests:

```python
def _make_close_cfg(planner_dry_run=False, planner_execute_closes=False):
    cfg = MagicMock()
    cfg.planner_dry_run = planner_dry_run
    cfg.planner_execute_closes = planner_execute_closes
    return cfg


def test_execute_close_returns_recommended_when_close_execution_disabled(self):
    planner, db = _make_close_planner()
    cfg = _make_close_cfg(planner_execute_closes=False)

    result = planner._execute_close("100x1x0", "peer1", cfg, "zombie")

    assert result["status"] == "recommended"
    planner.plugin.rpc.call.assert_not_called()
    db.update_planner_action.assert_called_once_with(99, status="recommended")


def test_execute_cycle_logs_close_recommendation_by_default(self):
    planner, plugin, prof, flow, pm = _make_cycle_planner(
        all_profitability={scid: loser_prof},
        all_flow={scid: loser_flow},
    )
    cfg = _make_cycle_cfg(planner_execute_closes=False)

    result = planner.execute_cycle(cfg)

    assert len(result["closes"]) == 1
    assert result["closes"][0]["status"] == "recommended"
    plugin.rpc.call.assert_not_called()
```

Also add config parsing assertions:

```python
def test_planner_execute_closes_defaults_false():
    cfg = Config()
    assert cfg.planner_execute_closes is False
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_capacity_planner.py -k "recommended_when_close_execution_disabled or logs_close_recommendation_by_default" -v
```

Expected: FAIL because no separate close-execution flag exists yet.

**Step 3: Write the minimal implementation**

In `modules/config.py`, add the new field in both mutable and snapshot config models:

```python
    planner_execute_closes: bool = False
```

Register the type in `CONFIG_FIELD_TYPES`.

In `cl-revenue-ops.py`, add a startup option:

```python
plugin.add_option(
    name='revenue-ops-planner-execute-closes',
    default='false',
    description='Allow the capacity planner to execute close RPCs (default: false)',
)
```

Parse it into config init:

```python
planner_execute_closes=options.get('revenue-ops-planner-execute-closes', 'false').lower() in ('true', '1', 'yes'),
```

In `modules/capacity_planner.py`, split close behavior:

```python
        if not getattr(cfg, "planner_execute_closes", False):
            if db and action_id:
                db.update_planner_action(action_id, status="recommended")
            self.plugin.log(
                f"[RECOMMEND] Close {channel_id} (peer: {peer_id[:16]}..., reason: {reason})",
                level="info",
            )
            return {"action_id": action_id, "status": "recommended", "channel_id": channel_id, "peer_id": peer_id}
```

Do this before any rebalancer stop or CLN mutation.

**Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_capacity_planner.py -k "recommended or planner_execute_closes" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/config.py cl-revenue-ops.py modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: make planner closes recommendation-only by default"
```

---

## Task 3: Enforce Safety Gates Conservatively For Live Close Execution

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

Add regression coverage for fee-gated closes and fail-closed cooldown behavior:

```python
def test_execute_cycle_does_not_execute_closes_when_fee_gate_fails(self):
    planner, plugin, prof, flow, pm = _make_cycle_planner(
        feerates_return={"perkb": {"opening": 200000}},
        all_profitability={scid: loser_prof},
        all_flow={scid: loser_flow},
    )
    cfg = _make_cycle_cfg(
        planner_execute_closes=True,
        planner_max_fee_rate_sat_vb=50.0,
    )

    result = planner.execute_cycle(cfg)

    assert len(result["closes"]) == 0
    assert any("exceeds max" in reason for reason in result["skipped_reasons"])
    plugin.rpc.call.assert_not_called()


def test_cooldown_blocks_when_database_errors(self):
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    prof_analyzer.database.get_recent_planner_actions.side_effect = Exception("db locked")
    planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

    ok, reason = planner._check_cooldown("peer1")

    assert ok is False
    assert "Cooldown check failed" in reason
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_capacity_planner.py -k "fee_gate_fails or cooldown_blocks_when_database_errors" -v
```

Expected: FAIL because closes still run outside the fee gate and cooldown DB errors currently allow execution.

**Step 3: Write the minimal implementation**

In `modules/capacity_planner.py`, change `_check_cooldown()`:

```python
        except Exception as e:
            return False, f"Cooldown check failed: {e}"
```

In `execute_cycle()`, gate live closes explicitly:

```python
        closes_allowed = fee_ok and bool(getattr(cfg, "planner_execute_closes", False))
```

Then inside the close loop:

```python
            if getattr(cfg, "planner_execute_closes", False):
                guards_ok, guards_reason = self._check_safety_guards(cfg, "close", peer_id)
                if not guards_ok:
                    summary["skipped_reasons"].append(
                        f"Close guard failed for {scid}: {guards_reason}"
                    )
                    continue
```

Do not suppress recommendation logging when `planner_execute_closes` is false; only suppress actual mutations.

**Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_capacity_planner.py -k "fee_gate_fails or cooldown_blocks_when_database_errors or TestSafetyGuards" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "fix: harden planner close safety gates"
```

---

## Task 4: Stop Failed Opens From Consuming The Rest Of The Cycle

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

Add a cycle regression test that simulates one failed open followed by one valid candidate:

```python
def test_failed_open_does_not_consume_open_slot_or_available_funds(self):
    planner, plugin, prof, flow, pm = _make_cycle_planner(
        all_profitability=all_prof,
        all_flow=all_flow,
    )
    cfg = _make_cycle_cfg(planner_max_opens_per_cycle=2)

    plugin.rpc.call.side_effect = [
        Exception("temporary fundchannel failure"),
        {"channel_id": "second-open"},
    ]

    result = planner.execute_cycle(cfg)

    assert any(open_rec["result"] == "failed" for open_rec in result["opens"])
    assert any(open_rec["result"] == "completed" for open_rec in result["opens"])
```

If your helper setup also uses `plugin.rpc.call()` for `connect` or `close`, make the side-effect function branch on the RPC method string instead of using a flat list.

**Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_capacity_planner.py -k "failed_open_does_not_consume_open_slot_or_available_funds" -v
```

Expected: FAIL because `execute_cycle()` increments `opens_this_cycle` and decrements `available_sats` even when the open fails.

**Step 3: Write the minimal implementation**

In `modules/capacity_planner.py`, only consume cycle capacity on successful or simulated-successful outcomes:

```python
                status = result.get("status", "unknown")
                summary["opens"].append({
                    "peer_id": peer_id,
                    "amount_sats": channel_size,
                    "ev": round(ev, 0),
                    "result": status,
                    "action_id": result.get("action_id"),
                })

                if status in ("completed", "dry_run"):
                    opens_this_cycle += 1
                    available_sats = max(0, available_sats - channel_size)
```

Leave failed attempts visible in the summary, but do not let them block later candidates.

**Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_capacity_planner.py -k "failed_open_does_not_consume_open_slot_or_available_funds or execute_cycle_opens_best_candidate" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "fix: keep planner cycle progressing after failed opens"
```

---

## Task 5: Expose The Missing Planner Cycle Limit Options

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/config.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

Add config-init coverage around the missing fields:

```python
def test_planner_cycle_limits_are_read_from_plugin_options():
    options = {
        "revenue-ops-planner-enabled": "true",
        "revenue-ops-planner-interval": "21600",
        "revenue-ops-planner-dry-run": "false",
        "revenue-ops-planner-max-opens-per-cycle": "3",
        "revenue-ops-planner-max-closes-per-cycle": "0",
        "revenue-ops-planner-min-channel-sats": "500000",
        "revenue-ops-planner-max-channel-sats": "10000000",
        "revenue-ops-planner-max-fee-rate": "50.0",
    }

    cfg = _build_test_config_from_options(options)

    assert cfg.planner_max_opens_per_cycle == 3
    assert cfg.planner_max_closes_per_cycle == 0
```

If no helper exists, add one in the test file that exercises the same parsing code path used in plugin init.

**Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_capacity_planner.py -k "cycle_limits_are_read_from_plugin_options" -v
```

Expected: FAIL because plugin init never parses those options.

**Step 3: Write the minimal implementation**

In `cl-revenue-ops.py`, add plugin options:

```python
plugin.add_option(
    name='revenue-ops-planner-max-opens-per-cycle',
    default='1',
    description='Maximum automated channel opens per planner cycle (default: 1)'
)
plugin.add_option(
    name='revenue-ops-planner-max-closes-per-cycle',
    default='0',
    description='Maximum planner close executions per cycle when close execution is enabled (default: 0)'
)
```

Parse them into config init:

```python
planner_max_opens_per_cycle=_safe_int('revenue-ops-planner-max-opens-per-cycle'),
planner_max_closes_per_cycle=_safe_int('revenue-ops-planner-max-closes-per-cycle'),
```

In `modules/config.py`, keep the config defaults aligned:

```python
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 0
```

Update the range table only if necessary; it already allows `0`.

**Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_capacity_planner.py -k "cycle_limits_are_read_from_plugin_options or respects_max_opens_per_cycle or respects_max_closes_per_cycle" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/config.py tests/test_capacity_planner.py
git commit -m "fix: expose planner cycle limits in plugin options"
```

---

## Task 6: Update Planner Status And Operator Docs

**Files:**
- Modify: `README.md`
- Modify: `config/cl-revenue-ops.conf.minimal`
- Modify: `config/cl-revenue-ops.conf.full`
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_operator_surface.py`

**Step 1: Write the failing tests**

Add an operator-surface test that documents the new default:

```python
def test_readme_states_planner_closes_are_recommendation_only_by_default():
    readme = Path("README.md").read_text()
    assert "closes are recommendation-only by default" in readme
```

Add a planner status assertion:

```python
def test_planner_status_reports_execute_closes_flag():
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, config=cfg)
    status = planner.get_status()
    assert status["execute_closes"] is False
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_operator_surface.py -k "planner_closes_are_recommendation_only_by_default" -v
pytest tests/test_capacity_planner.py -k "planner_status_reports_execute_closes_flag" -v
```

Expected: FAIL because docs and status output do not mention the new close behavior.

**Step 3: Write the minimal implementation**

In `modules/capacity_planner.py`, extend `get_status()`:

```python
        return {
            "enabled": getattr(cfg, 'planner_enabled', False) if cfg else False,
            "dry_run": getattr(cfg, 'planner_dry_run', False) if cfg else False,
            "execute_closes": getattr(cfg, 'planner_execute_closes', False) if cfg else False,
            "candidate_pool_size": len(db.get_planner_candidates()) if db else 0,
            "recent_actions": db.get_planner_actions(limit=5) if db else [],
        }
```

Update `README.md` and both config examples to say:

```text
Planner opens may be automated when enabled.
Planner closes are recommendation-only by default.
Set revenue-ops-planner-execute-closes=true only for explicit close execution.
```

**Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_capacity_planner.py -k "planner_status_reports_execute_closes_flag" -v
pytest tests/test_operator_surface.py -k "planner_closes_are_recommendation_only_by_default" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md config/cl-revenue-ops.conf.minimal config/cl-revenue-ops.conf.full modules/capacity_planner.py tests/test_operator_surface.py tests/test_capacity_planner.py
git commit -m "docs: clarify planner close defaults and status"
```

---

## Final Verification

### Task 7: Run Full Planner Regression Suite

**Files:**
- Test: `tests/test_capacity_planner.py`
- Test: `tests/test_operator_surface.py`

**Step 1: Run the focused planner suites**

Run:

```bash
pytest tests/test_capacity_planner.py tests/test_operator_surface.py -q
```

Expected: PASS.

**Step 2: Run the plugin tests most likely to catch config/init regressions**

Run:

```bash
pytest tests/test_boltz_integration.py tests/test_operator_surface.py tests/test_capacity_planner.py -q
```

Expected: PASS.

**Step 3: Spot-check the new operator-facing behavior**

Run these manually against a dev node with the plugin loaded:

```bash
lightning-cli revenue-planner-status
lightning-cli revenue-planner-execute
```

Expected:
- `revenue-planner-status` reports `execute_closes: false` unless explicitly enabled.
- A close candidate is logged/recorded as `recommended`, not executed, when close execution is disabled.
- Open execution uses generic JSON-RPC and no longer fails with `unexpected keyword argument 'id'`.

**Step 4: Commit any final test-only cleanup**

```bash
git add modules/capacity_planner.py modules/config.py cl-revenue-ops.py README.md config/cl-revenue-ops.conf.minimal config/cl-revenue-ops.conf.full tests/test_capacity_planner.py tests/test_operator_surface.py
git commit -m "test: verify planner safety and rpc hardening"
```

## Notes For Implementation

- Prefer generic `self.plugin.rpc.call(...)` over convenience wrapper methods anywhere the wrapper signature is not already proven by tests.
- Keep close recommendations in `planner_actions` with a distinct `recommended` status; the existing schema already supports arbitrary status strings.
- Do not bundle heuristic changes into this pass. Keep the implementation tightly scoped to safety, config, and execution correctness.
- If a deployed environment has an installed `pyln.client`, capture `inspect.signature(LightningRpc.fundchannel)` and `inspect.signature(LightningRpc.close)` once in the implementation notes, but do not depend on those convenience methods for planner mutations after this hardening pass.
