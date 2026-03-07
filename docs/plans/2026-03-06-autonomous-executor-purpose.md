# Autonomous Executor Surface Reduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `cl_revenue_ops` to an autonomous profit executor with a four-control public operator surface: `paused`, `daily_budget_sats`, `min_fee_ppm`, and `max_fee_ppm`.

**Architecture:** Keep the existing fee and rebalance engines initially, but wrap them in a smaller product boundary. First codify which config is public versus internal, then narrow the RPC surface, add operator-facing explainability, and route hive inputs through a single coordination-input adapter instead of mode-specific behavior.

**Tech Stack:** Python 3, Core Lightning plugin RPC methods in `cl-revenue-ops.py`, dataclass config in `modules/config.py`, SQLite-backed runtime overrides, pytest.

---

### Task 1: Codify The Public Safety Controls

**Files:**
- Create: `tests/test_operator_surface.py`
- Modify: `modules/config.py`

**Step 1: Write the failing test**

```python
from modules.config import Config


def test_public_runtime_keys_are_safety_only():
    cfg = Config()

    assert cfg.public_runtime_keys() == [
        "paused",
        "daily_budget_sats",
        "min_fee_ppm",
        "max_fee_ppm",
    ]


def test_internal_knobs_are_not_public():
    cfg = Config()

    assert "enable_vegas_reflex" not in cfg.public_runtime_keys()
    assert "thompson_prior_std_fee" not in cfg.public_runtime_keys()
    assert "sling_target_sink" not in cfg.public_runtime_keys()
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_operator_surface.py -k public_runtime_keys`

Expected: FAIL with `AttributeError` or assertion failure because `Config` does not expose a public safety-only key registry.

**Step 3: Write minimal implementation**

Add a `paused: bool = False` field to `Config` and `ConfigSnapshot`, plus explicit metadata in `modules/config.py`:

```python
PUBLIC_RUNTIME_KEYS = (
    "paused",
    "daily_budget_sats",
    "min_fee_ppm",
    "max_fee_ppm",
)


def public_runtime_keys(self) -> list[str]:
    return list(PUBLIC_RUNTIME_KEYS)
```

Also add helper methods for later tasks:

- `Config.is_public_runtime_key(key: str) -> bool`
- `Config.public_runtime_dict() -> dict`
- `Config.classify_runtime_key(key: str) -> str` returning `public`, `deprecated`, or `internal`

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_operator_surface.py -k public_runtime_keys`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_operator_surface.py modules/config.py
git commit -m "refactor: classify public safety controls"
```

### Task 2: Narrow `revenue-config` To The Public Surface

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/config.py`
- Test: `tests/test_operator_surface.py`

**Step 1: Write the failing test**

```python
def test_revenue_config_list_mutable_returns_public_controls_only(plugin):
    result = revenue_config(plugin, "list-mutable")

    assert result["mutable_keys"] == [
        "daily_budget_sats",
        "max_fee_ppm",
        "min_fee_ppm",
        "paused",
    ]


def test_revenue_config_rejects_internal_knob_updates(plugin):
    result = revenue_config(plugin, "set", "enable_vegas_reflex", "false")

    assert result["error"].startswith("Key 'enable_vegas_reflex' is not a public runtime control")
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_operator_surface.py -k "list_mutable or internal_knob_updates"`

Expected: FAIL because `revenue-config` still exposes the full mutable key set and still accepts internal knob updates.

**Step 3: Write minimal implementation**

In `cl-revenue-ops.py`, change `revenue-config` to:

- `get` with no key returns public controls by default
- `list-mutable` returns only `Config.public_runtime_keys()`
- `set` and `reset` reject non-public keys
- `get <key>` for internal keys returns a deprecation/internal warning instead of silently acting like it is operator-supported

Keep legacy internals available only through a clearly marked debug/admin path if needed later, but do not expose them as normal operator controls.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_operator_surface.py -k "list_mutable or internal_knob_updates"`

Expected: PASS

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/config.py tests/test_operator_surface.py
git commit -m "refactor: restrict revenue-config to safety controls"
```

### Task 3: Add Operator-Facing Status Instead Of Knob Dumps

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/config.py`
- Test: `tests/test_explainability.py`
- Test: `tests/test_operator_surface.py`

**Step 1: Write the failing test**

```python
def test_revenue_status_reports_operator_controls_not_full_config(plugin):
    result = revenue_status(plugin)

    assert "operator_controls" in result
    assert result["operator_controls"]["public_keys"] == [
        "paused",
        "daily_budget_sats",
        "min_fee_ppm",
        "max_fee_ppm",
    ]
    assert "enable_vegas_reflex" not in result["operator_controls"]["values"]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_explainability.py tests/test_operator_surface.py -k operator_controls`

Expected: FAIL because `revenue-status` currently reports broad config state rather than a safety-only operator view.

**Step 3: Write minimal implementation**

Add a dedicated operator-status block in `revenue-status`:

```python
result["operator_controls"] = {
    "public_keys": config.public_runtime_keys(),
    "values": config.public_runtime_dict(),
}
```

Do not dump internal algorithm knobs in the top-level operator section. Keep detailed config available only in a debug/admin-oriented path if still required for development.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_explainability.py tests/test_operator_surface.py -k operator_controls`

Expected: PASS

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/config.py tests/test_explainability.py tests/test_operator_surface.py
git commit -m "feat: expose operator-safe control status"
```

### Task 4: Replace Tuning Workflows With Decision Explainability

**Files:**
- Modify: `modules/fee_controller.py`
- Modify: `modules/rebalancer.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_explainability.py`
- Test: `tests/test_fee_controller.py`
- Test: `tests/test_rebalancer_module.py`

**Step 1: Write the failing test**

```python
def test_status_exposes_last_fee_decision_reason():
    result = revenue_status(plugin)

    assert result["fee_decision"]["action"] in {"hold", "raise", "lower", "suppressed"}
    assert "reason" in result["fee_decision"]
    assert "safety_block" in result["fee_decision"]


def test_status_exposes_last_rebalance_decision_reason():
    result = revenue_status(plugin)

    assert result["rebalance_decision"]["action"] in {"hold", "rebalance", "suppressed"}
    assert "reason" in result["rebalance_decision"]
    assert "budget_blocked" in result["rebalance_decision"]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_explainability.py tests/test_fee_controller.py tests/test_rebalancer_module.py -k "last_fee_decision_reason or last_rebalance_decision_reason"`

Expected: FAIL because decision summaries are not surfaced as first-class status output.

**Step 3: Write minimal implementation**

Persist lightweight last-decision summaries from fee and rebalance cycles:

- action
- reason
- dominant input or blocker
- whether a safety rail suppressed the action

Expose these summaries through `revenue-status` so operators inspect outcomes instead of changing knobs.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_explainability.py tests/test_fee_controller.py tests/test_rebalancer_module.py -k "last_fee_decision_reason or last_rebalance_decision_reason"`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py modules/rebalancer.py cl-revenue-ops.py tests/test_explainability.py tests/test_fee_controller.py tests/test_rebalancer_module.py
git commit -m "feat: surface operator-facing decision explainability"
```

### Task 5: Deprecate Tactical Operator Policy Controls

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/policy_manager.py`
- Test: `tests/test_policy_manager.py`
- Test: `tests/test_operator_surface.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_revenue_policy_set_is_rejected_for_normal_operator_use(plugin):
    result = revenue_policy(plugin, "set", "02" + "a" * 64, strategy="static", fee_ppm=500)

    assert result["error"].startswith("revenue-policy set is deprecated for normal operator use")
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_policy_manager.py tests/test_operator_surface.py -k deprecated_for_normal_operator_use`

Expected: FAIL because `revenue-policy set` is still treated as a normal operator tool.

**Step 3: Write minimal implementation**

Keep policy machinery available for internal use and compatibility, but change the public RPC contract:

- operator-facing docs mark it deprecated
- `revenue-policy set/delete/tag/untag/batch` reject normal operator use
- read-only inspection paths can remain for transition diagnostics

If the project wants an escape hatch later, add a clearly named debug/admin-only override rather than leaving tactical control on the main operator path.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_policy_manager.py tests/test_operator_surface.py -k deprecated_for_normal_operator_use`

Expected: PASS

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/policy_manager.py tests/test_policy_manager.py tests/test_operator_surface.py README.md
git commit -m "refactor: deprecate tactical operator policy controls"
```

### Task 6: Treat Hive As Coordination Input, Not Product Mode

**Files:**
- Modify: `modules/hive_bridge.py`
- Modify: `modules/fee_controller.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_hive_integration.py`
- Test: `tests/test_fee_controller.py`
- Test: `tests/test_rebalancer_module.py`

**Step 1: Write the failing test**

```python
def test_fee_controller_uses_empty_coordination_inputs_when_hive_unavailable():
    inputs = fee_controller._get_coordination_inputs("123x456x0", peer_id="02" + "a" * 64)

    assert inputs.mode == "local_only"
    assert inputs.priors == {}


def test_fee_controller_uses_coordination_priors_when_hive_available():
    inputs = fee_controller._get_coordination_inputs("123x456x0", peer_id="02" + "a" * 64)

    assert inputs.mode == "fleet_augmented"
    assert "peer_quality" in inputs.priors
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_hive_integration.py tests/test_fee_controller.py tests/test_rebalancer_module.py -k coordination_inputs`

Expected: FAIL because hive integration is still spread across direct mode checks and feature-specific branches.

**Step 3: Write minimal implementation**

Add a small adapter object or helper that gathers optional coordination inputs:

```python
CoordinationInputs(
    mode="local_only" or "fleet_augmented",
    priors={...},
    corridor_hint=...,
    peer_quality=...,
)
```

Use the same decision pipeline in both cases. Hive should only improve priors and coordination quality, not change the product boundary.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_hive_integration.py tests/test_fee_controller.py tests/test_rebalancer_module.py -k coordination_inputs`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/hive_bridge.py modules/fee_controller.py modules/rebalancer.py tests/test_hive_integration.py tests/test_fee_controller.py tests/test_rebalancer_module.py
git commit -m "refactor: route hive signals through coordination inputs"
```

### Task 7: Finish The Operator-Facing Documentation And Migration Notes

**Files:**
- Modify: `README.md`
- Create: `docs/plans/2026-03-06-autonomous-executor-purpose-migration.md`
- Test: `tests/test_operator_surface.py`

**Step 1: Write the failing test**

```python
def test_readme_examples_no_longer_advertise_internal_knob_tuning():
    readme = Path("README.md").read_text()

    assert "revenue-config set enable_vegas_reflex false" not in readme
    assert "revenue-policy set" not in readme
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_operator_surface.py -k readme_examples`

Expected: FAIL because the README still advertises the old operator surface.

**Step 3: Write minimal implementation**

Update docs to reflect:

- the new product purpose
- the four supported operator controls
- decision explainability workflow
- deprecated legacy controls and migration guidance

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_operator_surface.py -k readme_examples`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-03-06-autonomous-executor-purpose-migration.md tests/test_operator_surface.py
git commit -m "docs: document autonomous executor operator surface"
```

### Final Verification

Run the targeted suite after all tasks:

```bash
pytest -q \
  tests/test_operator_surface.py \
  tests/test_explainability.py \
  tests/test_fee_controller.py \
  tests/test_rebalancer_module.py \
  tests/test_hive_integration.py \
  tests/test_policy_manager.py
```

Expected: PASS

Run the broader safety net before merge or PR:

```bash
pytest -q tests/test_explainability.py tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py
```

Expected: PASS
