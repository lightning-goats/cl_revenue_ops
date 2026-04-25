# Hive Route Policy And Askrene Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the active rebalancer so it correctly uses `cl-hive` askrene layers for hive rebalancing, and add explicit `hive_only` / `hybrid` / `market_only` categorization plus priority handling driven by live state and fresh hive hints.

**Architecture:** Keep the active `RebalanceEngine` as the orchestration entry point, but introduce a first-class route-policy decision object before pricing. Route-policy classification should consume fresh hive hints and endpoint membership, then dispatch to either a strict hive router, a hybrid comparison flow, or the existing market router. Reuse the proven fleet-route logic from the legacy executor instead of leaving it stranded in `modules/rebalance_executor.py`.

**Tech Stack:** Python 3, Core Lightning RPC (`getroutes`, askrene layer RPCs, `listpeerchannels`, `listchannels`), `cl-hive` hint export RPC/datastore contract, `pytest`

---

### Task 1: Freeze The Broken Behavior With Failing Tests

**Files:**
- Create: `tests/test_rebalance_route_policy.py`
- Modify: `tests/test_rebalance_router_v3.py`
- Modify: `tests/test_rebalance_engine_v2.py`
- Reference: `modules/rebalance_router_v3.py`
- Reference: `modules/rebalance_engine_v2.py`

**Step 1: Write the failing route-policy classification tests**

```python
def test_decide_route_policy_requires_hive_for_hive_equalization():
    pair = PairCandidate(
        source_channel_id="1x1x1",
        dest_channel_id="2x2x2",
        source_peer_id="02aa",
        dest_peer_id="02bb",
        amount_sats=100_000,
        pair_budget_sats=10,
    )
    hints = FakeHiveHints(members={"02aa", "02bb"})

    decision = decide_route_policy(
        pair,
        reason_code="hive_equalization",
        hive_hints=hints,
    )

    assert decision.policy == RoutePolicy.HIVE_ONLY
    assert decision.allow_market_fallback is False
```

```python
def test_decide_route_policy_marks_hybrid_when_hints_prefer_fleet_but_do_not_require_it():
    hints = FakeHiveHints(
        members={"02aa"},
        recommendations=[{"source_peer_id": "02aa", "destination_peer_id": "03cc"}],
    )
    decision = decide_route_policy(pair, reason_code="ev_positive", hive_hints=hints)
    assert decision.policy == RoutePolicy.HYBRID
```

**Step 2: Write the failing router/engine integration tests**

```python
def test_router_v3_reprobes_layers_after_startup():
    router = RebalanceRouterV3(..., layer_names=["hive-fleet", "hive-reputation"])
    assert router.found_layers == []

    fake_rpc.layers = ["hive-fleet", "hive-reputation"]
    router.price_pair(...)

    assert fake_rpc.last_getroutes_kwargs["layers"][:2] == ["hive-fleet", "hive-reputation"]
```

```python
def test_engine_uses_hive_router_for_hive_only_pairs():
    engine = RebalanceEngine(...)
    engine._route_policy_service = FakePolicyService(policy=RoutePolicy.HIVE_ONLY)
    engine._hive_router = FakeHiveRouteRouter(success=True, route_cost_sats=0)

    candidates = engine.find_candidates()

    assert engine._hive_router.price_calls == 1
    assert engine.router_v3.price_calls == 0
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_rebalance_route_policy.py tests/test_rebalance_router_v3.py tests/test_rebalance_engine_v2.py -q`

Expected: FAIL because `RoutePolicy`, `decide_route_policy`, live re-probe behavior, and hive-router dispatch do not exist in the active engine.

**Step 4: Commit the red-test checkpoint**

```bash
git add tests/test_rebalance_route_policy.py tests/test_rebalance_router_v3.py tests/test_rebalance_engine_v2.py
git commit -m "test: lock missing hive route policy behavior"
```

### Task 2: Introduce Route Policy And Priority As First-Class Data

**Files:**
- Modify: `modules/rebalance_types_v2.py`
- Create: `modules/rebalance_route_policy.py`
- Test: `tests/test_rebalance_route_policy.py`

**Step 1: Write the failing data-model test**

```python
def test_pair_candidate_can_store_route_decision():
    decision = RouteDecision(
        policy=RoutePolicy.HIVE_ONLY,
        priority=RoutePriority.COORDINATED,
        reason="hive_equalization",
        allow_market_fallback=False,
    )
    pair = PairCandidate(..., route_decision=decision)
    assert pair.route_decision.policy is RoutePolicy.HIVE_ONLY
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_route_policy.py::test_pair_candidate_can_store_route_decision -q`

Expected: FAIL because the new enums/dataclasses are missing.

**Step 3: Write minimal implementation**

```python
class RoutePolicy(Enum):
    HIVE_ONLY = "hive_only"
    HYBRID = "hybrid"
    MARKET_ONLY = "market_only"


class RoutePriority(Enum):
    COORDINATED = "coordinated"
    HIVE_EQUALIZATION = "hive_equalization"
    EV_POSITIVE = "ev_positive"
    BACKGROUND = "background"


@dataclass(frozen=True)
class RouteDecision:
    policy: RoutePolicy
    priority: RoutePriority
    reason: str
    allow_market_fallback: bool = True
    hint_id: str = ""
    hint_type: str = ""
    prefer_hive_on_tie: bool = True
```

```python
@dataclass
class PairCandidate:
    ...
    route_decision: Optional[RouteDecision] = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rebalance_route_policy.py::test_pair_candidate_can_store_route_decision -q`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_types_v2.py modules/rebalance_route_policy.py tests/test_rebalance_route_policy.py
git commit -m "feat: add route policy decision model"
```

### Task 3: Implement Hint-Driven Categorization And Priority

**Files:**
- Modify: `modules/rebalance_route_policy.py`
- Modify: `modules/rebalance_engine_v2.py`
- Reference: `modules/hive_hints.py`
- Test: `tests/test_rebalance_route_policy.py`
- Test: `tests/test_rebalance_engine_v2.py`

**Step 1: Write the failing classification matrix tests**

```python
def test_hive_only_when_both_endpoints_are_hive_and_reason_is_equalization(): ...
def test_hybrid_when_one_endpoint_is_hive_and_fresh_hint_recommends_fleet_help(): ...
def test_market_only_when_hints_are_missing_or_stale(): ...
def test_coordination_hint_raises_priority_above_plain_ev_candidate(): ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rebalance_route_policy.py tests/test_rebalance_engine_v2.py -q -k "route_policy or priority"`

Expected: FAIL because the engine does not attach route decisions to pairs.

**Step 3: Implement the categorizer**

Use this rule order in `decide_route_policy(...)`:

```python
if reason_code == "hive_equalization" and src_is_hive and dst_is_hive:
    return RouteDecision(RoutePolicy.HIVE_ONLY, RoutePriority.HIVE_EQUALIZATION, ...)

if fresh_campaign_or_recommendation_requires_fleet:
    return RouteDecision(RoutePolicy.HIVE_ONLY, RoutePriority.COORDINATED, ...)

if fresh_campaign_or_recommendation_prefers_fleet:
    return RouteDecision(RoutePolicy.HYBRID, RoutePriority.COORDINATED, ...)

if src_is_hive or dst_is_hive:
    return RouteDecision(RoutePolicy.HYBRID, RoutePriority.EV_POSITIVE, ...)

return RouteDecision(RoutePolicy.MARKET_ONLY, RoutePriority.EV_POSITIVE, ...)
```

Use hive hints only when the snapshot is fresh. When hints are stale, degrade toward `HYBRID` or `MARKET_ONLY`; do not promote to `HIVE_ONLY` from stale data.

**Step 4: Attach route decisions before pricing**

In `RebalanceEngine.find_candidates()`:

```python
for pair in plan.selected:
    pair.route_decision = decide_route_policy(
        pair,
        reason_code=self._infer_reason_code(pair),
        hive_hints=self._hive_hints,
    )
```

Sort candidates by:

```python
priority_order = {
    RoutePriority.COORDINATED: 0,
    RoutePriority.HIVE_EQUALIZATION: 1,
    RoutePriority.EV_POSITIVE: 2,
    RoutePriority.BACKGROUND: 3,
}
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rebalance_route_policy.py tests/test_rebalance_engine_v2.py -q -k "route_policy or priority"`

Expected: PASS

**Step 6: Commit**

```bash
git add modules/rebalance_route_policy.py modules/rebalance_engine_v2.py tests/test_rebalance_route_policy.py tests/test_rebalance_engine_v2.py
git commit -m "feat: classify rebalance pairs by hive route policy"
```

### Task 4: Extract The Shared Hive Route Builder From The Legacy Executor

**Files:**
- Create: `modules/rebalance_hive_router.py`
- Modify: `modules/rebalance_executor.py`
- Modify: `modules/rebalance_engine_v2.py`
- Reference: `modules/hive_router.py`
- Test: `tests/test_rebalance_executor.py`
- Test: `tests/test_rebalance_engine_v2.py`

**Step 1: Write the failing shared-router tests**

```python
def test_hive_route_router_uses_all_live_hive_and_revenue_layers():
    router = RebalanceHiveRouter(...)
    fake_rpc.layers = ["hive-fleet", "hive-reputation", "hive-corridors", "hive-traffic", "revenue-local"]

    router.price_pair(pair, decision=RouteDecision(policy=RoutePolicy.HIVE_ONLY, ...))

    assert fake_rpc.last_getroutes_kwargs["layers"] == [
        "auto.localchans",
        "hive-fleet",
        "hive-reputation",
        "hive-corridors",
        "hive-traffic",
        "revenue-local",
        "auto.no_mpp_support",
    ]
```

```python
def test_hive_only_route_rejects_non_hive_intermediate():
    result = router.price_pair(...)
    assert result.success is False
    assert "non_hive_intermediate" in result.error
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rebalance_executor.py tests/test_rebalance_engine_v2.py -q -k "hive_route_router or non_hive_intermediate"`

Expected: FAIL because the shared hive router does not exist.

**Step 3: Implement the shared router**

Extract the proven fleet-route logic from:
- `modules/rebalance_executor.py:_get_layers`
- `modules/rebalance_executor.py:_compute_fleet_route`

Minimal API:

```python
class RebalanceHiveRouter:
    def price_pair(
        self,
        pair: PairCandidate,
        decision: RouteDecision,
        exclude: Optional[list[str]] = None,
    ) -> RouteResult:
        ...
```

Rules:
- Always build layer list live from `askrene-listlayers`
- Include `auto.localchans`
- Include every live `hive-*` and `revenue-*` layer
- Never include `auto.sourcefree`
- For `HIVE_ONLY`, verify every intermediate node is a hive member
- Preserve strict no-market-fallback semantics for `HIVE_ONLY`

**Step 4: Point the legacy executor at the shared router**

Replace duplicated fleet-route logic in `modules/rebalance_executor.py` with calls into `RebalanceHiveRouter`.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rebalance_executor.py tests/test_rebalance_engine_v2.py -q -k "hive_route_router or non_hive_intermediate"`

Expected: PASS

**Step 6: Commit**

```bash
git add modules/rebalance_hive_router.py modules/rebalance_executor.py modules/rebalance_engine_v2.py tests/test_rebalance_executor.py tests/test_rebalance_engine_v2.py
git commit -m "refactor: share hive-only route builder across engines"
```

### Task 5: Make The Active Engine Policy-Aware

**Files:**
- Modify: `modules/rebalance_engine_v2.py`
- Modify: `modules/rebalance_router_v3.py`
- Test: `tests/test_rebalance_engine_v2.py`
- Test: `tests/test_rebalance_router_v3.py`

**Step 1: Write the failing dispatch tests**

```python
def test_market_only_pair_uses_market_router(): ...
def test_hive_only_pair_uses_hive_router_only(): ...
def test_hybrid_pair_compares_hive_and_market_routes_and_picks_executable_best(): ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rebalance_engine_v2.py tests/test_rebalance_router_v3.py -q -k "market_only or hive_only or hybrid"`

Expected: FAIL because `find_candidates()` always prices through one generic router.

**Step 3: Implement route-policy dispatch**

In `RebalanceEngine.find_candidates()`:

```python
if pair.route_decision.policy is RoutePolicy.HIVE_ONLY:
    route_result = self._hive_router.price_pair(pair, pair.route_decision, exclude=...)
elif pair.route_decision.policy is RoutePolicy.HYBRID:
    hive_result = self._hive_router.price_pair(pair, pair.route_decision, exclude=...)
    market_result = self._market_router.price_pair(..., exclude=...)
    route_result = choose_best_route(hive_result, market_result, prefer_hive_on_tie=True)
else:
    route_result = self._market_router.price_pair(..., exclude=...)
```

Selection rule:
- discard failed routes
- require cost within effective budget
- if both succeed, pick lower cost
- if equal cost, prefer the hive route

**Step 4: Fix market-router layer refresh**

In `modules/rebalance_router_v3.py`, stop relying on init-only `self.found_layers`.

Minimal change:

```python
def _current_layers(self) -> list[str]:
    found = self._probe_layers()
    if "auto.no_mpp_support" not in found:
        found.append("auto.no_mpp_support")
    return found
```

Use `_current_layers()` inside `price_pair()`.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rebalance_engine_v2.py tests/test_rebalance_router_v3.py -q -k "market_only or hive_only or hybrid or reprobe"`

Expected: PASS

**Step 6: Commit**

```bash
git add modules/rebalance_engine_v2.py modules/rebalance_router_v3.py tests/test_rebalance_engine_v2.py tests/test_rebalance_router_v3.py
git commit -m "feat: route active engine by hive policy"
```

### Task 6: Extend Hive Hint Consumption For Explicit Priority Metadata

**Files:**
- Modify: `/home/sat/bin/cl-hive/modules/rpc_commands.py`
- Modify: `modules/hive_hints.py`
- Modify: `modules/rebalance_route_policy.py`
- Test: `/home/sat/bin/cl-hive/tests/test_export_hints.py`
- Test: `tests/test_hive_hints.py`
- Test: `tests/test_rebalance_route_policy.py`

**Step 1: Write the failing contract tests**

```python
def test_export_hints_emits_optional_route_policy_and_priority_fields(): ...
def test_hive_hint_adapter_accepts_route_policy_metadata_when_present(): ...
def test_route_policy_prefers_explicit_hint_metadata_over_local_guessing(): ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest /home/sat/bin/cl-hive/tests/test_export_hints.py tests/test_hive_hints.py tests/test_rebalance_route_policy.py -q`

Expected: FAIL because those fields are not yet exported or normalized.

**Step 3: Implement backward-compatible hint metadata**

Add optional fields to `rebalance_recommendations` / `rebalance_campaigns` when `cl-hive` knows them:

```python
item["route_policy"] = "hive_only"
item["priority"] = "coordinated"
item["allow_market_fallback"] = False
```

Consumer rule:
- accept these fields when present and valid
- derive locally when absent
- never break older snapshots

**Step 4: Run tests to verify they pass**

Run: `pytest /home/sat/bin/cl-hive/tests/test_export_hints.py tests/test_hive_hints.py tests/test_rebalance_route_policy.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add modules/rpc_commands.py tests/test_export_hints.py
git -C /home/sat/bin/cl-hive commit -m "feat: export explicit hive route policy hints"

git add modules/hive_hints.py modules/rebalance_route_policy.py tests/test_hive_hints.py tests/test_rebalance_route_policy.py
git commit -m "feat: consume explicit hive route policy hints"
```

### Task 7: Audit Logging, Docs, And End-To-End Verification

**Files:**
- Modify: `modules/rebalance_audit_v2.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `/home/sat/bin/cl-hive/README.md`
- Modify: `/home/sat/bin/cl-hive/CLAUDE.md`
- Test: `tests/test_rebalance_engine_v2.py`

**Step 1: Write the failing audit/docs tests**

```python
def test_audit_logs_route_policy_and_priority_for_selected_pair(): ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_engine_v2.py -q -k audit`

Expected: FAIL because route-policy metadata is not surfaced in audit output.

**Step 3: Implement audit surfacing and docs**

Audit fields to log:

```python
detail=(
    f"policy={pair.route_decision.policy.value} "
    f"priority={pair.route_decision.priority.value} "
    f"hint_type={pair.route_decision.hint_type}"
)
```

Update docs to describe:
- route-policy categories
- hint freshness requirements
- strict `hive_only` vs `hybrid` fallback behavior
- live layer refresh behavior

**Step 4: Run the focused verification suites**

Run: `pytest tests/test_rebalance_route_policy.py tests/test_rebalance_engine_v2.py tests/test_rebalance_router_v3.py tests/test_rebalance_executor.py tests/test_hive_hints.py -q`

Expected: PASS

Run: `pytest /home/sat/bin/cl-hive/tests/test_export_hints.py -q`

Expected: PASS

**Step 5: Run broader rebalance regression coverage**

Run: `pytest tests/test_rebalance_*.py tests/test_rebalancer_module.py tests/test_hive_live_contract.py tests/test_fee_hive_bias.py -q`

Expected: PASS

**Step 6: Commit**

```bash
git add modules/rebalance_audit_v2.py README.md CLAUDE.md tests/test_rebalance_engine_v2.py
git commit -m "docs: record hive route policy behavior"

git -C /home/sat/bin/cl-hive add README.md CLAUDE.md
git -C /home/sat/bin/cl-hive commit -m "docs: describe exported hive route policy hints"
```

### Notes And Guardrails

- Do not let stale hints force `HIVE_ONLY`.
- Do not regress the current `auto.no_mpp_support` behavior.
- Do not reintroduce `auto.sourcefree` for circular self-payments.
- Keep `HIVE_ONLY` routing strict: if no all-hive executable route exists, fail that candidate rather than silently turning it into a market rebalance.
- For `HYBRID`, prefer the hive route on ties, but do not pay a higher fee unless a later design explicitly adds a configurable premium.
- Reuse the legacy fleet-route code path; do not maintain two independent copies of hive-route construction logic.
