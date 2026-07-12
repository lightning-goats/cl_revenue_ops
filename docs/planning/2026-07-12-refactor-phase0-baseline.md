# Refactor Phase 0 — Baseline & Behavioral Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the complete Phase 0 baseline required by `docs/planning/refactor.md` — mutation-path inventory, decision-owner matrix, persistence map, compatibility catalog, golden characterization tests for every principal decision class, portability-hazard inventory, draft wire-contract spec, proposed `EconomicSnapshot` schema, conformance-fixture layout, and a PR sequence — WITHOUT changing any production behavior.

**Architecture:** Phase 0 is characterization, not refactoring. Deliverables are (a) inventory documents under `docs/refactor/phase0/`, each backed where possible by a **pin test** that fails when the inventory rots; (b) golden characterization tests under `tests/golden/` that freeze current decision outputs for fixed inputs; (c) draft language-neutral contracts under `schemas/` and `tests/conformance/` validated by a standalone tool that imports no plugin code. Much of the inventory content already exists in `docs/audit/` (June–July 2026 audit campaign) — those docs are pre-hive-removal (v2.17.0, 2026-07-10) and this plan re-pins their facts to current HEAD rather than redoing the research.

**Tech Stack:** Python 3.12, pytest, unittest.mock, JSON Schema (draft 2020-12) via `jsonschema` 4.10.3 (dev/test only), sqlite3. No new runtime dependencies.

## Global Constraints

- **Phase 0 only** (`docs/planning/refactor.md` line 1053): no production behavior changes; inventory, tests, schemas, and docs only. Do not start Phase 1 structures.
- "Return the inventory and proposed schemas for review before implementing Phase 1" (`refactor.md` line 1068) — the final task assembles the review packet; Phase 1 requires Sat's sign-off.
- "If the repository contradicts an assumption in this plan, document the contradiction and recommend the smallest correction; do not silently force the repository into the proposed shape" (`refactor.md` line 1068). Contradictions go in `docs/refactor/phase0/README.md` §Contradictions.
- **Never call action RPCs in tests** (AGENTS.md): `revenue-rebalance-cycle`, `revenue-fee-cycle`, `revenue-planner-execute`, `revenue-set-fee`, `revenue-rebalance`, `revenue-spend-*`, `revenue-analyze`, `revenue-wake-all`, `revenue-ignore/unignore`, `revenue-cleanup-closed`, `revenue-clear-reservations`, `revenue-policy set`, `revenue-config set`, `revenue-lnplus-*` mutations, any Boltz action RPC, any CLN open/close/pay/withdraw RPC. All golden tests use mocks; nothing touches a live node.
- **Never update a golden fixture merely to make a test pass** (`refactor.md` line 868). Intentional behavior changes require a dedicated test and rationale.
- All money amounts msat-native internally; conversions only at reporting boundaries (`refactor.md` invariant 4).
- No new entries in `requirements.txt` (exact-pinned production runtime; see its deploy gate header). `jsonschema` is used by tests via `pytest.importorskip("jsonschema")` and by the standalone validator tool with a clear error message if missing.
- Branch: `worktree-refactor` in the worktree at `/home/sat/bin/cl_revenue_ops/.claude/worktrees/refactor`. Commit after every task (small, reviewable commits).
- Test command for the full suite: `python3 -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider`. Baseline at commit `5e8f747`: **3114 passed, 1 skipped, 42.30s** (Python 3.12.3, 2026-07-12).
- Existing test conventions: `tests/conftest.py` mocks `pyln.client` at import time and autouse-injects `PermissivePolicyManager` into bare `CapacityPlanner`/`RebalanceEngine` instances. New tests follow the same patterns.

## Spec deliverable → task map (refactor.md "First task for the coding agent", lines 1055–1067)

| # | Spec deliverable | Task(s) |
|---|---|---|
| 1 | Map of every mutating CLN RPC / external write path | Task 2 |
| 2 | Decision-owner matrix | Task 3 |
| 3 | Persistence map (tables, datastore keys, restart recovery) | Task 4 |
| 4 | Public RPC/config/datastore compatibility catalog | Task 5 |
| 5 | Proposed canonical `EconomicSnapshot` schema mapped to sources | Task 16 |
| 6 | Golden test fixtures for production decision classes | Tasks 6–13 |
| 7 | Python-specific portability-hazard inventory | Task 14 |
| 8 | Draft wire-contract specification | Task 15 |
| 9 | Portable conformance-fixture layout | Task 17 |
| 10 | Sequence of small implementation PRs | Task 18 |
| — | Baseline test duration, migrations, RPC schemas (`refactor.md` line 770) | Task 1 |

---

### Task 1: Phase 0 scaffolding and baseline record

**Files:**
- Create: `docs/refactor/phase0/README.md`
- Create: `docs/refactor/phase0/baseline.md`

**Interfaces:**
- Produces: `docs/refactor/phase0/` directory that Tasks 2–5, 14–18 write into; README index that Task 18 completes.

- [ ] **Step 1: Create the directory and README index**

Write `docs/refactor/phase0/README.md`:

```markdown
# Refactor Phase 0 — Baseline & Behavioral Characterization

Deliverables for Phase 0 of `docs/planning/refactor.md`. Nothing in this
directory changes production behavior; it documents what exists at the
baseline commit and pins it with tests.

| Deliverable | File | Pin test |
|---|---|---|
| Baseline record | `baseline.md` | — |
| Mutation-path inventory | `mutation-paths.md` | `tests/test_mutation_path_inventory.py` |
| Decision-owner matrix | `decision-owners.md` | — |
| Persistence map | `persistence-map.md` | `tests/test_persistence_inventory.py` |
| Compatibility catalog | `compatibility-catalog.md` | `tests/test_rpc_surface_inventory.py` |
| Golden decision tests | — | `tests/golden/` |
| Portability hazards | `portability-hazards.md` | — |
| Wire-contract draft | `wire-contract-spec.md` | — |
| EconomicSnapshot schema | `../..//schemas/economic_snapshot.v0.schema.json` + `snapshot-mapping.md` | `tests/test_schema_validity.py` |
| Conformance corpus layout | `../../tests/conformance/README.md` | `tests/test_conformance_validator.py` |
| PR sequence | `pr-sequence.md` | — |

## Contradictions

(Filled in by later tasks: places where the repository contradicts an
assumption in `docs/planning/refactor.md`, with the smallest recommended
correction. Per the spec, contradictions are documented, never silently
forced.)

## Prior-art reuse

The 2026-06/07 audit campaign already produced most of the raw research.
Those docs predate the hive-removal (v2.17.0, 2026-07-10) — line numbers
and module lists drift. Phase 0 docs cite them and re-pin the facts to the
baseline commit rather than duplicating them:

- `docs/audit/deep/` — prod baseline T0, resource growth/retention,
  concurrency map, perf baseline, deferred ledger (94 findings), SBOM
- `docs/audit/contracts/` + `docs/audit/verification/` — 30 per-module
  intent contracts and verification reports
- `docs/audit/decision-loops.md` + `docs/audit/decision-loops/` — 7
  decision loops with verdicts
- `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md` — RPC classification
- `docs/contracts/` — the 3 public datastore telemetry contracts (current)
```

- [ ] **Step 2: Write the baseline record**

Write `docs/refactor/phase0/baseline.md`:

```markdown
# Phase 0 baseline record

Captured 2026-07-12 in worktree branch `worktree-refactor`.

- Baseline commit: `5e8f747` ("fix(planner): close-protection gate judges
  the 30d window, not lifetime history")
- Test suite: `python3 -m pytest tests/ -q --ignore=tests/integration
  -p no:cacheprovider` → **3114 passed, 1 skipped, 42.30s**
  (skip: `tests/test_pyln_integration.py` — pyln.testing not installed)
- Python: 3.12.3; runtime deps exact-pinned in `requirements.txt`
  (pyln-client 25.12.1, PyYAML 6.0.1, numpy 1.26.4); hash-pinned closure in
  `requirements.lock`; SBOM in `docs/audit/deep/sbom.cyclonedx.json`
- Plugin entry: `cl-revenue-ops.py` (9,911 lines), 64 registered RPC
  methods (`@plugin.method`), modules/ total ≈ 42,833 lines
- Database: 37 `CREATE TABLE IF NOT EXISTS` tables in
  `modules/database.py`; `schema_version` table is WRITE-ONLY by operator
  ruling DD9/MIG-3 (2026-07-02) — see `modules/database.py:606`
- Migrations: additive `CREATE TABLE/INDEX IF NOT EXISTS` +
  `ALTER TABLE` guards in `Database.__init__`; no migration framework
- Public datastore contracts (documented, tested by
  `tests/test_cross_plugin_contracts.py`):
  `["revenue","profitability-summary"]`, `["revenue","capex-summary"]`,
  `["revenue","segment-observations"]` — see `docs/contracts/`
- Production: single node `lnnode` (hive-nexus-01); prior audit baseline
  `docs/audit/deep/prod-baseline-T0.md` (53 MiB DB) — its "node 2 gap" is
  moot: fleet is single-node since 2026-07-11
- CLN runtime floor: v24.11.1 (`docs/CORE_LIGHTNING_COMPATIBILITY.md`)
```

- [ ] **Step 3: Verify the claims in the doc**

Run: `git log --oneline -1 && grep -c '@plugin.method' cl-revenue-ops.py && grep -c 'CREATE TABLE IF NOT EXISTS' modules/database.py`
Expected: `5e8f747 …`, `64`, `37`. If any number differs, fix the doc to match reality.

- [ ] **Step 4: Commit**

```bash
git add docs/refactor/phase0/
git commit -m "docs(refactor): Phase 0 scaffolding and baseline record"
```

---

### Task 2: Mutation-path inventory and enforcement test

**Files:**
- Create: `docs/refactor/phase0/mutation-paths.md`
- Create: `tests/test_mutation_path_inventory.py`

**Interfaces:**
- Produces: `MUTATING_CALL_SITES` allowlist (file → sorted verb list) that future PRs must consciously extend; the doc is the narrative behind it.

- [ ] **Step 1: Write the failing pin test**

Create `tests/test_mutation_path_inventory.py`:

```python
"""Phase 0 pin: every file that can invoke a mutating CLN RPC is inventoried.

A new mutating call site anywhere else — or a new verb in a known file —
fails this test until docs/refactor/phase0/mutation-paths.md and the
allowlist below are updated together. This is the enforcement half of the
Phase 0 mutation-path inventory (docs/planning/refactor.md, deliverable 1).

Scope: direct CLN RPC invocations only. Wrapper *callers* (e.g. modules
calling data_service.set_channel) are documented in the markdown inventory
but not scanned here — the wrapper itself is the choke point we pin.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# CLN RPC verbs that mutate node/network/external state. Read-only verbs
# (listpeerchannels, askrene-listlayers, getroutes, ...) are excluded.
MUTATING_ATTR_VERBS = (
    "setchannel|sendpay|waitsendpay|fundchannel|connect|invoice|delpay"
    "|delinvoice|signmessage|datastore|pay"
)
MUTATING_CALL_VERBS = (
    MUTATING_ATTR_VERBS
    + "|close|askrene-create-layer|askrene-remove-layer"
    + "|askrene-update-channel|askrene-bias-node|askrene-bias-channel"
    + "|askrene-reserve|askrene-unreserve|askrene-inform-channel"
    + "|askrene-age|askrene-disable-node"
)
PAT_ATTR = re.compile(r"rpc\.(" + MUTATING_ATTR_VERBS + r")\s*\(")
PAT_CALL = re.compile(
    r"""(?:\.call|_rpc_call)\(\s*['"](""" + MUTATING_CALL_VERBS + r""")['"]"""
)

# The complete inventory at baseline commit 5e8f747. Keys are repo-relative
# paths; values are the sorted set of mutating verbs the file may invoke.
MUTATING_CALL_SITES = {
    "modules/boltz_manager.py": ["pay"],
    "modules/capacity_planner.py": ["close", "fundchannel"],
    "modules/data_service.py": [
        "askrene-age", "askrene-bias-channel", "askrene-bias-node",
        "askrene-create-layer", "askrene-disable-node",
        "askrene-inform-channel", "askrene-remove-layer", "askrene-reserve",
        "askrene-unreserve", "askrene-update-channel", "close", "datastore",
        "delinvoice", "delpay", "fundchannel", "invoice", "pay", "sendpay",
        "setchannel", "waitsendpay",
    ],
    "modules/lnplus_swaps.py": ["connect", "fundchannel", "signmessage"],
    "modules/rebalance_engine_v2.py": [
        "askrene-remove-layer", "datastore", "delpay",
    ],
    "modules/rebalance_native_executor_v2.py": [
        "delinvoice", "delpay", "invoice", "sendpay", "waitsendpay",
    ],
    "modules/rebalance_router_v3.py": [
        "askrene-create-layer", "askrene-remove-layer",
        "askrene-update-channel",
    ],
}


def _scan():
    hits = {}
    files = sorted((REPO / "modules").glob("*.py"))
    files.append(REPO / "cl-revenue-ops.py")
    for f in files:
        text = f.read_text()
        verbs = set(PAT_ATTR.findall(text)) | set(PAT_CALL.findall(text))
        if verbs:
            hits[str(f.relative_to(REPO))] = sorted(verbs)
    return hits


def test_mutating_call_sites_match_inventory():
    actual = _scan()
    assert actual == MUTATING_CALL_SITES, (
        "Mutating CLN RPC call sites changed. Update BOTH this allowlist "
        "AND docs/refactor/phase0/mutation-paths.md, and say why in the "
        "commit message.\n"
        f"scan={actual!r}"
    )


def test_scanner_detects_known_seams():
    """Guard against the scanner regressing into matching nothing."""
    actual = _scan()
    assert "modules/data_service.py" in actual
    assert "setchannel" in actual["modules/data_service.py"]
```

- [ ] **Step 2: Run the test**

Run: `python3 -m pytest tests/test_mutation_path_inventory.py -v`
Expected: PASS (the allowlist above was generated by running this exact scan against `5e8f747`). If it fails, the tree moved since the plan was written: reconcile by updating the allowlist to the scan output **after confirming each new/changed entry is a real mutating call site** (read the cited file). Note: `cl-revenue-ops.py` legitimately has no direct entries — its datastore writes go through `data_service.datastore_push`.

- [ ] **Step 3: Write the narrative inventory**

Write `docs/refactor/phase0/mutation-paths.md` with this content (verify each line number with the given grep before committing; correct drift in place):

```markdown
# Mutation-path inventory (baseline 5e8f747)

Pin test: `tests/test_mutation_path_inventory.py` (file×verb allowlist).
Prior art: `docs/audit/deep/concurrency-map.md` (thread/lock graph),
`docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md` (RPC classification).

## Finding relevant to Workstream G

`modules/data_service.py` is ALREADY a partial central CLN execution
adapter ("NEVER cached" tier): `set_channel` (:275), `fund_channel`
(:281), `close_channel` (:288), `create_invoice` (:316), `send_pay`
(:320), `wait_send_pay` (:324), `delete_pay` (:328), `delete_invoice`
(:332), `pay` (:337), askrene mutations (:367–:425), `datastore_push`
(:461). The refactor's CLN adapter should grow from this seam.

Bypass sites (call plugin.rpc directly, must be routed through the
adapter during Phase 3):

- `modules/rebalance_native_executor_v2.py` — `_rpc_call` (:36) raw
  invoke: invoice build (:450), sendpay (:461), waitsendpay (:463),
  delpay (:386), delinvoice (:391)
- `modules/rebalance_engine_v2.py` — delpay (:2792), datastore fallback
  (:2869), askrene-remove-layer (:316)
- `modules/rebalance_router_v3.py` — askrene-create-layer (:615),
  askrene-update-channel (:631, :649), askrene-remove-layer (:668);
  each has a data_service-preferred branch
- `modules/lnplus_swaps.py` — connect (:1417), fundchannel (:1462),
  signmessage (:130, LN+ auth)
- `modules/capacity_planner.py` — fundchannel/close fallback paths when
  no data_service is wired (:3138 open; close execution path)

## Wrapper callers (economic writers)

- Fee broadcasts: `modules/fee_controller.py:7624` → data_service.set_channel
  (sole setchannel caller)
- Channel opens: `modules/capacity_planner.py:3138` → fund_channel
- Boltz loop-out invoice pay: `modules/boltz_manager.py:844` → pay
- Datastore telemetry: `cl-revenue-ops.py:3510,3515,3557,7242`,
  `modules/profitability_analyzer.py:758`,
  `modules/rebalance_engine_v2.py:2862` → datastore_push

## External write APIs (non-CLN)

- Boltz (`modules/boltz_manager.py`) — via `boltzcli` SUBPROCESS, not
  HTTP: `_run` (:444) / `_run_json` (:469). Writes: createreverseswap
  external-pay (:2017/:2031), createreverseswap (:2133, exec
  :2152–:2214), createswap loop-in (:1823/:1824), createchainswap
  (:2395/:2408), claimswaps (:2338), refund (:2312), withdraw/wallet
  send (:2429/:2471/:2477; also `cl-revenue-ops.py:7517`)
- LN+ (`modules/lnplus_swaps.py`) — HTTP POST via urllib
  (`LNPlusClient._request` :82, base https://lightningnetwork.plus/api/2):
  create_application (:200), delete_application (:205),
  complete_application (:210), create_rating (:229),
  mark_read_notifications (:220)

## Autonomous initiators (background threads)

All `threading.Thread` daemons started at `cl-revenue-ops.py:3428–3435`
(+ RPC-drain :597); each sleeps its interval with ±10–20% random jitter.

| Thread | Def | Interval | Can execute |
|---|---|---|---|
| flow-analysis | :3010 | flow_interval ≥60s | analytics, datastore writes |
| fee-adjustment | :3067 | fee_interval ≥60s | setchannel fee broadcasts |
| rebalance-check | :3108 | rebalance_interval ≥60s | circular sendpay, askrene layers, budget reservations |
| boltz-auto-cycle | :3148 | boltz_auto_cycle_interval_minutes (15m default) | Boltz loop-in/out/withdraw |
| capacity-planner | :3239 | planner_interval ≥600s (default 6h) | fundchannel opens, closes, reservations |
| lnplus-watcher | :3207 | lnplus_watcher_interval (1h default) | LN+ apply/complete/fundchannel/ratings |
| financial-snapshot | :3331 | 24h | DB snapshot writes only |
| startup-snapshot | :3431 | one-shot | peer snapshot to DB |

## Budget/spend enforcement points (Workstream D input)

Four distinct implementations gate spending today:

1. Generic spend ledger, `modules/database.py`: `reserve_spend` (:3895,
   atomic BEGIN IMMEDIATE; `_reserve_budget_atomic` :94),
   `mark_spend_reservation_spent` (:4019) + `record_spend_event` (:4072),
   `release_spend_reservation` (:4010), `cleanup_stale_spend_reservations`
   (:4168), `get_budget_status` (:4512)
2. Rebalance-specific, `modules/database.py`: `reserve_budget` (:3693),
   `release_budget_reservation` (:3734), `mark_budget_spent` (:3752)
3. Capex, `modules/capex_budget.py`: `budget_sats` (:76),
   `tactical_budget_sats` (:107), `get_channel_budget` (:332),
   `reserve/settle/release_boltz_swap_budget` (:429/:459/:504)
4. Growth/efficiency, `modules/growth_budget.py`
   `compute_growth_budget_status` (:90); `modules/capital_efficiency.py`
   `analyze` (:59)

Gate call sites before spend: `rebalancer.py:1451`,
`rebalance_engine_v2.py:1938`, `capacity_planner.py:3199/:3667`,
`boltz_manager.py:1642`, `lnplus_swaps.py:1439`,
`cl-revenue-ops.py:7321/:7351/:7396`.
```

- [ ] **Step 4: Spot-verify the cited line numbers**

Run: `grep -n "def set_channel" modules/data_service.py && grep -n "_rpc_call" modules/rebalance_native_executor_v2.py | head -2 && grep -n "def reserve_spend" modules/database.py`
Expected: line numbers matching the doc (275-ish, 36-ish, 3895-ish). Fix any drifted number in the doc.

- [ ] **Step 5: Run the full suite and commit**

Run: `python3 -m pytest tests/test_mutation_path_inventory.py -q`
Expected: 2 passed.

```bash
git add docs/refactor/phase0/mutation-paths.md tests/test_mutation_path_inventory.py
git commit -m "docs(refactor): Phase 0 mutation-path inventory with enforcement pin test"
```

---

### Task 3: Decision-owner matrix

**Files:**
- Create: `docs/refactor/phase0/decision-owners.md`

**Interfaces:**
- Produces: the authoritative "who decides what today" table that Tasks 6–13 use to pick golden-test seams, and that Workstreams A–F consume.

- [ ] **Step 1: Write the matrix**

Write `docs/refactor/phase0/decision-owners.md`:

```markdown
# Decision-owner matrix (baseline 5e8f747)

Prior art: `docs/audit/decision-loops.md` (loop verdicts),
`docs/audit/operator-decisions.md` + `docs/audit/deep/operator-decisions-deep.md`
(operator rulings D1–D4, DD1+).

Each row: the decision, its single current owner (or duplicated owners —
a refactor target), the exact entry-point seam, and what it consumes.

| Decision | Owner | Entry seam | Consumes |
|---|---|---|---|
| Fee target per channel | `FeeController` | `_adjust_channel_fee` (`modules/fee_controller.py:5465`); cycle `adjust_all_fees` (:4379) | db (DTS/cycle state), config snapshot, channel_info, profitability |
| Fee damping/rails/deadband | `FeeController` | `_apply_damped_fee_target` (:5241), `_get_fee_step_cap` (:5085), `_apply_zero_flow_ratchet_guard` (:5306), `_calculate_floor` (:7957) | fee profile, chain costs |
| Dynamic htlc_max | `FeeController` (embedded — Workstream F3 wants it out) | `_compute_dynamic_htlcmax_msat` (:2874), deadband `_htlcmax_delta_exceeds_deadband` (:2913) | cfg pcts, channel_info |
| Rebalance pair selection | `RebalanceEngine` + `RebalancePlanner` | `RebalanceEngine.find_candidates` (`modules/rebalance_engine_v2.py:1217`) → `RebalancePlanner.plan` (`modules/rebalance_planner_v2.py:110`) | StateSnapshot (pure), capex budgets, profitability |
| Rebalance execution & max-cost | `RebalanceEngine` | `execute_candidate` (:2879), `run_cycle` (:3025); `_pair_policy_allowed` (:2393); `_pair_max_fee_sats` (:1829) | policy_manager, budget ledger |
| Profitability class & ROI | `ChannelProfitabilityAnalyzer` | `analyze_channel` (`modules/profitability_analyzer.py:760`); `_classify_channel` (:2656) | db P&L, bookkeeper, config |
| Channel economic role (revenue) | `ChannelProfitabilityAnalyzer` | `ChannelRole` (:151); 30d window `ChannelProfitability.role_30d` (:396) | forward counts 30d/lifetime |
| Channel flow/balance state | `FlowAnalyzer` (**duplicate classification authority** vs profitability role — Workstream A target) | `_analyze_channel_impl` (`modules/flow_analysis.py:1792`), `_classify_balance_position` (:1904), `ChannelState` enum (:652) | kalman ratio, db flow state |
| Open candidates & ranking | `CapacityPlanner` | `generate_report` (`modules/capacity_planner.py:210`), `_score_candidate` (:2188), `get_candidate_sources` (:2384) | profitability, flow, policy |
| Close recommendation & protection | `CapacityPlanner` | `_close_protection_reason` (:1096, single source of truth), `_check_close_allowed` (:3426, policy tags, fail-closed), exec gate in `execute_cycle` (:339/:450) | profitability `role_30d`, flow confidence, policy tags |
| Boltz swap mode/plan | module-level in `cl-revenue-ops.py` (**not a class** — adapter boundary target) | `_run_boltz_auto_cycle_once` (:2019), `_select_boltz_auto_cycle_mode` (:1926), `_build_boltz_expansion_treasury_plan` (:8297), `_build_boltz_balance_plan` (:8475) | config snapshot, boltz_manager, planner |
| Boltz execution | `BoltzCliManager` | `loop_in` (:1751), `loop_out` (:1851), budget `check_tactical_budget` (:289) (`modules/boltz_manager.py`) | boltzcli subprocess, swap journal |
| LN+ qualification | `SwapEvaluator` | `run_cycle` (`modules/lnplus_swaps.py:268`), `_filter_swap` (:313), `_check_participants` (:352), `_select_and_apply` (:526) | LN+ HTTP client, db, policy bans, planner reputation |
| Budgets | FOUR implementations (see mutation-paths.md §budget) — Workstream D unifies | db spend ledger / db rebalance reservations / capex_budget / growth_budget | — |
| Protections | Policy tags (`PolicyManager`, no_close), close-protection gates, LN+ contract windows, hot-channel overrides (`hot_channel_protection_overrides` table) — DISTRIBUTED, Workstream F5 unifies | — | — |

## Known duplications (refactor targets, confirmed at baseline)

1. Channel classification: `FlowAnalyzer.ChannelState` (flow/balance) vs
   `ChannelRole`/`role_30d` (revenue) — two authorities, different enums.
2. Budgets: four implementations (above).
3. Rebalance modes: hot-channel protection, normal, structural drain,
   manual, diagnostic have distinct paths into the engine (Workstream F4).
4. Boltz decision logic lives in the plugin entry file, not the manager.
```

- [ ] **Step 2: Spot-verify three seams**

Run: `grep -n "def _close_protection_reason\|def _check_close_allowed" modules/capacity_planner.py && grep -n "def plan" modules/rebalance_planner_v2.py && grep -n "def role_30d" modules/profitability_analyzer.py`
Expected: line numbers ≈ the doc (1096/3426, 110, 396). Correct drift in the doc.

- [ ] **Step 3: Commit**

```bash
git add docs/refactor/phase0/decision-owners.md
git commit -m "docs(refactor): Phase 0 decision-owner matrix"
```

---

### Task 4: Persistence map and table pin test

**Files:**
- Create: `docs/refactor/phase0/persistence-map.md`
- Create: `tests/test_persistence_inventory.py`

**Interfaces:**
- Produces: `EXPECTED_TABLES` frozenset pin; persistence doc consumed by Workstream E (ledger design) and the migration plan.

- [ ] **Step 1: Write the failing pin test**

Create `tests/test_persistence_inventory.py`:

```python
"""Phase 0 pin: the set of sqlite tables the plugin creates is inventoried.

New tables (or renames) fail this test until
docs/refactor/phase0/persistence-map.md is updated in the same commit.
"""
import pathlib
import re

DATABASE_PY = pathlib.Path(__file__).resolve().parent.parent / "modules" / "database.py"

EXPECTED_TABLES = frozenset({
    "budget_reservations", "channel_closure_costs", "channel_costs",
    "channel_failures", "channel_probes", "channel_states",
    "closed_channels", "config_overrides", "daily_forwarding_stats",
    "daily_forwarding_stats_inbound", "dead_capital_stage", "fee_changes",
    "fee_strategy_state", "financial_snapshots", "forwards",
    "hot_channel_protection_overrides", "ignored_peers", "kalman_state",
    "lifetime_aggregates", "lnplus_peers", "lnplus_swaps",
    "mempool_fee_history", "pair_rebalance_failures",
    "peer_connection_history", "peer_policies", "peer_reputation",
    "planner_actions", "planner_candidates", "planner_recycle_ops",
    "plugin_flags", "rebalance_costs", "rebalance_history",
    "schema_version", "spend_events", "spend_reservations",
})


def _created_tables():
    text = DATABASE_PY.read_text()
    return frozenset(
        re.findall(r"CREATE TABLE IF NOT EXISTS\s+([a-z_0-9]+)", text)
    )


def test_table_inventory_matches():
    actual = _created_tables()
    assert actual == EXPECTED_TABLES, (
        "sqlite table set changed — update this pin AND "
        "docs/refactor/phase0/persistence-map.md together.\n"
        f"added={sorted(actual - EXPECTED_TABLES)} "
        f"removed={sorted(EXPECTED_TABLES - actual)}"
    )
```

- [ ] **Step 2: Run and reconcile**

Run: `python3 -m pytest tests/test_persistence_inventory.py -v`
Expected: PASS — the baseline scan found 37 `CREATE TABLE` statements but the allowlist above has 35 names; the scan at plan-writing time truncated at 60 lines. If the test FAILS listing `added=[...]`, those are the missing table names: add them to `EXPECTED_TABLES` **and** to the doc in Step 3, then re-run to PASS. Do not remove names unless the grep genuinely no longer finds them.

- [ ] **Step 3: Write the persistence map**

Write `docs/refactor/phase0/persistence-map.md`:

```markdown
# Persistence map (baseline 5e8f747)

Pin test: `tests/test_persistence_inventory.py`.
Prior art (authoritative for retention/growth): `docs/audit/deep/resource-growth.md`
(per-table BOUNDED/unbounded classification with retention line cites),
`docs/audit/deep/prod-baseline-T0.md` (production row counts, 53 MiB DB).

## SQLite database

Path: CLN lightning-dir `revenue_ops.db`; owner `modules/database.py`
(single `Database` class, 7,778 lines). All DDL is
`CREATE TABLE/INDEX IF NOT EXISTS` + guarded `ALTER TABLE` in
`Database.__init__` — additive-only, no migration framework.

`schema_version` is WRITE-ONLY by operator ruling DD9/MIG-3 (2026-07-02,
`modules/database.py:606`): the plugin stamps it but never refuses to run
on version mismatch. Any refactor migration tooling must not assume a
version gate exists.

Tables (paste the sorted list from tests/test_persistence_inventory.py's
EXPECTED_TABLES here, one per line, each annotated with its writer module
and — from resource-growth.md — its retention bound).

## Restart-recovery state (Workstream D/E input)

State that must survive restart and its current recovery path:

- `spend_reservations` / `spend_events` — generic spend ledger;
  stale-reservation cleanup `cleanup_stale_spend_reservations`
  (`modules/database.py:4168`) and RPC `revenue-spend-release-stale`
- `budget_reservations` — rebalance reservations (:3693 lineage)
- `lnplus_swaps` — external obligations (must be honored across restart;
  refactor invariant 6)
- Boltz swap journal (in `boltz_manager`; boltzcli owns swap state
  externally, journal reconciles)
- `dead_capital_stage` — staged closes pending execution
- In-flight sendpay: recovered via waitsendpay/listpays on next cycle

## CLN datastore keys (telemetry projections, read-only contracts)

Writers: `data_service.datastore_push` (`modules/data_service.py:461`) +
one fallback (`modules/rebalance_engine_v2.py:2869`).

- `["revenue","profitability-summary"]` —
  `docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md`
- `["revenue","capex-summary"]` — TTL 1800s —
  `docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md`
- `["revenue","segment-observations"]` — schema_version 1 —
  `docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md`

Enumerate any additional keys with:
`grep -rn "datastore_push\|rpc.datastore" modules/ cl-revenue-ops.py`
and document each (key, writer, schema doc or "undocumented").

Contract conformance is already tested by
`tests/test_cross_plugin_contracts.py` — the refactor's projection layer
must keep that test green (refactor invariant 3).
```

Fill in the two "paste/enumerate" sections while writing the doc — run the named commands and record actual results. The doc must ship complete; those instructions are for the implementer now, not a future reader.

- [ ] **Step 4: Run and commit**

Run: `python3 -m pytest tests/test_persistence_inventory.py -q`
Expected: 1 passed.

```bash
git add docs/refactor/phase0/persistence-map.md tests/test_persistence_inventory.py
git commit -m "docs(refactor): Phase 0 persistence map with table pin test"
```

---

### Task 5: Compatibility catalog and RPC-surface pin test

**Files:**
- Create: `docs/refactor/phase0/compatibility-catalog.md`
- Create: `tests/test_rpc_surface_inventory.py`

**Interfaces:**
- Produces: `EXPECTED_RPC_METHODS` pin (the public surface refactor invariant 2 protects); catalog consumed by Workstream I (RPC facade).

- [ ] **Step 1: Write the pin test**

Create `tests/test_rpc_surface_inventory.py`:

```python
"""Phase 0 pin: the registered RPC surface of cl-revenue-ops.

Adding/removing/renaming an RPC method fails this test until
docs/refactor/phase0/compatibility-catalog.md is updated in the same
commit. Refactor invariant 2: existing primary RPCs remain compatible
unless a separately approved migration changes them.
"""
import pathlib
import re

PLUGIN_PY = pathlib.Path(__file__).resolve().parent.parent / "cl-revenue-ops.py"

EXPECTED_RPC_METHODS = frozenset({
    "revenue-rebalance-cycle", "revenue-status", "revenue-rebalance-debug",
    "revenue-fee-debug", "revenue-fee-cycle", "revenue-analyze",
    "revenue-wake-all", "revenue-capacity-report", "revenue-planner-status",
    "revenue-lnplus-status", "revenue-lnplus-breaker-clear",
    "revenue-lnplus-abandon", "revenue-lnplus-backfill",
    "revenue-planner-candidate-sources", "revenue-planner-candidates",
    "revenue-planner-execute", "revenue-planner-history", "revenue-set-fee",
    "revenue-rebalance", "revenue-profitability", "revenue-history",
    "revenue-ignore", "revenue-unignore", "revenue-list-ignored",
    "revenue-ban", "revenue-unban", "revenue-list-banned", "revenue-policy",
    "revenue-report", "revenue-hot-channel-protection-peers",
    "revenue-config", "revenue-dashboard", "revenue-health",
    "revenue-cleanup-closed", "revenue-clear-reservations",
    "revenue-total-cost-budget", "revenue-capex-status",
    "revenue-spend-ledger", "revenue-spend-reserve", "revenue-spend-release",
    "revenue-spend-release-stale", "revenue-spend-settle",
    "revenue-boltz-quote", "revenue-boltz-loop-out", "revenue-boltz-loop-in",
    "revenue-boltz-status", "revenue-boltz-history",
    "revenue-boltz-external-pay-ignores", "revenue-boltz-budget",
    "revenue-boltz-wallet", "revenue-boltz-refund", "revenue-boltz-claim",
    "revenue-boltz-chainswap", "revenue-boltz-withdraw",
    "revenue-boltz-deposit", "revenue-boltz-backup",
    "revenue-boltz-backup-verify", "revenue-boltz-balance-recommendations",
    "revenue-boltz-auto-cycle-status", "revenue-boltz-auto-cycle-run-now",
    "revenue-boltz-balance-cycle", "revenue-boltz-expansion-treasury-status",
    "revenue-boltz-expansion-treasury-recommendations",
    "revenue-boltz-expansion-treasury-cycle",
})


def _registered_methods():
    text = PLUGIN_PY.read_text()
    return frozenset(re.findall(r'@plugin\.method\(\s*"([a-z-]+)"', text))


def test_rpc_surface_matches():
    actual = _registered_methods()
    assert actual == EXPECTED_RPC_METHODS, (
        "Registered RPC surface changed — update this pin AND "
        "docs/refactor/phase0/compatibility-catalog.md together.\n"
        f"added={sorted(actual - EXPECTED_RPC_METHODS)} "
        f"removed={sorted(EXPECTED_RPC_METHODS - actual)}"
    )


def test_expected_count():
    assert len(EXPECTED_RPC_METHODS) == 64
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/test_rpc_surface_inventory.py -v`
Expected: 2 passed (the 64 names were extracted from `5e8f747`). On failure, reconcile against `grep '@plugin.method' cl-revenue-ops.py` output.

- [ ] **Step 3: Write the catalog**

Write `docs/refactor/phase0/compatibility-catalog.md`:

```markdown
# Public compatibility catalog (baseline 5e8f747)

What the refactor MUST keep working (refactor invariants 2 and 3).
Pin test: `tests/test_rpc_surface_inventory.py` (64 methods).

## RPC surface

Primary operator surfaces (must remain schema-compatible; per
refactor.md Workstream I these become facades over projections):
`revenue-status`, `revenue-fee-debug`, `revenue-rebalance-debug`,
`revenue-config get|set`, `revenue-profitability`, `revenue-analyze`,
`revenue-wake-all`, `revenue-dashboard`, `revenue-health`,
planner/Boltz/LN+ diagnostics.

Action/mutation RPCs (AGENTS.md list; execution-gated): see AGENTS.md
"Action RPC warning" — that list plus every `revenue-boltz-*` action.
Classification per method: `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md`
(refresh header note: last updated 2026-07-09).

Full 64-method list: EXPECTED_RPC_METHODS in the pin test is normative.

## Config surface

Owner: `modules/config.py` (1,309 lines) + `config_overrides` table
(`revenue-config set` persists overrides; precedence documented in
README.md §revenue-config). The refactor's Workstream I risk-profile
work must preserve every currently-accepted key until the deprecation
window defined in refactor.md Phase 5.

Enumerate current keys: run
`python3 -c "import re;print(sorted(set(re.findall(r'getattr\(cfg, .([a-z_0-9]+).', open('modules/fee_controller.py').read()))))"`
style sweeps per module, or read modules/config.py's declared options —
record the full key list here with defaults (this is required content,
not optional).

## Datastore telemetry contracts (current, tested)

- `["revenue","profitability-summary"]`, `["revenue","capex-summary"]`
  (TTL 1800s), `["revenue","segment-observations"]` (schema_version 1)
- Docs: `docs/contracts/*.md`; conformance test:
  `tests/test_cross_plugin_contracts.py`; stale/malformed semantics are
  part of the contract (refactor invariant 3).

## External obligations

- LN+ swaps in flight (`lnplus_swaps` table) — contractual; honored even
  when new-obligation creation is disabled (invariant 6)
- Boltz swaps in flight (boltzcli journal) — reconciliation must complete
  across restart
```

Complete the config-key enumeration while writing (run the sweep, paste the keys). The committed doc must contain the actual list.

- [ ] **Step 4: Run and commit**

Run: `python3 -m pytest tests/test_rpc_surface_inventory.py tests/test_persistence_inventory.py tests/test_mutation_path_inventory.py -q`
Expected: 5 passed.

```bash
git add docs/refactor/phase0/compatibility-catalog.md tests/test_rpc_surface_inventory.py
git commit -m "docs(refactor): Phase 0 compatibility catalog with RPC-surface pin test"
```

---

### Task 6: Golden-test harness

**Files:**
- Create: `tests/golden/__init__.py`
- Create: `tests/golden/util.py`
- Create: `tests/golden/README.md`
- Create: `tests/golden/test_harness_selftest.py`

**Interfaces:**
- Produces: `golden_check(name: str, actual: Any) -> None` and `jsonify(obj) -> Any` in `tests/golden/util.py`, used by every Task 7–13 test module. Fixture files live at `tests/golden/fixtures/<name>.json`. Env var `GOLDEN_UPDATE=1` records; default asserts.

- [ ] **Step 1: Write the failing self-test**

Create `tests/golden/__init__.py` (empty file) and `tests/golden/test_harness_selftest.py`:

```python
"""Self-test for the golden harness (Task 6)."""
import json
import os
import pytest

from tests.golden.util import FIXTURE_DIR, golden_check, jsonify


def test_jsonify_handles_domain_shapes():
    import dataclasses
    import enum

    class Color(enum.Enum):
        RED = 1

    @dataclasses.dataclass
    class Point:
        x: int
        tag: Color

    assert jsonify(Point(1, Color.RED)) == {"x": 1, "tag": "RED"}
    assert jsonify((1, 2)) == [1, 2]


def test_golden_check_records_then_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.golden.util.FIXTURE_DIR", tmp_path)
    monkeypatch.setenv("GOLDEN_UPDATE", "1")
    golden_check("selftest/example", {"b": 2, "a": (1,)})
    monkeypatch.delenv("GOLDEN_UPDATE")
    golden_check("selftest/example", {"a": (1,), "b": 2})  # order-insensitive
    with pytest.raises(AssertionError):
        golden_check("selftest/example", {"a": [1], "b": 3})


def test_missing_fixture_fails_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.golden.util.FIXTURE_DIR", tmp_path)
    with pytest.raises(AssertionError, match="GOLDEN_UPDATE=1"):
        golden_check("selftest/absent", {"x": 1})
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/golden/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.golden.util'`.

- [ ] **Step 3: Implement the harness**

Create `tests/golden/util.py`:

```python
"""Golden characterization-test harness (refactor Phase 0).

golden_check(name, actual):
  - normal mode: compare `actual` (after jsonify canonicalization) to
    tests/golden/fixtures/<name>.json; exact equality required.
  - GOLDEN_UPDATE=1: (re)write the fixture from `actual`.

POLICY (docs/planning/refactor.md, Test strategy): never re-record a
fixture just to make a failing test pass. An intentional behavior change
needs a dedicated test and a rationale in the commit message.
"""
import dataclasses
import enum
import json
import os
import pathlib

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


def jsonify(obj):
    """Canonicalize domain objects to JSON-safe structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonify(dataclasses.asdict(obj))
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        return [jsonify(v) for v in seq]
    if isinstance(obj, float):
        # Round-trip via repr keeps exact float text stable across runs.
        return obj
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return repr(obj)


def golden_check(name: str, actual) -> None:
    path = FIXTURE_DIR / f"{name}.json"
    canonical = jsonify(actual)
    if os.environ.get("GOLDEN_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(canonical, sort_keys=True, indent=2) + "\n"
        )
        return
    assert path.exists(), (
        f"golden fixture missing: {path}\n"
        f"Record it with: GOLDEN_UPDATE=1 python3 -m pytest <this test> "
        f"— then REVIEW the recorded values for plausibility before "
        f"committing."
    )
    expected = json.loads(path.read_text())
    assert canonical == expected, (
        f"golden mismatch for {name!r}.\n"
        f"expected: {json.dumps(expected, sort_keys=True)[:2000]}\n"
        f"actual:   {json.dumps(canonical, sort_keys=True)[:2000]}\n"
        f"If this change is INTENTIONAL, re-record with GOLDEN_UPDATE=1 "
        f"and justify in the commit message; never re-record merely to "
        f"go green."
    )
```

Create `tests/golden/README.md`:

```markdown
# Golden characterization tests (refactor Phase 0)

These freeze the CURRENT behavior of each principal decision class
(docs/planning/refactor.md, Test strategy → Golden behavioral tests) so
the refactor can prove semantic parity.

- Fixtures: `fixtures/<class>/<scenario>.json` — canonical JSON
  (sorted keys) written by `GOLDEN_UPDATE=1`.
- Every module ALSO contains at least one hand-computed assertion (not
  golden) so a recorded fixture full of nonsense can't self-certify.
- Re-recording policy: see `util.py` docstring. Fixture diffs in review
  ARE the behavior-change review.

| Decision class | Test module |
|---|---|
| Fee damping/floor | `test_golden_fee_damping.py` |
| Dynamic htlc_max | `test_golden_htlcmax.py` |
| Rebalance planning | `test_golden_rebalance_planner.py` |
| Profitability class/role | `test_golden_profitability.py` |
| Close protection | `test_golden_close_protection.py` |
| Boltz auto-cycle plan | `test_golden_boltz_cycle.py` |
| LN+ qualification | `test_golden_lnplus_gates.py` |
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/golden/ -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/
git commit -m "test(refactor): golden characterization harness (Phase 0)"
```

---

### Task 7: Golden — fee damping and economic floor

**Files:**
- Create: `tests/golden/test_golden_fee_damping.py`
- Create: `tests/golden/fixtures/fee/` (recorded fixtures)

**Interfaces:**
- Consumes: `golden_check`, `jsonify` from `tests/golden/util.py` (Task 6).
- Covers spec class "Fee target and hold decisions" for the constraint stages (rails/rate/deadband seams). The unclamped DTS/PID target is deliberately NOT goldened in Phase 0 — it samples a Thompson posterior (`random`), a portability hazard recorded in Task 14; its determinism is a Phase 1 (clock/seed injection) outcome.

- [ ] **Step 1: Write the test module**

Create `tests/golden/test_golden_fee_damping.py`:

```python
"""Golden: fee damping (_apply_damped_fee_target/_get_fee_step_cap) and
economic floor (_calculate_floor). Pure constraint math — deterministic."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_controller import FeeController
from tests.golden.util import golden_check


PROFILE = SimpleNamespace(
    wake_cycle_max_delta_ratio=0.50,
    normal_cycle_max_delta_ratio=0.15,
    wake_cycle_min_delta_ppm=25,
    normal_cycle_min_delta_ppm=10,
)


@pytest.fixture
def fc():
    controller = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())
    # Pin the fee profile so damping math is exercised with fixed inputs.
    controller._resolve_fee_profile = lambda cfg=None: ("golden", PROFILE)
    return controller


DAMPING_SCENARIOS = [
    # (name, current_ppm, target_ppm, woke_from_sleep)
    ("small_raise_within_cap", 1000, 1100, False),
    ("large_raise_clamped", 1000, 5000, False),
    ("large_cut_clamped", 2000, 100, False),
    ("wake_cycle_wider_cap", 1000, 5000, True),
    ("low_fee_min_delta_floor", 10, 500, False),
    ("no_change", 750, 750, False),
]


@pytest.mark.parametrize("name,current,target,woke", DAMPING_SCENARIOS)
def test_golden_damped_fee_target(fc, name, current, target, woke):
    applied, diag = fc._apply_damped_fee_target(current, target, woke)
    golden_check(f"fee/damping_{name}", {
        "inputs": {"current": current, "target": target, "woke": woke},
        "applied_fee_ppm": applied,
        "diag": diag,
    })


def test_damping_hand_computed_anchor(fc):
    """Non-golden anchor: 1000 -> 5000 normal cycle caps at
    1000 + max(10, ceil(1000*0.15)) = 1150."""
    applied, diag = fc._apply_damped_fee_target(1000, 5000, False)
    assert applied == 1150
    assert diag["cap_applied"] is True
    assert diag["cap_reason"] == "normal_cycle_delta_cap"


FLOOR_SCENARIOS = [
    ("defaults_no_chain_costs", 2_000_000, None, "local"),
    ("live_chain_costs_local", 2_000_000,
     {"open_cost_sats": 2500, "close_cost_sats": 1500}, "local"),
    ("remote_opener_cheaper", 2_000_000,
     {"open_cost_sats": 2500, "close_cost_sats": 1500}, "remote"),
    ("small_channel", 200_000,
     {"open_cost_sats": 2500, "close_cost_sats": 1500}, "local"),
]


@pytest.mark.parametrize("name,capacity,chain_costs,opener", FLOOR_SCENARIOS)
def test_golden_calculate_floor(fc, name, capacity, chain_costs, opener):
    floor_ppm = fc._calculate_floor(
        capacity, chain_costs=chain_costs, peer_id=None, opener=opener
    )
    golden_check(f"fee/floor_{name}", {
        "inputs": {"capacity_sats": capacity, "chain_costs": chain_costs,
                   "opener": opener},
        "floor_ppm": floor_ppm,
    })
```

- [ ] **Step 2: Run to verify it fails for the right reason**

Run: `python3 -m pytest tests/golden/test_golden_fee_damping.py -v`
Expected: the hand-computed anchor PASSES; every golden test FAILS with "golden fixture missing". If instead you get an `AttributeError`/`TypeError` from `FeeController` construction or `_calculate_floor` internals (e.g. it reads an unmocked config value), mirror the setup in `tests/test_fee_controller.py::_make_fc` (line ~129) — stub the same attributes it stubs — and re-run until the only failures are missing fixtures.

- [ ] **Step 3: Record fixtures**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_fee_damping.py -q`
Then review every file under `tests/golden/fixtures/fee/` by hand: damping deltas must respect `max(min_delta, ceil(current*ratio))`; floors must be positive and ≤ 100,000 ppm; the remote-opener floor must be ≤ the local-opener floor for identical inputs. Do not commit implausible values — investigate instead.

- [ ] **Step 4: Run to verify green**

Run: `python3 -m pytest tests/golden/test_golden_fee_damping.py -q`
Expected: all pass, no GOLDEN_UPDATE in env.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_fee_damping.py tests/golden/fixtures/fee/
git commit -m "test(refactor): golden fixtures for fee damping and economic floor"
```

---

### Task 8: Golden — dynamic htlc_max

**Files:**
- Create: `tests/golden/test_golden_htlcmax.py`
- Create: `tests/golden/fixtures/htlcmax/`

**Interfaces:**
- Consumes: `golden_check` (Task 6); `FeeController` construction pattern from Task 7.
- Covers spec class "Dynamic htlc_max decisions" (refactor.md line 867) — the seam Workstream F3 will move into admission control.

- [ ] **Step 1: Write the test module**

Create `tests/golden/test_golden_htlcmax.py`:

```python
"""Golden: _compute_dynamic_htlcmax_msat + _htlcmax_delta_exceeds_deadband.
Pure functions of (cfg, channel_info, flow_state) — deterministic."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_controller import FeeController
from tests.golden.util import golden_check


@pytest.fixture
def fc():
    return FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())


def _cfg(**over):
    base = dict(
        enable_dynamic_htlcmax=True,
        htlcmax_source_pct=0.85,
        htlcmax_sink_pct=0.25,
        htlcmax_balanced_pct=0.50,
    )
    base.update(over)
    return SimpleNamespace(**base)


SCENARIOS = [
    # (name, cfg_overrides, channel_info, flow_state)
    ("disabled_returns_none", {"enable_dynamic_htlcmax": False},
     {"capacity": 2_000_000, "spendable_msat": 1_000_000_000}, "source"),
    ("string_true_enables", {"enable_dynamic_htlcmax": "true"},
     {"capacity": 2_000_000, "spendable_msat": 1_000_000_000}, "source"),
    ("source_ample_spendable", {},
     {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}, "source"),
    ("sink_small_share", {},
     {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}, "sink"),
    ("balanced_mid_share", {},
     {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}, "balanced"),
    ("depletion_caps_target", {},
     {"capacity": 2_000_000, "spendable_msat": 50_000_000}, "source"),
    ("floor_wins_when_depleted", {},
     {"capacity": 2_000_000, "spendable_msat": 1_000}, "source"),
    ("zero_capacity_none", {}, {"capacity": 0, "spendable_msat": 0}, "source"),
]


@pytest.mark.parametrize("name,over,chan,state", SCENARIOS)
def test_golden_htlcmax(fc, name, over, chan, state):
    result = fc._compute_dynamic_htlcmax_msat(_cfg(**over), chan, state)
    golden_check(f"htlcmax/{name}", {
        "inputs": {"cfg_overrides": over, "channel_info": chan,
                   "flow_state": state},
        "htlcmax_msat": result,
    })


def test_htlcmax_hand_computed_anchor(fc):
    """sink pct 0.25 of 2M sats = 500_000_000 msat; ample spendable so
    the depletion cap (spendable * fraction) doesn't bind below it only
    if fraction*1.9e9 >= 5e8 — verify against class constant."""
    result = fc._compute_dynamic_htlcmax_msat(
        _cfg(), {"capacity": 2_000_000, "spendable_msat": 1_900_000_000},
        "sink",
    )
    expected_uncapped = int(2_000_000 * 1000 * 0.25)
    depletion_cap = int(1_900_000_000 * fc.HTLCMAX_DEPLETION_SPENDABLE_FRACTION)
    assert result == max(fc.HTLCMAX_FLOOR_MSAT,
                         min(expected_uncapped, depletion_cap))


DEADBAND_CASES = [
    ("equal_no_broadcast", 500_000_000, 500_000_000),
    ("zero_current_always", 500_000_000, 0),
    ("tiny_move", 501_000_000, 500_000_000),
    ("big_move", 900_000_000, 500_000_000),
]


@pytest.mark.parametrize("name,new,current", DEADBAND_CASES)
def test_golden_htlcmax_deadband(fc, name, new, current):
    golden_check(f"htlcmax/deadband_{name}", {
        "inputs": {"new_msat": new, "current_msat": current},
        "exceeds": fc._htlcmax_delta_exceeds_deadband(new, current),
    })
```

- [ ] **Step 2: Run to verify failure mode**

Run: `python3 -m pytest tests/golden/test_golden_htlcmax.py -v`
Expected: anchor test PASSES; golden tests FAIL with "golden fixture missing".

- [ ] **Step 3: Record and review**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_htlcmax.py -q`
Review `tests/golden/fixtures/htlcmax/`: `disabled_returns_none` and `zero_capacity_none` must be `null`; source > balanced > sink for identical capacity; every non-null value ≥ `HTLCMAX_FLOOR_MSAT` (10,000 sats = 10,000,000 msat) and ≤ capacity in msat.

- [ ] **Step 4: Verify green**

Run: `python3 -m pytest tests/golden/test_golden_htlcmax.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_htlcmax.py tests/golden/fixtures/htlcmax/
git commit -m "test(refactor): golden fixtures for dynamic htlc_max decisions"
```

---

### Task 9: Golden — rebalance planner

**Files:**
- Create: `tests/golden/test_golden_rebalance_planner.py`
- Create: `tests/golden/fixtures/rebalance/`

**Interfaces:**
- Consumes: `golden_check`, `jsonify` (Task 6).
- Covers spec class "Rebalance source/target/amount/max-cost decisions" at the pure planner seam (`RebalancePlanner.plan` is documented "pure and intentionally free of plugin/RPC access" — the ideal golden seam; the engine's routing/execution layers are exercised by existing tests).

- [ ] **Step 1: Write the test module**

Create `tests/golden/test_golden_rebalance_planner.py`:

```python
"""Golden: RebalancePlanner.plan — source/dest pairing, amount sizing,
scoring, skip reasons. Pure snapshot-in / PlanResult-out."""
import pytest

from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_state_v2 import ChannelState, StateSnapshot
from tests.golden.util import golden_check


def _ch(cid, local_ratio, **over):
    base = dict(
        channel_id=cid,
        peer_id="02" + cid[0] * 64,
        capacity_sats=2_000_000,
        local_ratio=local_ratio,
        actual_inbound_fee_ppm=100,
        value_class="active",
        is_valuable=True,
        remaining_budget_sats=5_000,
        cooldown_active=False,
        source_eligible=True,
        dest_eligible=True,
        local_out_fee_ppm=250,
        is_active=True,
    )
    base.update(over)
    return ChannelState(**base)


def _snapshot(channels):
    return StateSnapshot(
        channels=tuple(channels),
        total_capacity_sats=sum(c.capacity_sats for c in channels),
        total_remaining_budget_sats=sum(
            c.remaining_budget_sats for c in channels),
        valuable_channel_count=sum(1 for c in channels if c.is_valuable),
    )


SCENARIOS = {
    "single_obvious_pair": [
        _ch("aaa", 0.90), _ch("bbb", 0.10),
    ],
    "no_over_remote_no_pairs": [
        _ch("aaa", 0.90), _ch("bbb", 0.50),
    ],
    "source_ineligible_skipped": [
        _ch("aaa", 0.90, source_eligible=False,
            source_reason="cooldown_active"),
        _ch("bbb", 0.10),
    ],
    "profitable_dest_preferred": [
        _ch("aaa", 0.92),
        _ch("bbb", 0.08, value_class="profitable"),
        _ch("ccc", 0.08, value_class="neutral", is_valuable=False),
    ],
    "custom_band_channel": [
        _ch("aaa", 0.70, target_band_high=0.60),  # over-local per own band
        _ch("bbb", 0.10),
    ],
    "amount_bounded_by_chunk": [
        _ch("aaa", 1.00, capacity_sats=50_000_000,
            remaining_budget_sats=100_000),
        _ch("bbb", 0.00, capacity_sats=50_000_000,
            remaining_budget_sats=100_000),
    ],
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_plan(name):
    planner = RebalancePlanner(
        target_band_low=0.35, target_band_high=0.65,
        max_chunk_sats=2_000_000, max_pairs=10, pair_fee_cap_ppm=0,
    )
    result = planner.plan(_snapshot(SCENARIOS[name]))
    golden_check(f"rebalance/plan_{name}", result)


def test_plan_hand_computed_anchor():
    """Non-golden anchor: one over-local + one over-remote channel with
    identical capacity must yield exactly one selected pair, source=aaa
    dest=bbb, amount > 0 and <= max_chunk_sats."""
    planner = RebalancePlanner()
    result = planner.plan(_snapshot([_ch("aaa", 0.90), _ch("bbb", 0.10)]))
    assert len(result.selected) == 1
    pair = result.selected[0]
    assert pair.source_channel_id == "aaa"
    assert pair.dest_channel_id == "bbb"
    assert 0 < pair.amount_sats <= planner.max_chunk_sats
```

- [ ] **Step 2: Run to verify failure mode**

Run: `python3 -m pytest tests/golden/test_golden_rebalance_planner.py -v`
Expected: anchor PASSES; golden tests FAIL with missing fixtures. A `TypeError` on `ChannelState(**base)` means a required field was added since baseline — check `modules/rebalance_state_v2.py` and extend `_ch`.

- [ ] **Step 3: Record and review**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_rebalance_planner.py -q`
Review `tests/golden/fixtures/rebalance/`: `no_over_remote_no_pairs` must have empty `selected` and a populated `drain_demand`; `source_ineligible_skipped` must show the skip reason `cooldown_active`; every selected pair's `score_decomposition.model_version` should read `v2-bootstrap-explainability`.

- [ ] **Step 4: Verify green, run whole suite**

Run: `python3 -m pytest tests/golden/ -q && python3 -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider 2>&1 | tail -2`
Expected: golden all pass; full suite still 3114+ passed (goldens add to the count; nothing else regresses).

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_rebalance_planner.py tests/golden/fixtures/rebalance/
git commit -m "test(refactor): golden fixtures for rebalance planner decisions"
```

---

### Task 10: Golden — profitability classification and 30d role

**Files:**
- Create: `tests/golden/test_golden_profitability.py`
- Create: `tests/golden/fixtures/profitability/`

**Interfaces:**
- Consumes: `golden_check` (Task 6).
- Covers spec class "Profitability classification and ROI" including `role_30d` (the signal the 2026-07-12 close-protection fix depends on — `5e8f747`).

- [ ] **Step 1: Write the test module**

Create `tests/golden/test_golden_profitability.py`. Base the object construction on `tests/test_profitability_analyzer.py::_make_profitability` (line 24) — copy that helper's shape, do not import it:

```python
"""Golden: ChannelProfitability derived signals (marginal_roi, role_30d)
and ChannelProfitabilityAnalyzer._classify_channel."""
from unittest.mock import MagicMock

import pytest

from modules.profitability_analyzer import (
    ChannelCosts,
    ChannelProfitability,
    ChannelProfitabilityAnalyzer,
    ChannelRevenue,
    ChannelRole,
    ProfitabilityClass,
)
from tests.golden.util import golden_check


def _make_prof(fees_earned_sats=2000, rebalance_cost_sats=1000,
               sourced_fee_contribution_sats=0):
    costs = ChannelCosts(
        channel_id="111x222x0", peer_id="02" + "a" * 64,
        open_cost_sats=500, rebalance_cost_sats=rebalance_cost_sats,
        effective_rebalance_cost_sats=0,
    )
    revenue = ChannelRevenue(
        channel_id="111x222x0",
        fees_earned_msat=fees_earned_sats * 1000,
        volume_routed_msat=1_000_000 * 1000,
        forward_count=100,
        sourced_fee_contribution_msat=sourced_fee_contribution_sats * 1000,
    )
    return ChannelProfitability(
        channel_id="111x222x0", peer_id="02" + "a" * 64,
        capacity_sats=2_000_000, costs=costs, revenue=revenue,
        net_profit_sats=fees_earned_sats - costs.total_cost_sats,
        roi_percent=10.0, classification=ProfitabilityClass.PROFITABLE,
        cost_per_sat_routed=0.001, fee_per_sat_routed=0.002,
        days_open=30, last_routed=None,
    )


ROLE_SCENARIOS = [
    # (name, window_avail, fwd_30d, sourced_fwd_30d, total_fwd_30d)
    ("no_window_falls_back_to_lifetime", False, 0, 0, 0),
    ("gateway_30d_dominant_sourced", True, 2, 40, 42),
    ("exit_30d_dominant_direct", True, 40, 2, 42),
    ("balanced_30d", True, 20, 22, 42),
    ("too_few_forwards_30d", True, 3, 4, 7),
]


@pytest.mark.parametrize(
    "name,avail,fwd,sourced,total", ROLE_SCENARIOS)
def test_golden_role_30d(name, avail, fwd, sourced, total):
    prof = _make_prof()
    prof.window_30d_available = avail
    prof.forward_count_30d = fwd
    prof.sourced_forward_count_30d = sourced
    prof.total_forward_count_30d = total
    golden_check(f"profitability/role30d_{name}", {
        "inputs": {"window_30d_available": avail, "forward_count_30d": fwd,
                   "sourced_forward_count_30d": sourced,
                   "total_forward_count_30d": total,
                   "lifetime_role": prof.channel_role},
        "role_30d": prof.role_30d,
    })


MARGINAL_ROI_SCENARIOS = [
    ("profit_over_cost", 500, 200),
    ("negative_profit", -300, 600),
    ("zero_cost_positive_profit", 500, 0),
    ("zero_cost_zero_profit", 0, 0),
]


@pytest.mark.parametrize("name,profit_30d,cost_30d", MARGINAL_ROI_SCENARIOS)
def test_golden_marginal_roi(name, profit_30d, cost_30d):
    prof = _make_prof()
    prof.marginal_profit_30d_sats = profit_30d
    prof.rebalance_cost_30d_sats = cost_30d
    golden_check(f"profitability/marginal_roi_{name}", {
        "inputs": {"marginal_profit_30d_sats": profit_30d,
                   "rebalance_cost_30d_sats": cost_30d},
        "marginal_roi": prof.marginal_roi,
    })


CLASSIFY_SCENARIOS = [
    # (name, roi, net_profit_sats, last_routed_ts, days_open, fwd_count)
    ("young_profitable", 25.0, 5000, 1_752_000_000, 10, 200),
    ("old_loser", -40.0, -8000, 1_740_000_000, 400, 15),
    ("never_routed_mature", 0.0, -500, None, 120, 0),
    ("breakeven_active", 0.5, 10, 1_752_300_000, 90, 500),
]


@pytest.mark.parametrize(
    "name,roi,net,last,days,fwd", CLASSIFY_SCENARIOS)
def test_golden_classify(name, roi, net, last, days, fwd):
    analyzer = ChannelProfitabilityAnalyzer(
        MagicMock(), MagicMock(), MagicMock())
    result = analyzer._classify_channel(
        roi, net, last, days, channel_id="111x222x0",
        peer_id="02" + "a" * 64, forward_count=fwd,
    )
    golden_check(f"profitability/classify_{name}", {
        "inputs": {"roi": roi, "net_profit": net, "last_routed": last,
                   "days_open": days, "forward_count": fwd},
        "classification": result,
    })


def test_marginal_roi_hand_computed_anchor():
    prof = _make_prof()
    prof.marginal_profit_30d_sats = 500
    prof.rebalance_cost_30d_sats = 200
    assert abs(prof.marginal_roi - 2.5) < 0.01
```

- [ ] **Step 2: Run to verify failure mode**

Run: `python3 -m pytest tests/golden/test_golden_profitability.py -v`
Expected: anchor PASSES; goldens FAIL on missing fixtures. If `_classify_channel` raises on MagicMock config/db attributes (e.g. compares a threshold), find how `tests/test_profitability_analyzer.py` constructs the analyzer for its `_classify_channel` tests (`grep -n "_classify_channel" tests/test_profitability_analyzer.py`) and mirror that setup exactly. If `_classify_channel` uses `time.time()` against `last_routed` (recency), freeze it: `monkeypatch.setattr("modules.profitability_analyzer.time.time", lambda: 1_752_400_000.0)` — fixed timestamps in scenarios were chosen near that instant. Record any such wall-clock dependency in Task 14's hazard doc.

- [ ] **Step 3: Record and review**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_profitability.py -q`
Review: `no_window_falls_back_to_lifetime` role must equal the recorded `lifetime_role`; `gateway_30d_dominant_sourced` must be `INBOUND_GATEWAY`-like; `too_few_forwards_30d` (7 < 10 threshold) must NOT be directional. `zero_cost_positive_profit` marginal_roi must be exactly `1.0` (per existing test `test_no_costs_returns_one_if_earning`).

- [ ] **Step 4: Verify green**

Run: `python3 -m pytest tests/golden/test_golden_profitability.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_profitability.py tests/golden/fixtures/profitability/
git commit -m "test(refactor): golden fixtures for profitability classification and role_30d"
```

---

### Task 11: Golden — close-protection gates

**Files:**
- Create: `tests/golden/test_golden_close_protection.py`
- Create: `tests/golden/fixtures/close_protection/`

**Interfaces:**
- Consumes: `golden_check` (Task 6); `ChannelRole` from `modules.profitability_analyzer`.
- Covers spec class "Close protections" — `_close_protection_reason` (capacity_planner.py:1096, the single source of truth) and the policy-tag gate `_check_close_allowed` (:3426).

- [ ] **Step 1: Write the test module**

Create `tests/golden/test_golden_close_protection.py`:

```python
"""Golden: CapacityPlanner close-protection gates.

_close_protection_reason inputs are duck-typed (prof, flow_metrics) —
SimpleNamespace stands in, matching how production passes
ChannelProfitability + FlowMetrics.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capacity_planner import CapacityPlanner
from modules.profitability_analyzer import ChannelRole
from tests.golden.util import golden_check


@pytest.fixture
def planner():
    return CapacityPlanner(MagicMock(), MagicMock(), MagicMock())


def _prof(role_30d=ChannelRole.BALANCED, marginal_roi_percent=0.0,
          window_30d_available=True, sourced_fee_30d_msat=0,
          lifetime_sourced_fee_sats=0):
    return SimpleNamespace(
        role_30d=role_30d,
        channel_role=ChannelRole.BALANCED,
        marginal_roi_percent=marginal_roi_percent,
        window_30d_available=window_30d_available,
        sourced_fee_30d_msat=sourced_fee_30d_msat,
        revenue=SimpleNamespace(
            sourced_fee_contribution_sats=lifetime_sourced_fee_sats),
    )


def _flow(confidence=0.9):
    return SimpleNamespace(confidence=confidence)


SCENARIOS = {
    "unprotected_dead_channel": (
        _prof(role_30d=ChannelRole.BALANCED, marginal_roi_percent=-90.0),
        _flow(0.9)),
    "gateway_30d_protected": (
        _prof(role_30d=ChannelRole.INBOUND_GATEWAY,
              marginal_roi_percent=-10.0),
        _flow(0.9)),
    "gateway_30d_but_deep_loser_unprotected": (
        _prof(role_30d=ChannelRole.INBOUND_GATEWAY,
              marginal_roi_percent=-60.0),
        _flow(0.9)),
    "sourced_fee_30d_protected": (
        _prof(window_30d_available=True, sourced_fee_30d_msat=500_000,
              marginal_roi_percent=-20.0),
        _flow(0.9)),
    "sourced_fee_lifetime_fallback_protected": (
        _prof(window_30d_available=False, lifetime_sourced_fee_sats=500,
              marginal_roi_percent=-20.0),
        _flow(0.9)),
    "stale_lifetime_sourcing_not_used_when_window_present": (
        _prof(window_30d_available=True, sourced_fee_30d_msat=0,
              lifetime_sourced_fee_sats=500,
              marginal_roi_percent=-90.0),
        _flow(0.9)),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_close_protection_reason(planner, name):
    prof, flow = SCENARIOS[name]
    reason = planner._close_protection_reason("111x222x0", prof, flow, set())
    golden_check(f"close_protection/{name}", {
        "reason": reason,
    })


def test_low_kalman_confidence_defers_to_inactivity_signal(planner):
    """Non-golden anchor for the confidence gate wiring: with low
    confidence and _inactivity_is_signal stubbed False, the gate must
    return KALMAN_LOW_CONFIDENCE; stubbed True must fall through it."""
    prof, flow = _prof(marginal_roi_percent=-90.0), _flow(confidence=0.2)
    planner._inactivity_is_signal = lambda p, f: False
    assert planner._close_protection_reason(
        "111x222x0", prof, flow, set()) == "KALMAN_LOW_CONFIDENCE"
    planner._inactivity_is_signal = lambda p, f: True
    assert planner._close_protection_reason(
        "111x222x0", prof, flow, set()) != "KALMAN_LOW_CONFIDENCE"
```

Then add golden coverage for `_check_close_allowed`: first run `grep -n "_check_close_allowed" tests/test_capacity_planner.py` and read one existing test (there are recent ones from the 2026-07-11/12 close-protection work) to copy its policy-manager mock shape exactly. Append to the module a parametrized golden test with scenarios: `no_policy_default_allowed`, `no_close_tag_blocks`, `protect_policy_blocks`, `policy_lookup_error_fails_closed` — each calling `planner._check_close_allowed(peer_id)` with the mocked policy manager and goldening the returned tuple.

- [ ] **Step 2: Run to verify failure mode**

Run: `python3 -m pytest tests/golden/test_golden_close_protection.py -v`
Expected: anchor PASSES; goldens FAIL on missing fixtures. If `_close_protection_reason` touches route-pair state beyond the 4th argument, pass the shape existing tests pass (check `grep -n "_close_protection_reason" tests/test_capacity_planner.py`).

- [ ] **Step 3: Record and review**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_close_protection.py -q`
Review against the intent of commit `5e8f747`: `gateway_30d_protected` → `"INBOUND_GATEWAY"`; `sourced_fee_30d_protected` → `"SOURCED_FEE_CONTRIBUTION"`; `stale_lifetime_sourcing_not_used_when_window_present` → `null` (this scenario IS the bug that commit fixed — lifetime sourcing must not protect when the 30d window is present and empty); `unprotected_dead_channel` → `null`.

- [ ] **Step 4: Verify green**

Run: `python3 -m pytest tests/golden/test_golden_close_protection.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_close_protection.py tests/golden/fixtures/close_protection/
git commit -m "test(refactor): golden fixtures for close-protection gates"
```

---

### Task 12: Golden — Boltz auto-cycle plan (dry run)

**Files:**
- Create: `tests/golden/test_golden_boltz_cycle.py`
- Create: `tests/golden/fixtures/boltz/`

**Interfaces:**
- Consumes: `golden_check` (Task 6); `load_plugin_module()` from `tests/plugin_test_utils.py`.
- Covers spec class "Boltz treasury/structural decisions" at the mode-selection/plan seam, dry-run only (never executes; AGENTS.md).

- [ ] **Step 1: Study the existing dry-run test and write the module**

First run: `sed -n 1,80p tests/test_boltz_auto_cycle_dry_run.py` and copy its exact setup (module load, `mod.boltz_manager` mock, `mod.config.snapshot.return_value` SimpleNamespace fields). Then create `tests/golden/test_golden_boltz_cycle.py`:

```python
"""Golden: Boltz auto-cycle mode selection and plan structure, dry-run.

Setup mirrors tests/test_boltz_auto_cycle_dry_run.py (module-level entry
point `_run_boltz_auto_cycle_once`). Volatile fields (timestamps,
durations) are stripped before goldening.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module
from tests.golden.util import golden_check

VOLATILE_KEYS = {"timestamp", "ts", "duration_ms", "elapsed", "now",
                 "checked_at", "updated_at"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


@pytest.fixture
def mod():
    m = load_plugin_module()
    m.boltz_manager = MagicMock(enabled=True)
    # Copy the SimpleNamespace config fields from
    # tests/test_boltz_auto_cycle_dry_run.py verbatim, then pin any
    # additional fields the cycle reads (AttributeError during Step 2
    # names each missing one).
    m.config.snapshot.return_value = SimpleNamespace(
        # ... copied fields ...
    )
    return m


def test_golden_auto_cycle_disabled_manager(mod):
    mod.boltz_manager = MagicMock(enabled=False)
    result = mod._run_boltz_auto_cycle_once(trigger="golden", dry_run=True)
    golden_check("boltz/cycle_disabled", _strip_volatile(result))


def test_golden_auto_cycle_idle_no_plans(mod):
    # With no treasury need and no balance recommendations the selector
    # must choose idle; stub the plan builders the way the existing
    # dry-run test does.
    result = mod._run_boltz_auto_cycle_once(trigger="golden", dry_run=True)
    golden_check("boltz/cycle_idle", _strip_volatile(result))


def test_golden_mode_selector_prefers_treasury():
    mod = load_plugin_module()
    treasury_plan = {"action": "loop_in", "amount_sats": 500_000,
                     "reason": "reserve_below_floor"}
    balance_plan = {"recommendations": [{"action": "loop_out",
                                         "channel_id": "111x222x0",
                                         "amount_sats": 250_000}]}
    mode = mod._select_boltz_auto_cycle_mode(
        treasury_plan=treasury_plan, balance_plan=balance_plan)
    golden_check("boltz/mode_treasury_vs_balance", {"mode": mode})


def test_dry_run_never_calls_executor(mod):
    """Non-golden anchor: dry_run must not invoke any boltz_manager
    execution method."""
    mod._run_boltz_auto_cycle_once(trigger="golden", dry_run=True)
    for method in ("loop_in", "loop_out", "withdraw"):
        assert not getattr(mod.boltz_manager, method).called
```

The two `...` config blocks are filled by copying from `tests/test_boltz_auto_cycle_dry_run.py` — that is a mechanical copy at implementation time, not a design decision. If `_select_boltz_auto_cycle_mode`'s parameters differ from `(treasury_plan=..., balance_plan=...)`, read its definition at `cl-revenue-ops.py:1926` and pass what it actually takes; golden the returned value whatever its shape.

- [ ] **Step 2: Run, iterating on missing config fields**

Run: `python3 -m pytest tests/golden/test_golden_boltz_cycle.py -v`
Expected end state: `test_dry_run_never_calls_executor` PASSES; golden tests FAIL only with "golden fixture missing". Iterate: each `AttributeError: ... SimpleNamespace ... 'x'` names a config field to add with its production default (find it in `modules/config.py`: `grep -n "x" modules/config.py`).

- [ ] **Step 3: Record and review**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_boltz_cycle.py -q`
Review: `cycle_disabled` must show a disabled/skip status, not a plan; `cycle_idle` must select no action; `mode_treasury_vs_balance` must prefer the treasury plan (treasury reserve outranks balance optimization).

- [ ] **Step 4: Verify green**

Run: `python3 -m pytest tests/golden/test_golden_boltz_cycle.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_boltz_cycle.py tests/golden/fixtures/boltz/
git commit -m "test(refactor): golden fixtures for Boltz auto-cycle dry-run decisions"
```

---

### Task 13: Golden — LN+ qualification gates

**Files:**
- Create: `tests/golden/test_golden_lnplus_gates.py`
- Create: `tests/golden/fixtures/lnplus/`

**Interfaces:**
- Consumes: `golden_check` (Task 6).
- Covers spec class "LN+ qualification and circuit-breaker behavior" at the pure gate seams `_filter_swap` (lnplus_swaps.py:313) and `_check_participants` (:352).

- [ ] **Step 1: Write the test module**

Create `tests/golden/test_golden_lnplus_gates.py`:

```python
"""Golden: LN+ SwapEvaluator qualification gates (fill state, terms,
participant quality). Pure functions of (swap dict, cfg)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.lnplus_swaps import SwapEvaluator
from tests.golden.util import golden_check

OUR_ID = "02" + "f" * 64


def _cfg(**over):
    base = dict(
        planner_min_channel_sats=1_000_000,
        planner_max_channel_sats=10_000_000,
        lnplus_max_duration_months=6,
        lnplus_max_participants=5,
        lnplus_min_participants=3,
        lnplus_min_peer_positive_ratings=5,
        lnplus_min_peer_rank=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _evaluator(policy_manager=None):
    rpc = MagicMock()
    rpc.getinfo.return_value = {"id": OUR_ID}
    db = MagicMock()
    db.lnplus_get_peer.return_value = None
    planner = MagicMock()
    planner._score_candidate.return_value = 1.0
    return SwapEvaluator(
        MagicMock(), rpc, db, MagicMock(), MagicMock(), planner,
        MagicMock(), policy_manager=policy_manager,
    )


def _swap(**over):
    base = dict(
        status="pending",
        participant_waiting_for_count=1,
        capacity_sats=2_000_000,
        duration_months=3,
        participant_max_count=4,
        platform="any",
    )
    base.update(over)
    return base


FILTER_SCENARIOS = {
    "qualifying_swap_passes": {},
    "not_pending": {"status": "completed"},
    "not_last_slot": {"participant_waiting_for_count": 2},
    "below_min_capacity": {"capacity_sats": 500_000},
    "above_max_capacity": {"capacity_sats": 50_000_000},
    "duration_too_long": {"duration_months": 12},
    "too_many_participants": {"participant_max_count": 9},
    "dual_swap_rejected": {"participant_max_count": 2},
    "lnd_platform_rejected": {"platform": "lnd"},
}


@pytest.mark.parametrize("name", sorted(FILTER_SCENARIOS))
def test_golden_filter_swap(name):
    ev = _evaluator()
    result = ev._filter_swap(_swap(**FILTER_SCENARIOS[name]), _cfg())
    golden_check(f"lnplus/filter_{name}", {
        "overrides": FILTER_SCENARIOS[name],
        "rejection": result,
    })


def _participant(**over):
    base = dict(
        pubkey="02" + "b" * 64,
        cancelled=False, banned=False,
        address_1="1.2.3.4:9735", address_2=None,
        positive_ratings_count=20, negative_ratings_count=0,
        lnplus_rank_number=5,
    )
    base.update(over)
    return base


PARTICIPANT_SCENARIOS = {
    "good_ring_passes": [_participant()],
    "own_node_in_ring": [_participant(pubkey=OUR_ID)],
    "no_address_rejected": [_participant(address_1=None)],
    "low_ratings_rejected": [_participant(positive_ratings_count=1)],
    "rank_below_floor": [_participant(lnplus_rank_number=1)],
    "cancelled_peer_skipped": [
        _participant(cancelled=True, positive_ratings_count=0),
        _participant(pubkey="02" + "c" * 64),
    ],
}


@pytest.mark.parametrize("name", sorted(PARTICIPANT_SCENARIOS))
def test_golden_check_participants(name):
    ev = _evaluator()
    swap = _swap(participants=PARTICIPANT_SCENARIOS[name])
    result = ev._check_participants(swap, _cfg())
    golden_check(f"lnplus/participants_{name}", {"rejection": result})


def test_operator_ban_vetoes_and_fails_closed():
    """Non-golden anchor (pins the 2026-07-12 ban-gate behavior)."""
    banned_pm = MagicMock()
    banned_pm.is_peer_banned.return_value = True
    ev = _evaluator(policy_manager=banned_pm)
    res = ev._check_participants(_swap(participants=[_participant()]), _cfg())
    assert res is not None and "operator-banned" in res

    broken_pm = MagicMock()
    broken_pm.is_peer_banned.side_effect = RuntimeError("db gone")
    ev = _evaluator(policy_manager=broken_pm)
    res = ev._check_participants(_swap(participants=[_participant()]), _cfg())
    assert res is not None and "fail closed" in res
```

- [ ] **Step 2: Run to verify failure mode**

Run: `python3 -m pytest tests/golden/test_golden_lnplus_gates.py -v`
Expected: the ban anchor PASSES (this behavior was test-covered on 2026-07-12 in `tests/test_lnplus_swaps.py` — if constructor arity differs, mirror that file's evaluator helper); goldens FAIL on missing fixtures.

- [ ] **Step 3: Record and review**

Run: `GOLDEN_UPDATE=1 python3 -m pytest tests/golden/test_golden_lnplus_gates.py -q`
Review: `qualifying_swap_passes` → `null`; every rejection string must start with its gate name (`fill_state:` / `terms:` / `peer_quality:` / `own_node:`); `dual_swap_rejected` must cite the 3-participant floor (operator ruling D-3); `cancelled_peer_skipped` → `null` (cancelled participant must not veto).

- [ ] **Step 4: Verify green and run the full suite**

Run: `python3 -m pytest tests/golden/ -q && python3 -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider 2>&1 | tail -2`
Expected: all golden modules pass; full suite green.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden_lnplus_gates.py tests/golden/fixtures/lnplus/
git commit -m "test(refactor): golden fixtures for LN+ qualification gates"
```

---

### Task 14: Portability-hazard inventory

**Files:**
- Create: `docs/refactor/phase0/portability-hazards.md`

**Interfaces:**
- Consumes: hazards observed while writing Tasks 7–13 (wall-clock in classification, Thompson sampling randomness, etc.).
- Produces: the Workstream J risk register that Phase 1's determinism work consumes.

- [ ] **Step 1: Write the hazard inventory**

Write `docs/refactor/phase0/portability-hazards.md`:

```markdown
# Python-portability hazard inventory (baseline 5e8f747)

Hazards that block cross-language decision parity (refactor.md
Workstream J / invariant 14). Counts from
`grep -c` sweeps; verify with the commands in each section.

## 1. Wall-clock reads in decision code (`time.time(`)

| Module | count | Notes |
|---|---|---|
| modules/database.py | 82 | timestamps for spend windows — AUTHORIZATION-RELEVANT (budget window boundaries) |
| modules/fee_controller.py | 37 | cycle cadence, cooldowns, zero-flow streaks |
| cl-revenue-ops.py | 29 | loop scheduling, heartbeats |
| modules/profitability_analyzer.py | 24 | recency vs last_routed in classification |
| modules/lnplus_swaps.py | 11 | breaker windows, deadline math |
| modules/policy_manager.py | 10 | — |
| capacity_planner / flow_analysis / rebalance_engine_v2 | 9 each | — |
| rebalancer.py 6 · boltz_manager.py 4 · data_service.py 3 · others ≤2 | | |

Verify: `grep -c "time.time(" modules/*.py cl-revenue-ops.py`
Refactor rule (J3): policies receive cycle time; direct reads must move
to snapshot/cycle context.

## 2. Randomness

- Loop-interval jitter: `random.randint` in every background loop tail
  (`cl-revenue-ops.py:3051,3091,3131,3191,3231,3289,3361`) — scheduling
  only, NOT decision-relevant; may remain.
- `modules/fee_controller.py` (5 uses): **Gaussian-Thompson posterior
  sampling inside `_adjust_channel_fee`** — DECISION-RELEVANT
  unseeded randomness. This is why Phase 0 goldens pin the damping/
  floor/deadband stages, not the raw DTS target. J3 requires seed
  injection recorded in cycle evidence.
- `uuid` in `modules/boltz_manager.py` (2) — swap/reservation IDs;
  idempotency keys must become deterministic (J3).

## 3. Binary floating point in authoritative paths (J2 violations)

- ROI/marginal_roi, confidence, kalman ratios, multipliers are Python
  floats end-to-end (e.g. `ChannelProfitability.marginal_roi`,
  `_close_protection_reason` ROI thresholds -30.0/-50.0,
  planner `score` floats rounded to 6dp).
- Fee floor math divides float chain costs into ppm
  (`_calculate_floor`).
- Budget/spend amounts ARE integer sats/msat (good), but window
  boundary comparisons mix float `time.time()`.

## 4. Unordered iteration feeding results

Sweep for dict/set iteration whose order can reach decisions or
serialized output: `grep -n "\.items()\|\.values()\|set(" modules/rebalance_planner_v2.py modules/capacity_planner.py modules/fee_controller.py | wc -l` — then
inspect ranking/serialization sites specifically. Record each confirmed
site here with file:line. (Known-safe: planner sorts candidates by
score; confirm tie-breaking is total — J3 requires a documented
tie-break sequence.)

## 5. Other hazards

- Untyped dicts cross every subsystem boundary (channel_info, swap
  dicts, plan dicts, RPC results) — the intent/schema work is the fix.
- Enum serialization: `ChannelRole`/`ProfitabilityClass`/`ChannelState`
  are Python enums; wire values must become stable strings (J1).
- Duck-typed `getattr(prof, 'role_30d', None)` fallbacks in
  capacity_planner tolerate legacy objects — schema versioning replaces
  this.
- `schema_version` DB table is write-only (no version gate) — replay/
  migration tooling cannot rely on it.
- Mutable module-level globals in `cl-revenue-ops.py` (managers wired at
  init; tests monkeypatch them) — Workstream H cycle context replaces.
- Boltz adapter shells out to `boltzcli` (subprocess) — outcome parsing
  is text/JSON from CLI; unknown-outcome handling must go through
  reconciliation (Workstream G).

## Hazards found while building the Phase 0 goldens

(Append here anything Tasks 7–13 uncovered, e.g. wall-clock freezing
needed in `_classify_channel`, config fields read via bare getattr with
inconsistent string/bool coercion in `_compute_dynamic_htlcmax_msat`.)
```

Fill §4's inspection results and the final section with actual findings before committing.

- [ ] **Step 2: Verify the counts**

Run: `grep -c "time.time(" modules/database.py modules/fee_controller.py cl-revenue-ops.py modules/profitability_analyzer.py`
Expected: 82/37/29/24 (fix the doc if drifted).

- [ ] **Step 3: Commit**

```bash
git add docs/refactor/phase0/portability-hazards.md
git commit -m "docs(refactor): Phase 0 portability-hazard inventory"
```

---

### Task 15: Draft wire-contract specification

**Files:**
- Create: `docs/refactor/phase0/wire-contract-spec.md`

**Interfaces:**
- Produces: the normative draft that Task 16's schema and Task 17's corpus layout instantiate; Workstream J1–J4 refine it in Phase 1.

- [ ] **Step 1: Write the draft spec**

Write `docs/refactor/phase0/wire-contract-spec.md`. This is a DRAFT for review (refactor.md Phase 0 deliverable 8) — decisions below are proposals with rationale, marked normative-candidate:

```markdown
# Draft wire-contract specification v0 (Phase 0 deliverable)

Status: DRAFT for operator review. Instantiates refactor.md Workstream
J1–J4 with concrete choices. Nothing here is enforced yet.

## Schema versioning (J1)

- Every canonical payload carries `schema_name` (string, e.g.
  "economic_snapshot") and `schema_version` (integer, starts at 0 while
  draft; 1 = first frozen).
- Encoding: JSON Schema draft 2020-12, one file per payload type at
  `schemas/<name>.v<version>.schema.json`.
- Unknown-field rule: readers MUST ignore unknown fields
  (`additionalProperties: true`) until version 1 freezes; version 1
  revisits per payload.
- Enum wire values: stable UPPER_SNAKE strings (`"INBOUND_GATEWAY"`),
  never ordinals or Python enum names implicitly.
- Change classes: backward-compatible (add optional field), forward-
  compatible (documented default), breaking (new schema_version +
  migration fixtures).

## Numeric and monetary semantics (J2)

- Money: integer millisatoshi. JSON representation: JSON number when
  |v| <= 2^53-1 is NOT trusted — canonical payloads encode msat as
  JSON integers and validators MUST reject non-integral or out-of-range
  values. Range: unsigned values in [0, 2^63-1] (checked u64 in Rust);
  signed P&L/deltas in [-(2^63), 2^63-1] (checked i64).
- Overflow: any checked-arithmetic failure in an authorization-relevant
  computation fails closed (reason code ARITHMETIC_OVERFLOW).
- Ratios/confidence/multipliers: scaled fixed-point integers,
  denominator 1_000_000 (field suffix `_ppm` for rates, `_micro` for
  generic ratios: confidence 0..1 → 0..1_000_000 micro). Binary floats
  are permitted ONLY in fields explicitly marked non-authoritative
  (suffix `_diag`).
- msat→sat reporting: floor division (`// 1000`) — matches current
  behavior (e.g. sourced_fee_30d_msat // 1000 in
  capacity_planner._close_protection_reason). Conformance vectors must
  encode this exactly.
- Division by zero / missing denominator: defined per field; default is
  "signal absent" (null + lower confidence), NEVER silent zero
  (refactor invariant 7). Precedent: marginal_roi with zero 30d cost
  and positive profit = 1.0 exactly (existing pinned behavior).
- Timestamps: integer unix seconds UTC (`_at` suffix); durations
  integer seconds.

## Determinism rules (J3)

- Canonical serialization for hashing: JSON with lexicographically
  sorted keys, no insignificant whitespace, UTF-8, integers only for
  authoritative numerics.
- Idempotency key: `sha256(canonical_json(envelope-subset))` where the
  subset is (intent type, target id, amount, snapshot id, budget
  bucket) — draft; finalize in Phase 1.
- Tie-break sequence (refactor.md J3): precedence class → requested
  priority → expected value → confidence → capital committed → stable
  target identifier → intent ID.
- Clock: cycle context supplies `cycle_time_at`; policies MUST NOT read
  wall clock (hazard inventory §1 lists current violations).
- Randomness: explicit seed in cycle context, recorded in ledger
  evidence (current violation: DTS posterior sampling, hazards §2).
- DB queries feeding decisions carry explicit ORDER BY.

## Reason-code catalog v0 (J4)

Seed catalog (code → owning layer → kind):

| Code | Layer | Kind |
|---|---|---|
| BUDGET_EXHAUSTED | governor | rejection |
| AUTHORITY_LEVEL_BLOCKED | governor | rejection |
| INTENT_STALE | arbiter | deferral |
| INTENT_SUPERSEDED | arbiter | rejection |
| CHANNEL_PROTECTED | governor | rejection |
| CONTRACT_OBLIGATION | arbiter | rejection |
| EV_BELOW_HOLD_MARGIN | policy | hold |
| INSUFFICIENT_CONFIDENCE | governor | hold |
| FEE_RAIL_CLAMPED | policy | clamp |
| COOLDOWN_ACTIVE | policy | hold |
| CONFLICT_CLOSE_REBALANCE | arbiter | rejection |
| EXTERNAL_CIRCUIT_BREAKER | executor | deferral |
| EXTERNAL_OUTCOME_UNKNOWN | reconciliation | unknown |
| ARITHMETIC_OVERFLOW | any | failure |
| SCHEMA_INVALID | any | failure |

Existing string reasons to map into the catalog (from goldens):
close-protection `KALMAN_LOW_CONFIDENCE`, `INBOUND_GATEWAY`,
`SOURCED_FEE_CONTRIBUTION`; LN+ gate prefixes `fill_state:`, `terms:`,
`peer_quality:`, `own_node:`; planner skip reasons
(`cooldown_active`, `source_ineligible`, ...); damping cap reasons
(`normal_cycle_delta_cap`, `wake_cycle_delta_cap`). Each code defines
required context fields in Phase 1.

## Domain wrappers (J2, Phase 1 implementation)

`Msat`, `Sat`, `Ppm`, `FixedRatio(micro)`, `UnixTime`, `IntentId`,
`ChannelId` (short-channel-id string form `NNNxNNNxN`), `PeerId`
(66-hex-char pubkey).
```

- [ ] **Step 2: Cross-check one rule against reality**

Run: `grep -n "// 1000" modules/capacity_planner.py | head -3`
Expected: the msat→sat floor-division sites exist as claimed. Adjust the spec if the codebase actually rounds differently anywhere that matters (check `modules/utils.py` `base_to_sats_ceil` — note in the spec that CEILING conversion exists for budget-conservatism paths and document where each rule applies).

- [ ] **Step 3: Commit**

```bash
git add docs/refactor/phase0/wire-contract-spec.md
git commit -m "docs(refactor): Phase 0 draft wire-contract specification v0"
```

---

### Task 16: Proposed EconomicSnapshot schema and source mapping

**Files:**
- Create: `schemas/economic_snapshot.v0.schema.json`
- Create: `docs/refactor/phase0/snapshot-mapping.md`
- Create: `tests/test_schema_validity.py`

**Interfaces:**
- Consumes: wire-contract rules (Task 15).
- Produces: the schema file Task 17's validator loads by `schema_name`+`schema_version`; the mapping doc Workstream A implements against.

- [ ] **Step 1: Write the failing schema test**

Create `tests/test_schema_validity.py`:

```python
"""Phase 0: schemas/ files are valid JSON Schema 2020-12 and the example
instance embedded in each schema validates against it."""
import json
import pathlib

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_DIR = pathlib.Path(__file__).resolve().parent.parent / "schemas"
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.schema.json"))


def test_schemas_exist():
    assert SCHEMA_FILES, "schemas/ must contain at least the v0 snapshot schema"


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_is_valid_and_example_validates(path):
    schema = json.loads(path.read_text())
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    for i, example in enumerate(schema.get("examples", [])):
        jsonschema.validate(example, schema)
    assert schema.get("examples"), f"{path.name} must embed >=1 example"
    assert schema["properties"]["schema_name"]["const"]
    assert schema["properties"]["schema_version"]["const"] == 0
```

Run: `python3 -m pytest tests/test_schema_validity.py -v`
Expected: FAIL — `schemas/` doesn't exist.

- [ ] **Step 2: Write the schema**

Create `schemas/economic_snapshot.v0.schema.json`. Field set from `refactor.md` Workstream A (lines 130–159), with wire rules from Task 15 (msat integers, micro ratios, stable enum strings, `additionalProperties: true` while draft). Structure:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "cl-revenue-ops/economic_snapshot.v0",
  "title": "EconomicSnapshot v0 (draft, Phase 0 proposal)",
  "type": "object",
  "required": ["schema_name", "schema_version", "snapshot_id",
               "observed_at", "evidence_window_seconds", "node", "channels"],
  "additionalProperties": true,
  "properties": {
    "schema_name": {"const": "economic_snapshot"},
    "schema_version": {"const": 0},
    "snapshot_id": {"type": "string", "minLength": 1},
    "observed_at": {"type": "integer", "minimum": 0},
    "evidence_window_seconds": {"type": "integer", "minimum": 0},
    "node": {
      "type": "object",
      "required": ["total_local_msat", "total_remote_msat",
                   "receivable_objective_msat", "onchain_confirmed_msat",
                   "reserved_msat", "daily_budget", "pending_operations",
                   "external_obligations"],
      "additionalProperties": true,
      "properties": {
        "total_local_msat": {"type": "integer", "minimum": 0},
        "total_remote_msat": {"type": "integer", "minimum": 0},
        "receivable_objective_msat": {"type": "integer", "minimum": 0},
        "onchain_confirmed_msat": {"type": "integer", "minimum": 0},
        "reserved_msat": {"type": "integer", "minimum": 0},
        "daily_budget": {
          "type": "object",
          "required": ["cap_msat", "reserved_msat", "spent_msat"],
          "properties": {
            "cap_msat": {"type": "integer", "minimum": 0},
            "reserved_msat": {"type": "integer", "minimum": 0},
            "spent_msat": {"type": "integer", "minimum": 0}
          }
        },
        "pending_operations": {"type": "array", "items": {"type": "object"}},
        "external_obligations": {"type": "array", "items": {"type": "object"}}
      }
    },
    "channels": {
      "type": "array",
      "items": {"$ref": "#/$defs/channel_snapshot"}
    }
  },
  "$defs": {
    "channel_snapshot": {
      "type": "object",
      "required": ["channel_id", "peer_id", "capacity_msat", "local_msat",
                   "remote_msat", "spendable_msat", "receivable_msat",
                   "exit_revenue_msat", "sourced_value_msat",
                   "rebalance_cost_msat", "capital_cost_msat",
                   "net_value_msat", "exit_volume_msat",
                   "sourced_volume_msat", "forward_count",
                   "sourced_forward_count", "role", "lifecycle",
                   "protections", "confidence_micro"],
      "additionalProperties": true,
      "properties": {
        "channel_id": {"type": "string", "pattern": "^[0-9]+x[0-9]+x[0-9]+$"},
        "peer_id": {"type": "string", "pattern": "^0[23][0-9a-f]{64}$"},
        "capacity_msat": {"type": "integer", "minimum": 0},
        "local_msat": {"type": "integer", "minimum": 0},
        "remote_msat": {"type": "integer", "minimum": 0},
        "spendable_msat": {"type": "integer", "minimum": 0},
        "receivable_msat": {"type": "integer", "minimum": 0},
        "exit_revenue_msat": {"type": "integer", "minimum": 0},
        "sourced_value_msat": {"type": "integer", "minimum": 0},
        "rebalance_cost_msat": {"type": "integer", "minimum": 0},
        "capital_cost_msat": {"type": "integer", "minimum": 0},
        "net_value_msat": {"type": "integer"},
        "exit_volume_msat": {"type": "integer", "minimum": 0},
        "sourced_volume_msat": {"type": "integer", "minimum": 0},
        "forward_count": {"type": "integer", "minimum": 0},
        "sourced_forward_count": {"type": "integer", "minimum": 0},
        "role": {"enum": ["SOURCE", "SINK", "ROUTER", "BALANCED",
                           "INBOUND_GATEWAY", "UNKNOWN"]},
        "lifecycle": {"enum": ["CANDIDATE", "OPENING", "EVALUATING",
                                "PRODUCTIVE", "PROTECTED",
                                "UNDERPERFORMING", "RECYCLING", "CLOSING"]},
        "protections": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["reason", "owner", "expires_at"],
            "properties": {
              "reason": {"type": "string"},
              "owner": {"type": "string"},
              "expires_at": {"type": ["integer", "null"]}
            }
          }
        },
        "confidence_micro": {"type": "integer", "minimum": 0,
                              "maximum": 1000000}
      }
    }
  },
  "examples": [
    {
      "schema_name": "economic_snapshot",
      "schema_version": 0,
      "snapshot_id": "cycle-000001",
      "observed_at": 1752300000,
      "evidence_window_seconds": 2592000,
      "node": {
        "total_local_msat": 500000000000,
        "total_remote_msat": 300000000000,
        "receivable_objective_msat": 400000000000,
        "onchain_confirmed_msat": 100000000000,
        "reserved_msat": 5000000000,
        "daily_budget": {"cap_msat": 10000000000,
                          "reserved_msat": 2000000000,
                          "spent_msat": 1000000000},
        "pending_operations": [],
        "external_obligations": []
      },
      "channels": [
        {
          "channel_id": "123x456x0",
          "peer_id": "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "capacity_msat": 2000000000000,
          "local_msat": 1200000000000,
          "remote_msat": 800000000000,
          "spendable_msat": 1180000000000,
          "receivable_msat": 780000000000,
          "exit_revenue_msat": 2000000,
          "sourced_value_msat": 1500000,
          "rebalance_cost_msat": 800000,
          "capital_cost_msat": 400000,
          "net_value_msat": 2300000,
          "exit_volume_msat": 900000000000,
          "sourced_volume_msat": 700000000000,
          "forward_count": 142,
          "sourced_forward_count": 96,
          "role": "ROUTER",
          "lifecycle": "PRODUCTIVE",
          "protections": [
            {"reason": "lnplus_contract", "owner": "lnplus",
             "expires_at": 1755000000}
          ],
          "confidence_micro": 850000
        }
      ]
    }
  ]
}
```

Fix the example `peer_id` to be exactly 66 hex chars (02 + 64) — count before committing.

- [ ] **Step 3: Write the source mapping doc**

Write `docs/refactor/phase0/snapshot-mapping.md` — every schema field mapped to today's data source:

```markdown
# EconomicSnapshot v0 → current data sources

| Field | Current source | Notes |
|---|---|---|
| channel_id/peer_id/capacity_msat/local_msat(=to_us)/spendable_msat/receivable_msat | `listpeerchannels` via data_service cached reads | remote = capacity - local |
| exit_revenue_msat | profitability `ChannelRevenue.fees_earned_msat` (db `forwards`/lifetime_aggregates) | |
| sourced_value_msat | `ChannelRevenue.sourced_fee_contribution_msat` | 30d window: `sourced_fee_30d_msat` |
| rebalance_cost_msat | `ChannelCosts.rebalance_cost_sats*1000` (db `rebalance_costs`) | plus effective_rebalance_cost |
| capital_cost_msat | `ChannelCosts.open_cost_sats*1000` + capital-efficiency carry | |
| net_value_msat | `ChannelProfitability.net_profit_sats*1000` | today computed in sats — precision note |
| exit/sourced_volume_msat, forward counts | `ChannelRevenue.volume_routed_msat`, `forward_count`, `sourced_forward_count_30d` etc. | |
| role | UNIFICATION REQUIRED: profitability `ChannelRole`/`role_30d` vs flow_analysis `ChannelState` (two authorities today — decision-owners.md) | snapshot role = the future single authority |
| lifecycle | DOES NOT EXIST today — derived: dead_capital_stage → RECYCLING/CLOSING, planner_actions → OPENING, lnplus/no_close tags → PROTECTED | Workstream F5 |
| protections | policy tags (peer_policies), lnplus contract windows, hot_channel_protection_overrides | become owned, expiring Protection records |
| confidence_micro | flow_analysis kalman confidence (float 0..1 → micro) | |
| node.daily_budget | db get_budget_status / spend ledger | four budget systems today — Workstream D |
| node.external_obligations | lnplus_swaps table + boltz journal in-flight | invariant 6 |
| observed_at/evidence_window | NEW: cycle context (J3) | replaces scattered time.time() |
```

- [ ] **Step 4: Run to verify green**

Run: `python3 -m pytest tests/test_schema_validity.py -v`
Expected: all pass (schema meta-valid, example validates, name/version consts present).

- [ ] **Step 5: Commit**

```bash
git add schemas/ docs/refactor/phase0/snapshot-mapping.md tests/test_schema_validity.py
git commit -m "feat(refactor): proposed EconomicSnapshot v0 schema with source mapping (Phase 0, draft)"
```

---

### Task 17: Conformance-corpus layout and standalone validator

**Files:**
- Create: `tools/conformance/validate_fixtures.py`
- Create: `tests/conformance/README.md`
- Create: `tests/conformance/scenarios/routine-cycle-smoke/snapshot.json`
- Create: `tests/test_conformance_validator.py`

**Interfaces:**
- Consumes: `schemas/*.schema.json` (Task 16).
- Produces: the corpus layout Rust later consumes unchanged (refactor.md J5); `validate_fixtures.py` runnable with zero `cl_revenue_ops` imports (J acceptance: "standalone schema validator can validate all canonical fixture payloads without importing cl_revenue_ops Python modules").

- [ ] **Step 1: Write the failing test**

Create `tests/test_conformance_validator.py`:

```python
"""Phase 0: the standalone conformance validator accepts the seed corpus
and rejects invalid payloads. The validator must import nothing from
modules/ or cl-revenue-ops.py (cross-language portability requirement)."""
import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("jsonschema")

REPO = pathlib.Path(__file__).resolve().parent.parent
VALIDATOR = REPO / "tools" / "conformance" / "validate_fixtures.py"
CORPUS = REPO / "tests" / "conformance" / "scenarios"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        capture_output=True, text=True, cwd=REPO,
    )


def test_validator_passes_seed_corpus():
    res = _run(str(CORPUS))
    assert res.returncode == 0, res.stdout + res.stderr


def test_validator_rejects_bad_payload(tmp_path):
    bad = tmp_path / "scenarios" / "broken" 
    bad.mkdir(parents=True)
    (bad / "snapshot.json").write_text(json.dumps({
        "schema_name": "economic_snapshot",
        "schema_version": 0,
        "snapshot_id": "x",
        # missing required fields entirely
    }))
    res = _run(str(tmp_path / "scenarios"))
    assert res.returncode != 0
    assert "broken" in res.stdout + res.stderr


def test_validator_imports_no_plugin_code():
    text = VALIDATOR.read_text()
    assert "import modules" not in text
    assert "from modules" not in text
    assert "cl-revenue-ops" not in text
```

Run: `python3 -m pytest tests/test_conformance_validator.py -v`
Expected: FAIL — validator and corpus don't exist.

- [ ] **Step 2: Implement the validator**

Create `tools/conformance/validate_fixtures.py`:

```python
#!/usr/bin/env python3
"""Standalone conformance-fixture validator (refactor Phase 0, J5).

Validates every *.json under the given scenarios directory against the
schema named by its `schema_name`/`schema_version` fields, loaded from
schemas/. Deliberately imports NOTHING from the plugin: a Rust
implementation must be able to reimplement this file from the schemas
alone.

Usage: python3 tools/conformance/validate_fixtures.py [scenarios_dir]
Exit 0 = all valid; 1 = any invalid/unknown-schema payload.
"""
import json
import pathlib
import sys

try:
    import jsonschema
except ImportError:
    sys.exit("jsonschema is required: pip install jsonschema")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = REPO / "schemas"


def load_schemas():
    schemas = {}
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text())
        key = (schema["properties"]["schema_name"]["const"],
               schema["properties"]["schema_version"]["const"])
        schemas[key] = schema
    return schemas


def main(scenarios_dir: str) -> int:
    schemas = load_schemas()
    failures = 0
    payloads = sorted(pathlib.Path(scenarios_dir).rglob("*.json"))
    if not payloads:
        print(f"no payloads found under {scenarios_dir}")
        return 1
    for payload_path in payloads:
        try:
            payload = json.loads(payload_path.read_text())
            key = (payload.get("schema_name"),
                   payload.get("schema_version"))
            if key not in schemas:
                raise ValueError(f"unknown schema {key}")
            jsonschema.validate(payload, schemas[key])
            print(f"OK   {payload_path}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {payload_path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "tests" / "conformance" / "scenarios")
    sys.exit(main(target))
```

- [ ] **Step 3: Create the corpus layout and seed scenario**

Create `tests/conformance/README.md`:

```markdown
# Cross-language conformance corpus (refactor Phase 0 layout, J5)

Layout per scenario (refactor.md lines 700–709):

    scenarios/<scenario-name>/
      snapshot.json                # required from Phase 1
      config.json                  # resolved cycle config
      cycle-context.json           # injected clock/seed
      expected-intents.json        # added as Workstream B lands
      expected-arbitration.json    # Workstream C
      expected-authorizations.json # Workstream D
      expected-projections.json    # Workstream E/I

Rules:
- Every payload declares `schema_name` + `schema_version` and validates
  against `schemas/` via `tools/conformance/validate_fixtures.py`
  (standalone; no plugin imports) — run in CI from Phase 1 onward.
- No live credentials, tokens, or unsanitized production identifiers.
- Comparison contract (Phase 1+): exact for integers, enums, ordering,
  reason codes, lifecycle, authorization outcomes; human-readable text
  and `_diag` fields excluded.
- Phase 0 ships only the layout + one smoke scenario; production-derived
  scenarios are captured during Phase 1 golden-parity work.
```

Create `tests/conformance/scenarios/routine-cycle-smoke/snapshot.json` — copy the `examples[0]` object verbatim from `schemas/economic_snapshot.v0.schema.json` into its own file.

- [ ] **Step 4: Run to verify green**

Run: `python3 -m pytest tests/test_conformance_validator.py -v && python3 tools/conformance/validate_fixtures.py`
Expected: 3 tests pass; validator prints `OK ... routine-cycle-smoke/snapshot.json`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add tools/conformance/ tests/conformance/ tests/test_conformance_validator.py
git commit -m "feat(refactor): conformance corpus layout and standalone validator (Phase 0)"
```

---

### Task 18: PR sequence, contradictions, and review packet

**Files:**
- Create: `docs/refactor/phase0/pr-sequence.md`
- Modify: `docs/refactor/phase0/README.md` (fill §Contradictions, mark deliverables complete)

**Interfaces:**
- Consumes: everything above.
- Produces: the Phase 0 review packet for Sat; Phase 1 does not start until this is reviewed (refactor.md line 1068).

- [ ] **Step 1: Write the PR sequence**

Write `docs/refactor/phase0/pr-sequence.md` — the spec's 19-PR sequence (refactor.md lines 993–1011) adjusted to repository reality found in Phase 0:

```markdown
# Proposed implementation PR sequence (adjusted to repo reality)

Adjustments from the spec's suggested sequence, with evidence:

1. PR-1 (ADR + mutation inventory) is COMPLETE as Phase 0 (this
   directory + pin tests) — no separate PR needed.
2. The CLN execution adapter should GROW FROM `modules/data_service.py`
   (already a partial adapter with 21 mutating verbs behind typed
   methods — mutation-paths.md) rather than be built new. Rebalance
   native executor, router v3, LN+ and planner fallback bypasses are
   the migration checklist.
3. The budget-reservation work (spec PR-8) must unify FOUR existing
   implementations (mutation-paths.md §budget), starting from the
   generic spend ledger (`reserve_spend`, already atomic
   BEGIN IMMEDIATE) — the other three become callers.
4. Classification unification (spec PR-13 lifecycle) must reconcile TWO
   live authorities (flow ChannelState vs profitability
   ChannelRole/role_30d — decision-owners.md).
5. Boltz decision logic (module-level in cl-revenue-ops.py) moves into
   the adapter boundary in the Boltz PR, not before.

Sequence (each PR: scope, non-scope, invariants, tests, rollback,
compat evidence — per refactor.md PR requirements):

 1. Canonical snapshot types + parity tests  (spec PR-2; golden
    fixtures from tests/golden/ are the parity oracle)
 2. Typed intents + structured explanations  (spec PR-3)
 3. Versioned schemas v1 freeze + reason codes + fixture harness in CI
    (spec PR-4; builds on schemas/ + tools/conformance/)
 4. Checked Msat/fixed-point types + cycle context (clock/seed
    injection)  (spec PR-5; kills hazards §1–§3 at decision seams)
 5. Append-only ledger schema + replay tests  (spec PR-6)
 6. Governor facade delegating to current checks  (spec PR-7)
 7. Durable reservations unification (4→1)  (spec PR-8)
 8. Intent arbiter in shadow mode  (spec PR-9)
 9. Fee policy migration  (spec PR-10)
10. Admission-control (htlc_max) extraction  (spec PR-11; seam already
    isolated: _compute_dynamic_htlcmax_msat)
11. Unified rebalancer migration  (spec PR-12; RebalancePlanner.plan is
    already pure — engine/executor consolidation is the work)
12. Lifecycle/protection ownership  (spec PR-13)
13. Capital planner migration  (spec PR-14)
14. Boltz adapter isolation  (spec PR-15)
15. LN+ adapter isolation  (spec PR-16)
16. Authority levels + risk profiles  (spec PR-17)
17. Legacy-path removal + docs  (spec PR-18)
18. Optional Rust shadow prototype  (spec PR-19; gated on frozen v1
    contracts)
```

- [ ] **Step 2: Fill the contradictions section**

Edit `docs/refactor/phase0/README.md` §Contradictions with at least these (verified during Tasks 2–17; add any others found):

```markdown
1. refactor.md assumes no central execution adapter exists; the repo
   already has one growing in `modules/data_service.py`. Smallest
   correction: Workstream G adopts data_service as the CLN adapter
   seed instead of creating a parallel module.
2. refactor.md's suggested `modules/core|policies|executors|projections`
   layout conflicts with the flat modules/ convention and 200-file test
   suite import paths. Smallest correction: introduce packages only at
   ownership-transition PRs (the spec itself allows this, line 93).
3. Boltz "API/authentication" (Workstream G) is actually a boltzcli
   subprocess, not HTTP. The adapter isolates subprocess + JSON-text
   parsing instead of HTTP formats.
4. The spec's phase-0 fixture capture from production assumed a fleet;
   production is one node (lnnode) since 2026-07-11. Golden fixtures
   are synthetic + code-derived in Phase 0; production-derived
   scenarios land with the Phase 1 validation pipeline
   (docs/plans/2026-04-23-production-revenue-validation-automation.md
   pipeline exists and needs only single-node cleanup).
5. `schema_version` table is write-only by operator ruling DD9/MIG-3 —
   ledger/migration tooling (Workstream E) must carry its own version
   gate rather than rely on the DB one.
```

Also update the README deliverable table statuses and note the final suite count.

- [ ] **Step 3: Full-suite verification**

Run: `python3 -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider 2>&1 | tail -3`
Expected: **> 3114 passed** (baseline plus all new pin/golden/schema/conformance tests), 1 skipped, 0 failed. Record the final counts in `docs/refactor/phase0/baseline.md` under a "Phase 0 exit" line.

- [ ] **Step 4: Commit**

```bash
git add docs/refactor/phase0/
git commit -m "docs(refactor): Phase 0 PR sequence, contradictions, and review packet"
```

- [ ] **Step 5: Prepare the review handoff**

Phase 0 exit gate (refactor.md line 772): "Every production mutation path and public contract is documented, and golden fixtures cover the principal decision classes." Present `docs/refactor/phase0/README.md` to Sat for review. **Do not begin Phase 1 without sign-off.**

---

## Verification checklist (run after all tasks)

1. `python3 -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider` — green, ≥ baseline count.
2. `python3 tools/conformance/validate_fixtures.py` — exit 0.
3. `git diff 5e8f747 --stat -- modules/ cl-revenue-ops.py` — **EMPTY** (Phase 0 changes zero production code; only docs/, tests/, tools/, schemas/).
4. Every doc in `docs/refactor/phase0/` contains no "TBD"/"enumerate here" residue: `grep -rn "TBD\|paste the\|Enumerate current\|Fill in" docs/refactor/phase0/` returns nothing.
