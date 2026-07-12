# Refactor Phase 1 — Wiring Tranche Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining Phase 1 bullets that require touching production files — shadow emission of typed intents from live fee decisions into the append-only ledger, and an on-demand canonical-snapshot preview RPC — while preserving every existing behavior and keeping all shadow paths fail-open.

**Architecture:** One new module (`modules/econ_shadow.py`) holds ALL shadow logic; `cl-revenue-ops.py` gains only three small, exception-guarded touchpoints (init construction, one call in the fee-cycle tail, one read-only RPC). The snapshot preview is built on demand inside the RPC handler from existing caches — no background cost. The intent ledger lives in its own sqlite file (`econ_ledger.db` beside `revenue_ops.db`); the production DB schema is untouched. Everything is gated by a new runtime config flag `econ_shadow_enabled`, **default False** — the deployed node's behavior is bit-identical until Sat flips the flag.

**Tech Stack:** existing plugin conventions; `modules/econ_*` foundations from the previous tranche.

## Global Constraints

- Phase 1 exit gate still holds: golden parity (fee controller, planner, etc. UNTOUCHED); no new component gains live authority — shadow records and previews only.
- Fail-open rule (J7 spirit): every shadow/preview code path is wrapped so an exception logs at `debug`/`warn` and returns None/0 — it must never break a fee cycle or an RPC.
- No behavior change with the flag off (default): the only unconditional additions are the flag itself, an RPC that reports "disabled", and an unused global.
- Pin tests are updated IN THE SAME COMMIT as the surfaces they pin (RPC surface 64→65; compatibility catalog config count 131→132/48→49).
- AGENTS.md: the new RPC is read-only; tests never trigger action RPCs.
- Full suite green after every task (baseline this tranche: 3300 passed).

---

### Task 1: Shadow module — `modules/econ_shadow.py`

**Files:** Create `modules/econ_shadow.py`, `tests/test_econ_shadow.py`.

**Interfaces:**
- `EconShadow(plugin, config, ledger_path: Optional[str] = None)` — resolves default ledger path as `<dir of config.db_path>/econ_ledger.db` (expanduser); ledger created LAZILY on first record; any ledger failure disables recording (`self._ledger_failed`) and logs once at warn.
- `enabled() -> bool` — reads `econ_shadow_enabled` from `config.snapshot()` (getattr default False, accepts bool or "true"/"1"/"yes" strings like other flags).
- `record_fee_intents(adjustments: list, now: int) -> int` — maps each `FeeAdjustment` (duck-typed: channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason_code) to a `SET_FEE` `IntentEnvelope`: `snapshot_id=f"fee-cycle-{now}"`, `created_at=now`, `expires_at=now+3600`, `target=channel_id`, `amount_msat=None`, `expected_benefit_msat=SignedMsat(0)`, `max_cost_msat=Msat(0)`, `capital_committed_msat=Msat(0)`, `confidence_micro=Micro(0)`, `reason_codes=()`, `explanation=Explanation("fee_adjustment", (("old_fee_ppm", old), ("new_fee_ppm", new), ("controller_reason_code", rc)))`, `priority=50`, `budget_bucket="fees"`, `origin_policy="fee_controller_shadow"`, `reversible=True`; appends one `intent_proposed` ledger event per intent. Returns count recorded; returns 0 (and never raises) when disabled, on ledger failure, or on a malformed adjustment (per-adjustment guard; one bad adjustment must not drop the rest).
- `build_snapshot_preview(*, channels: list[dict], profitability: dict, budget: dict, now: int, receivable_ratio_target: float = 0.0) -> tuple[Optional[dict], list[str]]` — assembles an `EconomicSnapshot` via `build_channel_snapshot` per channel (prof matched by short_channel_id; `role` from `prof.role_30d.name` when available else "UNKNOWN"; `lifecycle="PRODUCTIVE"`), node totals summed from channels, `daily_budget` from the budget dict (keys `cap_sats`/`reserved_sats`/`spent_sats`, msat via ×1000, missing→0), returns `(to_wire(snapshot), approximations)` where `approximations` names every placeholder field (e.g. `lifecycle`, `confidence_micro`, `onchain_confirmed_msat`, `sourced_volume_msat`) — missing evidence is labeled, never silently invented (invariant 7). Channels that fail to map are skipped and named in approximations. Returns `(None, [error])` on total failure.
- `intents_recorded_total: int` property (session counter for the RPC).

- [ ] Write `tests/test_econ_shadow.py` first: disabled flag → `record_fee_intents` returns 0 and creates no ledger file; enabled → records N events (read back via `EconLedger(path).events()`, event_type `intent_proposed`, idempotency stable across two identical calls with same `now` — same keys, ledger gets duplicate events which replay tolerates); one malformed adjustment among two valid → returns 2; ledger-path unwritable (`/nonexistent-dir/x.db`) → returns 0, no raise, warn logged once; `build_snapshot_preview` with two channels + one matching prof → wire dict validates against `schemas/economic_snapshot.v0.schema.json` (importorskip jsonschema), totals correct, approximations non-empty; preview with a channel missing `short_channel_id` → channel skipped + named; preview total failure (channels=None) → `(None, [...])`.
- [ ] Run FAIL → implement → run green → full suite → commit `feat(refactor): econ shadow module — fee-intent recording and snapshot preview (Phase 1 wiring)`.

### Task 2: Config flag `econ_shadow_enabled`

**Files:** Modify `modules/config.py` (dataclass field + `PUBLIC_RUNTIME_KEYS` entry), `docs/refactor/phase0/compatibility-catalog.md` (counts + tables); Create test in `tests/test_econ_shadow.py` (append).

- [ ] Add field `econ_shadow_enabled: bool = False` to the `Config` dataclass adjacent to other feature flags, with a comment citing this plan; add `'econ_shadow_enabled'` to `PUBLIC_RUNTIME_KEYS` (so Sat can flip it live via `revenue-config set`).
- [ ] Append test: `from modules.config import Config, PUBLIC_RUNTIME_KEYS`; `Config().econ_shadow_enabled is False`; `'econ_shadow_enabled' in PUBLIC_RUNTIME_KEYS`.
- [ ] Update compatibility-catalog.md: 132 fields / 49 runtime keys, add the row to both tables.
- [ ] Full suite green (config tests, `revenue-config` param validation matrix must still pass) → commit `feat(refactor): econ_shadow_enabled runtime flag (default off)`.

### Task 3: Plugin wiring + `revenue-econ-snapshot` RPC

**Files:** Modify `cl-revenue-ops.py` (3 touchpoints), `tests/test_rpc_surface_inventory.py` (65 methods), `docs/refactor/phase0/compatibility-catalog.md` (RPC note); Create `tests/test_econ_shadow_wiring.py`.

- [ ] Touchpoint 1 — global + init: `econ_shadow = None` near the other manager globals (~line 1013); in init after `fee_controller` construction (~line 2898): guarded `econ_shadow = EconShadow(safe_plugin, config)` (import at top with other module imports; construction failure logs warn and leaves None).
- [ ] Touchpoint 2 — fee-cycle tail: in `run_fee_adjustment()` immediately after `adjustments = fee_controller.adjust_all_fees()` (line ~3493), add:
  ```python
  # Phase 1 shadow (docs/planning/2026-07-12-refactor-phase1-wiring.md):
  # record the cycle's decisions as typed intents. Fail-open by contract.
  try:
      if econ_shadow is not None and econ_shadow.enabled():
          econ_shadow.record_fee_intents(adjustments, int(time.time()))
  except Exception as _shadow_err:
      plugin.log(f"econ shadow skipped: {_shadow_err}", level='debug')
  ```
- [ ] Touchpoint 3 — RPC: `@plugin.method("revenue-econ-snapshot")` read-only handler: returns `{"enabled": false}` when shadow off/None; else gathers `channels` via `data_service.listpeerchannels()`-equivalent cached read (use the same accessor other read paths use — find with `grep -n "def listpeerchannels\|get_channels" modules/data_service.py` and reuse), `profitability_analyzer.analyze_all_channels(force=False)`, `database.get_budget_status(...)` (guarded, `{}` on failure), calls `build_snapshot_preview`, returns `{"enabled": true, "snapshot": wire|None, "approximations": [...], "intents_recorded_total": n}`. Entire handler body guarded → error string in response, never an exception to the RPC layer.
- [ ] Update `tests/test_rpc_surface_inventory.py`: add `"revenue-econ-snapshot"`, count 65. Update compatibility-catalog.md (new read-only diagnostic RPC, no compatibility promise yet — internal preview).
- [ ] Write `tests/test_econ_shadow_wiring.py` using `load_plugin_module()` (pattern: `tests/test_boltz_auto_cycle_dry_run.py`): (a) `run_fee_adjustment` with `mod.econ_shadow = MagicMock(enabled=lambda: True)` and `mod.fee_controller` returning two adjustments → `record_fee_intents` called once with them; (b) shadow raising inside `record_fee_intents` → `run_fee_adjustment` still completes (returns adjustments, no raise); (c) `mod.econ_shadow = None` → no crash; (d) RPC handler with shadow None → `{"enabled": False}`; (e) RPC with mocked shadow returning a preview → response contains snapshot + approximations.
- [ ] Full suite green → commit `feat(refactor): wire econ shadow into fee cycle and add revenue-econ-snapshot RPC`.

### Task 4: Docs, verification, report

**Files:** Modify `docs/refactor/phase0/README.md`, `docs/refactor/phase0/persistence-map.md` (econ_ledger.db file noted), `docs/refactor/phase0/mutation-paths.md` (shadow is read-only; note the new datastore-free design).

- [ ] Verification: full suite green; `git diff 5e8f747 --name-only -- modules/ | grep -v econ_ | grep -v "reason_codes\|cycle_context\|governor_facade"` shows ONLY `modules/config.py`; goldens untouched; with flag default False, `run_fee_adjustment` behavior identical (test b/c above prove the guard); validator exit 0.
- [ ] Update README tranche section; note rollout instructions for Sat (`lightning-cli revenue-config set econ_shadow_enabled true`, then `lightning-cli revenue-econ-snapshot`).
- [ ] Commit `docs(refactor): Phase 1 wiring tranche status and rollout notes`.

## Self-review

- Spec coverage: "ledger alongside legacy persistence" (fee intents recorded, own file), "observe mode emits the same intents live mode considered" (Workstream B acceptance — shadow records intents FROM live decisions), "shadow comparison where feasible" (comparison target doesn't exist until policies migrate; emission is the feasible half — documented). "RPC output from projections" begins with the preview RPC; converting existing RPCs to projections is Phase 2+ work, deferred deliberately.
- The only modified pre-existing files: `modules/config.py` (one field + one tuple entry) and `cl-revenue-ops.py` (three guarded touchpoints). Everything else new.
