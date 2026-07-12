# Refactor Phase 2 Pilot A — Generic-Spend Ledger Journaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the append-only econ ledger a complete, replay-verifiable journal of the ENTIRE generic spend path (reservations, settlements, releases — from operator RPCs and every automated spender that uses `Database.reserve_spend`), with restart and duplicate-callback proofs. This is Phase 2's "ledger authoritative for new actions" entry, piloted on spend-path #1 of the four budget implementations.

**Architecture:** `Database` gains an optional `spend_journal` attribute (default None → zero behavior change). Its three spend-lifecycle methods invoke guarded journal hooks ONLY after a successful state change, so the journal can never affect authorization outcomes. `EconShadow` implements the journal interface (same fail-open contract and `econ_shadow_enabled` gate already proven in production). Wiring is one line at plugin init. Replay of `econ_ledger.db` then reconstructs reservation/spend state, verified against the DB.

**Why journaling before authorize():** the governor facade's `authorize()` needs IntentEnvelopes; automated spenders don't produce them yet (policy migrations are Phase 3/4). Journaling at the `Database` choke point captures 100% of generic-spend traffic NOW and gives Phase 2's audit/replay guarantees; per-spender migration to `authorize()` follows in later tranches.

**Tech Stack:** existing foundations (`econ_ledger`, `econ_shadow`); sqlite; pytest.

## Global Constraints

- Journal hooks fire ONLY on successful state changes and are individually guarded — a raising journal must never fail a reservation, settle, or release (fail-open, proven by tests).
- With `econ_shadow_enabled=false` (or `spend_journal=None`): behavior bit-identical, no ledger file created.
- No RPC schema changes; no new RPCs; no new config keys; pin tests unaffected (no new tables in database.py — the ledger stays in its own file).
- Modified pre-existing files this tranche: `modules/database.py` (hooks), `cl-revenue-ops.py` (one wiring line). Everything else new/tests.
- Full suite green after every task (baseline this tranche: 3318 passed).
- Golden fixtures untouched.

---

### Task 1: Spend-journal methods on EconShadow

**Files:** Modify `modules/econ_shadow.py`; Modify `tests/test_econ_shadow.py` (append test class).

**Interfaces (consumed by Task 2 hooks):**
- `note_spend_reserved(reservation_id: str, amount_sats: int, category: str) -> None`
- `note_spend_settled(reservation_id: str, actual_spent_sats: int, category: str = "") -> None`
- `note_spend_released(reservation_id: str, reason: str = "released") -> None`

All: no-ops when disabled/ledger-failed; never raise; timestamps via `int(time.time())` (audit timestamps, not decision inputs — J3 applies to decisions); `intent_id = idempotency_key = reservation_id` (legacy paths have no intent envelope; documented); `cycle_id = f"spend-{category or 'generic'}"`. Ledger events: reserved → `budget_reserved` (`amounts={"reserved_msat": sats*1000}`); settled → `cost_recorded` (`amounts={"cost_msat": sats*1000}`) then `execution_succeeded`; released → `reservation_released` (`details={"reason": reason}`).

- [ ] Append `TestSpendJournal` to `tests/test_econ_shadow.py`: disabled → no events, no file; enabled → reserve/settle/release sequence produces the 4 events above in order and `EconLedger.replay()` shows `reserved_msat=={}`, `spent_msat=={rid: amount*1000}`, `terminal=={rid: "execution_succeeded"}`; released-only reservation replays to zero reserved with no spend; methods never raise on garbage inputs (`note_spend_reserved(None, "x", 3)` → silently skipped).
- [ ] Run FAIL → implement (mirror the fail-open pattern of `record_fee_intents`) → green → full suite → commit `feat(refactor): spend-journal methods on EconShadow (Phase 2 pilot)`.

### Task 2: Journal hooks in Database spend lifecycle

**Files:** Modify `modules/database.py` (4 hook sites + attribute), Create `tests/test_spend_journal_hooks.py`.

**Hook sites (each: after the successful state change, wrapped in its own try/except that logs at debug and continues):**
1. `Database.__init__`: `self.spend_journal = None` (+ one comment line).
2. `reserve_spend` (:3895): after `conn.execute("COMMIT")`/before `return True` → `note_spend_reserved(rid, amount, cat)`.
3. `mark_spend_reservation_spent` (:4019): where the settled `amount` is known and the update committed → `note_spend_settled(rid, amount, category-if-available)`.
4. `release_spend_reservation` (:4010) → `note_spend_released(rid)`; `release_spend_reservations` (:4191, bulk/stale) → one `note_spend_released(rid, reason="stale")` per released id.

- [ ] Write `tests/test_spend_journal_hooks.py` first: real `Database` on temp path + `initialize()`; `db.spend_journal = MagicMock()`; (a) successful reserve → `note_spend_reserved` called once with (rid, amount, category); FAILED reserve (over budget with `effective_budget_sats`) → NOT called; (b) settle → `note_spend_settled` with the actual settled sats; settle of unknown rid (returns False) → NOT called; (c) release → `note_spend_released`; bulk stale release of 2 → called twice with reason="stale"; (d) journal raising `RuntimeError` on every hook → all four DB operations still succeed and return their normal values (fail-open proof); (e) `spend_journal=None` (default) → all operations succeed (regression guard).
- [ ] Run FAIL → implement hooks → green → full suite (the 1,051-case RPC matrix and all spender tests must stay green — they construct `Database` without a journal) → commit `feat(refactor): guarded spend-journal hooks in Database lifecycle (Phase 2 pilot)`.

### Task 3: Wiring + restart/replay/duplicate proofs

**Files:** Modify `cl-revenue-ops.py` (one line after `econ_shadow` construction: `database.spend_journal = econ_shadow` inside the same guarded try), Create `tests/test_spend_replay.py`.

- [ ] Wiring edit + append a wiring test to `tests/test_econ_shadow_wiring.py`-style module or new file: `load_plugin_module` not needed — assert via reading the init source is brittle; instead unit-test the composition directly in `tests/test_spend_replay.py` (real Database + real EconShadow with enabled config + real EconLedger file).
- [ ] `tests/test_spend_replay.py`:
  - **End-to-end journal:** Database(temp)+initialize, `db.spend_journal = EconShadow(MagicMock(), enabled-config, ledger_path=tmp)`; reserve 3 sats → settle 2 → ledger replay: spent 2000 msat, reserved 0 (settle consumes), terminal execution_succeeded.
  - **Restart proof:** after the above, construct a FRESH `EconLedger` on the same file (simulated restart) → `replay()` state identical; and DB `get_budget_status` spent matches replay spent (sats↔msat).
  - **Duplicate settle callback:** second `mark_spend_reservation_spent(rid)` returns False (terminal guard) → no second `cost_recorded`, replay spend unchanged.
  - **Stale-release recovery:** reserve then bulk stale-release → replay reserved 0, no spend, `reservation_released` with reason stale present.
  - **Crash-window honesty:** journal disabled mid-sequence (flip config mock to false between reserve and settle) → ledger shows reserve without settle; replay reports the outstanding reservation — document in the test docstring that ledger-vs-DB divergence detection is the reconciliation job (next tranche), not silent repair.
- [ ] Run green → full suite → commit `feat(refactor): wire spend journal at init with restart/duplicate replay proofs (Phase 2 pilot)`.

### Task 4: Docs, verification, report

**Files:** Modify `docs/refactor/phase0/README.md` (Phase 2 pilot section), `docs/refactor/phase0/persistence-map.md` (journal coverage note), `docs/refactor/phase0/pr-sequence.md` (mark PR-5/8 progress).

- [ ] Verification: full suite; `git diff 5e8f747 --name-only` pre-existing set grows only by `modules/database.py`; validator exit 0; goldens untouched; flag-off bit-identical (Task 2 test e).
- [ ] Update docs incl. rollout note (already-enabled flag on lnnode means journaling starts on deploy of this tranche; same instant-rollback flag).
- [ ] Commit `docs(refactor): Phase 2 pilot A status`. Report to Sat; deployment is a separate explicit step.

## Self-review

- Phase 2 coverage: "ledger authoritative for new actions" — journaling ALL generic-spend lifecycle is the entry; "restart, duplicate-callback tests" — Task 3; "route executions through governor" + "no executor reachable without authorization" — explicitly NEXT tranches (per-spender `authorize()` migration once intents exist for those paths); "durable reservations" — already durable in DB, now replay-verifiable.
- Interface consistency: hook names in Task 2 match Task 1 signatures; `EconLedger.replay` semantics (settle consumes reservation) already pinned by `tests/test_econ_ledger.py`.
