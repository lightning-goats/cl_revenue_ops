# Recorded-Price EV Replay — Implementation Plan

> **For agentic workers:** implement task-by-task with RED/GREEN evidence per
> the design at `docs/superpowers/specs/2026-08-22-recorded-price-ev-replay-design.md`.

**Goal:** Offline deterministic recomputation of the recorded sats-EV gate
from sealed v0 rebalance envelopes, byte-compared against recorded verdicts.

## Global constraints

Identical to Phase 1A: no behavioral change, no capture enablement, no RPC,
no schema change, no plugin/router/executor/database imports in the replay
path, default-off everything, malformed evidence fails closed (exit 2), a
genuine economic divergence fails as mismatch (exit 1).

---

### Task 1: Pure EV model module

Files:
- Create `modules/rebalance_ev_model.py`
- Create `tests/test_rebalance_ev_model.py`

Steps:

- [ ] RED: parity tests against real engine decompositions (stub router;
  validated/unvalidated dest fee, realized/prior utilization, effective-budget
  asymmetry, activity cap on/off, failure penalty, rejection reasons, zero-cost).
- [ ] RED: unknown model_version, non-finite, reserved float keys, missing keys.
- [ ] GREEN: implement `MODEL_VERSION` + `recompute_gate` mirroring the audit-F2
  arithmetic from recorded primitives only.
- [ ] Commit `feat: add pure recorded-price ev gate model`

### Task 2: Wire validation tightening

Files:
- Modify `modules/rebalance_cycle_replay_wire.py`
- Modify `tests/test_rebalance_cycle_replay_wire.py`
- Modify `schemas/rebalance_cycle_replay.v0.schema.json` only if the JSON
  Schema mirrors structural checks (keep v0 compatible; additive constraints)

Steps:

- [ ] RED: hostile decomposition shapes rejected at `validate_body`; valid
  fixtures unchanged.
- [ ] GREEN: bounded structural validation for final-pair decompositions.
- [ ] Prove fee wire untouched; commit `fix: validate ev decomposition structure`

### Task 3: Replay tool extension

Files:
- Modify `tools/rebalance_replay.py`
- Modify `tests/test_rebalance_replay.py`
- Modify `tests/test_architecture_guard.py`

Steps:

- [ ] RED: fixture envelope matches incl. new fields; tampered score/beats ->
  exit 1; unknown version -> exit 2; zero pairs trivial pass; import guards.
- [ ] GREEN: extend replay comparison and output fields additively.
- [ ] Commit `feat: replay recorded-price ev gate verdicts`

### Task 4: Evidence finding update

Files:
- Modify `docs/optimization/findings/2026-08-20-rebalance-replay-capture.md`

Steps:

- [ ] Run focused matrix + full functional suite (excl. supply-chain pins),
  py_compile, pyflakes, architecture guards.
- [ ] Record exact commands/totals, mark follow-up item delivered with its own
  narrow guarantee and remaining limits.
- [ ] Commit `docs: record recorded-price ev replay evidence`
