# ADR-001: DTS+PID is the authoritative fee controller (controller-neutral contract)

- Status: **ACCEPTED** (2026-07-13, operator-directed gap closure Phase C)
- Supersedes: the multiplicative fee decomposition in
  `docs/planning/refactor.md` §F2 (recorded there as an
  `approved_deviation`; original text preserved for history)
- Related: gap-closure plan §Gap 7; README contradiction #8;
  `docs/refactor/phase0/snapshot-dependency-audit.md` (3e)

## Context

The refactor specification proposed expressing the unclamped fee target
as `economic_baseline × liquidity_pressure × market_correction`. The
repository's actual controller is **DTS+PID**: a Gaussian
Thompson-sampling posterior over the fee→revenue relationship (with
contextual posteriors per time-bucket/corridor role) produces a sampled
revenue-optimal target; a PID controller on the channel's outbound-
liquidity ratio produces a bounded multiplier (`1.5^(p+i)`, clamped to
[0.5, 2.0], gains scaled by `1/log2(capacity/1e6 + 2)`); the two blend
into the requested target. This is not a product of three independent
factors, and forcing that decomposition would falsify the algorithm the
88 golden fixtures pin. The spec's own rule for such cases: document
the contradiction, do not silently force the proposal.

## Decision

The specification's fee-policy contract becomes **controller-neutral**:

```text
raw_fee_target = fee_controller(snapshot, controller_state, configuration)

final_fee_target =
    cooldown(
        deadband(
            rate_limit(
                rails(raw_fee_target)
            )
        )
    )
```

with DTS+PID as the authoritative `fee_controller` implementation. Any
future controller must satisfy the same contract; replacing DTS+PID is
NOT authorized by this ADR.

## Contract terms (normative)

1. **DTS+PID remains authoritative.** The Thompson posterior decides the
   revenue-seeking target; the PID multiplier manages liquidity balance;
   the blend is the controller's raw target. No parallel fee authority
   exists.
2. **Inputs are the canonical snapshot's observations plus explicit
   controller state.** Since PR 3e, every market/gossip/chain/channel
   observation the controller reads is frozen per cycle
   (`_frozen_observation` memo — one immutable computation per cycle).
   `controller_state` (Thompson posteriors, PID integrator, sleep
   state; persisted per channel in `v2_state_json`, algorithm version
   `dts_pid_v1`) is a DISTINCT input — it is memory, not observation,
   and is deliberately outside the snapshot freeze.
3. **Output is a typed `SET_FEE` intent.** Automated broadcasts pass the
   governor gate (`econ_governor_fees_enabled`, fail-closed, authority
   level `fees`) and are ledgered with `canonical_snapshot_id`
   evidence; the fee-shadow records the cycle's applied adjustments as
   intent proposals. Manual operator sets bypass neither ledger nor
   audit trail conventions but are explicitly operator-directed.
4. **Authoritative arithmetic is deterministic** given (snapshot,
   controller_state, configuration, seed). The DTS sampling seed is the
   one recorded portability hazard (unseeded `random` in production);
   cycle determinism where pinned (econ cycle) is byte-identical, and
   the constraint stages below are pure functions with golden coverage.
   Wire-contract numeric rules (checked integer msat/ppm, explicit
   rounding) apply to every value that crosses the intent boundary.
5. **Constraint stages run in the documented order** on the controller's
   raw target — each is real code with golden fixtures:
   - **rails**: floor/ceiling clamp (`_calculate_floor` incl. dynamic
     chain-cost floor and saturated carve-out; flow-adjusted ceiling,
     min/max_fee_ppm bounds);
   - **rate_limit**: per-cycle delta cap
     (`_apply_damped_fee_target` / `_get_fee_step_cap`, wake-cycle
     variant included, with target blending);
   - **deadband**: meaningful-change suppression
     (`is_meaningful_rate`, minimum-step and no-op suppression);
   - **cooldown**: hysteresis/sleep windows (`is_sleeping`/
     `sleep_until`, observation-window gates: min hours or min
     forwards).
6. **Auto fee bands are inputs/constraints, not authority.** Band-like
   mechanisms (congestion caps, exploration targets, supported-fee
   ceiling, market-boundary when enabled) enter as evidence or bounds
   on the controller's target; none may mutate channel fees outside the
   single `set_channel_fee` path and its governor gate.
7. **`revenue-fee-debug` exposes the real components** — DTS posterior
   state and last sampled target, PID term components, controller
   state/version, stage state, cycle decision summary, and stable
   reason codes — never fictional multiplicative factors.

## Consequences

- `docs/planning/refactor.md` §F2 is amended to the controller-neutral
  contract; the original decomposition is preserved there as an
  `approved_deviation` record with a pointer to this ADR.
- Conformance scenarios for fee behavior (Phase F corpus) fixture the
  STAGES and the controller contract, not the abandoned decomposition.
- The completion review scores the fee-formula item as
  `approved_deviation` — formalized here, not silently completed.
- Future work explicitly allowed: seed injection for DTS sampling
  (removing the portability hazard); per-stage trace persistence for
  richer debug output. Neither changes this contract.
