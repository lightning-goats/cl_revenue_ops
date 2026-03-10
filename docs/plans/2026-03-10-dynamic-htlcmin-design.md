# Dynamic HTLC Minimum Design

## Goal

Implement issue `#60` by adding a minimal, opt-in dynamic `htlc_minimum_msat` defense that raises the lower forwarding bound during HTLC-slot congestion and high mempool stress, then relaxes it back down automatically when conditions normalize.

## Context

`cl-revenue-ops` already has:

- channel congestion state and `htlc_utilization` from `FlowAnalyzer`
- a cycle-scoped Vegas mempool defense signal via `VegasReflexState`
- a centralized `set_channel_fee()` write path in `HillClimbingFeeController`
- existing dynamic `htlcmax` support attached late in `_adjust_channel_fee()`

That means the smallest coherent design is to reuse the same late-stage policy-write pattern for `htlcmin`.

Local Core Lightning documentation was also verified:

```bash
man lightning-setchannel | col -bx | rg -n "htlcmin|minimum_htlc_out_msat"
```

This confirms `setchannel` accepts `htlcmin` and advertises it as `minimum_htlc_out_msat`.

## Approaches Considered

### 1. Recommended: Compute dynamic `htlcmin` late in `_adjust_channel_fee()`

Compute the lower bound from current channel state immediately before the centralized `set_channel_fee()` call, and pass it through as an optional RPC kwarg.

Why this wins:

- matches the current dynamic `htlcmax` structure
- keeps fee/htlc policy writes centralized
- reversible by construction, because the value is recalculated every cycle from current state
- does not widen the scope of manual or policy-driven writes beyond the issue requirements

### 2. Separate HTLC policy helper module

Extract both `htlcmin` and `htlcmax` computation into a dedicated helper.

Why not now:

- more abstraction than the issue needs
- little evidence yet that HTLC policy needs to evolve independently from the fee controller

### 3. Apply `htlcmin` independently in `adjust_all_fees()`

Run a side policy update per channel outside `_adjust_channel_fee()`.

Why not:

- duplicates the policy-write path
- makes relaxation and coexistence with `htlcmax` harder to reason about
- increases the chance of drift between fee and HTLC settings

## Chosen Design

### Config

Add one new config field:

- `enable_dynamic_htlcmin: bool = False`

Wire it through:

- `modules/config.py`
- `cl-revenue-ops.py`
- `config/cl-revenue-ops.conf.full`
- `config/cl-revenue-ops.conf.minimal`

No additional operator tuning knobs are introduced in this issue.

### Fee Controller Integration

Add a small helper inside `HillClimbingFeeController`, for example:

- `_calculate_dynamic_htlcmin_msat(state, channel_info, cfg, vegas_multiplier, htlcmax_msat) -> Optional[int]`

Responsibilities:

1. Return `None` when the feature is disabled.
2. Read `state["htlc_utilization"]` with a safe default of `0.0`.
3. Compute a congestion defense once utilization crosses `cfg.htlc_congestion_threshold`.
4. Compute a Vegas defense when the floor multiplier rises above calm baseline.
5. Use the maximum of those defenses and the channel baseline.
6. Clamp below active `htlcmax_msat` if both policies are enabled.
7. Log failures at `debug` and fail open by returning `None`.

Then extend `set_channel_fee()` with an optional `htlcmin_msat` argument and attach it to the `rpc_params` dict before calling `self.plugin.rpc.setchannel(**rpc_params)`.

### Baseline and Relaxation

The baseline should be the channel's currently advertised `htlc_minimum_msat` when available, otherwise `0`.

To support that, `_get_channels_info()` should carry:

- `htlc_minimum_msat`
- optionally the compatibility alias `htlc_min_msat`

This matters because the defense must relax back down, not just ratchet upward. Recomputing from the current baseline on every cycle ensures normal conditions restore the normal lower bound.

### Internal Policy Shape

Keep the policy internal and deterministic.

Recommended shape:

- congestion defense:
  - inactive below `htlc_congestion_threshold`
  - exponential growth above threshold to aggressively suppress micro-HTLC spam as slot exhaustion approaches
- Vegas defense:
  - scale minimum routed amount upward from a low baseline when `vegas_multiplier > 1.0`
  - intended to make dust HTLCs less attractive when force-close sweep economics deteriorate

Final rule:

```text
dynamic_htlcmin = max(baseline_htlcmin, congestion_htlcmin, vegas_htlcmin)
```

If `htlcmax_msat` is also active:

```text
dynamic_htlcmin < htlcmax_msat
```

If the computed lower bound would meet or exceed the upper bound, clamp it just below the effective maximum and log the clamp.

## Data Flow

1. `adjust_all_fees()` updates Vegas state once per cycle.
2. `_adjust_channel_fee()` computes fee target and existing dynamic `htlcmax`.
3. `_adjust_channel_fee()` computes dynamic `htlcmin` from:
   - `state["htlc_utilization"]`
   - `cfg.htlc_congestion_threshold`
   - `self._vegas_state.get_floor_multiplier()`
   - channel baseline `htlc_minimum_msat`
   - active `htlcmax_msat`, if present
4. `_adjust_channel_fee()` passes `htlcmin_msat` and `htlcmax_msat` to `set_channel_fee()`.
5. `set_channel_fee()` assembles the `setchannel` kwargs:
   - `id`
   - `feebase`
   - `feeppm`
   - optional `htlcmin`
   - optional `htlcmax`

## Error Handling

- Missing `htlc_utilization` is treated as `0.0`.
- Missing baseline `htlc_minimum_msat` is treated as `0`.
- A computation failure must not fail the fee update; the controller logs at `debug` and omits `htlcmin`.
- The defense must never create an invalid `htlcmin >= htlcmax` pair.

## Testing Strategy

Add focused tests to:

- `tests/test_fee_setting_execution.py`
- `tests/test_plugin_audit_regressions.py`
- optionally `tests/test_fee_controller_audit_regressions.py` if an end-to-end regression is needed

Coverage should include:

- config snapshot/type wiring for `enable_dynamic_htlcmin`
- `set_channel_fee()` omits `htlcmin` when disabled
- congestion raises `htlcmin`
- Vegas pressure raises `htlcmin`
- the value relaxes back down to baseline when pressure disappears
- the value is clamped below active `htlcmax`

## Non-Goals

- no new database schema
- no new runtime tuning knobs for the congestion curve
- no changes to `FlowAnalyzer` classification logic
- no change to manual/policy-driven fee writes unless later testing shows that widening scope is necessary
