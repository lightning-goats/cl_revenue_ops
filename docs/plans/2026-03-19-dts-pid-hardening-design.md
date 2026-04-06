# DTS+PID Hardening Design

## Goal

Harden the standalone DTS+PID fee controller so it behaves conservatively under sparse and noisy Lightning data, avoids violent reversals, removes literal zero-fee probing, and gives operators a clearer explanation of how a fee move was chosen.

## Constraints

- Keep the standalone architecture: local policy override, hard bounds, DTS, PID, hysteresis.
- Do not reintroduce Hive.
- Do not reintroduce Thompson+AIMD fallback behavior.
- Do not redesign the plugin from scratch.
- Keep the implementation understandable and maintainable.

## Root Causes

### 1. DTS still produces targets that are too volatile for sparse Lightning data

`GaussianThompsonState.sample_fee()` explores aggressively before the posterior is trusted, and `apply_dts_discount(gamma=0.95)` widens uncertainty quickly enough that quiet channels drift back toward exploratory behavior after only a few hours. On low-volume LN channels this is too eager.

### 2. PID is acting like an amplifier, not a damper

`PIDState.calculate_multiplier()` uses an exponential multiplier with a strong derivative term (`kd=5.0`). On sparse and irregular fee-control intervals, the derivative is mostly measuring noise and regime discontinuities, not useful market dynamics. That makes source/sink corrections too jumpy.

### 3. Wake-up behavior is still too permissive

A waking channel immediately re-enters the full DTS+PID path in the same cycle. Even with a wake-cycle delta cap, the controller still recomputes a fresh target from noisy state and can lurch toward a new regime too quickly.

### 4. Bounds still act like ordinary targets

The controller samples inside `[floor, ceiling]`, then applies PID, then clamps again. This means min/max bounds are still frequent destinations rather than exceptional guard rails, especially on sink/source channels and on channels with raised floors.

### 5. Zero-fee probing is economically unsound

The current defibrillator path routes at `0` PPM until activity is detected. That attracts traffic that is not representative of real-priced demand, pollutes the learning signal, and causes abrupt regime transitions when the probe exits.

## Chosen Hardening Strategy

### DTS hardening

- Increase sparse-data conservatism by requiring more evidence before channels fully trust exploratory DTS behavior.
- Slow posterior forgetting so quiet channels do not become exploratory again too quickly.
- Add target blending before the final applied fee so a single cycle does not fully adopt the newly sampled target.

### PID hardening

- Remove the derivative term entirely.
- Narrow PID authority so it biases the target rather than dominating it.
- Keep directional behavior for source/sink channels, but make the correction gradual.

### Wake hardening

- Keep the existing wake detection.
- Apply stronger blending and stricter delta caps on the wake cycle.
- Treat wake-up as re-entry, not full correction.

### Exploration hardening

- Remove literal zero-fee probing.
- Replace it with bounded low-fee exploration above `min_fee_ppm`.
- Make exploration rare, conservative, and clearly labeled in logs/reasoning.

### Observability hardening

- Preserve the current target-path visibility.
- Add explanation for sparse-data conservatism, target blending, and bounded exploration.
- Make it obvious whether a move was held back by blending, by delta caps, by floor/ceiling pressure, or by wake damping.

## Non-Goals

- No new secondary fee controller.
- No broad plugin cleanup.
- No expansion of the operator-facing config surface unless required for correctness.

## Test Plan

Add regressions for:

- no near-floor to near-ceiling jump in one normal cycle
- no near-ceiling to near-floor jump in one normal cycle
- wake cycle more conservative than active cycle
- balanced channels monetized more cautiously than before
- source/sink direction still correct after PID hardening
- sparse-data channels update more conservatively
- zero-fee probing replaced by bounded low-fee exploration
- bounds no longer behave as uncontrolled default targets in common paths
