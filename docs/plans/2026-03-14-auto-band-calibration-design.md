# Auto Band Calibration Design

**Date**: 2026-03-14
**Status**: Approved

## Problem

Dynamic fee autobands already exist, but they are operator-configured per peer via
`fee_ppm_target` plus `fee_multiplier_min` / `fee_multiplier_max` in peer policy.
The fee controller also already learns a posterior optimum and uncertainty per
channel. The missing piece is to translate that learned uncertainty into a safe,
automatic exploration band.

The implementation now needs to happen under current traffic conditions, not as a
future placeholder. Auto-calibration should be enabled by default for channels
that have enough observations, while preserving the current manual autoband path
as an explicit operator override.

## Goals

- Enable automatic dynamic fee band calibration by default.
- Keep manual fee bands as the highest-precedence override.
- Derive auto bands from the channel's learned posterior optimum and uncertainty.
- Persist auto-band state where channel learning already lives.
- Expose the effective band and its source in debug output.

## Non-Goals

- No change to static, passive, or hive policy semantics.
- No migration of operator policy data into controller-owned state.
- No per-HTLC-size banding in this change; one effective band per channel.
- No database schema migration for a new table or dedicated columns.

## Decision Summary

### Band precedence

The controller resolves the active band in this order:

1. Manual policy autoband from `peer_policies`
2. Auto-calibrated per-channel band from fee-controller state
3. No band

Manual policy bands remain the operator escape hatch and are never overwritten by
auto-calibration.

### Persistence model

Auto-band metadata lives in fee-controller state inside `v2_state_json`, alongside
existing Thompson learning state. This keeps auto bands channel-scoped instead of
peer-scoped and avoids write churn in `peer_policies`.

### Default behavior

Auto-band calibration is enabled by default, but gated by minimum observation
count. Cold or low-signal channels behave exactly as they do today until they
cross the observation threshold.

## Data Model

Add an `auto_band` object to the serialized fee-controller state:

```json
{
  "auto_band": {
    "min_ppm": 160,
    "max_ppm": 320,
    "optimal_fee_ppm": 240,
    "posterior_std": 40.0,
    "sigma": 2.0,
    "min_width_ppm": 50,
    "observation_count": 27,
    "source": "auto",
    "last_calibrated": 1773506400
  }
}
```

Fields:

- `min_ppm`, `max_ppm`: computed band endpoints after all safety clamps
- `optimal_fee_ppm`: current posterior optimum used as anchor
- `posterior_std`: uncertainty used to derive the band
- `sigma`: multiplier used for the confidence interval
- `min_width_ppm`: enforced minimum width after clamping
- `observation_count`: observation count at calibration time
- `source`: `"auto"` for active learned band
- `last_calibrated`: unix timestamp for debug visibility

No new SQL columns are required.

## Configuration

Add config fields and plugin options for:

- `auto_band_enabled` default `True`
- `auto_band_min_observations` default `20`
- `auto_band_sigma` default `2.0`
- `auto_band_min_width_ppm` default `50`
- `auto_band_recalibrate_interval` default `10`

These defaults make auto-calibration live immediately on sufficiently observed
channels while keeping low-traffic channels on the existing unconstrained path.

## Controller Flow

### Calibration

During fee-controller cycles, periodically recalibrate eligible dynamic channels:

1. Skip if auto-band config is disabled.
2. Skip if policy strategy is not `dynamic`.
3. Skip calibration if manual policy multipliers are present; manual still wins.
4. Skip if the channel lacks a Thompson state or enough observations.
5. Compute `optimal_fee = predict_optimal_fee(min_fee_ppm, max_fee_ppm)`.
6. Compute a raw band from `optimal_fee ± auto_band_sigma * posterior_std`.
7. Clamp the band to global fee bounds.
8. Enforce `auto_band_min_width_ppm` around the optimal fee.
9. Persist auto-band metadata in controller state.

### Consumption

Existing call sites that currently read manual policy autobands switch to an
effective resolver:

- Initial fee setting on channel open
- Dynamic fee adjustment bounds in the main controller path
- Debug surfaces

The resolver returns both the effective ppm band and a source marker:
`manual`, `auto`, or `none`.

### Regime change behavior

When regime change detection resets Thompson beliefs, any persisted auto band is
cleared. That forces fresh exploration instead of pinning the controller to stale
historical bounds during a market shift.

## Error Handling

- Missing or invalid Thompson state: skip auto calibration, keep manual/none path.
- `predict_optimal_fee()` returns `None`: do not create an auto band.
- Degenerate or inverted bounds after clamping: normalize around the optimal fee
  with the configured minimum width.
- Manual policy configured without `fee_ppm_target`: behavior remains unchanged;
  existing manual-band validation rules still apply.

## Observability

Extend `revenue-fee-debug` to report:

- whether auto calibration is enabled
- effective band source (`manual`, `auto`, `none`)
- manual band values if configured
- auto-band eligibility and observation count
- calibrated `optimal_fee_ppm`
- `posterior_std`
- computed auto band
- `last_calibrated`

This keeps operator intent and learned state visible without mixing them together.

## Testing Strategy

### Fee controller

Extend `tests/test_fee_controller.py` to cover:

- manual band takes precedence over auto band
- auto band is inactive below minimum observations
- auto band activates with sufficient observations
- minimum width enforcement
- clamping to global min/max fee bounds
- regime change clearing auto-band state
- initial-fee path uses effective band resolver

### Config and policy behavior

Extend tests that exercise config defaults and policy resolution to confirm:

- auto-band config defaults are enabled and sane
- manual policy bands continue to work exactly as before

### Debug output

Add coverage for the debug payload if needed through operator-surface tests or
the closest existing debug-facing test file.
