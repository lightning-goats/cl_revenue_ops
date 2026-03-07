# Autonomous Executor Purpose Design

**Date:** 2026-03-06

## Context

`cl_revenue_ops` is currently described as a revenue-operations layer for Core Lightning, but the implementation has accumulated multiple product identities at once: fee optimizer, rebalancer, hive adapter, policy toolkit, telemetry layer, portfolio lab, and treasury manager. The runtime configuration surface in `modules/config.py` exposes many internal algorithm and subsystem knobs directly to operators.

That is misaligned with the intended operating model. The desired product is an autonomous executor with a very small manual override surface, optimized for maximum routing profit and strong fleet coordination when coupled with `cl-hive`.

## Product Purpose

`cl_revenue_ops` should autonomously allocate liquidity and set fees to maximize risk-adjusted routing profit with minimal operator babysitting.

More concretely:

- In standalone mode, it should optimize from local profitability, liquidity, and routing history.
- In fleet mode, it should optimize the same objective using better priors from hive coordination, corridor ownership, and shared flow prediction.
- The operator should control safety rails, not algorithm internals.

## Product Boundary

### Core responsibilities

- Observe local and fleet routing economics
- Decide whether to hold, change a fee, rebalance liquidity, or suppress action
- Execute the chosen action locally
- Record the outcome for later learning and diagnostics

### Supporting inputs

- Profitability classification and marginal ROI
- Liquidity position and flow state
- Kalman velocity and anticipatory prediction
- Portfolio opportunity cost
- Peer quality and route reliability
- Hive coordination signals when available

### Non-core responsibilities

These may still exist in the codebase, but they should not define the operator experience:

- exposing separate tuning families for Thompson, AIMD, scarcity, Vegas, Kelly, sling target shaping, cache TTLs, and coordination plumbing
- requiring the operator to decide how the model should think
- treating hive mode as a separate product with a separate control philosophy

## Control Model

The runtime model should reduce to three loops:

1. `observe`
   Collect profitability, liquidity state, flow velocity, peer quality, and hive coordination inputs.

2. `decide`
   Produce one of a small number of decisions:
   - hold
   - change fee
   - rebalance
   - suppress action
   - safety stop

3. `execute`
   Apply the action locally and persist outcome data for diagnostics and learning.

This is a single executor in both standalone and fleet mode. Hive should improve priors and coordination quality, not switch the plugin into a different behavioral product.

## Operator Surface

The supported operator surface should be reduced to four public controls:

- global pause
- daily budget cap
- hard fee floor
- hard fee ceiling

Everything else should be one of:

- internal implementation detail
- debug/admin-only diagnostic control
- deprecated compatibility shim during migration

Per-peer tactical policy, fee-algorithm hyperparameters, market-response toggles, rebalance-shaping knobs, and most hive cache/config switches should not remain public runtime controls.

## Technical Direction

### Fee decisions

Fee logic should collapse into one scoring outcome with internal stabilizers.

Internally, Thompson sampling, AIMD, scarcity handling, reputation weighting, Vegas reflex, and hysteresis can remain as signals or dampers. Externally, the system should emit a single fee decision with confidence and a reason summary.

### Rebalance decisions

Rebalancing should collapse into one capital-allocation function.

Liquidity thresholds, Kelly sizing, sling targets, flow state, portfolio metrics, and hive priors should all feed a single economic decision: whether moving capital from source to destination beats opportunity cost and fits within the global budget.

### Hive integration

Hive should be treated as a prior and coordination source, not as a separate executor mode.

Fleet data should improve:

- fee priors
- corridor ownership awareness
- expected rebalance value
- path selection

If hive becomes unavailable, the executor should degrade gracefully to local-only decisions without switching product behavior.

### Learning

The system should learn from outcomes rather than operator tuning. Internal parameters should be adjusted from realized revenue, liquidity preservation, failure cost, and routing quality instead of being exposed as public configuration because they were once useful during development.

## Public Knobs To Remove

These should be removed from the public operator surface first and either derived internally, moved behind admin/debug access, or deleted entirely.

### Fee algorithm tuning

- `thompson_prior_std_fee`
- `thompson_observation_decay_hours`
- `thompson_max_observations`
- `thompson_min_observations`
- `aimd_failure_threshold`
- `aimd_success_threshold`
- `aimd_multiplicative_decrease`
- `aimd_additive_increase_ppm`
- `aimd_min_decrease_interval`

### Market-response tuning

- `enable_vegas_reflex`
- `vegas_decay_rate`
- `enable_scarcity_pricing`
- `scarcity_threshold`
- `enable_reputation`
- `reputation_decay`
- `ema_smoothing_alpha`
- `enable_velocity_gate`
- `min_velocity_threshold`

### Rebalance-shaping tuning

- `enable_kelly`
- `kelly_fraction`
- `kelly_bypass_for_fleet`
- `low_liquidity_threshold`
- `high_liquidity_threshold`
- `rebalance_cooldown_hours`
- `inbound_fee_estimate_ppm`
- `sling_target_sink`
- `sling_target_source`
- `sling_target_balanced`
- `sling_deplete_pct_sink`
- `sling_deplete_pct_source`
- `sling_deplete_pct_balanced`

### Coordination plumbing

- most hive feature-enable toggles
- most hive and routing-intelligence cache TTLs
- mode-like switches that only exist because subsystems are loosely coupled

## Migration Strategy

The migration should avoid a flag-day rewrite.

1. Freeze the intended product boundary in docs.
2. Split config into public safety controls and internal tuning.
3. Keep legacy knobs temporarily as deprecated compatibility inputs.
4. Introduce a minimal operator API and status surface centered on decisions and safety rails.
5. Derive or absorb internal tuning from observed outcomes.
6. Remove deprecated runtime knobs once diagnostics show stable autonomous behavior.

## Success Criteria

- higher routing profit per unit of local capital
- fewer manual interventions per week
- fewer low-signal or no-op fee broadcasts
- better corridor discipline and capital coordination in hive mode
- a dramatically smaller supported config surface

## Implementation Guidance

The first implementation phase should focus on interface simplification, not algorithm replacement:

- define the new public config surface
- deprecate non-safety runtime knobs without changing core behavior
- improve explainability so decisions can be inspected without manual tuning
- keep standalone and fleet behavior on one executor path

Only after that should deeper model simplification remove or merge internal mechanisms.
