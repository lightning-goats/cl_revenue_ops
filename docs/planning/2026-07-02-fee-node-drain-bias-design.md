# Design + Plan: Node-liquidity-aware auto-drain-bias for the fee controller

Date: 2026-07-02. Branch: `fee-node-drain-bias`. Status: building.

## Problem
The fee controller optimizes per-channel revenue. It has a per-channel drain discount
(`_drain_fee_multiplier`, gated by `drain_fee_discount_max`, default 0.0) that lowers fees on
over-local zero-flow channels — but it's static and operator-toggled. The node-level
`receivable_ratio` starvation signal (source-heaviness) feeds the capacity planner but NOT the
fee controller. So the fee loop doesn't automatically lean into draining when the node as a
whole is source-heavy.

## Goal
Make the drain discount **node-liquidity-aware and self-activating**: when the node is
source-heavy (receivable ratio below target), automatically scale a drain bias on over-local
channels' fees, ramping with starvation; auto-deactivate as the node rebalances. Config-gated,
**default OFF** (merging changes no fee behavior until enabled). Rails (`min_fee_ppm`) still apply.

## Mechanism
1. **Node receivable ratio** `R = total_remote / total_capacity` over active (CHANNELD_NORMAL)
   channels (= 1 − total_local/total_capacity). Reuse the capacity-planner pattern
   (`capacity_planner.py:312-325`). Computed once per fee cycle.
2. **Node drain pressure** `P ∈ [0,1]`: 0 when `R >= receivable_ratio_target` (0.30); ramps
   linearly to 1.0 at/below `receivable_ratio_floor` (0.20). `P = clamp((target − R)/(target − floor), 0, 1)`.
3. **Effective drain bias**: extend the existing `_drain_fee_multiplier` so its effective
   discount cap becomes `max(drain_fee_discount_max, node_drain_bias_max * P)` when
   `node_drain_bias_enabled`. So a source-heavy node auto-applies up to `node_drain_bias_max`
   discount on its over-local zero-flow channels even if the static `drain_fee_discount_max` is 0;
   a balanced node (P=0) applies nothing.
4. **Observability**: surface `node_receivable_ratio`, `node_drain_pressure`, and the effective
   drain multiplier in the fee-debug / fee_decision reasoning.

## Config (new, runtime-settable, default off/neutral)
- `node_drain_bias_enabled: bool = False`
- `node_drain_bias_max: float = 0.3` (range 0.0–0.5; only active when enabled)
- reuse `receivable_ratio_target` (0.30) / `receivable_ratio_floor` (0.20).
Add both new keys to `PUBLIC_RUNTIME_KEYS`, `CONFIG_FIELD_TYPES`, `CONFIG_FIELD_RANGES`,
`Config`, `ConfigSnapshot`, and register `add_option`s (mirror the existing drain knobs so it's
not a silent no-op — P6-002 lesson).

## Safety
Default OFF → no behavior change on merge/deploy. Only touches the fee *discount* path (never
raises fees, never removes the `min_fee_ppm` rail). No spend/budget path touched. Reversible via
config at runtime.

## Plan (TDD, each task red-first, spec+quality review, config-gated)
1. **Pure helpers** `compute_node_receivable_ratio(channels)` + `node_drain_pressure(R, target, floor)`
   in fee_controller (or a small helper) + unit tests (starved→P=1, target→P=0, mid→ramp,
   zero-capacity safe).
2. **Config knobs** (2 new, wired end-to-end + registration test).
3. **Wire into `_drain_fee_multiplier`**: effective cap = `max(static, node_bias_max*P)` when
   enabled; feed node P from the cycle; expose in reasoning. TDD + mutation (disable the node
   term → test fails). Invariant: `node_drain_bias_enabled=False` → byte-identical to today.
4. **Regression + review**: full suite, `scorecard.py --deep-only`, spec+quality review, and
   confirm the fee money-path tests + `test_all_spenders_atomic.py` untouched.

Rollout: ship **default OFF**; enable per-node via `revenue-config set node_drain_bias_enabled true`
(runtime, like the other drain knobs) after review — same philosophy as the existing runtime drain knobs.
