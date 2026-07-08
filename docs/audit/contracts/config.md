# Intent Contract: modules/config.py

## Purpose
Holds every tunable parameter of the plugin in a mutable `Config` dataclass, plus an immutable
`ConfigSnapshot` that worker cycles capture at cycle start so a runtime config update can never
produce a torn read mid-cycle. Also provides transactional runtime updates (`update_runtime`:
validate → DB write → read-back verify → in-memory update under lock) and two small constant
classes, `ChainCostDefaults` (economic fee-floor math) and `LiquidityBuckets` (fee-tier
classification). Loaded at plugin startup by `cl-revenue-ops.py` and consumed by nearly every
module.

## Consumers / dependencies
- Consumers: `cl-revenue-ops.py` (startup options, `revenue-config` RPC), `modules/rebalancer.py`,
  `modules/fee_controller.py` (also uses `ChainCostDefaults`, `LiquidityBuckets`),
  `modules/capacity_planner.py` (`ChainCostDefaults`), `modules/__init__.py` re-export.
- Dependencies: `modules/database.py` (config-override persistence, TYPE_CHECKING import only);
  stdlib only otherwise — no RPC access.

## Invariants
- CFG-1: Keys in `IMMUTABLE_CONFIG_KEYS` (`db_path`, `dry_run`) can never be changed by
  `update_runtime`; the call returns an error dict, not success.
- CFG-2: `update_runtime` only reports success after the DB read-back equals the written value;
  on read-back mismatch the override is rolled back and an error is returned (no Ghost Config).
- CFG-3: Cross-field ordering constraints are enforced under `_lock` at update time
  (`update_runtime`, modules/config.py:1044–1079), checked from both sides of each pair so a
  same-cycle update to either field is caught:
  `min_fee_ppm <= max_fee_ppm`, `rebalance_min_amount <= rebalance_max_amount`,
  `low_liquidity_threshold < high_liquidity_threshold`,
  `rebalance_utilization_floor < rebalance_utilization_ceiling`,
  `hive_equalization_low_pct < hive_equalization_high_pct`,
  `sink_threshold < source_threshold`,
  `receivable_ratio_floor <= receivable_ratio_target`.
- CFG-4: `ConfigSnapshot.from_config` never raises on field drift: snapshot fields missing from a
  partially-deployed `Config` fall back to the snapshot field's declared default.
- CFG-5: Non-finite floats (NaN/Inf) are rejected both in `update_runtime` and in
  startup `_apply_override` (warning + skip).
- CFG-6: `Config.__post_init__` (modules/config.py:844–882) enforces a second, construction-time
  layer of the same invariant class so a bad value can never be instantiated directly (not just
  blocked at `update_runtime`): `hive_equalization_low_pct < hive_equalization_high_pct`;
  `receivable_ratio_target`, `receivable_ratio_floor`, `boltz_structural_budget_sats_per_day`,
  `drain_fee_discount_max`, and `node_drain_bias_max` are each range-checked against
  `CONFIG_FIELD_RANGES`; `receivable_ratio_floor <= receivable_ratio_target`; `fee_profile` is
  validated against `STRING_ENUM_VALID_VALUES` and normalized to lowercase; `rebalance_router` is
  restricted to `'v3'` only (raises `ValueError` — legacy `'v2'` routing was removed, so
  constructing a `Config` with `rebalance_router='v2'` is a hard error, not a silent fallback).
  Any violation raises `ValueError` at construction time rather than producing an invalid `Config`.

## Sanity check
`python3 -c "import sys; sys.path.insert(0,'.'); from modules.config import Config, ConfigSnapshot; s=ConfigSnapshot.from_config(Config()); print(s.version)"`
from the repo root must print `0`. Runtime-update behavior is exercised indirectly in
`tests/test_operator_surface.py` and other tests using `update_runtime`/`ConfigSnapshot`
(no dedicated `tests/test_config.py` exists).

## Public runtime surface additions (LN+ swap automation, zero-fee corridor, econ audit)

`PUBLIC_RUNTIME_KEYS` (modules/config.py:27–100) has grown a full LN+ liquidity-swap block plus
several econ-audit fields since this contract was first written:

- **`lnplus_*` block (13 runtime controls)**: `lnplus_swaps_enabled`, `lnplus_execute_applications`,
  `lnplus_swap_preference_margin`, `lnplus_max_duration_months`, `lnplus_min_peer_positive_ratings`,
  `lnplus_min_peer_rank`, `lnplus_max_participants`, `lnplus_min_participants`,
  `lnplus_apply_feerate_ceiling`, `lnplus_pending_timeout_days`, `lnplus_inbound_credit_factor`,
  `lnplus_fleet_pubkeys`, `lnplus_watcher_interval`. All are settable/resettable via
  `revenue-config` and refresh live from `setconfig`/config-file on the dynamic-config loop
  (subject to the same override-precedence rule as every other field — see README
  "revenue-config: actions and override precedence").
- **`hive_zero_fee_stale_grace_seconds`**: the zero-fee-corridor membership-grace window (default
  604800 = 7 days), tunable at runtime without a daemon restart (Z-2, 2026-07-08).
- **`min_fee_ppm_saturated`**: the class-aware saturated/source min-fee floor added by the 2026-07
  econ audit (E-2) IS a public runtime key (not internal-only) — settable/resettable exactly like
  `min_fee_ppm`.
- **`weekly_budget_sats`**: already a public runtime key before the econ audit, but worth calling
  out explicitly here since the audit (E-3) specifically required it be live-raisable — a daily
  budget increase silently capped by a stale weekly ceiling was the bug this closed.

None of these introduce a new CFG-3-style cross-field ordering check beyond what is listed above;
each is validated independently via `CONFIG_FIELD_TYPES`/`CONFIG_FIELD_RANGES`.

## Notes
- At 1490 lines (grown from 1182 as of this contract's last refresh, mainly from the LN+/econ-audit
  runtime-key additions above) this is large for a "config" module — it embeds validation tables
  (`CONFIG_FIELD_TYPES`, `CONFIG_FIELD_RANGES`, `STRING_ENUM_VALID_VALUES`), file-override parsing,
  and the transactional update machinery, not just constants.
- `Config` and `ConfigSnapshot` must stay field-aligned by hand; `from_config`'s default fallback
  masks drift silently (by design, for partial deploys) — drift is therefore not observable as an
  error.
- No dedicated unit-test file for this module; coverage is incidental via consumer tests.
