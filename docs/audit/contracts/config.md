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
- CFG-3: Cross-field ordering constraints are enforced under `_lock` at update time:
  `min_fee_ppm <= max_fee_ppm`, `rebalance_min_amount <= rebalance_max_amount`,
  `low_liquidity_threshold < high_liquidity_threshold`, `sink_threshold < source_threshold`,
  `receivable_ratio_floor <= receivable_ratio_target`.
- CFG-4: `ConfigSnapshot.from_config` never raises on field drift: snapshot fields missing from a
  partially-deployed `Config` fall back to the snapshot field's declared default.
- CFG-5: Non-finite floats (NaN/Inf) are rejected both in `update_runtime` and in
  startup `_apply_override` (warning + skip).

## Sanity check
`python3 -c "import sys; sys.path.insert(0,'.'); from modules.config import Config, ConfigSnapshot; s=ConfigSnapshot.from_config(Config()); print(s.version)"`
from the repo root must print `0`. Runtime-update behavior is exercised indirectly in
`tests/test_operator_surface.py` and other tests using `update_runtime`/`ConfigSnapshot`
(no dedicated `tests/test_config.py` exists).

## Notes
- At 1182 lines this is large for a "config" module — it embeds validation tables
  (`CONFIG_FIELD_TYPES`, `CONFIG_FIELD_RANGES`, `STRING_ENUM_VALID_VALUES`), file-override parsing,
  and the transactional update machinery, not just constants.
- `Config` and `ConfigSnapshot` must stay field-aligned by hand; `from_config`'s default fallback
  masks drift silently (by design, for partial deploys) — drift is therefore not observable as an
  error.
- No dedicated unit-test file for this module; coverage is incidental via consumer tests.
