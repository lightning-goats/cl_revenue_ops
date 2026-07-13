# Compatibility-Window Removal Checklist — execute ON/AFTER 2026-08-12

Announced 2026-07-13 (`contract-compatibility-policy.md`). Constraint:
NOTHING below executes before the date without an explicit operator
change to the compatibility policy. Prep (scanner, staged tests, this
checklist) shipped as PR 11.

## Preconditions (run first, must be clean)

1. `python3 tools/deprecation_scan.py` on lnnode → exit 0
   (no `rebalance_min_profit` in config file or config_overrides;
   zero ACTIVE `budget_reservations` rows).
2. DB backup: `sqlite3 revenue_ops.db ".backup ...-pre-removal-<ts>"`.
3. Full suite green at the removal commit's parent.

## Item 1: remove `rebalance_min_profit` (deprecated no-op)

Delete, by symbol (line numbers drift):

- `modules/config.py`: `CONFIG_FIELD_TYPES['rebalance_min_profit']`;
  the `DEPRECATED_RUNTIME_KEYS` member (leaving an empty frozenset and
  the machinery — it is the pattern for FUTURE deprecations);
  `DEPRECATED_KEY_REPLACEMENTS['rebalance_min_profit']`;
  `CONFIG_FIELD_RANGES['rebalance_min_profit']`; the `Config` dataclass
  field + its E-4.5 comment block; the `ConfigSnapshot` mirror field.
- `cl-revenue-ops.py`: the `revenue-ops-rebalance-min-profit` plugin
  option registration and the `_safe_int(...)` constructor line.
- `modules/risk_profiles.py`: the `FIELD_CLASSIFICATION` entry
  (`deprecated_transition` class becomes empty — keep the category).
- Tests: REPLACE (not merely delete) —
  `tests/test_econ_audit_wave.py::TestRebalanceMinProfitDeprecation`
  (pins the no-op EXISTS) and
  `tests/test_config_contradictions.py::TestDeprecatedOptions` (pins
  the warning) become the staged REJECTION tests in
  `tests/test_removal_readiness.py` (un-xfail them: unknown-key
  handling applies, startup ignores + warns on the stale override).
- Docs: compatibility catalog (key count 62 -> 61, field tables),
  `config-field-classification.md` regeneration, compatibility policy
  marked EXECUTED for item 1.

Post-removal semantics (pinned by the staged tests): the key behaves
like any unknown key — `revenue-config set` rejects it; a stale
persisted override is skipped with a warning at startup, never applied.

## Item 2: retire the legacy `budget_reservations` dual-path

- Precondition: scanner reports zero ACTIVE rows.
- Delete the transition-read fallbacks in `modules/database.py`
  (release/settle/cleanup dual-path branches marked "transition-only"
  since Phase 2J) and the `budget_reservations` half of
  `get_budget_status`.
- Drop or archive the table (archive: `ALTER TABLE budget_reservations
  RENAME TO budget_reservations_archived_20260812` — preferred, keeps
  history queryable).
- Update the table-inventory pin test in the same commit.

## Item 3: v0 -> v1 schema emission cutover

- Flip `schema_version` emission to 1 in `modules/econ_snapshot.py`
  (`to_wire`) and `modules/econ_intents.py` (`to_wire`) and tighten the
  emitted objects to the v1 closed shape (drop any fields not in v1 —
  audit first with the validator).
- Regenerate the conformance corpus (byte-identical pin will flag the
  version change — expected, review the diff).
- Keep the validator accepting v0 (read-side) for one further window;
  announce v0 READ removal separately if ever needed.

## Ordering

Item 1 and item 3 are independent; item 2 requires its precondition.
One PR per item, standard deploy procedure (backup, restart, verify
`revenue-status` + `revenue-config get rebalance_min_profit` returns
unknown-key for item 1), then update the completion review: DoD item
12 `pending_time_gate` -> `met`.
