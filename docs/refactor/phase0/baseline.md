# Phase 0 baseline record

Captured 2026-07-12 in worktree branch `worktree-refactor`.

- Baseline commit: `5e8f747` ("fix(planner): close-protection gate judges
  the 30d window, not lifetime history")
- Test suite: `python3 -m pytest tests/ -q --ignore=tests/integration
  -p no:cacheprovider` → **3114 passed, 1 skipped, 42.30s**
  (skip: `tests/test_pyln_integration.py` — pyln.testing not installed)
- Python: 3.12.3; runtime deps exact-pinned in `requirements.txt`
  (pyln-client 25.12.1, PyYAML 6.0.1, numpy 1.26.4); hash-pinned closure in
  `requirements.lock`; SBOM in `docs/audit/deep/sbom.cyclonedx.json`
- Plugin entry: `cl-revenue-ops.py` (9,911 lines), 64 registered RPC
  methods (`@plugin.method`), modules/ total ≈ 42,833 lines
- Database: 35 `CREATE TABLE IF NOT EXISTS` tables in
  `modules/database.py` (a bare `grep -c 'CREATE TABLE'` reports 37 — two
  hits are comments); `schema_version` table is WRITE-ONLY by operator
  ruling DD9/MIG-3 (2026-07-02) — see `modules/database.py:606`
- Migrations: additive `CREATE TABLE/INDEX IF NOT EXISTS` +
  `ALTER TABLE` guards in `Database.__init__`; no migration framework
- Public datastore contracts (documented, tested by
  `tests/test_cross_plugin_contracts.py`):
  `["revenue","profitability-summary"]`, `["revenue","capex-summary"]`,
  `["revenue","segment-observations"]` — see `docs/contracts/`
- Production: single node `lnnode` (hive-nexus-01); prior audit baseline
  `docs/audit/deep/prod-baseline-T0.md` (53 MiB DB) — its "node 2 gap" is
  moot: fleet is single-node since 2026-07-11
- CLN runtime floor: v24.11.1 (`docs/CORE_LIGHTNING_COMPATIBILITY.md`)

## Phase 0 exit (2026-07-12)

- Suite after Phase 0: **3212 passed, 1 skipped** (98 tests added:
  4 inventory pins, 88 golden/harness, 2 schema, 3 conformance,
  1 harness self-test)
- `git diff 5e8f747 --stat -- modules/ cl-revenue-ops.py` — EMPTY
  (zero production-code changes)
- `python3 tools/conformance/validate_fixtures.py` — exit 0
