"""
Phase 3A — DB schema migration integrity tests.

modules/database.py uses a "run all idempotent DDL on every startup" migration
model: `CREATE TABLE IF NOT EXISTS` plus `ALTER TABLE ... ADD COLUMN` wrapped in
try/except OperationalError (or PRAGMA table_info guards). The `schema_version`
table is seeded to 1 but never bumped or read — there is no version-driven
migration path. These tests therefore anchor on *historical git commits* of
modules/database.py (each representing a real deployed schema shape) and drive
the CURRENT initialize() forward, asserting:

  * every historical shape migrates forward to the full current schema
    (table + column + index parity with a fresh install),
  * migrations are idempotent (running twice is a no-op),
  * fresh-install == migrated-from-oldest (identical PRAGMA sets),
  * a DB whose schema_version is NEWER than the code is handled without
    corruption (documents the *absence* of downgrade refusal — see MIG-3),
  * an interrupted/partial migration completes cleanly on retry (crash-during
    -upgrade safety, which rests on idempotence rather than a transaction).

The historical database.py is loaded in an isolated synthetic package with a
stub `utils` module (initialize() only needs the 5 utility symbols at import
time, not at runtime), so we exercise the genuine old DDL.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import sqlite3
from unittest.mock import MagicMock

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Mock pyln.client before importing the current module (mirrors test_database.py)
_mock_pyln = MagicMock()
_mock_pyln.Plugin = MagicMock
_mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", _mock_pyln)
sys.modules.setdefault("pyln.client", _mock_pyln)

from modules.database import Database as CurrentDatabase  # noqa: E402


# Historical anchors: (commit, iso_date, note). Each is a real deployed shape of
# modules/database.py. The oldest predates the resolution_time/resolved_time
# columns and the planner tables, so it exercises the deepest migration path.
ANCHORS = [
    ("7ee0460", "2025-12-04", "first EV-rebalancer schema (no resolution_time)"),
    ("5b92d43", "2025-12-22", "htlc slot pricing era"),
    ("3fa9552", "2026-01-01", "ignored_peers introduced"),
    ("8f6783e", "2026-01-05", "policy-driven architecture + ignored->policies"),
    ("0b244d4", "2026-01-07", "forwards idempotent-insert migration"),
    ("193826f", "2026-03-19", "planner_candidates/planner_actions added"),
    ("2365c27", "2026-04-05", "planner_recycle_ops added"),
    ("204048b", "2026-04-24", "native route execution era"),
    ("2247370", "2026-06-27", "revenue-evidence freshness era"),
]

OLDEST_ANCHOR = ANCHORS[0][0]

_UTILS_STUB = '''
def normalize_scid(x):
    return str(x or "")
def base_to_sats_floor(m):
    return int(m) // 1000
def base_to_sats_ceil(m):
    return (int(m) + 999) // 1000
def base_delta_to_sats_toward_zero(m):
    return int(m) // 1000
def sats_to_base(s):
    return int(s) * 1000
'''

_HIST_CACHE: dict = {}


def _git_show(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", REPO, "show", f"{commit}:{path}"]
    ).decode()


def _load_historical_database(commit: str):
    """Load the historical modules/database.py at `commit` as an isolated
    package `histpkg_<commit>` with a stub utils module. Returns its Database
    class. Cached across tests."""
    if commit in _HIST_CACHE:
        return _HIST_CACHE[commit]

    src = _git_show(commit, "modules/database.py")
    tmpdir = tempfile.mkdtemp(prefix=f"histdb_{commit}_")
    pkgdir = os.path.join(tmpdir, f"histpkg_{commit}")
    os.makedirs(pkgdir)
    open(os.path.join(pkgdir, "__init__.py"), "w").close()
    with open(os.path.join(pkgdir, "utils.py"), "w") as fh:
        fh.write(_UTILS_STUB)
    with open(os.path.join(pkgdir, "database.py"), "w") as fh:
        fh.write(src)
    sys.path.insert(0, tmpdir)

    pkgname = f"histpkg_{commit}"
    pkg_spec = importlib.util.spec_from_file_location(
        pkgname,
        os.path.join(pkgdir, "__init__.py"),
        submodule_search_locations=[pkgdir],
    )
    pkgmod = importlib.util.module_from_spec(pkg_spec)
    sys.modules[pkgname] = pkgmod
    pkg_spec.loader.exec_module(pkgmod)

    u_spec = importlib.util.spec_from_file_location(
        f"{pkgname}.utils", os.path.join(pkgdir, "utils.py")
    )
    umod = importlib.util.module_from_spec(u_spec)
    sys.modules[f"{pkgname}.utils"] = umod
    u_spec.loader.exec_module(umod)

    d_spec = importlib.util.spec_from_file_location(
        f"{pkgname}.database", os.path.join(pkgdir, "database.py")
    )
    dmod = importlib.util.module_from_spec(d_spec)
    sys.modules[f"{pkgname}.database"] = dmod
    d_spec.loader.exec_module(dmod)

    _HIST_CACHE[commit] = dmod.Database
    return dmod.Database


def _close(db) -> None:
    for meth in ("close_all_connections", "close"):
        fn = getattr(db, meth, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                continue


def _build_old_db(commit: str, path: str) -> None:
    """Initialize a DB file at the historical schema shape."""
    Cls = _load_historical_database(commit)
    db = Cls(path, MagicMock())
    db.initialize()
    _close(db)


def _migrate_current(path: str) -> None:
    """Run the current initialize() (migration path) against `path`."""
    db = CurrentDatabase(path, MagicMock())
    db.initialize()
    _close(db)


def _snapshot(path: str):
    """Return (tables: {name: sorted[cols]}, indexes: sorted[names])."""
    conn = sqlite3.connect(path)
    try:
        tables = {}
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({name})")]
            tables[name] = sorted(cols)
        indexes = sorted(
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        )
        return tables, indexes
    finally:
        conn.close()


@pytest.fixture(scope="module")
def fresh_snapshot(tmp_path_factory):
    path = os.path.join(str(tmp_path_factory.mktemp("fresh")), "fresh.db")
    _migrate_current(path)
    return _snapshot(path)


@pytest.mark.parametrize("commit,date,note", ANCHORS, ids=[a[0] for a in ANCHORS])
def test_historical_shape_migrates_to_current(
    commit, date, note, fresh_snapshot, tmp_path
):
    """Every historical schema migrates forward to the full current schema in a
    SINGLE initialize() pass — no missing columns, tables, or indexes."""
    fresh_tables, fresh_indexes = fresh_snapshot
    path = os.path.join(str(tmp_path), f"old_{commit}.db")
    _build_old_db(commit, path)
    _migrate_current(path)
    mig_tables, mig_indexes = _snapshot(path)

    # No fresh table/column is missing after migration (the classic
    # "works on fresh install, breaks on upgrade" bug).
    missing_tables = set(fresh_tables) - set(mig_tables)
    assert not missing_tables, f"{commit}: migration dropped tables {missing_tables}"
    for table, cols in fresh_tables.items():
        missing_cols = set(cols) - set(mig_tables.get(table, []))
        assert not missing_cols, (
            f"{commit}: table {table} missing columns after migration: "
            f"{sorted(missing_cols)} (missing ALTER for already-deployed DBs)"
        )
    missing_idx = set(fresh_indexes) - set(mig_indexes)
    assert not missing_idx, f"{commit}: migration missing indexes {sorted(missing_idx)}"


@pytest.mark.parametrize("commit", [a[0] for a in ANCHORS], ids=[a[0] for a in ANCHORS])
def test_migration_is_idempotent(commit, tmp_path):
    """Running the full migration path twice is a no-op the second time:
    identical schema, and no duplicate columns/tables/errors."""
    path = os.path.join(str(tmp_path), f"idem_{commit}.db")
    _build_old_db(commit, path)
    _migrate_current(path)
    once = _snapshot(path)
    _migrate_current(path)  # second pass must not error or change anything
    twice = _snapshot(path)
    assert once == twice, f"{commit}: second migration pass changed the schema"


def test_fresh_install_equals_migrated_from_oldest(fresh_snapshot, tmp_path):
    """A freshly initialized DB must have a schema IDENTICAL to one migrated
    from the oldest historical version (table + column + index sets)."""
    path = os.path.join(str(tmp_path), "migrated_oldest.db")
    _build_old_db(OLDEST_ANCHOR, path)
    _migrate_current(path)
    migrated = _snapshot(path)
    assert migrated == fresh_snapshot, (
        "fresh-install schema diverges from migrated-from-oldest schema"
    )


def test_migration_twice_from_fresh_is_stable(tmp_path):
    """Fresh install then re-initialize is a no-op (restart safety)."""
    path = os.path.join(str(tmp_path), "fresh_twice.db")
    _migrate_current(path)
    once = _snapshot(path)
    _migrate_current(path)
    assert once == _snapshot(path)


def test_newer_schema_version_is_not_corrupted(tmp_path):
    """A DB whose schema_version is NEWER than the code recognizes must not be
    silently corrupted. NOTE: database.py has no downgrade/unknown-version
    refusal (schema_version is write-only, never read) — see finding MIG-3.
    This test locks in that a newer-version DB survives initialize() without
    data loss, and documents the absence of a refusal guard."""
    path = os.path.join(str(tmp_path), "newer.db")
    _migrate_current(path)

    # Simulate a DB written by a FUTURE code version: bump schema_version and
    # seed a canary row.
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_version SET version = 999999")
    conn.execute(
        "INSERT INTO peer_policies (peer_id, strategy, rebalance_mode, "
        "fee_ppm_target, tags, updated_at) VALUES ('canary', 'dynamic', "
        "'enabled', NULL, NULL, 1)"
    )
    conn.commit()
    conn.close()

    # Current code opening a "newer" DB must not corrupt or drop data.
    _migrate_current(path)

    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT peer_id FROM peer_policies WHERE peer_id = 'canary'"
    ).fetchone()
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    conn.close()
    assert row is not None, "newer-schema DB lost data on open"
    # Documents current behavior: version is left untouched (never read/reset).
    assert version == 999999


def test_interrupted_migration_completes_on_retry(monkeypatch, tmp_path):
    """Crash-during-upgrade safety: if initialize() aborts partway through
    (simulated by raising inside a late migration step), a retry completes
    cleanly and reaches the full schema. Safety rests on idempotence, since
    initialize() runs in autocommit and is not wrapped in one transaction."""
    path = os.path.join(str(tmp_path), "interrupted.db")
    _build_old_db(OLDEST_ANCHOR, path)

    # First attempt: blow up inside a late migration step (temporal profile is
    # called near the end of initialize(), after most tables are created but
    # before the planner tables), simulating a crash mid-upgrade.
    original = CurrentDatabase._migrate_temporal_profile_schema

    def _boom(self, conn):
        raise RuntimeError("simulated crash during upgrade")

    monkeypatch.setattr(CurrentDatabase, "_migrate_temporal_profile_schema", _boom)
    db = CurrentDatabase(path, MagicMock())
    with pytest.raises(RuntimeError):
        db.initialize()
    _close(db)

    # Sanity: the migration really was interrupted (planner tables, created
    # after the crash point, are absent).
    partial_tables, _ = _snapshot(path)
    assert "planner_recycle_ops" not in partial_tables

    # Retry with the real method: must complete and reach the full schema.
    monkeypatch.setattr(
        CurrentDatabase, "_migrate_temporal_profile_schema", original
    )
    _migrate_current(path)
    tables, _ = _snapshot(path)
    assert "planner_recycle_ops" in tables
    # channel_states must have the v2 columns added by the resumed migration.
    assert "temporal_profile_json" in tables["channel_states"]
