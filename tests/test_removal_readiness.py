"""PR 11 (gap-closure Phase H): compatibility-window removal prep.

Two halves:
1. The migration scanner works — it detects each blocker class and
   reports READY on a clean system (tested against synthetic DBs).
2. STAGED post-removal tests (xfail until the 2026-08-12 checklist
   executes): after removal, `rebalance_min_profit` behaves like any
   unknown key. Un-xfail these in the removal PR — they are its
   acceptance tests.
"""
import pathlib
import sqlite3
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCANNER = REPO / "tools" / "deprecation_scan.py"


def _run(db, config):
    return subprocess.run(
        [sys.executable, str(SCANNER), "--db", str(db),
         "--config", str(config)],
        capture_output=True, text=True)


def _mkdb(path, override=None, active_legacy=0):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE config_overrides "
                 "(key TEXT PRIMARY KEY, value TEXT)")
    if override is not None:
        conn.execute("INSERT INTO config_overrides VALUES (?, ?)",
                     ("rebalance_min_profit", str(override)))
    conn.execute("CREATE TABLE budget_reservations "
                 "(reservation_id TEXT, status TEXT)")
    for i in range(active_legacy):
        conn.execute("INSERT INTO budget_reservations VALUES (?, ?)",
                     (f"legacy-{i}", "active"))
    conn.commit()
    conn.close()


class TestScanner:
    def test_clean_system_is_ready(self, tmp_path):
        db = tmp_path / "r.db"
        _mkdb(db)
        config = tmp_path / "config"
        config.write_text("alias=node\n# rebalance_min_profit in a "
                          "comment does not count\n")
        result = _run(db, config)
        assert result.returncode == 0, result.stdout
        assert "READY" in result.stdout

    def test_config_file_usage_blocks(self, tmp_path):
        db = tmp_path / "r.db"
        _mkdb(db)
        config = tmp_path / "config"
        config.write_text("revenue-ops-rebalance-min-profit=25\n")
        result = _run(db, config)
        assert result.returncode == 1
        assert "config file" in result.stdout

    def test_persisted_override_blocks(self, tmp_path):
        db = tmp_path / "r.db"
        _mkdb(db, override="42")
        config = tmp_path / "config"
        config.write_text("")
        result = _run(db, config)
        assert result.returncode == 1
        assert "config_overrides" in result.stdout

    def test_active_legacy_reservations_block(self, tmp_path):
        db = tmp_path / "r.db"
        _mkdb(db, active_legacy=2)
        config = tmp_path / "config"
        config.write_text("")
        result = _run(db, config)
        assert result.returncode == 1
        assert "2 ACTIVE legacy" in result.stdout

    def test_missing_tables_do_not_block(self, tmp_path):
        db = tmp_path / "r.db"
        sqlite3.connect(db).close()  # empty db, no tables
        config = tmp_path / "missing-config"
        result = _run(db, config)
        assert result.returncode == 0

    def test_scanner_is_read_only(self, tmp_path):
        db = tmp_path / "r.db"
        _mkdb(db, override="42")
        before = db.read_bytes()
        _run(db, tmp_path / "config")
        assert db.read_bytes() == before


def test_no_other_deprecated_no_ops_undisclosed():
    """Phase H item 6: the announced window covers EVERYTHING deprecated.
    A new member of DEPRECATED_RUNTIME_KEYS must be added to the
    compatibility policy and this pin in the same commit."""
    from modules.config import DEPRECATED_RUNTIME_KEYS
    assert set(DEPRECATED_RUNTIME_KEYS) == {"rebalance_min_profit"}


class TestStagedPostRemoval:
    """Acceptance tests for the ≥2026-08-12 removal PR. xfail(strict)
    until then: they FAIL today (the shim exists), and the xfail marker
    comes off in the removal commit."""

    @pytest.mark.xfail(strict=True,
                       reason="staged for the 2026-08-12 removal: the "
                              "no-op shim still exists by design")
    def test_key_is_unknown_after_removal(self):
        from modules.config import CONFIG_FIELD_TYPES, Config
        assert "rebalance_min_profit" not in CONFIG_FIELD_TYPES
        assert not hasattr(Config(), "rebalance_min_profit")

    @pytest.mark.xfail(strict=True,
                       reason="staged for the 2026-08-12 removal")
    def test_stale_override_skipped_with_warning(self):
        from unittest.mock import MagicMock

        from modules.config import Config
        cfg = Config()
        database = MagicMock()
        database.get_all_config_overrides.return_value = {
            "rebalance_min_profit": "42"}
        database.get_config_version.return_value = 1
        warnings = cfg.load_overrides(database)
        # Post-removal: unknown key -> not applied, not fatal.
        assert not hasattr(cfg, "rebalance_min_profit")
        assert warnings  # startup surfaces the stale override
