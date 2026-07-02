"""DD8/RES-3: swap journal is bounded by a keep-last-180-days age cap.

The journal dedups by swap id but historically had no age cap, so completed
swap records accumulated on disk without bound. Old records are historical
only (non-behavioral), so pruning entries older than the retention window on
the next write keeps the file bounded without affecting behavior.
"""

import time
from unittest.mock import MagicMock

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager


def _make_manager(datadir):
    cfg = BoltzCliConfig(
        enabled=True,
        cli_path="/usr/local/bin/boltzcli",
        datadir=str(datadir),
        daily_budget_sats=3000,
        enforce_budget=True,
    )
    plugin = MagicMock()
    plugin.log = MagicMock()
    rpc = MagicMock()
    return BoltzCliManager(plugin, rpc, cfg)


def test_swap_journal_prunes_entries_older_than_window(tmp_path):
    mgr = _make_manager(tmp_path)
    now = int(time.time())
    window = mgr._SWAP_JOURNAL_MAX_AGE_SECONDS

    old_entry = {"id": "old-swap", "recorded_at": now - window - 86_400}
    recent_entry = {"id": "recent-swap", "recorded_at": now - 3_600}

    mgr._save_swap_journal([old_entry, recent_entry])

    loaded = mgr._load_swap_journal()
    ids = {rec.get("id") for rec in loaded}
    assert "recent-swap" in ids  # recent entry retained
    assert "old-swap" not in ids  # entry older than the window pruned


def test_swap_journal_retains_entry_dated_by_created_at(tmp_path):
    # An entry whose recorded_at is stale but whose created_at is recent is
    # still retained (uses the newest available timestamp).
    mgr = _make_manager(tmp_path)
    now = int(time.time())
    window = mgr._SWAP_JOURNAL_MAX_AGE_SECONDS

    entry = {
        "id": "boundary-swap",
        "recorded_at": now - window - 86_400,
        "created_at": now - 3_600,
    }
    mgr._save_swap_journal([entry])

    ids = {rec.get("id") for rec in mgr._load_swap_journal()}
    assert "boundary-swap" in ids
