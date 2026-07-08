"""Z-2: durable zero-fee hive corridor through hint staleness.

Covers:
- DB accessors hive_member_confirm / hive_member_last_confirmed (additive
  CREATE TABLE idiom, same shape as the lnplus tables).
- _hive_member_zero_fee_active falling back to a recent DB confirmation
  when live hive-hint data is stale/unavailable, and NOT falling back when
  hints are fresh and positively say "not a member".
- Confirmations surviving a simulated restart (new controller instance,
  same on-disk DB).
"""

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from modules.database import Database
from modules.config import Config
from modules.fee_controller import FeeController


def _make_db():
    path = os.path.join(tempfile.mkdtemp(prefix="hive_grace_test_"), "test.db")
    db = Database(path, MagicMock())
    db.initialize()
    return db, path


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


class TestHiveMemberConfirmationTable:
    def test_confirm_then_read_back(self):
        db, _ = _make_db()
        peer = "02" + "aa" * 32
        before = int(time.time())
        db.hive_member_confirm(peer)
        after = int(time.time())
        ts = db.hive_member_last_confirmed(peer)
        assert ts is not None
        assert before <= ts <= after

    def test_unknown_peer_returns_none(self):
        db, _ = _make_db()
        assert db.hive_member_last_confirmed("02" + "bb" * 32) is None

    def test_confirm_is_upsert_not_insert_error(self):
        db, _ = _make_db()
        peer = "02" + "cc" * 32
        db.hive_member_confirm(peer)
        first = db.hive_member_last_confirmed(peer)
        time.sleep(0.01)
        db.hive_member_confirm(peer)
        second = db.hive_member_last_confirmed(peer)
        assert second >= first

    def test_confirmations_persist_across_restart(self):
        """New Database instance pointed at the same file must see prior confirmations."""
        db, path = _make_db()
        peer = "02" + "dd" * 32
        db.hive_member_confirm(peer)
        ts = db.hive_member_last_confirmed(peer)

        db2 = Database(path, MagicMock())
        db2.initialize()
        assert db2.hive_member_last_confirmed(peer) == ts


class TestHiveZeroFeeGraceGate:
    """_hive_member_zero_fee_active durability through hint staleness."""

    def _fc(self, mock_plugin, db):
        cfg = Config(hive_zero_fee_stale_grace_seconds=604800)
        fc = FeeController(mock_plugin, cfg, db)
        return fc

    def test_member_fresh_hints_is_zero_fee_and_confirms(self, mock_plugin):
        db, _ = _make_db()
        fc = self._fc(mock_plugin, db)
        peer = "02" + "11" * 32
        adapter = MagicMock()
        adapter.get_membership_status.return_value = {
            "peer_id": peer, "known": True, "member": True,
            "fresh": True, "usable": True, "source": "datastore",
        }
        fc.hive_hints = adapter

        assert fc._hive_member_zero_fee_active(peer) is True
        # A positive membership result must persist a DB confirmation.
        assert db.hive_member_last_confirmed(peer) is not None

    def test_stale_within_grace_still_zero_fee_no_force_reprice(self, mock_plugin):
        db, _ = _make_db()
        fc = self._fc(mock_plugin, db)
        peer = "02" + "22" * 32
        # Prior confirmation 1 hour ago -- well within the 7-day grace.
        db.hive_member_confirm(peer)

        adapter = MagicMock()
        # Hints are now unavailable entirely (not usable).
        adapter.get_membership_status.return_value = {
            "peer_id": peer, "known": False, "member": False,
            "fresh": False, "usable": False, "source": "none",
        }
        fc.hive_hints = adapter

        assert fc._hive_member_zero_fee_active(peer) is True
        assert fc._check_hive_member_fee(peer) == 0
        # Must not have been queued for a forced reprice.
        assert fc._consume_hive_member_release(peer) is False

    def test_stale_beyond_grace_is_repriced(self, mock_plugin):
        db, _ = _make_db()
        fc = self._fc(mock_plugin, db)
        peer = "02" + "33" * 32
        db.hive_member_confirm(peer)
        # Backdate the confirmation beyond the 7-day grace window.
        conn = db._get_connection()
        conn.execute(
            "UPDATE hive_member_confirmations SET last_confirmed_at = ? WHERE pubkey = ?",
            (int(time.time()) - 604801, peer),
        )

        adapter = MagicMock()
        adapter.get_membership_status.return_value = {
            "peer_id": peer, "known": False, "member": False,
            "fresh": False, "usable": False, "source": "none",
        }
        fc.hive_hints = adapter

        assert fc._hive_member_zero_fee_active(peer) is False
        assert fc._check_hive_member_fee(peer) is None

    def test_fresh_hints_saying_not_member_ignore_recent_confirmation(self, mock_plugin):
        """A recent DB confirmation must never override fresh hints that
        positively say the peer is NOT currently a member."""
        db, _ = _make_db()
        fc = self._fc(mock_plugin, db)
        peer = "02" + "44" * 32
        db.hive_member_confirm(peer)  # very recent

        adapter = MagicMock()
        adapter.get_membership_status.return_value = {
            "peer_id": peer, "known": True, "member": False,
            "fresh": True, "usable": True, "source": "datastore",
        }
        fc.hive_hints = adapter

        assert fc._hive_member_zero_fee_active(peer) is False
        assert fc._check_hive_member_fee(peer) is None

    def test_grace_active_without_database_is_safe(self, mock_plugin):
        """No database wired -> no crash, no grace, standard release path."""
        fc = self._fc(mock_plugin, None)
        peer = "02" + "55" * 32
        adapter = MagicMock()
        adapter.get_membership_status.return_value = {
            "peer_id": peer, "known": False, "member": False,
            "fresh": False, "usable": False, "source": "none",
        }
        fc.hive_hints = adapter

        assert fc._hive_member_zero_fee_active(peer) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
