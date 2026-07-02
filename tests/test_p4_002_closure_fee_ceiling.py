"""P4-002: closure fee components (closure_fee/htlc_sweep/penalty) must NOT be
clamped to the 50000-sat _sanitize_fee ceiling. A real large force-close
(multi-HTLC sweep / fee spike) is legitimate; truncating it under-counts the
closure cost and over-states lifetime P&L. Use the closure-appropriate
(10 BTC / _sanitize_amount) ceiling instead.
"""

import os
import sys

from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


PEER = "a" * 66
CHAN = "111x222x0"


def _make_db(tmp_path):
    db_path = os.path.join(tmp_path, "closure.db")
    db = Database(db_path, MagicMock())
    db.initialize()
    return db


def _stored_row(db):
    row = db._get_connection().execute(
        "SELECT closure_fee_sats, htlc_sweep_fee_sats, penalty_fee_sats, "
        "total_closure_cost_sats FROM channel_closure_costs WHERE channel_id = ?",
        (CHAN,),
    ).fetchone()
    return row


def test_p4_002_large_force_close_fee_recorded_in_full(tmp_path):
    """A 120000-sat closure fee must be recorded in full, not clamped to
    50000."""
    db = _make_db(tmp_path)
    assert db.record_channel_closure(
        channel_id=CHAN,
        peer_id=PEER,
        close_type="local_unilateral",
        closure_fee_sats=120_000,
        htlc_sweep_fee_sats=80_000,
        penalty_fee_sats=0,
    ) is True

    row = _stored_row(db)
    assert row["closure_fee_sats"] == 120_000, "closure fee was truncated"
    assert row["htlc_sweep_fee_sats"] == 80_000, "htlc sweep fee was truncated"
    assert row["total_closure_cost_sats"] == 200_000


def test_p4_002_negative_component_still_clamped_to_zero(tmp_path):
    """Behaviour preserved: a negative component is still floored at 0."""
    db = _make_db(tmp_path)
    assert db.record_channel_closure(
        channel_id=CHAN,
        peer_id=PEER,
        close_type="mutual",
        closure_fee_sats=-5,
        htlc_sweep_fee_sats=0,
        penalty_fee_sats=0,
    ) is True
    row = _stored_row(db)
    assert row["closure_fee_sats"] == 0


def test_p4_002_absurd_component_bounded_by_amount_ceiling(tmp_path):
    """The 10 BTC amount ceiling still bounds an absurd value."""
    db = _make_db(tmp_path)
    huge = db.MAX_AMOUNT_SATS + 1
    assert db.record_channel_closure(
        channel_id=CHAN,
        peer_id=PEER,
        close_type="local_unilateral",
        closure_fee_sats=huge,
        htlc_sweep_fee_sats=0,
        penalty_fee_sats=0,
    ) is True
    row = _stored_row(db)
    assert row["closure_fee_sats"] == db.MAX_AMOUNT_SATS
