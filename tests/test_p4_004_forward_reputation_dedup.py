"""P4-004: record_forward_and_reputation must not re-increment peer reputation
when the forward INSERT OR IGNORE was a duplicate (rowcount==0). The fee side
is already deduped by the unique index; the reputation upsert ran
unconditionally, double-counting on startup-hydration/live overlap replays.
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


PEER = "b" * 66


def _make_db(tmp_path):
    db_path = os.path.join(tmp_path, "fwd_rep.db")
    db = Database(db_path, MagicMock())
    db.initialize()
    return db


def _forward():
    return {
        "in_channel": "111x1x0",
        "out_channel": "222x2x0",
        "in_msat": 1_000_000,
        "out_msat": 999_000,
        "fee_msat": 1000,
        "received_time": 1_700_000_000,
        "resolved_time": 1_700_000_001,
        "resolution_time": 1.0,
    }


def test_p4_004_duplicate_forward_increments_reputation_once(tmp_path):
    db = _make_db(tmp_path)
    fwd = _forward()

    db.record_forward_and_reputation(fwd, PEER, success=True)
    rep1 = db.get_peer_reputation(PEER)
    assert rep1["successes"] == 1

    # Replaying the exact same forward is a duplicate (INSERT OR IGNORE
    # rowcount==0); reputation must NOT increment again.
    db.record_forward_and_reputation(dict(fwd), PEER, success=True)
    rep2 = db.get_peer_reputation(PEER)
    assert rep2["successes"] == 1, "duplicate forward double-counted reputation"


def test_p4_004_duplicate_failure_not_double_counted(tmp_path):
    db = _make_db(tmp_path)
    fwd = _forward()
    db.record_forward_and_reputation(fwd, PEER, success=False)
    db.record_forward_and_reputation(dict(fwd), PEER, success=False)
    rep = db.get_peer_reputation(PEER)
    assert rep["failures"] == 1


def test_p4_004_distinct_forwards_still_accumulate(tmp_path):
    db = _make_db(tmp_path)
    for i in range(3):
        fwd = _forward()
        fwd["out_msat"] = 999_000 - i  # make each row unique
        fwd["received_time"] = 1_700_000_000 + i
        db.record_forward_and_reputation(fwd, PEER, success=True)
    rep = db.get_peer_reputation(PEER)
    assert rep["successes"] == 3
