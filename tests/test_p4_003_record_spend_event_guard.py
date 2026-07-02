"""P4-003: record_spend_event must reject a non-positive amount_sats (mirror
reserve_spend's amount<=0 guard). A negative amount SUM()s into the committed
spend and *lowers* it, raising remaining budget in the overspend-permitting
direction.
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


def _make_db(tmp_path):
    db_path = os.path.join(tmp_path, "spend.db")
    db = Database(db_path, MagicMock())
    db.initialize()
    return db


def test_p4_003_negative_amount_rejected_and_does_not_lower_committed(tmp_path):
    db = _make_db(tmp_path)
    assert db.record_spend_event("evt-pos", "boltz", 1000) is True
    assert db.get_category_spend_sats("boltz") == 1000

    # A negative amount must be rejected outright.
    assert db.record_spend_event("evt-neg", "boltz", -500) is False
    # And must NOT have subtracted from committed spend.
    assert db.get_category_spend_sats("boltz") == 1000


def test_p4_003_zero_amount_rejected(tmp_path):
    db = _make_db(tmp_path)
    assert db.record_spend_event("evt-zero", "boltz", 0) is False
    assert db.get_category_spend_sats("boltz") == 0


def test_p4_003_positive_amount_still_recorded(tmp_path):
    db = _make_db(tmp_path)
    assert db.record_spend_event("evt-ok", "rebalance", 250) is True
    assert db.get_category_spend_sats("rebalance") == 250
