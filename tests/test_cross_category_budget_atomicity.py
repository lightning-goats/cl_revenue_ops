"""DD1 / P1-003: the unified budget must be enforced ACROSS categories.

The generic spend ledger (spend_reservations, via ``reserve_spend``) and the
rebalance ledger (budget_reservations, via ``reserve_budget`` /
``_reserve_budget_atomic``) share ONE unified budget. Historically each enforced
only its own table under a different lock, so the two categories could jointly
exceed the budget (a cross-category TOCTOU). The fix enforces the full
cross-category total inside each reserve's ``BEGIN IMMEDIATE`` transaction, which
the WAL single-writer lock serializes — so no interleaving can overshoot.
"""

import os
import sys
import threading
from unittest.mock import MagicMock

# Mock pyln.client before importing modules.
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402


def _make_db(tmp_path):
    db = Database(os.path.join(tmp_path, "xcat.db"), MagicMock())
    db.initialize()
    return db


def _active_total(db):
    conn = db._get_connection()
    gen = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    reb = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM budget_reservations WHERE status='active'"
    ).fetchone()[0]
    return int(gen or 0) + int(reb or 0)


def test_generic_then_rebalance_cannot_exceed_budget(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    since = 0
    # Generic reserve takes 600 of the 1000 budget.
    assert db.reserve_spend("g1", 600, "boltz",
                            effective_budget_sats=budget, since_timestamp=since) is True
    # Rebalance reserve of 600 would bring the cross-category total to 1200 > 1000.
    ok, _ = db.reserve_budget("r1", 600, "chan", budget_limit=budget, since_timestamp=since)
    assert ok is False
    assert _active_total(db) <= budget


def test_rebalance_then_generic_cannot_exceed_budget(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    since = 0
    ok, _ = db.reserve_budget("r1", 600, "chan", budget_limit=budget, since_timestamp=since)
    assert ok is True
    # Generic reserve of 600 would jointly exceed the budget.
    assert db.reserve_spend("g1", 600, "boltz",
                            effective_budget_sats=budget, since_timestamp=since) is False
    assert _active_total(db) <= budget


def test_generic_reserve_within_budget_passes(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    assert db.reserve_spend("g1", 400, "boltz",
                            effective_budget_sats=budget, since_timestamp=0) is True
    ok, _ = db.reserve_budget("r1", 500, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True
    assert _active_total(db) == 900


def test_concurrent_cross_category_reserves_never_exceed(tmp_path):
    """Two threads racing a generic reserve and a rebalance reserve against the
    same budget must never jointly overshoot it."""
    budget = 1000
    db = _make_db(tmp_path)
    results = {}
    barrier = threading.Barrier(2)

    def gen():
        barrier.wait()
        results["gen"] = db.reserve_spend("g1", 600, "boltz",
                                          effective_budget_sats=budget, since_timestamp=0)

    def reb():
        barrier.wait()
        ok, _ = db.reserve_budget("r1", 600, "chan", budget_limit=budget, since_timestamp=0)
        results["reb"] = ok

    for _ in range(50):
        # fresh db each iteration to re-run the race cleanly
        db = _make_db(tmp_path)
        results = {}
        barrier = threading.Barrier(2)
        t1 = threading.Thread(target=gen)
        t2 = threading.Thread(target=reb)
        t1.start(); t2.start()
        t1.join(); t2.join()
        # At most one of the two 600-sat reserves can win (1200 > 1000).
        assert not (results.get("gen") and results.get("reb")), (
            "both cross-category reserves succeeded and jointly exceeded the budget"
        )
        assert _active_total(db) <= budget
