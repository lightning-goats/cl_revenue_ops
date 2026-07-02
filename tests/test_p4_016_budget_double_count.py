"""P4-016: rebalance reserve double-counted the generic ledger.

The rebalance reserve pre-subtracted non-rebalance costs into ``budget_limit``
(effective_budget - ext_spent - ext_reserved) AND ``_reserve_budget_atomic``
independently SUMs the same spend_events + spend_reservations into committed →
each generic/boltz spend counted twice. Direction is conservative (it starves
the rebalance budget on a capex-active node) but it double-counts, refuting the
clean single-count partition. DD1's authoritative rail is the in-txn sum, so the
caller must pass the full effective budget and let the atomic transaction count
each category exactly once (mirroring the generic reserve_spend path).
"""

import os
import sys
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402


def _make_db(tmp_path):
    db = Database(os.path.join(tmp_path, "p4016.db"), MagicMock())
    db.initialize()
    return db


def _make_engine(db):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    cfg = Config(dry_run=True, daily_budget_sats=1000)
    engine = RebalanceEngine(plugin=plugin, config=cfg, database=db)
    # Full unified budget = 1000.
    engine.global_budget_limit_provider = lambda: {"effective_budget_sats": 1000}
    # The non-rebalance cost provider reports the SAME 300-sat generic spend
    # event that also lives in spend_events (the double-representation the bug
    # counts twice).
    engine.external_liquidity_cost_provider = lambda: {
        "spent_24h_sats": 300,
        "reserved_24h_sats": 0,
    }
    return engine


def test_generic_spend_counted_once_in_rebalance_reserve(tmp_path):
    from modules.rebalance_types_v2 import PairCandidate

    db = _make_db(tmp_path)
    # A 300-sat generic ledger spend event, inside the daily window.
    assert db.record_spend_event(
        event_id="gen1", category="ledger", amount_sats=300,
        timestamp=int(time.time()),
    )

    engine = _make_engine(db)
    # Per-attempt fee ceiling = 500. Budget 1000, one generic spend of 300.
    # Counted once: 1000 - 300 = 700 headroom → 500 fits → reserve succeeds.
    # Counted twice (bug): 1000 - 300(ext) - 300(atomic) = 400 → 500 rejected.
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=50_000,
        pair_budget_sats=500,
    )

    reserved, result = engine._reserve_execution_budget(pair, reservation_id="r1")
    assert reserved is True, (
        f"generic spend double-counted → rebalance budget starved: {result}"
    )
