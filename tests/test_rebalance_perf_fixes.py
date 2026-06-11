"""Regression tests for the per-cycle RPC/DB scaler fixes.

Covers:
1. The post-rebalance anchor row is fetched once per cooled channel in
   _build_snapshot and shared by the fill-fraction cooldown helper and the
   drift override (was 2 point queries per cooled channel).
2. v3 router reuses a cycle-scoped exclude layer per unique exclude set
   (was create + N updates + remove per price_pair).
3. Final-hop policy lookups are served from the broadcast listpeerchannels
   cache instead of an uncached per-peer RPC.
4. Hive router caches askrene-listlayers for the cycle.
5. The empirical dest success-rate DB aggregate is memoized per cycle.
6. A pending-settlement result without a payment_hash is recorded as a
   terminal failure (never an unsweepable parked row).
7. Skip-log emission is aggregated for non-actionable reasons.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.rebalance_router_v2 import RouteResult

OUR_ID = "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3"
SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64


# ---------------------------------------------------------------------------
# 1. Anchor row fetched once per cooled channel
# ---------------------------------------------------------------------------


class _AnchorCountingDb:
    """Counts get_last_post_rebalance_state point queries."""

    def __init__(self, last_ts, anchor):
        self._last_ts = last_ts
        self._anchor = anchor
        self.anchor_calls = 0

    def get_last_rebalance_times(self):
        return {"100x1x0": self._last_ts} if self._last_ts else {}

    def get_last_rebalance_time(self, channel_id, reason_code=None):
        return self._last_ts

    def get_last_post_rebalance_state(self, channel_id):
        self.anchor_calls += 1
        return self._anchor


def _cooled_channels():
    return {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": SRC_PEER,
                "short_channel_id": "100x1x0",
                "total_msat": 1_000_000_000,
                "our_amount_msat": 200_000_000,
                "updates": {"remote": {"fee_proportional_millionths": 10}},
            }
        ]
    }


def _snapshot_engine(mock_plugin, db):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": []}
    mock_plugin.rpc.listpeerchannels.return_value = _cooled_channels()
    return RebalanceEngine(
        plugin=mock_plugin, config=Config(dry_run=True), database=db
    )


def test_build_snapshot_fetches_anchor_once_per_cooled_channel(mock_plugin):
    """The cooldown fill-fraction helper AND the drift override must share
    one anchor fetch — not issue one point query each."""
    anchor = {"post_local_ratio": 0.6, "amount_sats": 50_000}
    db = _AnchorCountingDb(last_ts=int(time.time()) - 60, anchor=anchor)
    engine = _snapshot_engine(mock_plugin, db)

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert db.anchor_calls == 1


def test_build_snapshot_anchor_drives_both_cooldown_and_drift(mock_plugin):
    """The shared anchor still feeds both consumers: a large drift below the
    anchored post-rebalance ratio sets the override flag."""
    # Channel local ratio is 0.2; anchor says it was 0.6 after the last
    # rebalance -> drift of 0.4 >= default 0.30 threshold.
    anchor = {"post_local_ratio": 0.6, "amount_sats": 50_000}
    db = _AnchorCountingDb(last_ts=int(time.time()) - 60, anchor=anchor)
    engine = _snapshot_engine(mock_plugin, db)

    captured = {}
    import modules.rebalance_engine_v2 as engine_mod

    real_builder = engine_mod.build_state_snapshot

    def capture(normalized, *args, **kwargs):
        captured["normalized"] = normalized
        return real_builder(normalized, *args, **kwargs)

    with patch.object(engine_mod, "build_state_snapshot", side_effect=capture):
        engine._build_snapshot()

    entry = captured["normalized"][0]
    assert entry["cooldown_active"] is True
    assert entry["cooldown_override"] is True
    assert db.anchor_calls == 1


def test_build_snapshot_no_anchor_query_when_not_cooled(mock_plugin):
    """Channels outside the base cooldown window never pay the point query."""
    db = _AnchorCountingDb(last_ts=None, anchor=None)
    engine = _snapshot_engine(mock_plugin, db)

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert db.anchor_calls == 0
