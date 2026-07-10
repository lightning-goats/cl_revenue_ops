"""Upgrade A: adaptive base_fee_msat per channel role (2026-04-22).

Two-bucket v1: hive fleet members get base_fee_msat_intra_fleet (0);
all other peers get base_fee_msat_non_hive (default 1000 msat).

Back-compat: when base_fee_policy="off" (default), the legacy
cfg.base_fee_msat path is used for every channel.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_data_service(mock_plugin):
    ds = MagicMock()
    ds.get_peer_channels.side_effect = lambda peer_id=None, **kw: (
        mock_plugin.rpc.listpeerchannels(peer_id) if peer_id is not None
        else mock_plugin.rpc.listpeerchannels()
    )
    ds.get_channels.side_effect = lambda **kw: mock_plugin.rpc.listchannels(**kw)
    ds.get_node_id.side_effect = lambda: mock_plugin.rpc.getinfo().get("id", "")
    ds.set_channel.side_effect = lambda **kw: mock_plugin.rpc.setchannel(**kw)
    ds.get_feerates.side_effect = lambda **kw: mock_plugin.rpc.feerates(**kw)
    ds.get_askrene_layers.side_effect = lambda: mock_plugin.rpc.call("askrene-listlayers", {})
    return ds


def _lpc_payload(scid, peer_id, fee_ppm=100):
    return {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "short_channel_id": scid,
                "peer_id": peer_id,
                "spendable_msat": 500_000_000,
                "receivable_msat": 500_000_000,
                "total_msat": 1_000_000_000,
                "updates": {
                    "local": {"fee_base_msat": 0, "fee_proportional_millionths": fee_ppm}
                },
            }
        ]
    }


def _strategy_state():
    return {
        "last_revenue_rate": 0.0,
        "last_fee_ppm": 0,
        "trend_direction": 1,
        "step_ppm": 50,
        "last_update": 0,
        "consecutive_same_direction": 0,
        "is_sleeping": 0,
        "sleep_until": 0,
        "stable_cycles": 0,
        "last_broadcast_fee_ppm": 0,
        "last_state": "balanced",
        "forward_count_since_update": 0,
        "last_volume_sats": 0,
        "v2_state_json": "{}",
    }


class _FakeHints:
    """Minimal hive_hints stand-in for classification tests."""
    def __init__(self, hive_peers):
        self._hive = set(hive_peers)

    def is_hive_member(self, peer_id):
        return peer_id in self._hive


def test_policy_off_uses_legacy_base_fee_msat(mock_plugin, mock_database):
    from modules.config import Config
    from modules.fee_controller import FeeController

    scid = "123x456x0"
    peer = "02" + "a" * 64
    cfg = Config(
        min_fee_ppm=10, max_fee_ppm=5000,
        base_fee_msat=777,
        base_fee_policy="off",
        dry_run=False,
    )
    mock_plugin.rpc.listpeerchannels.return_value = _lpc_payload(scid, peer)
    mock_plugin.rpc.setchannel = MagicMock()
    mock_database.get_fee_strategy_state.return_value = _strategy_state()
    mock_database.record_fee_change = MagicMock()

    fc = FeeController(mock_plugin, cfg, mock_database)
    fc.data_service = _make_data_service(mock_plugin)

    fc.set_channel_fee(scid, 200, manual=True)

    assert mock_plugin.rpc.setchannel.call_args.kwargs["feebase"] == 777


def test_policy_adaptive_without_hints_falls_back_to_legacy(mock_plugin, mock_database):
    """Plugin startup race: hive_hints is injected AFTER FeeController is
    constructed. If the first fee tick lands in that window, we must NOT
    charge intra-fleet channels the non-hive base. Falling back to the
    legacy cfg.base_fee_msat (default 0) is the safe path."""
    from modules.config import Config
    from modules.fee_controller import FeeController

    scid = "123x456x0"
    peer = "02" + "c" * 64
    cfg = Config(
        min_fee_ppm=10, max_fee_ppm=5000,
        base_fee_msat=0,
        base_fee_policy="adaptive",
        base_fee_msat_intra_fleet=0,
        base_fee_msat_non_hive=1000,
        dry_run=False,
    )
    mock_plugin.rpc.listpeerchannels.return_value = _lpc_payload(scid, peer)
    mock_plugin.rpc.setchannel = MagicMock()
    mock_database.get_fee_strategy_state.return_value = _strategy_state()
    mock_database.record_fee_change = MagicMock()

    fc = FeeController(mock_plugin, cfg, mock_database)
    fc.data_service = _make_data_service(mock_plugin)
    # hive_hints intentionally left as None

    fc.set_channel_fee(scid, 200, manual=True)

    # Legacy fallback while hive_hints is unset, not 1000 msat.
    assert mock_plugin.rpc.setchannel.call_args.kwargs["feebase"] == 0


def test_explicit_override_beats_policy(mock_plugin, mock_database):
    """base_fee_msat_override on set_channel_fee is absolute — it
    should win over both the legacy fallback and the adaptive policy
    so capacity_planner or manual callers can force a value."""
    from modules.config import Config
    from modules.fee_controller import FeeController

    scid = "123x456x0"
    peer = "02" + "d" * 64
    cfg = Config(
        min_fee_ppm=10, max_fee_ppm=5000,
        base_fee_msat=0,
        base_fee_policy="adaptive",
        base_fee_msat_intra_fleet=0,
        base_fee_msat_non_hive=1000,
        dry_run=False,
    )
    mock_plugin.rpc.listpeerchannels.return_value = _lpc_payload(scid, peer)
    mock_plugin.rpc.setchannel = MagicMock()
    mock_database.get_fee_strategy_state.return_value = _strategy_state()
    mock_database.record_fee_change = MagicMock()

    fc = FeeController(mock_plugin, cfg, mock_database)
    fc.data_service = _make_data_service(mock_plugin)
    fc.hive_hints = _FakeHints(hive_peers=set())

    fc.set_channel_fee(scid, 200, manual=True, base_fee_msat_override=5555)

    assert mock_plugin.rpc.setchannel.call_args.kwargs["feebase"] == 5555
