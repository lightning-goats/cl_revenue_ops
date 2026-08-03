"""Upgrade A: base_fee_msat policy (2026-04-22).

The per-role split (intra-fleet vs everyone else) retired with cl-mycelium:
both "off" (default) and "adaptive" apply the legacy cfg.base_fee_msat to
every channel.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from modules.fee_authority import FeeAuthorityGate

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

    fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
    fc.data_service = _make_data_service(mock_plugin)

    fc.set_channel_fee(scid, 200, manual=True)

    assert mock_plugin.rpc.setchannel.call_args.kwargs["feebase"] == 777


def test_policy_adaptive_uses_legacy_base_fee(mock_plugin, mock_database):
    """The per-role split retired with cl-mycelium: "adaptive" now applies
    the legacy cfg.base_fee_msat (default 0) to every peer."""
    from modules.config import Config
    from modules.fee_controller import FeeController

    scid = "123x456x0"
    peer = "02" + "c" * 64
    cfg = Config(
        min_fee_ppm=10, max_fee_ppm=5000,
        base_fee_msat=0,
        base_fee_policy="adaptive",
        dry_run=False,
    )
    mock_plugin.rpc.listpeerchannels.return_value = _lpc_payload(scid, peer)
    mock_plugin.rpc.setchannel = MagicMock()
    mock_database.get_fee_strategy_state.return_value = _strategy_state()
    mock_database.record_fee_change = MagicMock()

    fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
    fc.data_service = _make_data_service(mock_plugin)

    fc.set_channel_fee(scid, 200, manual=True)

    assert mock_plugin.rpc.setchannel.call_args.kwargs["feebase"] == 0


def test_explicit_override_beats_policy(mock_plugin, mock_database):
    """base_fee_msat_override on set_channel_fee is absolute — it
    should win over both the legacy fallback and the adaptive policy
    so explicit manual callers can force a value."""
    from modules.config import Config
    from modules.fee_controller import FeeController

    scid = "123x456x0"
    peer = "02" + "d" * 64
    cfg = Config(
        min_fee_ppm=10, max_fee_ppm=5000,
        base_fee_msat=0,
        base_fee_policy="adaptive",
        dry_run=False,
    )
    mock_plugin.rpc.listpeerchannels.return_value = _lpc_payload(scid, peer)
    mock_plugin.rpc.setchannel = MagicMock()
    mock_database.get_fee_strategy_state.return_value = _strategy_state()
    mock_database.record_fee_change = MagicMock()

    fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
    fc.data_service = _make_data_service(mock_plugin)

    fc.set_channel_fee(scid, 200, manual=True, base_fee_msat_override=5555)

    assert mock_plugin.rpc.setchannel.call_args.kwargs["feebase"] == 5555
