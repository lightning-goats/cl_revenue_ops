import sys
import os
import time
from unittest.mock import MagicMock
from typing import Union

import pytest

# Ensure project root is importable (matches other tests)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from modules.fee_authority import FeeAuthorityGate

def _make_data_service(mock_plugin):
    """Build a data_service MagicMock that delegates to mock_plugin.rpc.

    This lets tests configure mock_plugin.rpc.* as before while the
    FeeController calls data_service.* internally.  set_channel also
    forwards to mock_plugin.rpc.setchannel so existing call assertions work.
    """
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


def _listpeerchannels_payload(
    channel_id: str,
    peer_id: str,
    fee_ppm: int = 100,
    htlc_minimum_msat: Union[int, str] = 0,
    htlc_maximum_msat: Union[int, str] = 0,
):
    return {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "short_channel_id": channel_id,
                "peer_id": peer_id,
                "spendable_msat": 500_000_000,
                "receivable_msat": 500_000_000,
                "total_msat": 1_000_000_000,
                "htlc_minimum_msat": htlc_minimum_msat,
                "htlc_maximum_msat": htlc_maximum_msat,
                "updates": {
                    "local": {
                        "fee_base_msat": 0,
                        "fee_proportional_millionths": fee_ppm
                    }
                }
            }
        ]
    }


def _listpeerchannels_current_payload(
    channel_id: str,
    peer_id: str,
    fee_ppm: int = 100,
    full_channel_id: str = "",
    minimum_htlc_out_msat: Union[int, str] = 0,
    maximum_htlc_out_msat: Union[int, str] = 0,
):
    return {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "short_channel_id": channel_id,
                "channel_id": full_channel_id or ("11" * 32),
                "peer_id": peer_id,
                "spendable_msat": 500_000_000,
                "receivable_msat": 500_000_000,
                "total_msat": 1_000_000_000,
                "minimum_htlc_out_msat": minimum_htlc_out_msat,
                "maximum_htlc_out_msat": maximum_htlc_out_msat,
                "updates": {
                    "local": {
                        "fee_base_msat": 0,
                        "fee_proportional_millionths": fee_ppm,
                        "htlc_minimum_msat": minimum_htlc_out_msat,
                        "htlc_maximum_msat": maximum_htlc_out_msat,
                    }
                }
            }
        ]
    }


def _fee_strategy_state_dict():
    # Minimal dict for _get_cycle_state fee strategy loaders.
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


def _setchannel_kwargs(mock_plugin):
    return mock_plugin.rpc.setchannel.call_args.kwargs


def _disabled_fee_authority_gate(now: int = 11_000):
    from modules.fee_authority import FeeAuthorityGate

    gate = FeeAuthorityGate(enabled=True, now_fn=lambda: now)
    gate.set_enabled(False, reason="setconfig")
    return gate


def _blocked_fee_result(channel_id: str, fee_ppm: int, now: int = 11_000):
    return {
        "success": False,
        "channel_id": channel_id,
        "fee_ppm": fee_ppm,
        "message": "Fee authority disabled",
        "status": "blocked",
        "reason": "fee_authority_disabled",
        "operation": "set_channel_fee",
        "generation": 1,
        "transitioned_at": now,
    }


class TestFeeAuthorityExecutionBoundary:
    def test_constructor_accepts_shared_fee_authority_gate(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.fee_controller import FeeController

        gate = _disabled_fee_authority_gate()
        controller = FeeController(
            mock_plugin,
            Config(fee_authority_enabled=False),
            mock_database,
            fee_authority_gate=gate,
        )

        assert controller.fee_authority_gate is gate

    def test_disabled_gate_prevents_manual_state_and_rpc_mutation(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.fee_controller import ChannelCycleState, ChannelFeeState, FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        cfg = Config(
            min_fee_ppm=10,
            max_fee_ppm=5000,
            dry_run=False,
            fee_authority_enabled=False,
        )
        gate = _disabled_fee_authority_gate()
        controller = FeeController(
            mock_plugin,
            cfg,
            mock_database,
            fee_authority_gate=gate,
        )
        controller.data_service = _make_data_service(mock_plugin)
        controller._cycle_states[channel_id] = ChannelCycleState(
            is_sleeping=True,
            sleep_until=99_999,
            stable_cycles=4,
        )
        controller._channel_fee_states[channel_id] = ChannelFeeState(
            is_sleeping=True,
            sleep_until=99_999,
            stable_cycles=4,
        )
        mock_plugin.rpc.setchannel = MagicMock(return_value={})

        result = controller.set_channel_fee(
            channel_id,
            125,
            manual=True,
            channel_info={
                "short_channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            },
        )

        assert result == _blocked_fee_result(channel_id, 125)
        assert controller._cycle_states[channel_id].is_sleeping is True
        assert controller._cycle_states[channel_id].sleep_until == 99_999
        assert controller._cycle_states[channel_id].stable_cycles == 4
        assert controller._channel_fee_states[channel_id].is_sleeping is True
        assert controller._channel_fee_states[channel_id].sleep_until == 99_999
        assert controller._channel_fee_states[channel_id].stable_cycles == 4
        mock_database.update_fee_strategy_state.assert_not_called()
        controller.data_service.set_channel.assert_not_called()

    def test_disabled_gate_prevents_governor_and_dynamic_htlcmax_rpc_work(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        cfg = Config(
            min_fee_ppm=10,
            max_fee_ppm=5000,
            dry_run=False,
            fee_authority_enabled=False,
        )
        gate = _disabled_fee_authority_gate()
        controller = FeeController(
            mock_plugin,
            cfg,
            mock_database,
            fee_authority_gate=gate,
        )
        controller.data_service = _make_data_service(mock_plugin)
        controller._fee_governor_enabled = MagicMock(return_value=True)
        controller._governed_authorize_fee_broadcast = MagicMock(
            return_value=(True, "authorized")
        )
        mock_plugin.rpc.setchannel = MagicMock(return_value={})

        result = controller.set_channel_fee(
            channel_id,
            125,
            manual=False,
            htlcmax_msat=21_000_000,
            channel_info={
                "short_channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            },
        )

        assert result == _blocked_fee_result(channel_id, 125)
        controller._fee_governor_enabled.assert_not_called()
        controller._governed_authorize_fee_broadcast.assert_not_called()
        controller.data_service.set_channel.assert_not_called()


class TestChannelInfoShaping:
    def test_get_channels_info_preserves_htlc_minimum_and_maximum_msat(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        advertised_htlc_minimum_msat = "42000msat"
        advertised_htlc_maximum_msat = "21000000msat"

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(
            channel_id,
            peer_id,
            fee_ppm=100,
            htlc_minimum_msat=advertised_htlc_minimum_msat,
            htlc_maximum_msat=advertised_htlc_maximum_msat,
        )

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)
        channel_info = fc._get_channels_info()[channel_id]

        assert channel_info["htlc_minimum_msat"] == 42_000
        assert channel_info["htlc_min_msat"] == 42_000
        assert channel_info["htlc_maximum_msat"] == 21_000_000
        assert channel_info["htlc_max_msat"] == 21_000_000

    def test_get_channels_info_reads_current_listpeerchannels_htlc_fields(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_current_payload(
            channel_id,
            peer_id,
            fee_ppm=100,
            minimum_htlc_out_msat="42000msat",
            maximum_htlc_out_msat="21000000msat",
        )

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)
        channel_info = fc._get_channels_info()[channel_id]

        assert channel_info["htlc_minimum_msat"] == 42_000
        assert channel_info["htlc_min_msat"] == 42_000
        assert channel_info["htlc_maximum_msat"] == 21_000_000
        assert channel_info["htlc_max_msat"] == 21_000_000


class TestSetChannelFeeLimits:
    def test_set_channel_fee_enforces_limits_by_default(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        # State loaders/savers
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        fc.set_channel_fee(channel_id, 1, manual=True, enforce_limits=True)

        # Should clamp up to min_fee_ppm
        mock_plugin.rpc.setchannel.assert_called()
        call_kwargs = _setchannel_kwargs(mock_plugin)
        assert call_kwargs["feebase"] == 0
        assert call_kwargs["feeppm"] == 10

    def test_set_channel_fee_can_bypass_limits_for_force(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        fc.set_channel_fee(channel_id, 1, manual=True, enforce_limits=False)

        assert _setchannel_kwargs(mock_plugin)["feeppm"] == 1

    def test_force_operator_set_wins(self, mock_plugin, mock_database):
        # DD6/DEF-081: an explicit operator force=true set-fee
        # applies the operator's fee (clamped to [min,max] per DD2), NOT the
        # automatic fleet zero-fee policy.
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=1000, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(
            channel_id, peer_id, fee_ppm=100
        )
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        # Mirror the revenue-set-fee force=true call path: manual operator set,
        # economic clamp bypassed (enforce_limits=False).
        result = fc.set_channel_fee(
            channel_id, 250, manual=True, enforce_limits=False
        )

        assert result["success"] is True
        call_kwargs = _setchannel_kwargs(mock_plugin)
        # Operator's explicit fee wins.
        assert call_kwargs["feeppm"] == 250
        assert result["fee_ppm"] == 250

    def test_set_channel_fee_normalizes_colon_scid(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        result = fc.set_channel_fee("123:456:0", 125, manual=True)

        assert result["success"] is True
        assert _setchannel_kwargs(mock_plugin)["id"] == channel_id

    def test_set_channel_fee_resolves_full_channel_id_to_scid(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        full_channel_id = "ad" * 32
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_current_payload(
            channel_id,
            peer_id,
            fee_ppm=100,
            full_channel_id=full_channel_id,
        )
        mock_plugin.rpc.setchannel = MagicMock()
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        result = fc.set_channel_fee(full_channel_id, 125, manual=True)

        assert result["success"] is True
        assert _setchannel_kwargs(mock_plugin)["id"] == channel_id

    def test_manual_full_channel_id_syncs_canonical_scid_state(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import ChannelCycleState, ChannelFeeState, FeeController

        channel_id = "123x456x0"
        full_channel_id = "ad" * 32
        peer_id = "02" + "a" * 64
        now = int(time.time())

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_current_payload(
            channel_id,
            peer_id,
            fee_ppm=100,
            full_channel_id=full_channel_id,
        )
        mock_plugin.rpc.setchannel = MagicMock()
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_fee_ppm=100,
            last_broadcast_fee_ppm=100,
            last_update=now - 7200,
            is_sleeping=True,
            sleep_until=now + 3600,
            stable_cycles=2,
        )
        fc._channel_fee_states[channel_id] = ChannelFeeState(
            last_fee_ppm=100,
            last_broadcast_fee_ppm=100,
            last_update=now - 7200,
            is_sleeping=True,
            sleep_until=now + 3600,
            stable_cycles=2,
        )

        result = fc.set_channel_fee(full_channel_id, 125, manual=True)

        assert result["success"] is True
        assert result["channel_id"] == channel_id
        assert _setchannel_kwargs(mock_plugin)["id"] == channel_id
        assert full_channel_id not in fc._cycle_states
        assert full_channel_id not in fc._channel_fee_states
        assert fc._cycle_states[channel_id].last_broadcast_fee_ppm == 125
        assert fc._cycle_states[channel_id].is_sleeping is False
        assert fc._channel_fee_states[channel_id].last_broadcast_fee_ppm == 125
        assert fc._channel_fee_states[channel_id].is_sleeping is False
        assert mock_database.record_fee_change.call_args.kwargs["channel_id"] == channel_id

    def test_set_channel_fee_rejects_ambiguous_peer_id(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        peer_id = "02" + "a" * 64
        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                _listpeerchannels_payload("123x456x0", peer_id, fee_ppm=100)["channels"][0],
                _listpeerchannels_payload("123x456x1", peer_id, fee_ppm=110)["channels"][0],
            ]
        }
        mock_plugin.rpc.setchannel = MagicMock()
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        result = fc.set_channel_fee(peer_id, 125, manual=True)

        assert result["success"] is False
        assert "active channels" in result["message"]
        mock_plugin.rpc.setchannel.assert_not_called()

class TestSetChannelFeeHtlcMin:
    def _make_controller(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)
        return fc

    def test_set_channel_fee_omits_htlcmin_when_not_requested(self, mock_plugin, mock_database):
        fc = self._make_controller(mock_plugin, mock_database)

        fc.set_channel_fee("123x456x0", 125, manual=True)

        assert "htlcmin" not in _setchannel_kwargs(mock_plugin)

    def test_set_channel_fee_includes_htlcmin_when_requested(self, mock_plugin, mock_database):
        fc = self._make_controller(mock_plugin, mock_database)

        fc.set_channel_fee("123x456x0", 125, manual=True, htlcmin_msat=42_000)

        assert _setchannel_kwargs(mock_plugin)["htlcmin"] == "42000msat"

    def test_set_channel_fee_surfaces_applied_htlc_bounds_and_warnings(self, mock_plugin, mock_database):
        fc = self._make_controller(mock_plugin, mock_database)
        mock_plugin.rpc.setchannel.return_value = {
            "channels": [{
                "short_channel_id": "123x456x0",
                "fee_proportional_millionths": 125,
                "minimum_htlc_out_msat": "50000msat",
                "maximum_htlc_out_msat": "21000000msat",
                "warning_htlcmin_too_low": "peer floor applied",
            }]
        }

        result = fc.set_channel_fee("123x456x0", 125, manual=True, htlcmin_msat=42_000)

        assert result["success"] is True
        assert result["applied_htlcmin_msat"] == 50_000
        assert result["applied_htlcmax_msat"] == 21_000_000
        assert result["warnings"]["warning_htlcmin_too_low"] == "peer floor applied"


class TestDynamicHtlcMinPersistence:
    def test_fee_state_roundtrip_preserves_dynamic_htlcmin_baseline(self):
        from modules.fee_controller import ChannelFeeState

        state = ChannelFeeState()
        state.dynamic_htlcmin_baseline_msat = 42_000

        v2_data = state.to_v2_dict()

        assert v2_data["dynamic_htlcmin_baseline_msat"] == 42_000

        roundtrip = ChannelFeeState.from_v2_dict(v2_data, {"v2_state_json": "{}"})

        assert roundtrip.dynamic_htlcmin_baseline_msat == 42_000


class TestGossipRefreshExecution:
    def test_gossip_refresh_executes_setchannel_and_returns_fee_adjustment(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController, ChannelCycleState, FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        # set_channel_fee verifies on-chain fee by calling listpeerchannels again.
        # Simulate that the fee actually changes after setchannel.
        mock_plugin.rpc.listpeerchannels.side_effect = [
            _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100),  # initial read
            _listpeerchannels_payload(channel_id, peer_id, fee_ppm=101),  # verify read
            _listpeerchannels_payload(channel_id, peer_id, fee_ppm=101),  # (possible) second verify
        ]
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()
        mock_database.get_last_forward_time.return_value = int(time.time()) - 86400 * 2

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        # Provide a real-ish state and ensure the fee change will be applied.
        st = ChannelCycleState(
            last_update=int(time.time()) - 86400 * 2,
            last_broadcast_fee_ppm=100,
            last_fee_ppm=100,
            last_gossip_refresh=0
        )

        adj = fc._create_gossip_refresh_adjustment(
            channel_id=channel_id,
            peer_id=peer_id,
            state=st,
            current_fee_ppm=100,
            current_time=int(time.time())
        )

        assert adj is not None
        assert adj.reason_code == FeeReasonCode.GOSSIP_REFRESH.value
        assert adj.new_fee_ppm in (99, 101)
        mock_plugin.rpc.setchannel.assert_called()


class TestBoundedExplorationEndToEnd:
    def test_exploration_can_land_on_floor_when_channel_is_already_near_floor(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        cfg = Config(min_fee_ppm=40, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())

        target = fc._get_exploration_fee_target(
            current_fee_ppm=42,
            floor_ppm=40,
            cfg=cfg.snapshot(),
            sparse_data_conservative=False,
        )

        assert target == 40

    def test_exploration_target_preserves_meaningful_headroom_above_floor(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        cfg = Config(min_fee_ppm=40, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())

        target = fc._get_exploration_fee_target(
            current_fee_ppm=400,
            floor_ppm=40,
            cfg=cfg.snapshot(),
            sparse_data_conservative=False,
        )

        assert target > int(40 * fc.EXPLORATION_FEE_MULTIPLIER)
        assert target < 400

    def test_exploration_uses_bounded_low_fee_above_floor(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController, FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        # set_channel_fee verifies with listpeerchannels after setchannel.
        fee_holder = {"fee": 100}
        mock_plugin.rpc.setchannel = MagicMock(
            side_effect=lambda *args, **kwargs: fee_holder.__setitem__("fee", kwargs["feeppm"])
        )
        mock_plugin.rpc.listpeerchannels = MagicMock(side_effect=lambda: _listpeerchannels_payload(channel_id, peer_id, fee_ppm=fee_holder["fee"]))

        # Minimal DB stubs used by _adjust_channel_fee
        db_state = _fee_strategy_state_dict()
        db_state["last_update"] = int(time.time()) - 7200
        mock_database.get_fee_strategy_state.return_value = db_state
        mock_database.update_fee_strategy_state = MagicMock()
        mock_database.record_fee_change = MagicMock()

        mock_database.get_channel_probe.return_value = {"started": int(time.time()) - 7200}
        mock_database.clear_channel_probe = MagicMock()
        mock_database.get_last_rebalance_cost.return_value = None
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0}
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_recent_fee_changes.return_value = []
        mock_database.get_last_forward_time.return_value = int(time.time()) - 86400 * 10
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        channel_info = {
            "channel_id": channel_id,
            "peer_id": peer_id,
            "capacity": 1_000_000,
            "spendable_msat": 500_000_000,
            "receivable_msat": 500_000_000,
            "fee_base_msat": 0,
            "fee_proportional_millionths": 100,
            "opener": "local",
        }
        flow_state = {"state": "balanced", "forward_count": 0, "sats_out": 0}

        adj = fc._adjust_channel_fee(channel_id, peer_id, flow_state, channel_info, chain_costs=None, cfg=cfg.snapshot())
        assert adj is not None
        assert adj.reason_code == FeeReasonCode.LOW_FEE_EXPLORATION.value
        assert adj.new_fee_ppm >= cfg.min_fee_ppm
        assert adj.new_fee_ppm > 0
        assert adj.new_fee_ppm < channel_info["fee_proportional_millionths"]
        assert _setchannel_kwargs(mock_plugin)["feeppm"] == adj.new_fee_ppm

    def test_exploration_success_holds_safe_low_fee_and_clears_probe(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController, FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        # Channel is already at a low fee and exploration sees traffic.
        fee_holder = {"fee": 15}
        mock_plugin.rpc.setchannel = MagicMock(
            side_effect=lambda *args, **kwargs: fee_holder.__setitem__("fee", kwargs["feeppm"])
        )
        mock_plugin.rpc.listpeerchannels = MagicMock(side_effect=lambda: _listpeerchannels_payload(channel_id, peer_id, fee_ppm=fee_holder["fee"]))

        db_state = _fee_strategy_state_dict()
        db_state["last_update"] = int(time.time()) - 7200
        db_state["last_broadcast_fee_ppm"] = 0
        mock_database.get_fee_strategy_state.return_value = db_state
        mock_database.update_fee_strategy_state = MagicMock()
        mock_database.record_fee_change = MagicMock()

        mock_database.get_channel_probe.return_value = {"started": int(time.time()) - 7200}
        mock_database.clear_channel_probe = MagicMock()
        mock_database.get_volume_since.return_value = 100_000  # any >0 means probe success
        mock_database.get_forward_count_since.return_value = 1
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0}
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_recent_fee_changes.return_value = []
        mock_database.get_last_forward_time.return_value = int(time.time()) - 3600
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        channel_info = {
            "channel_id": channel_id,
            "peer_id": peer_id,
            "capacity": 1_000_000,
            "spendable_msat": 500_000_000,
            "receivable_msat": 500_000_000,
            "fee_base_msat": 0,
            "fee_proportional_millionths": 15,
            "opener": "local",
        }
        flow_state = {"state": "balanced", "forward_count": 0, "sats_out": 0}

        adj = fc._adjust_channel_fee(channel_id, peer_id, flow_state, channel_info, chain_costs=None, cfg=cfg.snapshot())
        assert adj is not None
        assert adj.reason_code == FeeReasonCode.LOW_FEE_EXPLORATION_SUCCESS.value
        assert adj.reason.startswith("EXPLORATION:")
        assert adj.new_fee_ppm >= 10
        assert adj.new_fee_ppm > 0

        mock_database.clear_channel_probe.assert_called()
        assert _setchannel_kwargs(mock_plugin)["feeppm"] == adj.new_fee_ppm

    def test_set_channel_fee_syncs_state_for_low_fee_exploration_reasons(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import (
            ChannelCycleState,
            ChannelFeeState,
            FeeController,
            FeeReasonCode,
        )

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        now = int(time.time())

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=25)
        mock_plugin.rpc.setchannel = MagicMock()
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_fee_ppm=200,
            last_broadcast_fee_ppm=200,
            last_update=now - 7200,
            is_sleeping=True,
            sleep_until=now + 3600,
            stable_cycles=2,
        )
        fc._channel_fee_states[channel_id] = ChannelFeeState(
            last_fee_ppm=200,
            last_broadcast_fee_ppm=200,
            last_update=now - 7200,
            is_sleeping=True,
            sleep_until=now + 3600,
            stable_cycles=2,
        )

        result = fc.set_channel_fee(
            channel_id,
            25,
            reason="bounded exploration",
            reason_code=FeeReasonCode.LOW_FEE_EXPLORATION.value,
            channel_info={"peer_id": peer_id, "fee_proportional_millionths": 200},
        )

        assert result["success"] is True
        assert fc._cycle_states[channel_id].last_broadcast_fee_ppm == 25
        assert fc._cycle_states[channel_id].last_state == FeeReasonCode.LOW_FEE_EXPLORATION.value
        assert fc._cycle_states[channel_id].is_sleeping is False
        assert fc._channel_fee_states[channel_id].last_broadcast_fee_ppm == 25
        assert fc._channel_fee_states[channel_id].last_state == FeeReasonCode.LOW_FEE_EXPLORATION.value
        assert fc._channel_fee_states[channel_id].is_sleeping is False


class TestSetInitialFee:
    """Tests for set_initial_fee - immediate fee setting on channel open."""

    def _make_controller(self, mock_plugin, mock_database, policy_manager=None):
        from modules.config import Config
        from modules.fee_controller import FeeController

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = FeeController(
            mock_plugin, cfg, mock_database, policy_manager,
            fee_authority_gate=FeeAuthorityGate(),
        )
        fc.data_service = _make_data_service(mock_plugin)
        return fc

    def test_initial_fee_sets_dts_prior_sample(self, mock_plugin, mock_database):
        """New dynamic channel gets a fee from the DTS prior."""
        from modules.fee_controller import FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # After setchannel is called, verification re-queries listpeerchannels.
        # Simulate the fee actually taking effect.
        mock_plugin.rpc.setchannel = MagicMock()

        def fake_listpeerchannels(*args, **kwargs):
            if mock_plugin.rpc.setchannel.called:
                last_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
                return _listpeerchannels_payload(channel_id, peer_id, fee_ppm=last_fee)
            return _listpeerchannels_payload(channel_id, peer_id, fee_ppm=0)

        mock_plugin.rpc.listpeerchannels.side_effect = fake_listpeerchannels

        fc = self._make_controller(mock_plugin, mock_database)
        result = fc.set_initial_fee(channel_id, peer_id)

        assert result is not None
        assert result["success"] is True
        mock_plugin.rpc.setchannel.assert_called()
        applied_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
        # Fee should be within configured bounds
        assert 10 <= applied_fee <= 5000

    def test_initial_fee_passes_parsed_htlc_bounds_in_channel_info(self, mock_plugin, mock_database):
        """Manual channel_info shaping preserves the advertised HTLC bounds."""
        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_current_payload(
            channel_id,
            peer_id,
            fee_ppm=0,
            minimum_htlc_out_msat="42000msat",
            maximum_htlc_out_msat="21000000msat",
        )

        fc = self._make_controller(mock_plugin, mock_database)
        fc.set_channel_fee = MagicMock(return_value={"success": True})

        result = fc.set_initial_fee(channel_id, peer_id)

        assert result == {"success": True}
        channel_info = fc.set_channel_fee.call_args.kwargs["channel_info"]
        assert channel_info["htlc_minimum_msat"] == 42_000
        assert channel_info["htlc_min_msat"] == 42_000
        assert channel_info["htlc_maximum_msat"] == 21_000_000
        assert channel_info["htlc_max_msat"] == 21_000_000

    def test_initial_fee_respects_passive_policy(self, mock_plugin, mock_database):
        """PASSIVE policy channels are skipped entirely."""
        from modules.policy_manager import PolicyManager, FeeStrategy, PeerPolicy

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(
            channel_id, peer_id
        )

        pm = MagicMock(spec=PolicyManager)
        pm.get_policy.return_value = PeerPolicy(
            peer_id=peer_id, strategy=FeeStrategy.PASSIVE
        )

        fc = self._make_controller(mock_plugin, mock_database, policy_manager=pm)
        result = fc.set_initial_fee(channel_id, peer_id)

        assert result is None
        mock_plugin.rpc.setchannel = MagicMock()
        mock_plugin.rpc.setchannel.assert_not_called()

    def test_initial_fee_respects_static_policy(self, mock_plugin, mock_database):
        """STATIC policy sets the exact target fee."""
        from modules.policy_manager import PolicyManager, FeeStrategy, PeerPolicy

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_plugin.rpc.setchannel = MagicMock()

        def fake_listpeerchannels(*args, **kwargs):
            if mock_plugin.rpc.setchannel.called:
                last_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
                return _listpeerchannels_payload(channel_id, peer_id, fee_ppm=last_fee)
            return _listpeerchannels_payload(channel_id, peer_id, fee_ppm=0)

        mock_plugin.rpc.listpeerchannels.side_effect = fake_listpeerchannels

        pm = MagicMock(spec=PolicyManager)
        pm.get_policy.return_value = PeerPolicy(
            peer_id=peer_id, strategy=FeeStrategy.STATIC, fee_ppm_target=250
        )

        fc = self._make_controller(mock_plugin, mock_database, policy_manager=pm)
        result = fc.set_initial_fee(channel_id, peer_id)

        assert result is not None
        assert result["success"] is True
        applied_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
        assert applied_fee == 250

    def test_initial_fee_matches_by_funding_txid(self, mock_plugin, mock_database):
        """Channel can be resolved using the funding txid (channel_id field)."""
        scid = "800x1x0"
        funding_txid = "ad723c457ceb425d3f6833cc35402c84b178df1778e6ba37fd73354ad5c15c6f"
        peer_id = "02" + "a" * 64

        mock_plugin.rpc.setchannel = MagicMock()

        def fake_listpeerchannels(*args, **kwargs):
            if mock_plugin.rpc.setchannel.called:
                last_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
            else:
                last_fee = 0
            return {
                "channels": [{
                    "state": "CHANNELD_NORMAL",
                    "short_channel_id": scid,
                    "channel_id": funding_txid,
                    "peer_id": peer_id,
                    "spendable_msat": 500_000_000,
                    "receivable_msat": 500_000_000,
                    "total_msat": 1_000_000_000,
                    "updates": {"local": {"fee_base_msat": 0, "fee_proportional_millionths": last_fee}},
                }]
            }

        mock_plugin.rpc.listpeerchannels.side_effect = fake_listpeerchannels

        fc = self._make_controller(mock_plugin, mock_database)
        result = fc.set_initial_fee(funding_txid, peer_id)

        assert result is not None
        assert result["success"] is True
        # Should use the SCID for the setchannel call
        called_id = _setchannel_kwargs(mock_plugin)["id"]
        assert called_id == scid

    def test_initial_fee_fallback_single_normal_channel(self, mock_plugin, mock_database):
        """Falls back to the only NORMAL channel if ID doesn't match."""
        scid = "800x1x0"
        event_id = "some_unrecognized_id"
        peer_id = "02" + "a" * 64

        mock_plugin.rpc.setchannel = MagicMock()

        def fake_listpeerchannels(*args, **kwargs):
            if mock_plugin.rpc.setchannel.called:
                last_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
            else:
                last_fee = 0
            return {
                "channels": [{
                    "state": "CHANNELD_NORMAL",
                    "short_channel_id": scid,
                    "peer_id": peer_id,
                    "spendable_msat": 500_000_000,
                    "receivable_msat": 500_000_000,
                    "total_msat": 1_000_000_000,
                    "updates": {"local": {"fee_base_msat": 0, "fee_proportional_millionths": last_fee}},
                }]
            }

        mock_plugin.rpc.listpeerchannels.side_effect = fake_listpeerchannels

        fc = self._make_controller(mock_plugin, mock_database)
        result = fc.set_initial_fee(event_id, peer_id)

        assert result is not None
        assert result["success"] is True

    def test_initial_fee_returns_none_on_rpc_error(self, mock_plugin, mock_database):
        """Gracefully handles RPC failures without raising."""
        peer_id = "02" + "a" * 64
        mock_plugin.rpc.listpeerchannels.side_effect = Exception("RPC timeout")

        fc = self._make_controller(mock_plugin, mock_database)
        result = fc.set_initial_fee("123x456x0", peer_id)

        assert result is None


class TestSetInitialFeePersistentPriorSeeding:
    """Audit F5: set_initial_fee used to seed a THROWAWAY thompson state
    with the fleet/gossip prior, sample one fee, and discard it. The
    channel's PERSISTENT state still started at the default prior
    (200/100), so the first regular fee cycle walked away from the best
    available evidence. The chosen prior must now be written into the
    persistent state, with a durable posterior nudge so the sample-time
    bias machinery carries the signal through early cycles."""

    CHANNEL_ID = "123x456x0"
    PEER_ID = "02" + "a" * 64

    def _make_controller(self, mock_plugin, mock_database, network_prior=None):
        from modules.config import Config
        from modules.fee_controller import FeeController

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        mock_plugin.rpc.setchannel = MagicMock()

        def fake_listpeerchannels(*args, **kwargs):
            if mock_plugin.rpc.setchannel.called:
                last_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
                return _listpeerchannels_payload(self.CHANNEL_ID, self.PEER_ID, fee_ppm=last_fee)
            return _listpeerchannels_payload(self.CHANNEL_ID, self.PEER_ID, fee_ppm=0)

        mock_plugin.rpc.listpeerchannels.side_effect = fake_listpeerchannels

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = _make_data_service(mock_plugin)

        fc._get_network_fee_prior = MagicMock(return_value=network_prior)
        return fc

    def test_seeded_state_pulls_first_cycle_sample_toward_prior(
        self, mock_plugin, mock_database
    ):
        """First regular cycle's sample must be measurably pulled toward the
        seeded prior compared to an unseeded control (fixed seed)."""
        import random
        from modules.fee_controller import GaussianThompsonState

        fc = self._make_controller(
            mock_plugin, mock_database,
            network_prior={"mean": 2500, "std": 80},
        )
        fc.set_initial_fee(self.CHANNEL_ID, self.PEER_ID)

        seeded = fc._channel_fee_states[self.CHANNEL_ID].thompson
        control = GaussianThompsonState()  # default 200/100 prior

        random.seed(4242)
        seeded_fee = seeded.sample_fee(10, 5000)
        random.seed(4242)
        control_fee = control.sample_fee(10, 5000)

        assert abs(seeded_fee - 2500) < abs(control_fee - 2500), (
            f"Seeded sample {seeded_fee} not pulled toward 2500 "
            f"vs control {control_fee}"
        )
        assert seeded_fee > 1500, "Seeded sample should be near the seeded prior"

    def test_gossip_prior_seeds_persistent_state_with_own_std(
        self, mock_plugin, mock_database
    ):
        """Without a fleet prior, the gossip prior seeds the persistent
        state keeping its own quality-adjusted std."""
        fc = self._make_controller(
            mock_plugin, mock_database,
            network_prior={"mean": 700, "std": 120},
        )
        fc.set_initial_fee(self.CHANNEL_ID, self.PEER_ID)

        state = fc._channel_fee_states.get(self.CHANNEL_ID)
        assert state is not None
        assert state.thompson.prior_mean_fee == 700
        assert state.thompson.prior_std_fee == 120
        targets = [entry[0] for entry in state.thompson.posterior_bias]
        assert 700.0 in targets

    def test_default_prior_path_leaves_persistent_state_unseeded(
        self, mock_plugin, mock_database
    ):
        """No fleet/gossip prior: the persistent state keeps its defaults
        and no posterior_bias entry is recorded."""
        fc = self._make_controller(
            mock_plugin, mock_database, network_prior=None,
        )
        result = fc.set_initial_fee(self.CHANNEL_ID, self.PEER_ID)

        assert result is not None
        state = fc._channel_fee_states.get(self.CHANNEL_ID)
        if state is not None:
            assert state.thompson.prior_mean_fee == 200
            assert state.thompson.posterior_bias == []

    def test_passive_policy_skip_preserved(self, mock_plugin, mock_database):
        """PASSIVE policy still skips entirely — no fee set, no seeding."""
        from modules.policy_manager import PolicyManager, FeeStrategy, PeerPolicy

        fc = self._make_controller(mock_plugin, mock_database)
        pm = MagicMock(spec=PolicyManager)
        pm.get_policy.return_value = PeerPolicy(
            peer_id=self.PEER_ID, strategy=FeeStrategy.PASSIVE
        )
        fc.policy_manager = pm

        result = fc.set_initial_fee(self.CHANNEL_ID, self.PEER_ID)

        assert result is None
        mock_plugin.rpc.setchannel.assert_not_called()
        assert self.CHANNEL_ID not in fc._channel_fee_states

    def test_static_policy_skips_prior_seeding(self, mock_plugin, mock_database):
        """STATIC policy sets its target without touching the thompson prior."""
        from modules.policy_manager import PolicyManager, FeeStrategy, PeerPolicy

        fc = self._make_controller(mock_plugin, mock_database)
        pm = MagicMock(spec=PolicyManager)
        pm.get_policy.return_value = PeerPolicy(
            peer_id=self.PEER_ID, strategy=FeeStrategy.STATIC, fee_ppm_target=250
        )
        fc.policy_manager = pm

        result = fc.set_initial_fee(self.CHANNEL_ID, self.PEER_ID)

        assert result is not None
        applied_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
        assert applied_fee == 250
        state = fc._channel_fee_states.get(self.CHANNEL_ID)
        if state is not None:
            assert state.thompson.prior_mean_fee == 200
            assert state.thompson.posterior_bias == []
