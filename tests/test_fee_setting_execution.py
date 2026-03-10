import sys
import os
import time
from unittest.mock import MagicMock
from typing import Union

import pytest

# Ensure project root is importable (matches other tests)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _fee_strategy_state_dict():
    # Minimal dict for _get_hill_climb_state/_get_thompson_aimd_state loaders.
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


class TestChannelInfoShaping:
    def test_get_channels_info_preserves_htlc_minimum_and_maximum_msat(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

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

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)
        channel_info = fc._get_channels_info()[channel_id]

        assert channel_info["htlc_minimum_msat"] == 42_000
        assert channel_info["htlc_min_msat"] == 42_000
        assert channel_info["htlc_maximum_msat"] == 21_000_000
        assert channel_info["htlc_max_msat"] == 21_000_000


class TestSetChannelFeeLimits:
    def test_set_channel_fee_enforces_limits_by_default(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        # State loaders/savers
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)

        fc.set_channel_fee(channel_id, 1, manual=True, enforce_limits=True)

        # Should clamp up to min_fee_ppm
        mock_plugin.rpc.setchannel.assert_called()
        call_kwargs = _setchannel_kwargs(mock_plugin)
        assert call_kwargs["feebase"] == 0
        assert call_kwargs["feeppm"] == 10

    def test_set_channel_fee_can_bypass_limits_for_force(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)

        fc.set_channel_fee(channel_id, 1, manual=True, enforce_limits=False)

        assert _setchannel_kwargs(mock_plugin)["feeppm"] == 1

    def test_set_channel_fee_allows_zero_for_hive_policy(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController, FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)

        fc.set_channel_fee(
            channel_id,
            0,
            reason="Policy: HIVE",
            reason_code=FeeReasonCode.POLICY_HIVE.value,
            enforce_limits=False
        )

        assert _setchannel_kwargs(mock_plugin)["feeppm"] == 0


class TestSetChannelFeeHtlcMin:
    def _make_controller(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)
        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)
        mock_plugin.rpc.setchannel = MagicMock()

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        return HillClimbingFeeController(mock_plugin, cfg, mock_database)

    def test_set_channel_fee_omits_htlcmin_when_not_requested(self, mock_plugin, mock_database):
        fc = self._make_controller(mock_plugin, mock_database)

        fc.set_channel_fee("123x456x0", 125, manual=True)

        assert "htlcmin" not in _setchannel_kwargs(mock_plugin)

    def test_set_channel_fee_includes_htlcmin_when_requested(self, mock_plugin, mock_database):
        fc = self._make_controller(mock_plugin, mock_database)

        fc.set_channel_fee("123x456x0", 125, manual=True, htlcmin_msat=42_000)

        assert _setchannel_kwargs(mock_plugin)["htlcmin"] == "42000msat"


class TestDynamicHtlcMin:
    def _make_controller(self, mock_plugin, mock_database, **cfg_overrides):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

        cfg = Config(
            min_fee_ppm=10,
            max_fee_ppm=5000,
            base_fee_msat=0,
            dry_run=False,
            enable_dynamic_htlcmin=True,
            htlc_congestion_threshold=0.8,
            **cfg_overrides,
        )
        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        return HillClimbingFeeController(mock_plugin, cfg, mock_database), cfg

    def test_dynamic_htlcmin_rises_under_congestion_exponentially(self, mock_plugin, mock_database):
        fc, cfg = self._make_controller(mock_plugin, mock_database)
        capacity_sats = 483_000
        protective_seed_msat = (capacity_sats * 1000) // 483

        htlcmin_msat = fc._calculate_dynamic_htlcmin_msat(
            state={"htlc_utilization": 0.95},
            channel_info={"htlc_minimum_msat": 42_000, "capacity": capacity_sats},
            cfg=cfg.snapshot(),
            vegas_multiplier=1.0,
            htlcmax_msat=None,
        )

        assert htlcmin_msat == protective_seed_msat * 8

    def test_dynamic_htlcmin_zero_baseline_congestion_yields_non_zero_value(self, mock_plugin, mock_database):
        fc, cfg = self._make_controller(mock_plugin, mock_database)

        htlcmin_msat = fc._calculate_dynamic_htlcmin_msat(
            state={"htlc_utilization": 0.95},
            channel_info={"htlc_minimum_msat": 0, "capacity": 483_000},
            cfg=cfg.snapshot(),
            vegas_multiplier=1.0,
            htlcmax_msat=None,
        )

        assert htlcmin_msat == 8_000_000

    def test_dynamic_htlcmin_zero_baseline_vegas_yields_non_zero_value(self, mock_plugin, mock_database):
        fc, cfg = self._make_controller(mock_plugin, mock_database)
        fc._vegas_state.intensity = 1.0

        htlcmin_msat = fc._calculate_dynamic_htlcmin_msat(
            state={"htlc_utilization": 0.0},
            channel_info={"htlc_minimum_msat": 0, "capacity": 483_000},
            cfg=cfg.snapshot(),
            vegas_multiplier=fc._vegas_state.get_floor_multiplier(),
            htlcmax_msat=None,
        )

        assert htlcmin_msat == 3_000_000

    def test_dynamic_htlcmin_relaxes_back_to_baseline(self, mock_plugin, mock_database):
        fc, cfg = self._make_controller(mock_plugin, mock_database)

        htlcmin_msat = fc._calculate_dynamic_htlcmin_msat(
            state={"htlc_utilization": 0.2},
            channel_info={"htlc_minimum_msat": 42_000, "capacity": 483_000},
            cfg=cfg.snapshot(),
            vegas_multiplier=1.0,
            htlcmax_msat=None,
        )

        assert htlcmin_msat == 42_000

    def test_dynamic_htlcmin_clamps_below_active_htlcmax(self, mock_plugin, mock_database):
        fc, cfg = self._make_controller(mock_plugin, mock_database)
        fc._vegas_state.intensity = 1.0

        htlcmin_msat = fc._calculate_dynamic_htlcmin_msat(
            state={"htlc_utilization": 0.95},
            channel_info={"htlc_minimum_msat": 42_000},
            cfg=cfg.snapshot(),
            vegas_multiplier=fc._vegas_state.get_floor_multiplier(),
            htlcmax_msat=60_000,
        )

        assert htlcmin_msat == 59_000

    def test_dynamic_htlcmin_clamps_below_advertised_htlcmax_when_explicit_max_missing(self, mock_plugin, mock_database):
        fc, cfg = self._make_controller(mock_plugin, mock_database)

        htlcmin_msat = fc._calculate_dynamic_htlcmin_msat(
            state={"htlc_utilization": 0.95},
            channel_info={
                "htlc_minimum_msat": 0,
                "htlc_maximum_msat": "60000msat",
                "capacity": 483_000,
            },
            cfg=cfg.snapshot(),
            vegas_multiplier=1.0,
            htlcmax_msat=None,
        )

        assert htlcmin_msat == 59_000


class TestGossipRefreshExecution:
    def test_gossip_refresh_executes_setchannel_and_returns_fee_adjustment(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController, HillClimbState, FeeReasonCode

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

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)

        # Provide a real-ish state and ensure the fee change will be applied.
        st = HillClimbState(
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


class TestZeroFeeProbeEndToEnd:
    def test_zero_fee_probe_sets_fee_to_zero_bypassing_min_fee(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController, FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False, enable_reputation=False)

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
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0}
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_recent_fee_changes.return_value = []
        mock_database.get_last_forward_time.return_value = int(time.time()) - 86400 * 10
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)
        # Keep the test focused on probe behavior (avoid Thompson path complexity).
        fc.ENABLE_THOMPSON_AIMD = False
        fc.ENABLE_DYNAMIC_WINDOWS = False
        fc.ENABLE_SATURATION_FLOOR = False
        fc.ENABLE_BALANCE_FLOOR = False
        fc.ENABLE_REBALANCE_FLOOR = False
        fc.ENABLE_FLOW_CEILING = False

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
        assert adj.reason_code == FeeReasonCode.ZERO_FEE_PROBE.value

        assert _setchannel_kwargs(mock_plugin)["feeppm"] == 0

    def test_zero_fee_probe_success_exits_to_floor_and_clears_probe(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController, FeeReasonCode

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False, enable_reputation=False)

        # Fee is currently 0 (probe already active), and we observed volume -> success.
        fee_holder = {"fee": 0}
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

        fc = HillClimbingFeeController(mock_plugin, cfg, mock_database)
        fc.ENABLE_THOMPSON_AIMD = False
        fc.ENABLE_DYNAMIC_WINDOWS = False
        fc.ENABLE_SATURATION_FLOOR = False
        fc.ENABLE_BALANCE_FLOOR = False
        fc.ENABLE_REBALANCE_FLOOR = False
        fc.ENABLE_FLOW_CEILING = False

        channel_info = {
            "channel_id": channel_id,
            "peer_id": peer_id,
            "capacity": 1_000_000,
            "spendable_msat": 500_000_000,
            "receivable_msat": 500_000_000,
            "fee_base_msat": 0,
            "fee_proportional_millionths": 0,
            "opener": "local",
        }
        flow_state = {"state": "balanced", "forward_count": 0, "sats_out": 0}

        adj = fc._adjust_channel_fee(channel_id, peer_id, flow_state, channel_info, chain_costs=None, cfg=cfg.snapshot())
        assert adj is not None
        assert adj.reason_code == FeeReasonCode.ZERO_FEE_PROBE_SUCCESS.value
        assert adj.new_fee_ppm >= 10

        mock_database.clear_channel_probe.assert_called()
        assert _setchannel_kwargs(mock_plugin)["feeppm"] == adj.new_fee_ppm


class TestSetInitialFee:
    """Tests for set_initial_fee - immediate fee setting on channel open."""

    def _make_controller(self, mock_plugin, mock_database, policy_manager=None):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000, base_fee_msat=0, dry_run=False)

        mock_database.get_fee_strategy_state.return_value = _fee_strategy_state_dict()
        mock_database.record_fee_change = MagicMock()

        fc = HillClimbingFeeController(
            mock_plugin, cfg, mock_database, policy_manager
        )
        return fc

    def test_initial_fee_sets_thompson_prior_sample(self, mock_plugin, mock_database):
        """New dynamic channel gets a fee from the Thompson prior."""
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

        mock_plugin.rpc.listpeerchannels.return_value = _listpeerchannels_payload(
            channel_id,
            peer_id,
            fee_ppm=0,
            htlc_minimum_msat="42000msat",
            htlc_maximum_msat="21000000msat",
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

    def test_initial_fee_respects_hive_policy(self, mock_plugin, mock_database):
        """HIVE policy sets zero fee regardless of min_fee_ppm."""
        from modules.policy_manager import PolicyManager, FeeStrategy, PeerPolicy

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_plugin.rpc.setchannel = MagicMock()

        def fake_listpeerchannels(*args, **kwargs):
            if mock_plugin.rpc.setchannel.called:
                last_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
                return _listpeerchannels_payload(channel_id, peer_id, fee_ppm=last_fee)
            return _listpeerchannels_payload(channel_id, peer_id, fee_ppm=100)

        mock_plugin.rpc.listpeerchannels.side_effect = fake_listpeerchannels

        pm = MagicMock(spec=PolicyManager)
        pm.get_policy.return_value = PeerPolicy(
            peer_id=peer_id, strategy=FeeStrategy.HIVE
        )

        fc = self._make_controller(mock_plugin, mock_database, policy_manager=pm)
        result = fc.set_initial_fee(channel_id, peer_id)

        assert result is not None
        assert result["success"] is True
        applied_fee = _setchannel_kwargs(mock_plugin)["feeppm"]
        assert applied_fee == 0

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
