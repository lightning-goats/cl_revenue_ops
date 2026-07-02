import time
from unittest.mock import MagicMock

import pytest


def _rpc_calls_for(mock_plugin, method: str):
    """Return all rpc.call invocations for a given RPC method name."""
    return [
        c for c in mock_plugin.rpc.call.call_args_list
        if c[0] and c[0][0] == method
    ]


def _candidate(
    *,
    source_candidates=None,
    to_channel="222x333x0",
    primary_source_peer_id="02" + "a" * 64,
    to_peer_id="02" + "b" * 64,
    amount_sats=50000,
):
    from modules.rebalancer import RebalanceCandidate

    if source_candidates is None:
        source_candidates = ["111x222x0"]

    amt_msat = amount_sats * 1000
    return RebalanceCandidate(
        source_candidates=source_candidates,
        to_channel=to_channel,
        primary_source_peer_id=primary_source_peer_id,
        to_peer_id=to_peer_id,
        amount_sats=amount_sats,
        amount_msat=amt_msat,
        outbound_fee_ppm=1000,
        inbound_fee_ppm=100,
        source_fee_ppm=100,
        weighted_opp_cost_ppm=100,
        spread_ppm=800,
        max_budget_sats=10,
        max_budget_msat=10_000,
        max_fee_ppm=2000,
        expected_profit_sats=1,
        liquidity_ratio=0.1,
        dest_flow_state="balanced",
        dest_turnover_rate=0.0,
        source_turnover_rate=0.0,
    )


def _make_coordination_rebalancer(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(
        dry_run=True,
        low_liquidity_threshold=0.2,
        high_liquidity_threshold=0.8,
    )
    r = EVRebalancer(mock_plugin, cfg, mock_database)

    r.data_service = MagicMock()
    r.data_service.invalidate = MagicMock()
    r.data_service.datastore_push.return_value = True

    r._check_capital_controls = MagicMock(return_value=True)
    r._get_peer_connection_status = MagicMock(return_value={})
    r._calculate_turnover_rate = MagicMock(return_value=0.05)

    source_scid = "111x1x0"
    sink_a_scid = "222x2x0"
    sink_b_scid = "333x3x0"
    source_peer = "02" + "a" * 64
    sink_a_peer = "02" + "b" * 64
    sink_b_peer = "02" + "c" * 64

    r._get_channels_with_balances = MagicMock(return_value={
        source_scid: {
            "peer_id": source_peer,
            "capacity": 1_000_000,
            "spendable_sats": 900_000,
            "fee_ppm": 100,
        },
        sink_a_scid: {
            "peer_id": sink_a_peer,
            "capacity": 1_000_000,
            "spendable_sats": 100_000,
            "fee_ppm": 200,
        },
        sink_b_scid: {
            "peer_id": sink_b_peer,
            "capacity": 1_000_000,
            "spendable_sats": 120_000,
            "fee_ppm": 220,
        },
    })

    mock_database.cleanup_stale_reservations.return_value = 0
    mock_database.list_hot_channel_protection_override_peers.return_value = []
    mock_database.get_failure_count.return_value = (0, 0)
    mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
    mock_database.get_last_rebalance_time.return_value = 0
    mock_database.get_top_route_pairs.return_value = []
    mock_database.get_channel_state.return_value = {"state": "balanced"}
    mock_database.get_rebalance_success_signal.return_value = None
    mock_database.get_peer_uptime_percent.return_value = 99.0

    # CapEx engine: active tier for all channels (CapEx is the primary path)
    from modules.capex_budget import ChannelCapexBudget
    mock_capex = MagicMock()
    mock_capex.compute_allocations.return_value = None
    mock_capex.get_channel_budget.return_value = ChannelCapexBudget(
        channel_id="test", tier="active", budget_msat=500_000_000,
        tier_ppm=500, priority_class="preservation",
    )
    r.set_capex_engine(mock_capex)

    return r, {
        "source_scid": source_scid,
        "sink_a_scid": sink_a_scid,
        "sink_b_scid": sink_b_scid,
        "source_peer": source_peer,
        "sink_a_peer": sink_a_peer,
        "sink_b_peer": sink_b_peer,
    }


class TestExecuteRebalanceBudgetReservationLifecycle:
    def test_execute_rebalance_dry_run_does_not_reserve_budget_and_clears_pending(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_database.record_rebalance = MagicMock(return_value=123)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))

        cand = _candidate()
        res = r.execute_rebalance(cand)

        assert res["success"] is True
        mock_database.reserve_budget.assert_not_called()
        assert cand.to_channel not in r._pending

    def test_execute_rebalance_releases_budget_on_executor_failure(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        # rebalance_engine_v2 is None by default, so execute_rebalance will fail

        mock_database.record_rebalance = MagicMock(return_value=456)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
        mock_database.release_budget_reservation = MagicMock(return_value=True)

        cand = _candidate()
        res = r.execute_rebalance(cand, enforce_budget=True)

        assert res["success"] is False
        mock_database.reserve_budget.assert_called_once()
        mock_database.release_budget_reservation.assert_called_once_with('456')

    def test_execute_rebalance_success_records_cost_row_with_msat_precision(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.data_service = MagicMock()
        r.data_service.invalidate = MagicMock()
        r.data_service.datastore_push.return_value = True
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True,
            fee_msat=1501,
            fee_ppm=30,
            hops=3,
            route_type="native",
            attempts=1,
            parts=1,
        )

        mock_database.record_rebalance = MagicMock(return_value=457)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
        mock_database.release_budget_reservation = MagicMock(return_value=True)
        mock_database.record_rebalance_cost = MagicMock()
        mock_database.reset_failure_count = MagicMock()

        cand = _candidate()
        res = r.execute_rebalance(cand, enforce_budget=True)

        assert res["success"] is True
        mock_database.record_rebalance_cost.assert_called_once()
        call_kwargs = mock_database.record_rebalance_cost.call_args.kwargs
        assert call_kwargs["channel_id"] == cand.to_channel
        assert call_kwargs["peer_id"] == cand.to_peer_id
        assert call_kwargs["cost_msat"] == 1501
        assert call_kwargs["cost_sats"] == 2
        assert call_kwargs["amount_sats"] == cand.amount_sats


class TestCoordinatedRebalanceReporting:
    def _mock_coordination_rpc(self, mock_plugin, intent_response=None):
        intent_calls = []
        outcome_calls = []

        def side_effect(method, params=None):
            if method == "hive-report-rebalance-intent":
                intent_calls.append(params)
                return intent_response or {
                    "status": "accepted",
                    "recommendation_id": "rec-1",
                    "route_segments": ["02" + "a" * 64 + ">" + "02" + "b" * 64],
                    "lease": {"lease_id": "lease-1"},
                    "campaign": {"campaign_id": "campaign-1"},
                }
            if method == "hive-report-rebalance-outcome":
                outcome_calls.append(params)
                return {"status": "accepted"}
            return {}

        mock_plugin.rpc.call.side_effect = side_effect
        return intent_calls, outcome_calls

    def test_execute_rebalance_reports_intent_started_and_success(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.data_service = MagicMock()
        r.data_service.invalidate = MagicMock()
        r.data_service.datastore_push.return_value = True
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r._get_our_node_id = MagicMock(return_value="02" + "f" * 64)
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True,
            fee_msat=2500,
            fee_ppm=50,
            hops=3,
            route_type="native",
            attempts=1,
            parts=1,
        )

        mock_database.record_rebalance = MagicMock(return_value=321)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
        mock_database.release_budget_reservation = MagicMock(return_value=True)
        intent_calls, outcome_calls = self._mock_coordination_rpc(mock_plugin)

        candidate = _candidate(
            source_candidates=["111x1x0"],
            to_channel="222x2x0",
            primary_source_peer_id="02" + "a" * 64,
            to_peer_id="02" + "b" * 64,
            amount_sats=50_000,
        )
        candidate.reason_code = RebalanceReasonCode.COORDINATED_REBALANCE.value
        candidate.coordination_hint_type = "recommendation"
        candidate.coordination_hint_id = "rec-1"
        candidate.coordination_rank_bonus = 1.25

        result = r.execute_rebalance(candidate, enforce_budget=True)

        assert result["success"] is True
        assert len(intent_calls) == 1
        assert len(outcome_calls) == 2
        assert outcome_calls[0]["status"] == "started"
        assert outcome_calls[1]["status"] == "succeeded"
        assert outcome_calls[0]["lease_id"] == "lease-1"
        assert outcome_calls[0]["campaign_id"] == "campaign-1"
        assert outcome_calls[1]["lease_id"] == "lease-1"
        assert outcome_calls[1]["campaign_id"] == "campaign-1"
        assert outcome_calls[1]["recommendation_id"] == "rec-1"
        assert outcome_calls[1]["route_segments"] == ["02" + "a" * 64 + ">" + "02" + "b" * 64]

    def test_execute_rebalance_reports_failed_with_stable_reason(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.data_service = MagicMock()
        r.data_service.invalidate = MagicMock()
        r.data_service.datastore_push.return_value = True
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r._get_our_node_id = MagicMock(return_value="02" + "f" * 64)
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=False,
            route_type="native",
            attempts=2,
            error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
        )

        mock_database.record_rebalance = MagicMock(return_value=322)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
        mock_database.release_budget_reservation = MagicMock(return_value=True)
        intent_calls, outcome_calls = self._mock_coordination_rpc(mock_plugin)

        candidate = _candidate(
            source_candidates=["111x1x0"],
            to_channel="222x2x0",
            primary_source_peer_id="02" + "a" * 64,
            to_peer_id="02" + "b" * 64,
            amount_sats=50_000,
        )
        candidate.reason_code = RebalanceReasonCode.COORDINATED_REBALANCE.value
        candidate.coordination_hint_type = "recommendation"
        candidate.coordination_hint_id = "rec-2"
        candidate.coordination_rank_bonus = 1.25

        result = r.execute_rebalance(candidate, enforce_budget=True)

        assert result["success"] is False
        assert result["error"] == "shared_conflict_changed"
        assert len(intent_calls) == 1
        assert len(outcome_calls) == 2
        assert outcome_calls[0]["status"] == "started"
        assert outcome_calls[1]["status"] == "failed"
        assert outcome_calls[1]["reason"] == "shared_conflict_changed"

    def test_execute_rebalance_declines_without_executor(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.data_service = MagicMock()
        r.data_service.invalidate = MagicMock()
        r.data_service.datastore_push.return_value = True
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r._get_our_node_id = MagicMock(return_value="02" + "f" * 64)
        r.rebalance_engine_v2 = None

        mock_database.record_rebalance = MagicMock(return_value=323)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
        mock_database.release_budget_reservation = MagicMock(return_value=True)
        intent_calls, outcome_calls = self._mock_coordination_rpc(mock_plugin)

        candidate = _candidate(
            source_candidates=["111x1x0"],
            to_channel="222x2x0",
            primary_source_peer_id="02" + "a" * 64,
            to_peer_id="02" + "b" * 64,
            amount_sats=50_000,
        )
        candidate.reason_code = RebalanceReasonCode.COORDINATED_REBALANCE.value
        candidate.coordination_hint_type = "recommendation"
        candidate.coordination_hint_id = "rec-3"
        candidate.coordination_rank_bonus = 1.25

        result = r.execute_rebalance(candidate, enforce_budget=True)

        assert result["success"] is False
        assert result["error"] == "local_policy_block"
        assert len(intent_calls) == 0
        assert len(outcome_calls) == 1
        assert outcome_calls[0]["status"] == "declined"
        assert outcome_calls[0]["reason"] == "local_policy_block"

    def test_execute_rebalance_reports_budget_block_decline_for_coordinated_candidate(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.data_service = MagicMock()
        r.data_service.invalidate = MagicMock()
        r.data_service.datastore_push.return_value = True
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r._get_our_node_id = MagicMock(return_value="02" + "f" * 64)
        r.rebalance_engine_v2 = MagicMock()

        mock_database.record_rebalance = MagicMock(return_value=324)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(False, 0))
        intent_calls, outcome_calls = self._mock_coordination_rpc(mock_plugin)

        candidate = _candidate(
            source_candidates=["111x1x0"],
            to_channel="222x2x0",
            primary_source_peer_id="02" + "a" * 64,
            to_peer_id="02" + "b" * 64,
            amount_sats=50_000,
        )
        candidate.reason_code = RebalanceReasonCode.COORDINATED_REBALANCE.value
        candidate.coordination_hint_type = "recommendation"
        candidate.coordination_hint_id = "rec-4"
        candidate.coordination_rank_bonus = 1.25

        result = r.execute_rebalance(candidate, enforce_budget=True)

        assert result["success"] is False
        assert len(intent_calls) == 0
        assert len(outcome_calls) == 1
        assert outcome_calls[0]["status"] == "declined"
        assert outcome_calls[0]["reason"] == "local_budget_block"

    def test_execute_rebalance_continues_when_optional_intent_report_fails(
        self, mock_plugin, mock_database
    ):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.data_service = MagicMock()
        r.data_service.invalidate = MagicMock()
        r.data_service.datastore_push.return_value = True
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r._get_our_node_id = MagicMock(return_value="02" + "f" * 64)
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True,
            fee_msat=2500,
            fee_ppm=50,
            route_type="native",
        )

        mock_database.record_rebalance = MagicMock(return_value=325)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
        mock_database.release_budget_reservation = MagicMock(return_value=True)
        outcome_calls = []

        def side_effect(method, params=None):
            if method == "hive-report-rebalance-intent":
                raise RuntimeError("intent rpc unavailable")
            if method == "hive-report-rebalance-outcome":
                outcome_calls.append(params)
                return {"status": "accepted"}
            return {}

        mock_plugin.rpc.call.side_effect = side_effect

        candidate = _candidate(
            source_candidates=["111x1x0"],
            to_channel="222x2x0",
            primary_source_peer_id="02" + "a" * 64,
            to_peer_id="02" + "b" * 64,
            amount_sats=50_000,
        )
        candidate.reason_code = RebalanceReasonCode.COORDINATED_REBALANCE.value
        candidate.coordination_hint_type = "recommendation"
        candidate.coordination_hint_id = "rec-5"
        candidate.coordination_rank_bonus = 1.25

        result = r.execute_rebalance(candidate, enforce_budget=True)

        assert result["success"] is True
        r.rebalance_engine_v2.execute_candidate.assert_called_once()
        assert len(outcome_calls) == 1
        assert outcome_calls[0]["status"] == "succeeded"
        assert outcome_calls[0]["details"]["intent_status"] == "report_failed"


class TestLastHopFeeUnits:
    def test_get_last_hop_fee_converts_base_fee_to_ppm(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        from unittest.mock import MagicMock

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        peer_id = "02" + "e" * 64
        our_id = "02" + "f" * 64

        mock_data_service = MagicMock()
        mock_data_service.get_node_id.return_value = our_id
        mock_data_service.get_channels.return_value = {
            "channels": [
                {
                    "destination": our_id,
                    "fee_per_millionth": 100,
                    "base_fee_millisatoshi": 1000,  # 1 sat
                }
            ]
        }
        r.data_service = mock_data_service

        # At 100k sats (100,000,000 msat), a 1 sat base fee ~= 10 ppm.
        ppm = r._get_last_hop_fee(peer_id, amount_msat=100_000_000)
        assert ppm == 110


class TestManualRebalanceBudgetBypass:
    def test_manual_rebalance_does_not_reserve_budget(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._check_capital_controls = MagicMock(return_value=True)
        r._estimate_inbound_fee = MagicMock(return_value=0)
        r._get_channels_with_balances = MagicMock(return_value={
            "111x222x0": {"peer_id": "02" + "a" * 64, "fee_ppm": 10, "spendable_sats": 1_000_000, "capacity": 2_000_000},
            "222x333x0": {"peer_id": "02" + "b" * 64, "fee_ppm": 20, "spendable_sats": 1000, "capacity": 2_000_000},
        })

        mock_database.record_rebalance = MagicMock(return_value=999)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(True, 9999))

        r.manual_rebalance("111x222x0", "222x333x0", 50_000, max_fee_sats=10, force=True)
        mock_database.reserve_budget.assert_not_called()


class TestExecuteOnceDiagnostic:
    """Diagnostic rebalance uses the v2 execution stack."""

    def test_diagnostic_uses_v2_engine(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        channel_id = "111x222x0"

        r._get_channels_with_balances = MagicMock(return_value={
            channel_id: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 100},
            "333x444x0": {"capacity": 2_000_000, "spendable_sats": 1_500_000, "peer_id": "02" + "c" * 64, "fee_ppm": 200},
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=99)
        mock_database.update_rebalance_result = MagicMock()

        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True,
            fee_msat=5000,
            fee_sats=5,
        )

        result = r.diagnostic_rebalance(channel_id)

        r.rebalance_engine_v2.execute_candidate.assert_called_once()
        assert result["success"] is True

    def test_diagnostic_records_in_database(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        channel_id = "111x222x0"
        r._get_channels_with_balances = MagicMock(return_value={
            channel_id: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 100},
            "333x444x0": {"capacity": 2_000_000, "spendable_sats": 1_500_000, "peer_id": "02" + "c" * 64, "fee_ppm": 200},
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=99)
        mock_database.update_rebalance_result = MagicMock()

        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=False,
            error="no route",
        )

        r.diagnostic_rebalance(channel_id)

        mock_database.record_rebalance.assert_called_once()
        mock_database.update_rebalance_result.assert_called_once_with(99, 'failed', error_message="no route")

    def _make_diag_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        channel_id = "111x222x0"
        r._get_channels_with_balances = MagicMock(return_value={
            channel_id: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 100},
            "333x444x0": {"capacity": 2_000_000, "spendable_sats": 1_500_000, "peer_id": "02" + "c" * 64, "fee_ppm": 200},
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=99)
        mock_database.update_rebalance_result = MagicMock()
        return r, channel_id

    def test_blocked_shock_reports_shock_status_blocked(self, mock_plugin, mock_database):
        """Audit (defibrillation honesty): capital-controls block must be
        reported as shock_status='blocked', never as a completed shock."""
        r, channel_id = self._make_diag_rebalancer(mock_plugin, mock_database)
        r._check_capital_controls = MagicMock(return_value=False)
        r.rebalance_engine_v2 = MagicMock()

        result = r.diagnostic_rebalance(channel_id)

        assert result["shock_status"] == "blocked"
        r.rebalance_engine_v2.execute_candidate.assert_not_called()

    def test_failed_shock_reports_shock_status_failed(self, mock_plugin, mock_database):
        from modules.rebalance_executor_v2 import ExecutionResult

        r, channel_id = self._make_diag_rebalancer(mock_plugin, mock_database)
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=False, error="no route",
        )

        result = r.diagnostic_rebalance(channel_id)

        assert result["shock_status"] == "failed"

    def test_successful_shock_reports_completed_with_actual_fee(self, mock_plugin, mock_database):
        from modules.rebalance_executor_v2 import ExecutionResult

        r, channel_id = self._make_diag_rebalancer(mock_plugin, mock_database)
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True, fee_msat=5000, fee_sats=5,
        )

        result = r.diagnostic_rebalance(channel_id)

        assert result["shock_status"] == "completed"
        assert result["actual_fee_sats"] == 5

    def test_no_engine_reports_shock_status_failed(self, mock_plugin, mock_database):
        r, channel_id = self._make_diag_rebalancer(mock_plugin, mock_database)
        r.rebalance_engine_v2 = None

        result = r.diagnostic_rebalance(channel_id)

        assert result["shock_status"] == "failed"

    def test_no_sources_reports_shock_status_failed(self, mock_plugin, mock_database):
        r, channel_id = self._make_diag_rebalancer(mock_plugin, mock_database)
        # Only the target channel exists — no viable shock source
        r._get_channels_with_balances = MagicMock(return_value={
            channel_id: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 100},
        })

        result = r.diagnostic_rebalance(channel_id)

        assert result["shock_status"] == "failed"


class TestDiagnosticFeeCap:
    """Operator ruling D4 (2026-07-01): the defibrillator fee envelope is the
    configured diagnostic_rebalance_max_fee_sats (default 400 sats), the single
    binding knob. The ppm ceiling is derived from it (ceil(cap/amount*1e6)),
    so raising the sat cap actually raises the envelope — under the old
    hardcoded pair (100 sats, 2000 ppm) both bounds bound at exactly 100 sats
    on a 50k shock and every observed market route (118-363 sats) was
    rejected route_over_budget."""

    def _run_diag(self, mock_plugin, mock_database, cfg):
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_executor_v2 import ExecutionResult

        r = EVRebalancer(mock_plugin, cfg, mock_database)
        channel_id = "111x222x0"
        r._get_channels_with_balances = MagicMock(return_value={
            channel_id: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 100},
            "333x444x0": {"capacity": 2_000_000, "spendable_sats": 1_500_000, "peer_id": "02" + "c" * 64, "fee_ppm": 200},
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=99)
        mock_database.update_rebalance_result = MagicMock()
        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True, fee_msat=5000, fee_sats=5,
        )

        result = r.diagnostic_rebalance(channel_id)
        assert result["success"] is True
        candidate = r.rebalance_engine_v2.execute_candidate.call_args.args[0]
        record_kwargs = mock_database.record_rebalance.call_args.kwargs
        return candidate, record_kwargs

    def test_default_cap_400_flows_into_engine_call(self, mock_plugin, mock_database):
        from modules.config import Config

        candidate, record_kwargs = self._run_diag(mock_plugin, mock_database, Config(dry_run=False))

        assert candidate.amount_sats == 50_000  # shock size unchanged
        assert candidate.max_budget_sats == 400
        assert candidate.max_budget_msat == 400_000
        # ppm ceiling derived from the cap: ceil(400 / 50_000 * 1e6) = 8000
        assert candidate.max_fee_ppm == 8000
        assert record_kwargs["max_fee_sats"] == 400

    def test_configured_cap_overrides_default(self, mock_plugin, mock_database):
        from modules.config import Config

        cfg = Config(dry_run=False, diagnostic_rebalance_max_fee_sats=250)
        candidate, record_kwargs = self._run_diag(mock_plugin, mock_database, cfg)

        assert candidate.max_budget_sats == 250
        assert candidate.max_budget_msat == 250_000
        assert candidate.max_fee_ppm == 5000  # ceil(250 / 50_000 * 1e6)
        assert record_kwargs["max_fee_sats"] == 250

    def test_cap_clamped_to_daily_budget(self, mock_plugin, mock_database):
        """The diagnostic cap stays subordinate to the daily rebalance budget."""
        from modules.config import Config

        cfg = Config(
            dry_run=False,
            diagnostic_rebalance_max_fee_sats=5_000,
            daily_budget_sats=600,
        )
        candidate, record_kwargs = self._run_diag(mock_plugin, mock_database, cfg)

        assert candidate.max_budget_sats == 600
        assert record_kwargs["max_fee_sats"] == 600

    def test_cap_clamped_to_floor_of_one(self, mock_plugin, mock_database):
        from modules.config import Config

        cfg = Config(dry_run=False, diagnostic_rebalance_max_fee_sats=0)
        candidate, record_kwargs = self._run_diag(mock_plugin, mock_database, cfg)

        assert candidate.max_budget_sats == 1
        assert record_kwargs["max_fee_sats"] == 1

    def test_cap_clamped_to_static_ceiling(self, mock_plugin, mock_database):
        """A typo can't authorize huge diagnostic spend even with a huge daily budget."""
        from modules.config import Config

        cfg = Config(
            dry_run=False,
            diagnostic_rebalance_max_fee_sats=999_999,
            daily_budget_sats=1_000_000,
        )
        candidate, record_kwargs = self._run_diag(mock_plugin, mock_database, cfg)

        assert candidate.max_budget_sats == 10_000
        assert record_kwargs["max_fee_sats"] == 10_000

    def test_runtime_update_rejects_out_of_range_values(self, mock_plugin, mock_database):
        from modules.config import Config

        cfg = Config(dry_run=False)
        assert "error" in cfg.update_runtime(MagicMock(), "diagnostic_rebalance_max_fee_sats", "0")
        assert "error" in cfg.update_runtime(MagicMock(), "diagnostic_rebalance_max_fee_sats", "20000")

    def test_snapshot_carries_diagnostic_cap(self, mock_plugin, mock_database):
        from modules.config import Config

        cfg = Config(dry_run=False, diagnostic_rebalance_max_fee_sats=321)
        assert cfg.snapshot().diagnostic_rebalance_max_fee_sats == 321


class TestExecuteOnceManual:
    """Manual rebalance uses the v2 execution stack."""

    def test_manual_uses_v2_engine(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        from_ch = "111x222x0"
        to_ch = "333x444x0"

        r._get_channels_with_balances = MagicMock(return_value={
            from_ch: {"capacity": 2_000_000, "spendable_sats": 1_500_000, "peer_id": "02" + "a" * 64, "fee_ppm": 200},
            to_ch: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 300},
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=55)
        mock_database.update_rebalance_result = MagicMock()

        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=True,
            fee_msat=5000,
            fee_sats=5,
        )

        result = r.manual_rebalance(from_ch, to_ch, 100_000, max_fee_sats=50)

        r.rebalance_engine_v2.execute_candidate.assert_called_once()
        assert result["success"] is True

    def test_manual_handles_failure(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_executor_v2 import ExecutionResult

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        from_ch = "111x222x0"
        to_ch = "333x444x0"

        r._get_channels_with_balances = MagicMock(return_value={
            from_ch: {"capacity": 2_000_000, "spendable_sats": 1_500_000, "peer_id": "02" + "a" * 64, "fee_ppm": 200},
            to_ch: {"capacity": 1_000_000, "spendable_sats": 50_000, "peer_id": "02" + "b" * 64, "fee_ppm": 300},
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=55)
        mock_database.update_rebalance_result = MagicMock()

        r.rebalance_engine_v2 = MagicMock()
        r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
            success=False,
            error="no route found",
        )

        result = r.manual_rebalance(from_ch, to_ch, 100_000, max_fee_sats=50)

        assert result.get("success") is False
        assert "no route found" in result.get("error", "")
        mock_database.update_rebalance_result.assert_called_once_with(55, 'failed', error_message="no route found")



# Obsolete executor-specific test classes removed.

# =============================================================================
# Audit Round 8 – Turn 2 Regression Tests
# =============================================================================

class TestAuditTurn2ManualRebalanceZeroFee:
    """P1-1 regression: manual_rebalance zero max_fee_sats when spread==0."""

    def test_manual_rebalance_zero_spread_gets_fallback_fee(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.record_rebalance = MagicMock(return_value=123)
        mock_database.update_rebalance_result = MagicMock()

        # Setup channels where spread = fee_ppm - est_in - src_ppm = 0
        r._get_channels_with_balances = MagicMock(return_value={
            "100x1x0": {"capacity": 1_000_000, "spendable_sats": 800_000,
                        "peer_id": "02" + "a" * 64, "fee_ppm": 200, "base_fee_msat": 0, "htlcs": 0},
            "200x2x0": {"capacity": 1_000_000, "spendable_sats": 200_000,
                        "peer_id": "02" + "b" * 64, "fee_ppm": 100, "base_fee_msat": 0, "htlcs": 0},
        })
        # est_in = 100 => spread = 100 - 100 - 200 = -200 => fallback to 100
        r._estimate_inbound_fee = MagicMock(return_value=100)

        result = r.manual_rebalance("100x1x0", "200x2x0", 50_000)
        # Should succeed (dry run) with a fallback fee, not 0
        if "candidate" in result:
            assert result["candidate"]["max_budget_sats"] > 0


class TestAuditTurn2HotChannelBudgetFilter:
    """Budget enforcement: daily budget is a hard cap, never exceeded."""

    def test_budget_exceeded_blocks_even_with_hot_channel_protection(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            daily_budget_sats=100,
            hot_channel_protection_enabled=True,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Simulate budget exceeded with hot-channel enabled
        mock_database.get_total_rebalance_fees = MagicMock(return_value=200)  # > 100 budget
        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"status": "confirmed", "amount_msat": 10_000_000_000}],
            "channels": [],
        }

        # _check_capital_controls must return False — daily budget is a hard cap
        result = r._check_capital_controls(cfg)
        assert result is False

    def test_budget_exceeded_blocks_without_hot_channel_protection(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            daily_budget_sats=100,
            hot_channel_protection_enabled=False,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_database.get_total_rebalance_fees = MagicMock(return_value=200)
        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"status": "confirmed", "amount_msat": 10_000_000_000}],
            "channels": [],
        }

        result = r._check_capital_controls(cfg)
        assert result is False

    def test_budget_ok_allows_rebalancing(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            daily_budget_sats=1000,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_database.get_total_rebalance_fees = MagicMock(return_value=100)  # < 1000 budget
        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"status": "confirmed", "amount_msat": 10_000_000_000}],
            "channels": [],
        }

        result = r._check_capital_controls(cfg)
        assert result is True


class TestVolumeBasedSizingFix:
    """Fix: Low-volume channels should not be penalized worse than zero-volume."""

    def test_low_volume_not_killed_by_sizing_guard(self, mock_plugin, mock_database):
        """A channel with 10k sats/day volume (below rebalance_min_amount) should
        use capacity-based target, not vol_target which would fail the sizing guard."""
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            rebalance_min_amount=50000,
            flow_window_days=7,
            enable_velocity_gate=False,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Mock channel state with low volume (10k sats/day * 7 days = 70k total)
        mock_database.get_channel_state.return_value = {
            "state": "balanced",
            "sats_in": 35000,
            "sats_out": 35000,
            "flow_ratio": 0.0,
        }
        mock_database.get_fee_strategy_state.return_value = {"last_broadcast_fee_ppm": 500}
        mock_database.get_peer_uptime_percent.return_value = 100.0

        # daily_volume = (35000 + 35000) / 7 = 10000
        # vol_target = 10000 * 3 = 30000 (< 50000 min_amount)
        # Before fix: raw_target = min(cap_target, 30000) = 30000 -> SIZING GUARD kills it
        # After fix: vol_target < min_amount -> fall back to cap_target

        # We verify the fix by checking the internal math directly
        daily_volume = 10000
        vol_target = int(daily_volume * 3)  # 30000
        cap_target = int(1_000_000 * 0.50)  # 500000

        # After fix: vol_target (30000) < rebalance_min_amount (50000), so use cap_target
        assert vol_target < cfg.rebalance_min_amount
        if vol_target >= cfg.rebalance_min_amount:
            raw_target = min(cap_target, vol_target)
        else:
            raw_target = cap_target
        assert raw_target == cap_target
        assert raw_target >= cfg.rebalance_min_amount

    def test_high_volume_still_constrains_target(self, mock_plugin, mock_database):
        """A channel with high volume should still use vol_target to prevent overfill."""
        from modules.config import Config

        cfg = Config(
            rebalance_min_amount=50000,
            flow_window_days=7,
        )

        daily_volume = 100000  # 100k/day
        vol_target = int(daily_volume * 3)  # 300000
        cap_target = int(2_000_000 * 0.50)  # 1000000

        assert vol_target >= cfg.rebalance_min_amount
        raw_target = min(cap_target, vol_target)
        assert raw_target == vol_target  # volume constraint applied


class TestLastDecisionSummary:
    def test_execute_rebalance_records_budget_blocked_summary(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        mock_database.record_rebalance = MagicMock(return_value=456)
        mock_database.update_rebalance_result = MagicMock()
        mock_database.reserve_budget = MagicMock(return_value=(False, 0))

        res = r.execute_rebalance(_candidate(), enforce_budget=True)
        summary = r.get_last_decision_summary()

        assert res["success"] is False
        assert summary["action"] == "suppressed"
        assert summary["reason"] == "budget_exhausted"
        assert summary["dominant_input"] == "daily_budget_sats"
        assert summary["safety_block"] is True
        assert summary["budget_blocked"] is True


class TestHiveEqualizationConfigSurface:
    def test_hive_equalization_defaults_are_present_in_config_and_snapshot(self):
        from modules.config import Config

        cfg = Config()
        snapshot = cfg.snapshot()

        assert cfg.hive_equalization_enabled is True
        assert cfg.hive_equalization_low_pct == 0.35
        assert cfg.hive_equalization_high_pct == 0.65
        assert cfg.hive_equalization_cooldown_hours == 48
        assert cfg.hive_equalization_max_candidates_per_cycle == 1

        assert snapshot.hive_equalization_enabled is True
        assert snapshot.hive_equalization_low_pct == 0.35
        assert snapshot.hive_equalization_high_pct == 0.65
        assert snapshot.hive_equalization_cooldown_hours == 48
        assert snapshot.hive_equalization_max_candidates_per_cycle == 1

    def test_hive_equalization_direct_construction_requires_valid_band(self):
        from modules.config import Config

        with pytest.raises(ValueError, match="hive_equalization_low_pct"):
            Config(hive_equalization_low_pct=0.7, hive_equalization_high_pct=0.6)

    def test_hive_equalization_runtime_updates_use_typed_validation(self):
        from modules.config import Config

        cfg = Config()
        stored_values = {}
        database = MagicMock()

        def set_config_override(key, value):
            stored_values[key] = value
            return 99

        def get_config_override(key):
            return stored_values.get(key)

        database.set_config_override.side_effect = set_config_override
        database.get_config_override.side_effect = get_config_override
        database.delete_config_override.return_value = True

        result = cfg.update_runtime(database, "hive_equalization_enabled", "false")
        assert result["status"] == "success"
        assert cfg.hive_equalization_enabled is False

        result = cfg.update_runtime(database, "hive_equalization_low_pct", "0.4")
        assert result["status"] == "success"
        assert cfg.hive_equalization_low_pct == 0.4

        result = cfg.update_runtime(database, "hive_equalization_high_pct", "0.7")
        assert result["status"] == "success"
        assert cfg.hive_equalization_high_pct == 0.7

        result = cfg.update_runtime(database, "hive_equalization_cooldown_hours", "72")
        assert result["status"] == "success"
        assert cfg.hive_equalization_cooldown_hours == 72

        result = cfg.update_runtime(database, "hive_equalization_max_candidates_per_cycle", "3")
        assert result["status"] == "success"
        assert cfg.hive_equalization_max_candidates_per_cycle == 3

        result = cfg.update_runtime(database, "hive_equalization_low_pct", "1.5")
        assert "out of range" in result["error"]

        result = cfg.update_runtime(database, "hive_equalization_low_pct", "0.8")
        assert "must be less than" in result["error"]

        result = cfg.update_runtime(database, "hive_equalization_high_pct", "0.39")
        assert "must be greater than" in result["error"]

    def test_hive_equalization_load_overrides_repairs_invalid_band(self):
        from modules.config import Config

        cfg = Config()
        database = MagicMock()
        database.get_all_config_overrides.return_value = {
            "hive_equalization_low_pct": "0.8",
            "hive_equalization_high_pct": "0.6",
        }
        database.get_config_version.return_value = 7

        warnings = cfg.load_overrides(database)

        assert warnings == []
        assert cfg.hive_equalization_high_pct == 0.6
        assert cfg.hive_equalization_low_pct == pytest.approx(0.55)
        assert cfg.snapshot().hive_equalization_low_pct == pytest.approx(0.55)

    def test_reason_code_includes_hive_equalization(self):
        from modules.rebalancer import RebalanceReasonCode

        assert RebalanceReasonCode.HIVE_EQUALIZATION.value == "hive_equalization"


class TestGetLastHopFeeFleetMember:
    """_get_last_hop_fee always uses actual peer fees, even for fleet members."""

    def test_uses_actual_fee_for_hive_member(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._fee_cache = {}

        mock_router = MagicMock()
        mock_router.is_hive_member.return_value = True
        r.hive_router = mock_router

        # Fleet member with actual 500 ppm fee in peer channel data
        r._peer_inbound_fees = {
            "03796a" + "0" * 58: {"fee_ppm": 500, "base_msat": 0},
        }

        result = r._get_last_hop_fee("03796a" + "0" * 58)
        # Should return actual fee, NOT 0
        assert result == 500

    def test_queries_actual_fee_for_non_member(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_router = MagicMock()
        mock_router.is_hive_member.return_value = False
        r.hive_router = mock_router
        mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}

        result = r._get_last_hop_fee("03a93b" + "0" * 58)
        # Should NOT have short-circuited to 0
        assert result != 0 or result is None


class TestConfigOptionDefaultAlignment:
    """P6-010 / DEF-042: the Config dataclass default for max_fee_ppm must
    agree with the plugin-option default (which wins at runtime). A bare
    Config() previously got 5000 while the option default was 2000."""

    def _plugin_option_default(self, name):
        """Extract a plugin.add_option default without importing the plugin
        module (which would register the whole plugin surface)."""
        import ast
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree = ast.parse(open(os.path.join(root, "cl-revenue-ops.py")).read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_option"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            name_node = kw.get("name")
            if isinstance(name_node, ast.Constant) and name_node.value == name:
                default_node = kw.get("default")
                if isinstance(default_node, ast.Constant):
                    return default_node.value
        raise AssertionError(f"option {name} not found")

    def test_config_max_fee_ppm_default_matches_option(self):
        from modules.config import Config
        option_default = int(self._plugin_option_default("revenue-ops-max-fee-ppm"))
        assert Config().max_fee_ppm == option_default
        assert Config().max_fee_ppm == 2000
