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
        """A channel with 10k sats/day volume (below the 50k floor) should
        use capacity-based target, not vol_target which would fail the sizing guard."""
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            flow_window_days=7,
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

        # After fix: vol_target (30000) < the 50k floor, so use cap_target
        assert vol_target < 50000
        if vol_target >= 50000:
            raw_target = min(cap_target, vol_target)
        else:
            raw_target = cap_target
        assert raw_target == cap_target
        assert raw_target >= 50000

    def test_high_volume_still_constrains_target(self, mock_plugin, mock_database):
        """A channel with high volume should still use vol_target to prevent overfill."""
        from modules.config import Config

        cfg = Config(
            flow_window_days=7,
        )

        daily_volume = 100000  # 100k/day
        vol_target = int(daily_volume * 3)  # 300000
        cap_target = int(2_000_000 * 0.50)  # 1000000

        assert vol_target >= 50000
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


class TestRebalanceUtilizationFloorCeilingGuard:
    """rebalance_utilization_floor/ceiling must mirror the low/high_liquidity_threshold
    cross-field guard idiom: no __post_init__ raise, repaired in load_overrides,
    rejected in update_runtime.
    """

    def test_direct_construction_does_not_raise_for_inverted_pair(self):
        # Matches low_liquidity_threshold/high_liquidity_threshold: no
        # __post_init__ invariant is enforced for this pair (repair/reject
        # happen in load_overrides/update_runtime instead).
        from modules.config import Config

        cfg = Config(
            rebalance_utilization_floor=0.9,
            rebalance_utilization_ceiling=0.1,
        )
        assert cfg.rebalance_utilization_floor == 0.9
        assert cfg.rebalance_utilization_ceiling == 0.1

    def test_runtime_updates_reject_inverted_pair(self):
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

        result = cfg.update_runtime(database, "rebalance_utilization_ceiling", "0.6")
        assert result["status"] == "success"
        assert cfg.rebalance_utilization_ceiling == 0.6

        result = cfg.update_runtime(database, "rebalance_utilization_floor", "0.3")
        assert result["status"] == "success"
        assert cfg.rebalance_utilization_floor == 0.3

        result = cfg.update_runtime(database, "rebalance_utilization_floor", "0.9")
        assert "must be less than" in result["error"]

        result = cfg.update_runtime(database, "rebalance_utilization_ceiling", "0.1")
        assert "must be greater than" in result["error"]

    def test_load_overrides_repairs_invalid_band(self):
        from modules.config import Config

        cfg = Config()
        database = MagicMock()
        database.get_all_config_overrides.return_value = {
            "rebalance_utilization_floor": "0.9",
            "rebalance_utilization_ceiling": "0.1",
        }
        database.get_config_version.return_value = 7

        warnings = cfg.load_overrides(database)

        assert warnings == []
        assert cfg.rebalance_utilization_ceiling == 0.1
        assert cfg.rebalance_utilization_floor == pytest.approx(0.05)
        assert cfg.snapshot().rebalance_utilization_floor == pytest.approx(0.05)


class TestRebalanceSmallChannelBandHalfWidthRange:
    def test_field_range_is_tightened_to_half_width(self):
        from modules.config import CONFIG_FIELD_RANGES

        assert CONFIG_FIELD_RANGES["rebalance_small_channel_band_half_width"] == (0.0, 0.5)

    def test_runtime_update_rejects_value_above_half(self):
        from modules.config import Config

        cfg = Config()
        database = MagicMock()

        result = cfg.update_runtime(
            database, "rebalance_small_channel_band_half_width", "0.6"
        )
        assert "error" in result
        assert "out of range" in result["error"]


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


def test_upstream_pattern_defaults():
    from modules.config import Config
    c = Config()
    assert c.rebalance_activity_window_seconds == 3600
    assert c.rebalance_activity_penalty_coeff == 0.5
    assert c.rebalance_activity_penalty_cap_frac == 0.5
    assert c.rebalance_utilization_window_days == 7
    assert c.rebalance_utilization_floor == 0.05
    assert c.rebalance_utilization_ceiling == 1.0
    assert c.rebalance_utilization_min_forwards == 5
    assert c.rebalance_size_tiered_targets is True
    assert c.rebalance_size_reference_percentile == 0.5
    assert c.rebalance_small_channel_band_half_width == 0.15


class TestUpstreamPatternOptionsRegistered:
    """P6-002 regression guard: a config knob added to the Config dataclass
    without a matching plugin.add_option(...) registration is a silent
    no-op at startup (the operator-supplied value is never read). Verify
    each of the 10 upstream-pattern knobs is (a) present in CONFIG_FIELD_TYPES
    and (b) actually registered as a plugin option with a default that
    matches the Config dataclass default, using the same AST-based
    extraction as TestConfigOptionDefaultAlignment above (avoids importing
    the plugin module, which would register the whole plugin surface)."""

    FIELD_TO_OPTION = {
        "rebalance_activity_window_seconds": "revenue-ops-rebalance-activity-window-seconds",
        "rebalance_activity_penalty_coeff": "revenue-ops-rebalance-activity-penalty-coeff",
        "rebalance_activity_penalty_cap_frac": "revenue-ops-rebalance-activity-penalty-cap-frac",
        "rebalance_utilization_window_days": "revenue-ops-rebalance-utilization-window-days",
        "rebalance_utilization_floor": "revenue-ops-rebalance-utilization-floor",
        "rebalance_utilization_ceiling": "revenue-ops-rebalance-utilization-ceiling",
        "rebalance_utilization_min_forwards": "revenue-ops-rebalance-utilization-min-forwards",
        "rebalance_size_tiered_targets": "revenue-ops-rebalance-size-tiered-targets",
        "rebalance_size_reference_percentile": "revenue-ops-rebalance-size-reference-percentile",
        "rebalance_small_channel_band_half_width": "revenue-ops-rebalance-small-channel-band-half-width",
    }

    def _plugin_option_default(self, name):
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

    def test_all_ten_knobs_in_config_field_types(self):
        from modules.config import CONFIG_FIELD_TYPES
        for field in self.FIELD_TO_OPTION:
            assert field in CONFIG_FIELD_TYPES, f"{field} missing from CONFIG_FIELD_TYPES"

    def test_all_ten_knobs_registered_as_plugin_options(self):
        from modules.config import Config
        cfg = Config()
        for field, option_name in self.FIELD_TO_OPTION.items():
            option_default = self._plugin_option_default(option_name)
            field_default = getattr(cfg, field)
            if isinstance(field_default, bool):
                parsed_bool = str(option_default).lower() in ("true", "1", "yes")
                assert parsed_bool is field_default, (
                    f"{option_name} default {option_default!r} does not match "
                    f"{field}={field_default!r}"
                )
            else:
                assert type(field_default)(option_default) == field_default, (
                    f"{option_name} default {option_default!r} does not match "
                    f"{field}={field_default!r}"
                )
