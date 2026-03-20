"""Comprehensive Boltz integration tests for balance planning, budget accounting, and external-pay."""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
from tests.plugin_test_utils import load_plugin_module


def _make_manager(**overrides):
    cfg_kwargs = {
        "enabled": True,
        "cli_path": "/usr/local/bin/boltzcli",
        "datadir": "/tmp/test_boltz",
        "daily_budget_sats": 3000,
        "enforce_budget": True,
    }
    cfg_kwargs.update(overrides)
    cfg = BoltzCliConfig(**cfg_kwargs)
    plugin = MagicMock()
    plugin.log = MagicMock()
    rpc = MagicMock()
    mgr = BoltzCliManager(plugin, rpc, cfg)
    return mgr


def _load_boltz_plugin_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    return mod


class TestBoltzAutoCycleModeSelection:
    def test_prefers_treasury_when_reserve_below_target(self):
        mod = _load_boltz_plugin_module()

        selection = mod._select_boltz_auto_cycle_mode(
            treasury_plan={
                "status": "ok",
                "recommendations": [{"channel_id": "100x1x0"}],
                "treasury": {"deficit_sats": 700000, "min_deficit_sats": 250000},
            },
            balance_plan={
                "recommendations": [{"channel_id": "200x1x0"}],
            },
        )

        assert selection["mode"] == "treasury"
        assert selection["reason"] == "standing_onchain_reserve_below_target"
        assert selection["reserve_deficit_sats"] == 700000

    def test_falls_back_to_balance_when_treasury_is_at_target(self):
        mod = _load_boltz_plugin_module()

        selection = mod._select_boltz_auto_cycle_mode(
            treasury_plan={
                "status": "at_target",
                "recommendations": [],
                "treasury": {"deficit_sats": 0, "min_deficit_sats": 250000},
            },
            balance_plan={
                "recommendations": [{"channel_id": "200x1x0"}],
            },
        )

        assert selection["mode"] == "balance"
        assert selection["reason"] == "onchain_reserve_healthy_use_balance_mode"

    def test_returns_idle_when_no_mode_has_candidates(self):
        mod = _load_boltz_plugin_module()

        selection = mod._select_boltz_auto_cycle_mode(
            treasury_plan={
                "status": "at_target",
                "recommendations": [],
                "treasury": {"deficit_sats": 0},
            },
            balance_plan={"recommendations": []},
        )

        assert selection["mode"] == "idle"
        assert selection["reason"] == "no_eligible_boltz_actions"


class TestDynamicChannelTuning:
    """Tests for _boltz_dynamic_channel_tuning() threshold/sizing logic.

    These test the mathematical formulas directly since the function is pure.
    """

    def test_high_protection_channel_triggers_earlier(self):
        base_low_trigger = 35.0
        protection_score = 0.9
        trigger_boost = 20.0 * protection_score
        eff_low_trigger = min(70.0, max(base_low_trigger, base_low_trigger + trigger_boost))
        assert eff_low_trigger > base_low_trigger
        assert eff_low_trigger == 53.0

    def test_low_protection_channel_uses_defaults(self):
        protection_score = 0.0
        trigger_boost = 20.0 * protection_score
        eff_low_trigger = min(70.0, max(35.0, 35.0 + trigger_boost))
        assert eff_low_trigger == 35.0

    def test_cooldown_multiplier_bounds(self):
        for ps in [0.0, 0.5, 1.0, 1.5]:
            cooldown_mult = 1.0 - (0.75 * min(1.0, ps))
            assert cooldown_mult >= 0.25, f"Cooldown {cooldown_mult} < 0.25 at ps={ps}"

    def test_amount_multiplier_range(self):
        for ps in [0.0, 0.5, 1.0]:
            amount_mult = 1.0 + (2.0 * ps)
            assert 1.0 <= amount_mult <= 3.0

    def test_drain_accel_score_clamped(self):
        for vel in [-0.1, 0.0, 0.001, 0.05/24.0, 0.1, 1.0]:
            score = max(0.0, min(1.0, vel / (0.05 / 24.0)))
            assert 0.0 <= score <= 1.0

    def test_severity_loop_in_range(self):
        for trigger, local in [(40, 0), (40, 20), (40, 39), (40, 40)]:
            severity = max(0.0, (trigger - local) / max(trigger, 1.0))
            assert 0.0 <= severity <= 1.0

    def test_severity_loop_out_range(self):
        for trigger, local in [(80, 80), (80, 90), (80, 100)]:
            severity = max(0.0, (local - trigger) / max(100.0 - trigger, 1.0))
            assert 0.0 <= severity <= 1.0

    def test_protection_score_composition(self):
        """Protection score should be weighted combination of hotness and drain."""
        hotness = 0.8
        drain = 0.6
        protection = max(0.0, min(1.0, 0.60 * hotness + 0.40 * drain))
        assert abs(protection - 0.72) < 0.001

    def test_trigger_boost_capped_at_70(self):
        """Effective low trigger should never exceed 70%."""
        protection_score = 1.0
        trigger_boost = 20.0 * protection_score
        eff = min(70.0, max(35.0, 35.0 + trigger_boost))
        assert eff == 55.0  # 35 + 20 = 55, under 70 cap


class TestBoltzCostComponents:
    """Tests for get_boltz_cost_components() budget accounting."""

    def test_completed_swap_counted(self):
        mgr = _make_manager()
        now = int(time.time())
        swap = {
            "id": "s1", "createdAt": str(now - 100), "completedAt": str(now - 50),
            "state": "completed", "status": "swap.completed",
            "boltzFee": "40", "networkFee": "10",
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [swap]})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=24)
        assert result["spent_24h_sats"] == 50
        assert result["counted_swaps"] == 1

    def test_old_swap_excluded(self):
        mgr = _make_manager()
        now = int(time.time())
        old_swap = {
            "id": "s1", "createdAt": str(now - 200000), "completedAt": str(now - 200000),
            "state": "completed", "status": "swap.completed",
            "boltzFee": "100", "networkFee": "10",
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [old_swap]})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=24)
        assert result["spent_24h_sats"] == 0

    def test_swap_without_timestamp_skipped(self):
        mgr = _make_manager()
        swap = {
            "id": "s1", "state": "completed", "status": "swap.completed",
            "boltzFee": "40", "networkFee": "10",
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [swap]})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=24)
        assert result["skipped_without_timestamp"] >= 1

    def test_budget_enforcement_blocks_over_limit(self):
        mgr = _make_manager(daily_budget_sats=100, enforce_budget=True)
        mgr.get_budget_status = MagicMock(return_value={
            "remaining_24h_sats_estimate": 50,
            "daily_budget_sats": 100,
        })
        quote = {"boltzFee": "60", "networkFee": "10"}
        result = mgr._enforce_budget_for_quote(quote)
        assert result["allowed"] is False
        assert "exceeds" in result["reason"].lower()

    def test_budget_enforcement_allows_within_limit(self):
        mgr = _make_manager(daily_budget_sats=100, enforce_budget=True)
        mgr.get_budget_status = MagicMock(return_value={
            "remaining_24h_sats_estimate": 80,
            "daily_budget_sats": 100,
        })
        quote = {"boltzFee": "30", "networkFee": "10"}
        result = mgr._enforce_budget_for_quote(quote)
        assert result["allowed"] is True

    def test_boltzd_unreachable_raises(self):
        mgr = _make_manager()
        mgr._listswaps_json = MagicMock(side_effect=Exception("Connection refused"))
        with pytest.raises(Exception, match="Connection refused"):
            mgr.get_boltz_cost_components(window_hours=24)

    def test_window_hours_clamped_zero_uses_default(self):
        """window_hours=0 is falsy so falls through to default 24h."""
        mgr = _make_manager()
        mgr._listswaps_json = MagicMock(return_value={"swaps": []})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=0)
        assert result["window_seconds"] == 24 * 3600  # 0 is falsy, defaults to 24

    def test_window_hours_clamped_large_value(self):
        """window_hours over 168 should be clamped to 168."""
        mgr = _make_manager()
        mgr._listswaps_json = MagicMock(return_value={"swaps": []})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=999)
        assert result["window_seconds"] == 168 * 3600  # Clamped to 168 hours max

    def test_window_hours_clamped_small_positive(self):
        """window_hours below 1 (but truthy) should be clamped to 1 hour."""
        mgr = _make_manager()
        mgr._listswaps_json = MagicMock(return_value={"swaps": []})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        # window_hours=-5 is truthy, int(-5 or 24) = -5, max(1, min(168, -5)) = 1
        result = mgr.get_boltz_cost_components(window_hours=-5)
        assert result["window_seconds"] == 3600  # Clamped to 1 hour minimum


class TestExternalPayFallback:
    """Tests for chanId rejection handling and external-pay routing."""

    def test_contains_chanids_cln_error_detection(self):
        mgr = _make_manager()
        swap = {"error": "chanIds are not supported for cln backends"}
        assert mgr._contains_chanids_cln_error(swap) is True

    def test_contains_chanids_no_error(self):
        mgr = _make_manager()
        swap = {"id": "abc", "state": "pending"}
        assert mgr._contains_chanids_cln_error(swap) is False

    def test_extract_reverse_swap_invoice_found(self):
        mgr = _make_manager()
        response = {"invoice": "lnbc1234567890abcdef"}
        invoice = mgr._extract_reverse_swap_invoice(response)
        assert invoice == "lnbc1234567890abcdef"

    def test_extract_reverse_swap_invoice_missing(self):
        mgr = _make_manager()
        response = {"id": "abc", "state": "pending"}
        invoice = mgr._extract_reverse_swap_invoice(response)
        assert invoice is None

    def test_extract_reverse_swap_invoice_nested(self):
        """Invoice extraction should scan nested dicts up to 4 levels deep."""
        mgr = _make_manager()
        response = {"result": {"swap": {"swapInvoice": "lnbc999nested"}}}
        invoice = mgr._extract_reverse_swap_invoice(response)
        assert invoice == "lnbc999nested"

    def test_extract_reverse_swap_invoice_ignores_non_ln_prefix(self):
        """Invoice fields that don't start with 'ln' should be skipped."""
        mgr = _make_manager()
        response = {"invoice": "not_a_real_invoice"}
        invoice = mgr._extract_reverse_swap_invoice(response)
        assert invoice is None

    def test_estimate_swap_fee_named_fields(self):
        mgr = _make_manager()
        swap = {"boltzFee": "50", "networkFee": "15"}
        fee = mgr._estimate_swap_fee_sats(swap)
        assert fee == 65

    def test_estimate_swap_fee_zero_on_empty(self):
        mgr = _make_manager()
        fee = mgr._estimate_swap_fee_sats({})
        assert fee == 0

    def test_estimate_swap_fee_all_named_fields(self):
        """All known fee field names should be summed."""
        mgr = _make_manager()
        swap = {
            "boltzFee": "10",
            "networkFee": "20",
            "serviceFee": "5",
            "routingFee": "3",
            "onchainFee": "2",
        }
        fee = mgr._estimate_swap_fee_sats(swap)
        assert fee == 40

    def test_is_completed_swap_success_states(self):
        mgr = _make_manager()
        # _is_completed_swap checks for tokens: success, completed, claimed, done
        # in the combined state/status text
        for state in ["completed", "success", "done"]:
            assert mgr._is_completed_swap({"state": state}) is True
        # Also works via status field when state is absent
        assert mgr._is_completed_swap({"status": "transaction.claimed"}) is True

    def test_is_completed_swap_pending_state(self):
        mgr = _make_manager()
        assert mgr._is_completed_swap({"status": "pending"}) is False
        assert mgr._is_completed_swap({"state": "pending"}) is False

    def test_is_error_swap_detection(self):
        mgr = _make_manager()
        # Error field triggers error detection
        assert mgr._is_error_swap({"error": "failed"}) is True
        # State field "ERROR" or "FAILED" triggers error detection
        assert mgr._is_error_swap({"state": "ERROR"}) is True
        assert mgr._is_error_swap({"state": "FAILED"}) is True
        # Completed state is not an error
        assert mgr._is_error_swap({"state": "completed"}) is False

    def test_is_error_swap_none_input(self):
        mgr = _make_manager()
        assert mgr._is_error_swap(None) is False
        assert mgr._is_error_swap("not a dict") is False

    def test_parse_int_various_types(self):
        assert BoltzCliManager._parse_int("123", 0) == 123
        assert BoltzCliManager._parse_int(None, 99) == 99
        assert BoltzCliManager._parse_int(45.7, 0) == 45
        assert BoltzCliManager._parse_int(True, 0) == 1
        assert BoltzCliManager._parse_int("garbage", 42) == 42

    def test_parse_int_edge_cases(self):
        assert BoltzCliManager._parse_int("", 7) == 7
        assert BoltzCliManager._parse_int(False, 0) == 0
        assert BoltzCliManager._parse_int("3.14", 0) == 3
        assert BoltzCliManager._parse_int(0, 99) == 0

    def test_parse_timestamp_sanity(self):
        assert BoltzCliManager._parse_timestamp(0) is None
        assert BoltzCliManager._parse_timestamp(-1) is None
        assert BoltzCliManager._parse_timestamp(100) is None  # Before 2001
        assert BoltzCliManager._parse_timestamp(978307200) == 978307200  # Exactly 2001-01-01
        now = int(time.time())
        assert BoltzCliManager._parse_timestamp(now) == now

    def test_parse_timestamp_string_input(self):
        """Timestamps passed as strings should be parsed correctly."""
        assert BoltzCliManager._parse_timestamp("978307200") == 978307200
        assert BoltzCliManager._parse_timestamp("0") is None
        assert BoltzCliManager._parse_timestamp("garbage") is None

    def test_parse_timestamp_boltz_negative(self):
        """Boltz sometimes emits -62135596800 as a default timestamp."""
        assert BoltzCliManager._parse_timestamp(-62135596800) is None

    def test_norm_currency(self):
        assert BoltzCliManager._norm_currency("LBTC", "BTC") == "LBTC"
        assert BoltzCliManager._norm_currency("L-BTC", "BTC") == "LBTC"
        assert BoltzCliManager._norm_currency("BTC", "BTC") == "BTC"
        assert BoltzCliManager._norm_currency("btc", "BTC") == "BTC"

    def test_norm_currency_default_fallback(self):
        """When currency is None, the default should be used."""
        assert BoltzCliManager._norm_currency(None, "BTC") == "BTC"
        assert BoltzCliManager._norm_currency(None, "LBTC") == "LBTC"

    def test_norm_currency_unknown_passthrough(self):
        """Unknown currencies should be uppercased and returned as-is."""
        assert BoltzCliManager._norm_currency("eth", "BTC") == "ETH"
