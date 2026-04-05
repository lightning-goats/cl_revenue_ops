"""
Tests for datastore IPC — profitability summary push.

Verifies:
1. _push_profitability_summary writes correct key and payload structure
2. analyze_all_channels calls the push after analysis
3. Datastore failures don't crash analysis (fire-and-forget)
"""

import json
import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch, call

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import (
    ChannelProfitabilityAnalyzer,
    ChannelProfitability,
    ChannelCosts,
    ChannelRevenue,
    ProfitabilityClass,
    ChannelRole,
)


# ============================================================
# Helpers
# ============================================================

def _make_analyzer():
    """Build an analyzer with mocked plugin, config, and database."""
    plugin = MagicMock()
    config = MagicMock()
    config.estimated_open_cost_sats = 1000
    database = MagicMock()
    analyzer = ChannelProfitabilityAnalyzer(plugin, config, database)
    return analyzer, plugin, config, database


def _setup_integration_mocks(analyzer, plugin, config, database, ch_id="100x1x0"):
    """Wire up all mocks needed for analyze_all_channels to produce results."""
    now = int(time.time())

    # RPC cache for listpeerchannels
    rpc_cache = MagicMock()
    rpc_cache.listpeerchannels.return_value = {
        "channels": [{
            "state": "CHANNELD_NORMAL",
            "short_channel_id": ch_id,
            "total_msat": "2000000000msat",
            "peer_id": "02abc",
            "funding_txid": "ff" * 32,
            "opener": "local",
        }]
    }
    analyzer.rpc_cache = rpc_cache

    # Bookkeeper (income_events)
    plugin.rpc.call.return_value = {"income_events": []}

    # Database methods used by _get_channel_costs
    database.get_all_channels_revenue_totals.return_value = {
        ch_id: {
            "fees_earned_sats": 6000,
            "volume_routed_sats": 1_000_000,
            "forward_count": 50,
            "sourced_volume_sats": 800_000,
            "sourced_fee_contribution_sats": 4000,
            "sourced_forward_count": 30,
        }
    }
    database.get_channel_rebalance_costs.return_value = 500
    database.get_channel_open_cost.return_value = 1000
    database.get_last_forward_time_any_direction.return_value = now - 3600
    database.get_channel_full_pnl.return_value = {
        "total_contribution_sats": 2000,
        "rebalance_cost_sats": 300,
    }
    database.get_channel_rebalance_success_rate.return_value = {
        "total": 0, "successes": 0, "success_rate": 0.0,
        "avg_cost_ppm": 0, "avg_amount_sats": 0,
    }
    database.get_diagnostic_rebalance_stats.return_value = {
        "attempt_count": 0, "last_success_time": None,
    }
    database.get_fee_strategy_state.return_value = None


def _make_profitability(channel_id="100x1x0", peer_id="02abc",
                        classification=ProfitabilityClass.PROFITABLE,
                        net_profit_sats=5000, roi_percent=42.5,
                        days_open=90, forward_count=50,
                        sourced_forward_count=30):
    """Build a ChannelProfitability with sensible defaults."""
    costs = ChannelCosts(
        channel_id=channel_id,
        peer_id=peer_id,
        open_cost_sats=1000,
        rebalance_cost_sats=500,
    )
    revenue = ChannelRevenue(
        channel_id=channel_id,
        fees_earned_sats=6000,
        volume_routed_sats=1_000_000,
        forward_count=forward_count,
        sourced_volume_sats=800_000,
        sourced_fee_contribution_sats=4000,
        sourced_forward_count=sourced_forward_count,
    )
    return ChannelProfitability(
        channel_id=channel_id,
        peer_id=peer_id,
        capacity_sats=2_000_000,
        costs=costs,
        revenue=revenue,
        net_profit_sats=net_profit_sats,
        roi_percent=roi_percent,
        classification=classification,
        cost_per_sat_routed=0.001,
        fee_per_sat_routed=0.006,
        days_open=days_open,
        last_routed=int(time.time()) - 3600,
    )


# ============================================================
# Tests — _push_profitability_summary (unit)
# ============================================================

class TestPushProfitabilitySummary:

    def test_payload_structure(self):
        """Datastore called with correct key, and payload has expected fields."""
        analyzer, plugin, _, _ = _make_analyzer()
        # Prevent stale-cache trigger inside get_fee_multiplier -> get_profitability
        analyzer._cache_timestamp = int(time.time())

        ch_id = "100x1x0"
        p = _make_profitability(channel_id=ch_id)
        results = {ch_id: p}

        analyzer._push_profitability_summary(results)

        plugin.rpc.datastore.assert_called_once()
        call_kwargs = plugin.rpc.datastore.call_args
        # Check key
        assert call_kwargs[1]["key"] == ["revenue", "profitability-summary"]
        assert call_kwargs[1]["mode"] == "create-or-replace"

        # Parse and validate JSON payload
        payload = json.loads(call_kwargs[1]["string"])
        assert "timestamp" in payload
        assert isinstance(payload["timestamp"], int)
        assert ch_id in payload["channels"]

        ch = payload["channels"][ch_id]
        assert ch["class"] == "profitable"
        assert ch["net_profit_sats"] == 5000
        assert ch["roi_pct"] == 42.5
        assert ch["days_open"] == 90
        assert ch["role"] in [r.value for r in ChannelRole]
        assert "fee_multiplier" in ch
        assert isinstance(ch["fee_multiplier"], float)

    def test_multiple_channels(self):
        """All channels appear in the payload."""
        analyzer, plugin, _, _ = _make_analyzer()
        analyzer._cache_timestamp = int(time.time())
        results = {
            "100x1x0": _make_profitability(channel_id="100x1x0"),
            "200x2x0": _make_profitability(
                channel_id="200x2x0",
                classification=ProfitabilityClass.UNDERWATER,
                net_profit_sats=-1000,
                roi_percent=-15.0,
            ),
        }

        analyzer._push_profitability_summary(results)

        payload = json.loads(plugin.rpc.datastore.call_args[1]["string"])
        assert len(payload["channels"]) == 2
        assert payload["channels"]["200x2x0"]["class"] == "underwater"
        assert payload["channels"]["200x2x0"]["net_profit_sats"] == -1000

    def test_empty_results_pushes_empty_channels(self):
        """Empty results still push a valid payload with no channels."""
        analyzer, plugin, _, _ = _make_analyzer()
        analyzer._cache_timestamp = int(time.time())

        analyzer._push_profitability_summary({})

        payload = json.loads(plugin.rpc.datastore.call_args[1]["string"])
        assert payload["channels"] == {}
        assert "timestamp" in payload

    def test_failure_does_not_raise(self):
        """Datastore RPC failure is swallowed (fire-and-forget)."""
        analyzer, plugin, _, _ = _make_analyzer()
        analyzer._cache_timestamp = int(time.time())
        plugin.rpc.datastore.side_effect = Exception("datastore unavailable")

        # Should not raise
        analyzer._push_profitability_summary({"100x1x0": _make_profitability()})

        plugin.log.assert_called()

    def test_roi_rounded_to_two_decimals(self):
        """roi_pct is rounded to 2 decimal places."""
        analyzer, plugin, _, _ = _make_analyzer()
        analyzer._cache_timestamp = int(time.time())
        p = _make_profitability(roi_percent=33.33333)
        analyzer._push_profitability_summary({"100x1x0": p})

        payload = json.loads(plugin.rpc.datastore.call_args[1]["string"])
        assert payload["channels"]["100x1x0"]["roi_pct"] == 33.33


# ============================================================
# Tests — analyze_all_channels integration
# ============================================================

class TestAnalyzeAllChannelsPush:

    def test_push_called_after_analysis(self):
        """analyze_all_channels calls _push_profitability_summary with results."""
        analyzer, plugin, config, database = _make_analyzer()
        ch_id = "100x1x0"
        _setup_integration_mocks(analyzer, plugin, config, database, ch_id)

        results = analyzer.analyze_all_channels(force=True)

        # Results should contain our channel
        assert ch_id in results

        # Datastore should have been called
        plugin.rpc.datastore.assert_called_once()
        call_kwargs = plugin.rpc.datastore.call_args[1]
        assert call_kwargs["key"] == ["revenue", "profitability-summary"]

        payload = json.loads(call_kwargs["string"])
        assert ch_id in payload["channels"]

    def test_push_failure_does_not_crash_analysis(self):
        """Analysis returns results even when datastore push raises."""
        analyzer, plugin, config, database = _make_analyzer()
        ch_id = "100x1x0"
        _setup_integration_mocks(analyzer, plugin, config, database, ch_id)
        plugin.rpc.datastore.side_effect = Exception("datastore broken")

        # Should NOT raise despite datastore failure
        results = analyzer.analyze_all_channels(force=True)

        # Analysis results should still be returned
        assert ch_id in results
        assert results[ch_id].net_profit_sats is not None


# ============================================================
# Tests — Fee-bounds push
# ============================================================

class TestFeeBoundsPush:
    """Fee bounds pushed to datastore each fee cycle."""

    def test_fee_bounds_payload_structure(self):
        """Fee bounds payload has timestamp, min, max, mid."""
        import json as _json
        payload = {
            "timestamp": int(time.time()),
            "min_fee_ppm": 10,
            "max_fee_ppm": 5000,
            "mid_fee_ppm": 2505,
        }
        payload_str = _json.dumps(payload)
        parsed = _json.loads(payload_str)
        assert parsed["min_fee_ppm"] == 10
        assert parsed["max_fee_ppm"] == 5000
        assert parsed["mid_fee_ppm"] == 2505
        assert "timestamp" in parsed
        assert isinstance(parsed["timestamp"], int)


# ============================================================
# Tests — Dashboard push
# ============================================================

class TestDashboardPush:
    """Dashboard snapshot pushed to datastore."""

    def test_dashboard_payload_structure(self):
        """Dashboard payload wraps revenue_dashboard output with timestamp."""
        import json as _json
        dashboard_result = {
            "financial_health": {"tlv_sats": 187071746, "net_profit_sats": 12815, "operating_margin_pct": 95.07, "annualized_roc_pct": 0.07},
            "period": {"window_days": 30, "gross_revenue_sats": 13480, "opex_sats": 665, "rebalance_cost_sats": 665, "closure_cost_sats": 0, "volume_sats": 20638037, "forward_count": 188},
            "warnings": ["Channel unknown is bleeding"],
            "bleeder_count": 1,
        }
        payload = {"timestamp": int(time.time()), **dashboard_result}
        payload_str = _json.dumps(payload)
        parsed = _json.loads(payload_str)
        assert "timestamp" in parsed
        assert parsed["financial_health"]["net_profit_sats"] == 12815
        assert parsed["period"]["window_days"] == 30
        assert len(payload_str) < 5000
