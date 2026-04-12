"""
Tests for BookkeeperCache — batch bkpr-listincome indexing.

Verifies:
1. Single onchain_fee event indexed by txid
2. Wallet fallback when channel-account missing
3. Channel-account preferred over wallet
4. Unknown txid returns None
5. Bookkeeper unavailable returns None
6. Only one RPC call regardless of lookup count
7. Non-fee events ignored
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import BookkeeperCache, ChannelProfitabilityAnalyzer


# ============================================================
# Helpers
# ============================================================

def _make_rpc(events):
    """Build a mock RPC that returns the given income_events list."""
    rpc = MagicMock()
    rpc.call.return_value = {"income_events": events}
    return rpc


def _onchain_fee_event(account, txid, credit_msat="0msat", debit_msat="0msat"):
    """Build a single onchain_fee income event."""
    return {
        "account": account,
        "tag": "onchain_fee",
        "txid": txid,
        "credit_msat": credit_msat,
        "debit_msat": debit_msat,
    }


# ============================================================
# Tests
# ============================================================

class TestBookkeeperCache:

    def test_single_onchain_fee_indexed_by_txid(self):
        """Single consolidated onchain_fee event retrievable by txid."""
        txid = "aabb" * 16
        events = [
            _onchain_fee_event("channel:123x1x0", txid, credit_msat="0msat", debit_msat="450000msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        assert cache.available is True
        # debit - credit = 450000 msat → 450 sats
        assert cache.get_open_cost_by_txid(txid) == 450

    def test_wallet_fallback_when_channel_account_missing(self):
        """Wallet perspective used when no channel-account fee exists."""
        txid = "ccdd" * 16
        events = [
            _onchain_fee_event("wallet", txid, credit_msat="0msat", debit_msat="300000msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        assert cache.available is True
        # wallet: debit - credit = 300000 msat → 300 sats
        assert cache.get_open_cost_by_txid(txid) == 300

    def test_channel_account_preferred_over_wallet(self):
        """Channel-account fee takes priority over wallet perspective."""
        txid = "eeff" * 16
        events = [
            _onchain_fee_event("channel:456x2x0", txid, credit_msat="0msat", debit_msat="500000msat"),
            _onchain_fee_event("wallet", txid, credit_msat="0msat", debit_msat="600000msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        # Channel-account: debit - credit = 500000 → 500 sats
        # Wallet: debit - credit = 600000 → 600 sats
        # Channel-account should win
        assert cache.get_open_cost_by_txid(txid) == 500

    def test_channel_account_netting_debit_minus_credit(self):
        """Multiple onchain_fee updates for a channel account are netted correctly."""
        txid = "abcd" * 16
        events = [
            _onchain_fee_event("channel:789x3x0", txid, credit_msat="0msat", debit_msat="700000msat"),
            _onchain_fee_event("channel:789x3x0", txid, credit_msat="200000msat", debit_msat="0msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        # debit - credit = 500000 msat → 500 sats
        assert cache.get_open_cost_by_txid(txid) == 500

    def test_zero_net_channel_account_suppresses_wallet_fallback(self):
        """A channel-account onchain_fee entry should block wallet fallback even when net zero."""
        txid = "beef" * 16
        events = [
            _onchain_fee_event("channel:999x9x0", txid, credit_msat="400000msat", debit_msat="0msat"),
            _onchain_fee_event("channel:999x9x0", txid, credit_msat="0msat", debit_msat="400000msat"),
            _onchain_fee_event("wallet", txid, credit_msat="0msat", debit_msat="600000msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        # Channel-account presence should win, even though net fee is 0.
        assert cache.get_open_cost_by_txid(txid) == 0

    def test_unknown_txid_returns_none(self):
        """Missing txid returns None."""
        events = [
            _onchain_fee_event("channel:111x1x0", "aa" * 32, credit_msat="100000msat", debit_msat="0msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        assert cache.get_open_cost_by_txid("unknown_txid") is None

    def test_bookkeeper_unavailable_returns_none(self):
        """RPC failure -> available=False, all lookups return None."""
        rpc = MagicMock()
        rpc.call.side_effect = Exception("bkpr-listincome: command not found")
        cache = BookkeeperCache(rpc)

        assert cache.available is False
        assert cache.get_open_cost_by_txid("anything") is None

    def test_only_one_rpc_call(self):
        """Exactly 1 RPC call regardless of how many lookups are performed."""
        txid1 = "1111" * 16
        txid2 = "2222" * 16
        events = [
            _onchain_fee_event("channel:100x0x0", txid1, credit_msat="200000msat", debit_msat="0msat"),
            _onchain_fee_event("channel:200x0x0", txid2, credit_msat="300000msat", debit_msat="0msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        # Multiple lookups
        cache.get_open_cost_by_txid(txid1)
        cache.get_open_cost_by_txid(txid2)
        cache.get_open_cost_by_txid("nonexistent")
        cache.get_open_cost_by_txid(txid1)

        # Still only 1 RPC call from __init__
        assert rpc.call.call_count == 1

    def test_non_fee_events_ignored(self):
        """Routed/invoice events are not indexed."""
        txid = "abcd" * 16
        events = [
            {
                "account": "channel:111x1x0",
                "tag": "routed",
                "txid": txid,
                "credit_msat": 1000000,
                "debit_msat": 0,
            },
            {
                "account": "channel:111x1x0",
                "tag": "invoice",
                "txid": txid,
                "credit_msat": 500000,
                "debit_msat": 0,
            },
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        assert cache.available is True
        assert cache.get_open_cost_by_txid(txid) is None


# ============================================================
# Helpers for analyzer tests
# ============================================================

def _make_analyzer():
    """Build analyzer with mocked dependencies."""
    plugin = MagicMock()
    config = MagicMock()
    config.estimated_open_cost_sats = 2000
    config.rpc_timeout_seconds = 15
    database = MagicMock()
    analyzer = ChannelProfitabilityAnalyzer(plugin, config, database)
    return analyzer


# ============================================================
# Task 2: Open cost from cache
# ============================================================

class TestOpenCostFromCache:
    def test_open_cost_from_cache_skips_rpc(self):
        """When bkpr_cache has the fee, no per-channel RPC is made."""
        analyzer = _make_analyzer()
        analyzer.database.get_channel_open_cost.return_value = None
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }

        cache = MagicMock()
        cache.available = True
        cache.get_open_cost_by_txid.return_value = 350

        costs = analyzer._get_channel_costs(
            "100x1x0", "peer1", "funding_tx_1", 2_000_000,
            opener="local", open_timestamp=1700000000,
            bkpr_cache=cache
        )

        assert costs.open_cost_sats == 350
        analyzer.plugin.rpc.call.assert_not_called()

    def test_open_cost_falls_back_to_config_when_cache_empty(self):
        """When cache has no data for this txid, uses config fallback."""
        analyzer = _make_analyzer()
        analyzer.database.get_channel_open_cost.return_value = None
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }

        cache = MagicMock()
        cache.available = True
        cache.get_open_cost_by_txid.return_value = None

        costs = analyzer._get_channel_costs(
            "100x1x0", "peer1", "funding_tx_1", 2_000_000,
            opener="local", open_timestamp=1700000000,
            bkpr_cache=cache
        )

        assert costs.open_cost_sats == 2000  # config.estimated_open_cost_sats


# ============================================================
# Task 3: Open timestamp without bookkeeper RPC
# ============================================================

class TestOpenTimestampNoBkprRpc:
    def test_scid_block_height_estimate(self):
        """Timestamp is derived from SCID block height without any RPC."""
        analyzer = _make_analyzer()
        ts = analyzer._get_channel_open_timestamp("900000x123x0", "sometxid")
        expected = 1231006505 + (900000 * 600)
        assert ts == expected
        analyzer.plugin.rpc.call.assert_not_called()

    def test_fallback_when_scid_unparseable(self):
        """Bad SCID format falls back to 30 days ago."""
        analyzer = _make_analyzer()
        ts = analyzer._get_channel_open_timestamp("badformat", "sometxid")
        now = int(time.time())
        thirty_days_ago = now - (86400 * 30)
        assert abs(ts - thirty_days_ago) < 5


# ============================================================
# Task 4: Rebalance costs from DB only
# ============================================================

class TestRebalanceCostsDbOnly:
    def test_rebalance_cost_from_db_no_bkpr_rpc(self):
        """_get_channel_costs uses DB rebalance cost without bookkeeper call."""
        analyzer = _make_analyzer()
        analyzer.database.get_channel_open_cost.return_value = 300
        analyzer.database.get_channel_rebalance_costs.return_value = 1500
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }

        costs = analyzer._get_channel_costs(
            "100x1x0", "peer1", "funding_tx_1", 2_000_000,
            opener="local", open_timestamp=1700000000,
            bkpr_cache=MagicMock(available=True, get_open_cost_by_txid=MagicMock(return_value=None))
        )

        assert costs.rebalance_cost_sats == 1500
        for call_args in analyzer.plugin.rpc.call.call_args_list:
            assert "bkpr-listaccountevents" not in str(call_args)


# ============================================================
# Task 5: analyze_all_channels creates one BookkeeperCache
# ============================================================

class TestAnalyzeAllChannelsBatchFetch:
    """analyze_all_channels creates one BookkeeperCache for all channels."""

    def test_single_bkpr_call_for_all_channels(self):
        """Analyzing 3 channels makes exactly 1 bkpr-listincome call."""
        analyzer = _make_analyzer()

        # Set up 3 channels
        analyzer.plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "state": "CHANNELD_NORMAL",
                    "short_channel_id": f"{900000 + i}x1x0",
                    "total_msat": "2000000000msat",
                    "funding_txid": f"txid_{i}",
                    "peer_id": f"peer_{i}",
                    "opener": "local"
                }
                for i in range(3)
            ]
        }

        # Set up bkpr-listincome response
        from unittest.mock import call
        analyzer.plugin.rpc.call.return_value = {
            "income_events": [
                {
                    "account": f"reversed_{i}",
                    "tag": "onchain_fee",
                    "credit_msat": f"{(i + 1) * 100000}msat",
                    "debit_msat": "0msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": f"txid_{i}"
                }
                for i in range(3)
            ]
        }

        # DB returns nothing cached
        analyzer.database.get_channel_open_cost.return_value = None
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }
        analyzer.database.get_total_routing_revenue.return_value = 0
        analyzer.database.get_channel_routing_revenue.return_value = (0, 0, 0)
        analyzer.database.get_last_routing_time.return_value = None
        analyzer.database.get_channel_full_pnl.return_value = None
        analyzer.database.get_channel_sourced_revenue.return_value = (0, 0, 0)

        analyzer.analyze_all_channels(force=True)

        # Only ONE bkpr-listincome call, not 3+ bkpr-listaccountevents calls
        bkpr_calls = [
            c for c in analyzer.plugin.rpc.call.call_args_list
            if "bkpr" in str(c)
        ]
        assert len(bkpr_calls) == 1
        assert bkpr_calls[0] == call("bkpr-listincome", {"consolidate_fees": True})
