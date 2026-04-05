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
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import BookkeeperCache


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
            _onchain_fee_event("channel:123x1x0", txid, credit_msat="450000msat", debit_msat="0msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        assert cache.available is True
        # credit - debit = 450000 msat → 450 sats
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
            _onchain_fee_event("channel:456x2x0", txid, credit_msat="500000msat", debit_msat="0msat"),
            _onchain_fee_event("wallet", txid, credit_msat="0msat", debit_msat="600000msat"),
        ]
        rpc = _make_rpc(events)
        cache = BookkeeperCache(rpc)

        # Channel-account: credit - debit = 500000 → 500 sats
        # Wallet: debit - credit = 600000 → 600 sats
        # Channel-account should win
        assert cache.get_open_cost_by_txid(txid) == 500

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
