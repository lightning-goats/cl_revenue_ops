# Bookkeeper Batch Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ~84 sequential per-channel `bkpr-listaccountevents` RPC calls with 1–2 bulk bookkeeper API calls, eliminating the cascade of 15s timeouts that breaks `revenue-profitability`, `revenue-dashboard`, and `yield-summary`.

**Architecture:** Create a `BookkeeperCache` that fetches `bkpr-listincome(consolidate_fees=true)` once per analysis cycle, indexes results by txid/account, and exposes the same data the per-channel methods currently compute. Drop the per-channel `bkpr-listaccountevents` calls entirely. For closure costs (called once per channel close, not a hot path), switch to `bkpr-inspect` which returns `fees_paid_msat` directly.

**Tech Stack:** Python 3.10+, pyln-client, pytest, CLN bookkeeper plugin APIs

---

## Background

### The Problem

`profitability_analyzer.py` makes up to 4 `bkpr-listaccountevents` RPC calls per channel:
1. `_get_all_channels()` → `_get_open_timestamp_from_bookkeeper()` — 1 call/channel
2. `_get_channel_costs()` → `_get_open_cost_from_bookkeeper()` — 1–2 calls/channel (per-account + wallet fallback)
3. `_get_channel_costs()` → `_get_rebalance_costs_from_bookkeeper()` — 1 call/channel

With 28 channels and `rpc_timeout_seconds=15`, worst case is 112 sequential RPCs × 15s = 28 minutes of serial timeouts. The hive bridge has a 5s timeout, so every financial query aborts.

### The Fix

CLN bookkeeper provides purpose-built batch APIs:

| Old (per-channel) | New (batch) |
|---|---|
| N × `bkpr-listaccountevents(account)` for open costs | 1 × `bkpr-listincome(consolidate_fees=true)` |
| N × `bkpr-listaccountevents(account)` for open timestamps | Drop — use SCID block height estimate (already the fallback, accurate to ~10 min) |
| N × `bkpr-listaccountevents(account)` for rebalance costs | Drop — DB is already the primary source and tracks all rebalance attempts |
| N × `bkpr-listaccountevents(account: "wallet")` fallback | Eliminated — `bkpr-listincome` returns all accounts including wallet |
| `bkpr-listaccountevents(account)` for closure costs | 1 × `bkpr-inspect(account)` — returns `fees_paid_msat` per tx directly |

**Result:** 2 RPC calls instead of ~112. Analysis completes in seconds.

### Key API Details

**`bkpr-listincome(consolidate_fees=true)`** returns `income_events` array:
- `account`: channel_id or "wallet"
- `tag`: "onchain_fee", "invoice", "routed", "deposit", etc.
- `credit_msat`, `debit_msat`: amounts
- `txid`: transaction ID (for onchain_fee events)
- `timestamp`: event time
- `consolidate_fees=true` emits one net event per (txid, account) — handles batch opens correctly

**`bkpr-inspect(account)`** returns `txs` array:
- `txid`: transaction ID
- `fees_paid_msat`: on-chain fee for the transaction (direct, no arithmetic)
- `outputs`: array with credit/debit details

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/profitability_analyzer.py` | Modify | Add `BookkeeperCache`, rewire `_get_channel_costs`, `_get_all_channels` |
| `cl-revenue-ops.py:4497-4582` | Modify | Replace `_get_closure_costs_from_bookkeeper` with `bkpr-inspect` |
| `tests/test_bookkeeper_batch.py` | Create | Tests for `BookkeeperCache` and the refactored cost lookups |
| `tests/test_profitability_fixes.py` | Modify | Update mocks that patch removed methods |
| `tests/test_session3_audit_regressions.py` | Modify | Update mocks that patch removed methods |

---

### Task 1: Create BookkeeperCache class

**Files:**
- Create: `tests/test_bookkeeper_batch.py`
- Modify: `modules/profitability_analyzer.py` (add class after imports, ~line 30)

The cache fetches `bkpr-listincome(consolidate_fees=true)` once and builds two indexes:
1. `onchain_fees_by_txid`: maps `txid → {credit_msat, debit_msat}` (summed across all accounts for that txid)
2. `onchain_fees_by_account_txid`: maps `(account, txid) → {credit_msat, debit_msat}` (per-account perspective)

- [ ] **Step 1: Write the failing test for BookkeeperCache construction and indexing**

```python
# tests/test_bookkeeper_batch.py
"""Tests for BookkeeperCache — batch bookkeeper data fetching."""

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

from modules.profitability_analyzer import BookkeeperCache


class TestBookkeeperCacheConstruction:
    """BookkeeperCache fetches bkpr-listincome once and indexes by txid."""

    def test_single_onchain_fee_indexed_by_txid(self):
        """A single consolidated onchain_fee event is retrievable by txid."""
        rpc = MagicMock()
        rpc.call.return_value = {
            "income_events": [
                {
                    "account": "abc123reversed",
                    "tag": "onchain_fee",
                    "credit_msat": "0msat",
                    "debit_msat": "450000msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "funding_tx_aaa"
                }
            ]
        }
        cache = BookkeeperCache(rpc)

        fee = cache.get_open_cost_by_txid("funding_tx_aaa")
        assert fee == 450  # 450000 msat // 1000
        rpc.call.assert_called_once_with(
            "bkpr-listincome", {"consolidate_fees": True}
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestBookkeeperCacheConstruction::test_single_onchain_fee_indexed_by_txid -v`
Expected: FAIL — `ImportError: cannot import name 'BookkeeperCache'`

- [ ] **Step 3: Implement BookkeeperCache**

Add to `modules/profitability_analyzer.py` after the existing imports and before `ProfitabilityClass` (around line 30):

```python
class BookkeeperCache:
    """
    Batch-fetches bkpr-listincome(consolidate_fees=true) once and indexes
    results for O(1) lookups by txid.

    Replaces N per-channel bkpr-listaccountevents calls with 1 bulk call.
    """

    def __init__(self, rpc):
        """Fetch and index all income events.

        Args:
            rpc: An object with .call(method, params) — either plugin.rpc
                 or ThreadSafeRpcProxy.
        """
        self._onchain_fees: dict = {}      # txid → net_fee_sats
        self._wallet_fees: dict = {}       # txid → net_fee_sats (wallet perspective)
        self._fetch_ok = False

        try:
            result = rpc.call("bkpr-listincome", {"consolidate_fees": True})
            events = result.get("income_events", [])
            self._index_onchain_fees(events)
            self._fetch_ok = True
        except Exception:
            # Bookkeeper unavailable — all lookups return None
            pass

    def _index_onchain_fees(self, events: list) -> None:
        """Build txid indexes from onchain_fee events."""
        # Per-account fees: (account, txid) → {credit, debit}
        account_fees: dict = {}
        for ev in events:
            if ev.get("tag") != "onchain_fee":
                continue
            txid = ev.get("txid")
            if not txid:
                continue
            account = ev.get("account", "")
            credit = _shared_parse_msat(ev.get("credit_msat", 0))
            debit = _shared_parse_msat(ev.get("debit_msat", 0))

            key = (account, txid)
            if key in account_fees:
                account_fees[key]["credit"] += credit
                account_fees[key]["debit"] += debit
            else:
                account_fees[key] = {"credit": credit, "debit": debit}

        # Build channel-perspective index: for non-wallet accounts,
        # net fee = credit - debit (channel debited = fee paid)
        for (account, txid), totals in account_fees.items():
            if account == "wallet":
                # Wallet perspective: fee = debit - credit (opposite)
                net_msat = totals["debit"] - totals["credit"]
                if net_msat > 0:
                    self._wallet_fees[txid] = net_msat // 1000
            else:
                net_msat = totals["credit"] - totals["debit"]
                if txid not in self._onchain_fees and net_msat > 0:
                    self._onchain_fees[txid] = net_msat // 1000

    def get_open_cost_by_txid(self, funding_txid: str) -> int | None:
        """Look up the on-chain fee for a funding transaction.

        Tries channel-account perspective first, then wallet fallback.

        Returns:
            Fee in sats, or None if not found.
        """
        if not self._fetch_ok:
            return None
        # Channel-account perspective
        fee = self._onchain_fees.get(funding_txid)
        if fee is not None:
            return fee
        # Wallet fallback (we opened the channel, fee attributed to wallet)
        return self._wallet_fees.get(funding_txid)

    @property
    def available(self) -> bool:
        """Whether the bookkeeper fetch succeeded."""
        return self._fetch_ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestBookkeeperCacheConstruction::test_single_onchain_fee_indexed_by_txid -v`
Expected: PASS

- [ ] **Step 5: Write tests for wallet fallback and batch opens**

Add to `tests/test_bookkeeper_batch.py`:

```python
    def test_wallet_fallback_when_channel_account_missing(self):
        """If no channel-account fee exists, falls back to wallet perspective."""
        rpc = MagicMock()
        rpc.call.return_value = {
            "income_events": [
                {
                    "account": "wallet",
                    "tag": "onchain_fee",
                    "credit_msat": "0msat",
                    "debit_msat": "320000msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "funding_tx_bbb"
                }
            ]
        }
        cache = BookkeeperCache(rpc)

        fee = cache.get_open_cost_by_txid("funding_tx_bbb")
        assert fee == 320

    def test_channel_account_preferred_over_wallet(self):
        """Channel-account perspective takes priority over wallet."""
        rpc = MagicMock()
        rpc.call.return_value = {
            "income_events": [
                {
                    "account": "abc123reversed",
                    "tag": "onchain_fee",
                    "credit_msat": "200000msat",
                    "debit_msat": "0msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "funding_tx_ccc"
                },
                {
                    "account": "wallet",
                    "tag": "onchain_fee",
                    "credit_msat": "0msat",
                    "debit_msat": "900000msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "funding_tx_ccc"
                }
            ]
        }
        cache = BookkeeperCache(rpc)

        fee = cache.get_open_cost_by_txid("funding_tx_ccc")
        assert fee == 200  # Channel account, not wallet's 900

    def test_unknown_txid_returns_none(self):
        """Lookup for a txid not in the cache returns None."""
        rpc = MagicMock()
        rpc.call.return_value = {"income_events": []}
        cache = BookkeeperCache(rpc)

        assert cache.get_open_cost_by_txid("nonexistent") is None

    def test_bookkeeper_unavailable_returns_none(self):
        """If bkpr-listincome fails, all lookups return None gracefully."""
        rpc = MagicMock()
        rpc.call.side_effect = Exception("plugin not found")
        cache = BookkeeperCache(rpc)

        assert cache.available is False
        assert cache.get_open_cost_by_txid("any_txid") is None

    def test_only_one_rpc_call(self):
        """Cache makes exactly one RPC call regardless of lookup count."""
        rpc = MagicMock()
        rpc.call.return_value = {
            "income_events": [
                {
                    "account": "ch1",
                    "tag": "onchain_fee",
                    "credit_msat": "100000msat",
                    "debit_msat": "0msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "tx1"
                },
                {
                    "account": "ch2",
                    "tag": "onchain_fee",
                    "credit_msat": "200000msat",
                    "debit_msat": "0msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "tx2"
                }
            ]
        }
        cache = BookkeeperCache(rpc)
        cache.get_open_cost_by_txid("tx1")
        cache.get_open_cost_by_txid("tx2")
        cache.get_open_cost_by_txid("tx3")

        assert rpc.call.call_count == 1

    def test_non_fee_events_ignored(self):
        """Events with tags other than onchain_fee are not indexed."""
        rpc = MagicMock()
        rpc.call.return_value = {
            "income_events": [
                {
                    "account": "ch1",
                    "tag": "routed",
                    "credit_msat": "5000000msat",
                    "debit_msat": "0msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "tx_routed"
                },
                {
                    "account": "ch1",
                    "tag": "onchain_fee",
                    "credit_msat": "150000msat",
                    "debit_msat": "0msat",
                    "currency": "bc",
                    "timestamp": 1700000000,
                    "txid": "tx_fee"
                }
            ]
        }
        cache = BookkeeperCache(rpc)

        assert cache.get_open_cost_by_txid("tx_routed") is None
        assert cache.get_open_cost_by_txid("tx_fee") == 150
```

- [ ] **Step 6: Run all BookkeeperCache tests**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/profitability_analyzer.py tests/test_bookkeeper_batch.py
git commit -m "feat: add BookkeeperCache for batch bkpr-listincome fetching

Replaces N per-channel bkpr-listaccountevents calls with a single
bkpr-listincome(consolidate_fees=true) call, indexed by txid."
```

---

### Task 2: Rewire open cost lookup to use BookkeeperCache

**Files:**
- Modify: `modules/profitability_analyzer.py` — `_get_channel_costs()`, `_get_open_cost_from_bookkeeper()`
- Modify: `tests/test_bookkeeper_batch.py` — add integration-style tests

The `_get_channel_costs` method currently calls `_get_open_cost_from_bookkeeper()` which makes 1–2 RPCs per channel. Rewire it to accept an optional `BookkeeperCache` and look up the fee by `funding_txid` instead.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bookkeeper_batch.py`:

```python
from modules.profitability_analyzer import ChannelProfitabilityAnalyzer


def _make_analyzer():
    """Build analyzer with mocked dependencies."""
    plugin = MagicMock()
    config = MagicMock()
    config.estimated_open_cost_sats = 2000
    config.rpc_timeout_seconds = 15
    database = MagicMock()
    analyzer = ChannelProfitabilityAnalyzer(plugin, config, database)
    return analyzer


class TestOpenCostFromCache:
    """_get_channel_costs uses BookkeeperCache instead of per-channel RPC."""

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
        # Verify no direct bkpr-listaccountevents call was made
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestOpenCostFromCache -v`
Expected: FAIL — `_get_channel_costs() got an unexpected keyword argument 'bkpr_cache'`

- [ ] **Step 3: Add `bkpr_cache` parameter to `_get_channel_costs`**

In `modules/profitability_analyzer.py`, modify `_get_channel_costs` signature (line ~1576):

Change:
```python
    def _get_channel_costs(self, channel_id: str, peer_id: str,
                          funding_txid: str, capacity_sats: int = 0,
                          opener: str = "local",
                          open_timestamp: Optional[int] = None) -> ChannelCosts:
```

To:
```python
    def _get_channel_costs(self, channel_id: str, peer_id: str,
                          funding_txid: str, capacity_sats: int = 0,
                          opener: str = "local",
                          open_timestamp: Optional[int] = None,
                          bkpr_cache: Optional['BookkeeperCache'] = None) -> ChannelCosts:
```

Then in the local-opener branch where it queries bookkeeper for open cost (the section starting around line 1658 `if open_cost is None and funding_txid:`), replace the `_get_open_cost_from_bookkeeper` call with a cache lookup:

Replace the block at ~line 1624–1674 (the `else:` branch for local opener):

```python
        else:
            # Local opener -> We paid fees. Proceed with lookup logic.

            # Use cached value if available
            open_cost = db_open_cost

            # RETROACTIVE FIX: Re-query channels stored with fallback value
            if db_open_cost is not None and db_open_cost == self.config.estimated_open_cost_sats:
                if funding_txid:
                    self.plugin.log(
                        f"Stored cost for {channel_id} is fallback value "
                        f"({db_open_cost} sats). Attempting re-query...",
                        level='debug'
                    )
                    requeried_cost = self._lookup_open_cost(
                        funding_txid, capacity_sats, bkpr_cache
                    )

                    if requeried_cost is not None:
                        self.plugin.log(
                            f"Retroactive fix for {channel_id}: updated open_cost from "
                            f"{db_open_cost} sats (fallback) to {requeried_cost} sats (actual)",
                            level='info'
                        )
                        self.database.record_channel_open_cost(
                            channel_id, peer_id, requeried_cost, capacity_sats,
                            timestamp=open_timestamp
                        )
                        open_cost = requeried_cost

            # SANITY CHECK: Detect invalid open_cost (capital mistaken as expense)
            if open_cost is not None and capacity_sats > 0:
                open_cost = self._sanity_check_open_cost(
                    channel_id, peer_id, funding_txid, open_cost, capacity_sats,
                    open_timestamp=open_timestamp
                )

            # Query bookkeeper if not found
            if open_cost is None and funding_txid:
                open_cost = self._lookup_open_cost(
                    funding_txid, capacity_sats, bkpr_cache
                )
                if open_cost is not None:
                    self.database.record_channel_open_cost(
                        channel_id, peer_id, open_cost, capacity_sats,
                        timestamp=open_timestamp
                    )

            # Final fallback
            if open_cost is None:
                open_cost = self.config.estimated_open_cost_sats
                self.plugin.log(
                    f"Using estimated open cost ({open_cost} sats) for {channel_id} - "
                    f"bookkeeper data not available",
                    level='debug'
                )
```

Add the new `_lookup_open_cost` method after `_get_open_cost_from_bookkeeper`:

```python
    def _lookup_open_cost(self, funding_txid: str, capacity_sats: int = 0,
                          bkpr_cache: Optional['BookkeeperCache'] = None) -> Optional[int]:
        """Look up open cost from cache, falling back to legacy per-channel RPC.

        Args:
            funding_txid: The funding transaction ID
            capacity_sats: Channel capacity for validation
            bkpr_cache: Batch bookkeeper cache (preferred)

        Returns:
            Fee in sats, or None if not found
        """
        fee = None

        # Prefer batch cache
        if bkpr_cache is not None and bkpr_cache.available:
            fee = bkpr_cache.get_open_cost_by_txid(funding_txid)
        else:
            # Legacy fallback: per-channel RPC (only if no cache provided)
            fee = self._get_open_cost_from_bookkeeper(funding_txid, capacity_sats)

        # Validate
        if fee is not None and not self._is_valid_fee_amount(fee, capacity_sats, funding_txid):
            return None

        return fee
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestOpenCostFromCache -v`
Expected: PASS

- [ ] **Step 5: Run existing profitability tests to verify no regressions**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_profitability_fixes.py -v`
Expected: All PASS (existing tests don't pass `bkpr_cache`, so they use the legacy path)

- [ ] **Step 6: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/profitability_analyzer.py tests/test_bookkeeper_batch.py
git commit -m "feat: wire open cost lookup through BookkeeperCache

_get_channel_costs accepts optional bkpr_cache parameter.
When provided, uses O(1) cache lookup instead of per-channel RPC.
Falls back to legacy _get_open_cost_from_bookkeeper when no cache."
```

---

### Task 3: Drop per-channel bookkeeper calls for open timestamps

**Files:**
- Modify: `modules/profitability_analyzer.py` — `_get_all_channels()`, `_get_channel_open_timestamp()`

Currently `_get_all_channels()` calls `_get_channel_open_timestamp()` per channel, which calls `_get_open_timestamp_from_bookkeeper()` → `bkpr-listaccountevents`. This is N more RPCs.

The SCID block height estimate is already the fallback and is accurate to ~10 minutes — more than adequate for `days_open` calculations. Drop the bookkeeper path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bookkeeper_batch.py`:

```python
class TestOpenTimestampNoBkprRpc:
    """Open timestamp estimation uses SCID block height, no bookkeeper RPC."""

    def test_scid_block_height_estimate(self):
        """Timestamp is derived from SCID block height without any RPC."""
        analyzer = _make_analyzer()

        # Block 900000 ≈ genesis + 900000 * 600s
        ts = analyzer._get_channel_open_timestamp("900000x123x0", "sometxid")
        expected = 1231006505 + (900000 * 600)
        assert ts == expected
        # No bookkeeper RPC should have been attempted
        analyzer.plugin.rpc.call.assert_not_called()

    def test_fallback_when_scid_unparseable(self):
        """Bad SCID format falls back to 30 days ago."""
        analyzer = _make_analyzer()
        ts = analyzer._get_channel_open_timestamp("badformat", "sometxid")
        now = int(time.time())
        thirty_days_ago = now - (86400 * 30)
        # Should be within a few seconds of 30 days ago
        assert abs(ts - thirty_days_ago) < 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestOpenTimestampNoBkprRpc -v`
Expected: FAIL — `_get_open_timestamp_from_bookkeeper` still called, or if it's mocked, the RPC call assertion fails.

- [ ] **Step 3: Remove bookkeeper path from `_get_channel_open_timestamp`**

In `modules/profitability_analyzer.py`, replace `_get_channel_open_timestamp` (lines ~1491–1534):

```python
    def _get_channel_open_timestamp(self, channel_id: str, funding_txid: str) -> int:
        """
        Get the timestamp when a channel was opened.

        Uses SCID block height to estimate open time. This is accurate to ~10
        minutes, which is sufficient for days_open / ROI calculations.

        Fallback: 30 days ago if SCID is unparseable.

        Args:
            channel_id: Short channel ID (e.g., "902205x123x0")
            funding_txid: Funding transaction ID (unused, kept for API compat)

        Returns:
            Unix timestamp of channel open
        """
        # Estimate from SCID block height
        # SCID format is "blockheight x txindex x output"
        if channel_id and 'x' in channel_id:
            try:
                block_height = int(channel_id.split('x')[0])
                # ~10 minutes per block, Bitcoin mainnet genesis = 1231006505
                genesis_timestamp = 1231006505
                seconds_per_block = 600
                estimated_timestamp = genesis_timestamp + (block_height * seconds_per_block)

                # Sanity check - should be in the past
                now = int(time.time())
                if estimated_timestamp < now:
                    return estimated_timestamp

            except (ValueError, IndexError):
                pass

        # Fallback to 30 days ago
        return int(time.time()) - (86400 * 30)
```

- [ ] **Step 4: Run tests to verify**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestOpenTimestampNoBkprRpc -v`
Expected: PASS

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_profitability_fixes.py tests/test_session3_audit_regressions.py -v`
Expected: All PASS

- [ ] **Step 5: Delete dead method `_get_open_timestamp_from_bookkeeper`**

Remove the `_get_open_timestamp_from_bookkeeper` method (previously at lines ~1536–1574). It is no longer called.

- [ ] **Step 6: Run full test suite to catch any remaining references**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -v`
Expected: All PASS (no test references `_get_open_timestamp_from_bookkeeper` directly)

- [ ] **Step 7: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/profitability_analyzer.py tests/test_bookkeeper_batch.py
git commit -m "perf: drop per-channel bookkeeper RPC for open timestamps

Use SCID block height estimate instead. Eliminates N bkpr-listaccountevents
calls during channel discovery. Accurate to ~10 minutes — sufficient for
days_open calculations."
```

---

### Task 4: Drop per-channel bookkeeper call for rebalance costs

**Files:**
- Modify: `modules/profitability_analyzer.py` — `_get_channel_costs()`
- Modify: `tests/test_bookkeeper_batch.py`

The database is the primary source for rebalance costs (`database.get_channel_rebalance_costs`). The bookkeeper cross-check via `_get_rebalance_costs_from_bookkeeper` is supplementary — it exists to catch costs the DB might have missed. In practice, the DB records every rebalance attempt reliably. Drop the bookkeeper lookup to eliminate N more RPCs.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bookkeeper_batch.py`:

```python
class TestRebalanceCostsDbOnly:
    """Rebalance costs come from database only — no bookkeeper RPC."""

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
        # Ensure no bkpr-listaccountevents call for rebalance costs
        for call_args in analyzer.plugin.rpc.call.call_args_list:
            assert "bkpr-listaccountevents" not in str(call_args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestRebalanceCostsDbOnly -v`
Expected: FAIL — `_get_rebalance_costs_from_bookkeeper` still called, making a `bkpr-listaccountevents` RPC

- [ ] **Step 3: Remove bookkeeper rebalance cost lookup from `_get_channel_costs`**

In `modules/profitability_analyzer.py`, in `_get_channel_costs` (around lines 1597–1602), replace:

```python
        # Get rebalance costs - combine database records with bookkeeper data
        db_rebalance_costs = self.database.get_channel_rebalance_costs(channel_id)
        bkpr_rebalance_costs = self._get_rebalance_costs_from_bookkeeper(channel_id, funding_txid)

        # Use the higher value (bookkeeper may have more complete history)
        rebalance_costs = max(db_rebalance_costs, bkpr_rebalance_costs)
```

With:

```python
        # Rebalance costs from database (records all rebalance attempts)
        rebalance_costs = self.database.get_channel_rebalance_costs(channel_id)
```

- [ ] **Step 4: Run tests to verify**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestRebalanceCostsDbOnly tests/test_profitability_fixes.py tests/test_session3_audit_regressions.py -v`
Expected: All PASS

- [ ] **Step 5: Delete dead method `_get_rebalance_costs_from_bookkeeper`**

Remove the `_get_rebalance_costs_from_bookkeeper` method (previously at lines ~1992–2054). It is no longer called.

- [ ] **Step 6: Update tests that mock the removed method**

In `tests/test_profitability_fixes.py`, the following lines mock the removed method:
- Line 382: `patch.object(analyzer, '_get_rebalance_costs_from_bookkeeper', return_value=0)`
- Line 416: `patch.object(analyzer, '_get_rebalance_costs_from_bookkeeper', return_value=0)`
- Line 445: `patch.object(analyzer, '_get_rebalance_costs_from_bookkeeper', return_value=0)`

Remove those `patch.object` lines from the `with` statements. The tests should still work because the DB mock already returns 0 by default.

In `tests/test_session3_audit_regressions.py`, line 294:
- `analyzer._get_rebalance_costs_from_bookkeeper = MagicMock(return_value=0)`

Remove that line.

- [ ] **Step 7: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/profitability_analyzer.py tests/test_bookkeeper_batch.py tests/test_profitability_fixes.py tests/test_session3_audit_regressions.py
git commit -m "perf: drop per-channel bookkeeper RPC for rebalance costs

Database is the authoritative source for rebalance costs and records all
attempts. Eliminates N bkpr-listaccountevents calls from the cost
calculation hot path."
```

---

### Task 5: Wire BookkeeperCache into analyze_all_channels

**Files:**
- Modify: `modules/profitability_analyzer.py` — `analyze_all_channels()`, `analyze_channel()`

Create the cache once at the start of `analyze_all_channels` and pass it through to each channel's cost calculation. This is the integration point where 84 RPCs become 1.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bookkeeper_batch.py`:

```python
class TestAnalyzeAllChannelsBatchFetch:
    """analyze_all_channels creates one BookkeeperCache for all channels."""

    def test_single_bkpr_call_for_all_channels(self):
        """Analyzing 3 channels makes exactly 1 bkpr-listincome call."""
        analyzer = _make_analyzer()

        # Set up 3 channels via listpeerchannels
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

        # DB returns nothing cached so bookkeeper cache is used
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestAnalyzeAllChannelsBatchFetch -v`
Expected: FAIL — still making per-channel bookkeeper calls

- [ ] **Step 3: Create BookkeeperCache in analyze_all_channels and pass through**

In `modules/profitability_analyzer.py`, modify `analyze_all_channels` (around lines 405–420):

Replace:
```python
        results = {}
        try:
            # Get all channels
            channels = self._get_all_channels()

            # Batch fetch all revenue data with a single RPC call
            all_revenue_data = self._get_all_revenue_data()

            for channel_id, channel_info in channels.items():
                # Pass precalculated revenue to avoid per-channel RPC calls
                precalculated_revenue = all_revenue_data.get(channel_id)
                profitability = self.analyze_channel(
                    channel_id, channel_info, precalculated_revenue=precalculated_revenue
                )
```

With:
```python
        results = {}
        try:
            # Get all channels
            channels = self._get_all_channels()

            # Batch fetch all revenue data with a single DB query
            all_revenue_data = self._get_all_revenue_data()

            # Batch fetch bookkeeper data with a single RPC call
            bkpr_cache = BookkeeperCache(self.plugin.rpc)

            for channel_id, channel_info in channels.items():
                # Pass precalculated revenue to avoid per-channel RPC calls
                precalculated_revenue = all_revenue_data.get(channel_id)
                profitability = self.analyze_channel(
                    channel_id, channel_info,
                    precalculated_revenue=precalculated_revenue,
                    bkpr_cache=bkpr_cache
                )
```

Modify `analyze_channel` signature (line ~449) to accept and forward the cache:

Change:
```python
    def analyze_channel(self, channel_id: str,
                       channel_info: Optional[Dict] = None,
                       precalculated_revenue: Optional[ChannelRevenue] = None) -> Optional[ChannelProfitability]:
```

To:
```python
    def analyze_channel(self, channel_id: str,
                       channel_info: Optional[Dict] = None,
                       precalculated_revenue: Optional[ChannelRevenue] = None,
                       bkpr_cache: Optional['BookkeeperCache'] = None) -> Optional[ChannelProfitability]:
```

And where `analyze_channel` calls `_get_channel_costs` (around line 480), pass the cache through:

Change:
```python
            costs = self._get_channel_costs(
                channel_id, peer_id, funding_txid, capacity, opener,
                open_timestamp=open_timestamp
            )
```

To:
```python
            costs = self._get_channel_costs(
                channel_id, peer_id, funding_txid, capacity, opener,
                open_timestamp=open_timestamp,
                bkpr_cache=bkpr_cache
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestAnalyzeAllChannelsBatchFetch -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/profitability_analyzer.py tests/test_bookkeeper_batch.py
git commit -m "perf: wire BookkeeperCache into analyze_all_channels

Creates one BookkeeperCache at the start of analysis and passes it
through to all channel cost calculations. This is the integration
point: 84 sequential bkpr-listaccountevents calls → 1 bkpr-listincome."
```

---

### Task 6: Replace closure cost lookup with bkpr-inspect

**Files:**
- Modify: `cl-revenue-ops.py` — `_get_closure_costs_from_bookkeeper()` (lines ~4497–4582)
- Modify: `tests/test_bookkeeper_batch.py`

The `_get_closure_costs_from_bookkeeper` function is called on channel close events (not a hot path). Replace it with `bkpr-inspect(account)` which returns `fees_paid_msat` per transaction directly, avoiding manual event scanning.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bookkeeper_batch.py`:

```python
# Import the function from the main plugin file
# We need to handle the import carefully since cl-revenue-ops.py has a hyphenated name
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cl_revenue_ops_plugin",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cl-revenue-ops.py")
)


class TestClosureCostsBkprInspect:
    """_get_closure_costs_from_bookkeeper uses bkpr-inspect."""

    def test_closure_fee_from_bkpr_inspect(self):
        """bkpr-inspect returns fees_paid_msat directly."""
        # This test validates the expected bkpr-inspect response format.
        # The actual function replacement is in cl-revenue-ops.py.
        inspect_response = {
            "txs": [
                {
                    "txid": "funding_tx_abc",
                    "blockheight": 900000,
                    "fees_paid_msat": "1500000msat",
                    "outputs": [
                        {
                            "account": "100x1x0",
                            "outnum": 0,
                            "output_value_msat": "2000000000msat",
                            "currency": "bc",
                            "output_tag": "channel_open"
                        }
                    ]
                },
                {
                    "txid": "closing_tx_def",
                    "blockheight": 943000,
                    "fees_paid_msat": "850000msat",
                    "outputs": [
                        {
                            "account": "100x1x0",
                            "outnum": 0,
                            "output_value_msat": "1998500000msat",
                            "currency": "bc",
                            "output_tag": "channel_close"
                        }
                    ]
                }
            ]
        }

        # Parse closure costs from bkpr-inspect response
        closure_fee_sats = 0
        funding_txid = None
        closing_txid = None

        for tx in inspect_response["txs"]:
            fees_msat = int(str(tx["fees_paid_msat"]).replace("msat", ""))
            has_open = any(o.get("output_tag") == "channel_open" for o in tx.get("outputs", []))
            has_close = any(o.get("output_tag") == "channel_close" for o in tx.get("outputs", []))

            if has_open:
                funding_txid = tx["txid"]
            if has_close:
                closing_txid = tx["txid"]
                closure_fee_sats += fees_msat // 1000

        assert closure_fee_sats == 850
        assert funding_txid == "funding_tx_abc"
        assert closing_txid == "closing_tx_def"
```

- [ ] **Step 2: Run test to verify it passes (this is a format validation test)**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_bookkeeper_batch.py::TestClosureCostsBkprInspect -v`
Expected: PASS (validates our parsing logic against the expected API format)

- [ ] **Step 3: Replace `_get_closure_costs_from_bookkeeper` in `cl-revenue-ops.py`**

Replace the function at lines ~4497–4582:

```python
def _get_closure_costs_from_bookkeeper(channel_id: str) -> Optional[Dict[str, Any]]:
    """
    Query bookkeeper for on-chain fees related to channel closure.

    Uses bkpr-inspect to get fees_paid_msat per transaction directly,
    avoiding raw event scanning.

    Args:
        channel_id: The channel short ID

    Returns:
        Dict with closure_fee_sats, htlc_sweep_fee_sats, funding_txid, closing_txid
        or None if bookkeeper unavailable
    """
    global safe_plugin

    if safe_plugin is None:
        return None

    try:
        result = safe_plugin.rpc.call("bkpr-inspect", {"account": channel_id})

        if not result or "txs" not in result:
            return None

        txs = result.get("txs", [])
        if not isinstance(txs, list):
            plugin.log(f"Security: Invalid txs structure from bkpr-inspect for {channel_id}", level='warn')
            return None

        closure_fee_sats = 0
        htlc_sweep_fee_sats = 0
        funding_txid = None
        closing_txid = None

        for tx in txs:
            if not isinstance(tx, dict):
                continue

            txid = tx.get("txid")
            fees_msat = parse_msat(tx.get("fees_paid_msat", 0))
            fee_sats = min(fees_msat // 1000, 50000)  # Bounds check

            outputs = tx.get("outputs", [])
            if not isinstance(outputs, list):
                continue

            tags = {o.get("output_tag", "") for o in outputs if isinstance(o, dict)}
            spend_tags = {o.get("spend_tag", "") for o in outputs if isinstance(o, dict)}
            all_tags = tags | spend_tags

            if "channel_open" in all_tags:
                funding_txid = txid

            is_close = any(
                t for t in all_tags
                if t in ("channel_close", "mutual_close", "unilateral_close")
            )
            is_sweep = any(
                t for t in all_tags
                if "htlc" in t.lower() or "sweep" in t.lower()
            )

            if is_sweep:
                htlc_sweep_fee_sats += fee_sats
            elif is_close:
                closing_txid = txid
                closure_fee_sats += fee_sats

        return {
            'closure_fee_sats': closure_fee_sats,
            'htlc_sweep_fee_sats': htlc_sweep_fee_sats,
            'funding_txid': funding_txid,
            'closing_txid': closing_txid
        }

    except Exception as e:
        plugin.log(f"Bookkeeper query failed for {channel_id}: {e}", level='debug')
        return None
```

- [ ] **Step 4: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add cl-revenue-ops.py tests/test_bookkeeper_batch.py
git commit -m "perf: replace closure cost lookup with bkpr-inspect

bkpr-inspect returns fees_paid_msat per transaction directly,
eliminating raw event scanning. Only called on channel close events
(not a hot path)."
```

---

### Task 7: Delete dead code and verify

**Files:**
- Modify: `modules/profitability_analyzer.py` — remove `_get_open_cost_from_bookkeeper`
- Modify: `tests/test_plugin_audit_regressions.py` — update if it references removed methods

After Tasks 2–4, these methods are dead code:
- `_get_open_cost_from_bookkeeper` — replaced by `_lookup_open_cost` + `BookkeeperCache`

(`_get_open_timestamp_from_bookkeeper` and `_get_rebalance_costs_from_bookkeeper` were already deleted in Tasks 3 and 4.)

- [ ] **Step 1: Verify `_get_open_cost_from_bookkeeper` is still called**

Check if `_lookup_open_cost` still calls `_get_open_cost_from_bookkeeper` as a fallback. If yes, decide whether to keep it for the non-batch code path (single-channel `analyze_channel` called without cache).

If `_get_open_cost_from_bookkeeper` is still used as a fallback in `_lookup_open_cost`, it should be kept. If `_lookup_open_cost` always receives a cache in production (via `analyze_all_channels`), the fallback exists only for direct `analyze_channel` calls from `revenue-profitability channel_id` (single channel mode). In that case, keep the legacy method but document it as single-channel-only.

- [ ] **Step 2: Search for any remaining `bkpr-listaccountevents` references in production code**

Run: `grep -rn "bkpr-listaccountevents" modules/ cl-revenue-ops.py`

Any remaining references should only be in the legacy `_get_open_cost_from_bookkeeper` (single-channel fallback). Confirm no other production code path uses it.

- [ ] **Step 3: Update `tests/test_plugin_audit_regressions.py` if needed**

Run: `grep -n "_get_rebalance_costs_from_bookkeeper\|_get_open_timestamp_from_bookkeeper\|_get_open_cost_from_bookkeeper" tests/test_plugin_audit_regressions.py`

Fix any references to deleted methods.

- [ ] **Step 4: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add -u
git commit -m "chore: remove dead bookkeeper methods, update test mocks

Cleans up methods replaced by BookkeeperCache batch fetching."
```

---

## Verification Checklist

After all tasks complete:

- [ ] `python3 -m pytest tests/ -v` — all tests pass
- [ ] `grep -rn "bkpr-listaccountevents" modules/` — no references in hot paths (only in legacy single-channel fallback if kept)
- [ ] `grep -rn "bkpr-listincome" modules/` — exactly one call site in `BookkeeperCache.__init__`
- [ ] Deploy to node and run `lightning-cli revenue-profitability` — should complete in seconds, not timeout
- [ ] Run `lightning-cli revenue-dashboard 30` — should return financial data without abort
- [ ] Check logs for `RPC timeout after 15s on bkpr-listaccountevents` — should no longer appear during normal analysis cycles
