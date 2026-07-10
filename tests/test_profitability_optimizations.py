"""
Tests for profitability analyzer performance optimizations.

Covers:
1. Open-cost write skip: record_channel_open_cost is NOT called on a second
   analysis pass when cost/opened_at/peer/capacity are unchanged.
2. Batch P&L + fee-state prefetch in analyze_all_channels: when the database
   exposes get_all_channels_full_pnl / get_all_fee_strategy_states, the
   per-channel variants are not called; without them, the per-channel
   fallback works; results are identical either way (behavior parity).
3. Bleeder batching: identify_bleeders and identify_bleeders_v2 use the
   batch P&L method when available, with per-channel fallback and identical
   verdicts.
"""

import json
import os
import sys
import time
from collections import Counter
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import (
    ChannelProfitabilityAnalyzer, ProfitabilityClass
)


# ============================================================
# Seed helpers
# ============================================================

PEER_A = "02" + "a" * 64
PEER_B = "02" + "b" * 64


def make_pnl(channel_id, window_days, contribution_sats=0,
             rebalance_cost_sats=0, direct_forwards=0, sourced_forwards=0):
    """Build a dict shaped exactly like Database.get_channel_full_pnl."""
    contribution_msat = contribution_sats * 1000
    rebalance_cost_msat = rebalance_cost_sats * 1000
    net_msat = contribution_msat - rebalance_cost_msat
    return {
        'channel_id': channel_id,
        'window_days': window_days,
        'direct_revenue_msat': contribution_msat,
        'direct_forward_count': direct_forwards,
        'sourced_volume_msat': 0,
        'sourced_fee_contribution_msat': 0,
        'sourced_forward_count': sourced_forwards,
        'total_contribution_msat': contribution_msat,
        'rebalance_cost_msat': rebalance_cost_msat,
        'rebalance_cost_sats': rebalance_cost_sats,
        'net_pnl_msat': net_msat,
        'direct_revenue_sats': contribution_sats,
        'total_contribution_sats': contribution_sats,
        'net_pnl_sats': contribution_sats - rebalance_cost_sats,
        'sourced_fee_contribution_sats': 0,
        'sourced_volume_sats': 0,
        'revenue_sats': contribution_sats,
        'forward_count': direct_forwards,
    }


def make_channel_seed(channel_id, peer_id=PEER_A, capacity=2_000_000,
                      fees_earned_sats=5000, rebalance_costs=0,
                      pnl_30d=None, pnl_7d=None, fee_state=None,
                      last_forward_age=3600):
    return {
        'peer_id': peer_id,
        'capacity': capacity,
        'funding_txid': 'ff' * 32,
        'rebalance_costs': rebalance_costs,
        'revenue': {
            'fees_earned_msat': fees_earned_sats * 1000,
            'volume_routed_msat': 1_000_000_000,
            'forward_count': 50,
            'sourced_volume_msat': 0,
            'sourced_fee_contribution_msat': 0,
            'sourced_forward_count': 10,
        },
        'pnl': {
            30: pnl_30d or make_pnl(channel_id, 30, contribution_sats=500,
                                    rebalance_cost_sats=100,
                                    direct_forwards=20),
            7: pnl_7d or make_pnl(channel_id, 7, contribution_sats=100,
                                  rebalance_cost_sats=20, direct_forwards=5),
        },
        'fee_state': fee_state,
        'last_forward': int(time.time()) - last_forward_age,
    }


class FakeDatabase:
    """Stateful fake DB with per-channel methods only (no batch P&L)."""

    def __init__(self, channels):
        self.channels = channels  # channel_id -> seed dict
        self.cost_rows = {}       # channel_id -> channel_costs row
        self.calls = Counter()

    # --- channel_costs table ---

    def seed_cost_row(self, channel_id, peer_id, open_cost_sats,
                      capacity_sats, opened_at):
        self.cost_rows[channel_id] = {
            'channel_id': channel_id,
            'peer_id': peer_id,
            'open_cost_sats': open_cost_sats,
            'capacity_sats': capacity_sats,
            'opened_at': opened_at,
        }

    def get_channel_cost(self, channel_id):
        self.calls['get_channel_cost'] += 1
        row = self.cost_rows.get(channel_id)
        return dict(row) if row else None

    def get_channel_open_cost(self, channel_id):
        self.calls['get_channel_open_cost'] += 1
        row = self.cost_rows.get(channel_id)
        return row['open_cost_sats'] if row else None

    def record_channel_open_cost(self, channel_id, peer_id, open_cost_sats,
                                 capacity_sats, timestamp=None):
        self.calls['record_channel_open_cost'] += 1
        self.seed_cost_row(channel_id, peer_id, open_cost_sats,
                           capacity_sats, timestamp or int(time.time()))

    # --- rebalance costs / success rate ---

    def get_channel_rebalance_costs(self, channel_id):
        self.calls['get_channel_rebalance_costs'] += 1
        return self.channels[channel_id].get('rebalance_costs', 0)

    def get_channel_rebalance_success_rate(self, channel_id, window_days=30):
        self.calls['get_channel_rebalance_success_rate'] += 1
        return None  # no rebalance history

    # --- P&L ---

    def get_channel_full_pnl(self, channel_id, window_days=30):
        self.calls['get_channel_full_pnl'] += 1
        return dict(self.channels[channel_id]['pnl'][window_days])

    # --- fee strategy state ---

    def get_fee_strategy_state(self, channel_id):
        self.calls['get_fee_strategy_state'] += 1
        state = self.channels[channel_id].get('fee_state')
        if state:
            return dict(state)
        # Same shape as Database default-state return
        return {'channel_id': channel_id, 'v2_state_json': '{}'}

    def get_all_fee_strategy_states(self):
        self.calls['get_all_fee_strategy_states'] += 1
        return [dict(c['fee_state']) for c in self.channels.values()
                if c.get('fee_state')]

    # --- revenue / forwards ---

    def get_all_channels_revenue_totals(self):
        self.calls['get_all_channels_revenue_totals'] += 1
        return {ch: dict(c['revenue']) for ch, c in self.channels.items()}

    def get_last_forward_time_any_direction(self, channel_id):
        self.calls['get_last_forward_time_any_direction'] += 1
        return self.channels[channel_id].get('last_forward')

    def get_diagnostic_rebalance_stats(self, channel_id, days=14):
        self.calls['get_diagnostic_rebalance_stats'] += 1
        return {'attempt_count': 0, 'last_success_time': None}


class BatchFakeDatabase(FakeDatabase):
    """Fake DB additionally exposing the batch full-P&L method."""

    def get_all_channels_full_pnl(self, window_days):
        self.calls['get_all_channels_full_pnl'] += 1
        return {
            ch: dict(c['pnl'][window_days])
            for ch, c in self.channels.items()
            if window_days in c['pnl']
        }


def make_analyzer(db, channels, estimated_open_cost=1000):
    """Build an analyzer with a fake plugin wired to the seeded channels."""
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"income_events": [], "events": []}
    plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "short_channel_id": ch,
                "total_msat": c['capacity'] * 1000,
                "peer_id": c['peer_id'],
                "funding_txid": c['funding_txid'],
                "opener": "local",
            }
            for ch, c in channels.items()
        ]
    }
    config = MagicMock()
    config.estimated_open_cost_sats = estimated_open_cost
    return ChannelProfitabilityAnalyzer(plugin, config, db)


def seed_steady_state_cost_rows(analyzer, db, channels):
    """Seed channel_costs rows matching what analysis would compute."""
    for ch, c in channels.items():
        opened_at = analyzer._get_channel_open_timestamp(ch, c['funding_txid'])
        db.seed_cost_row(ch, c['peer_id'], 2000, c['capacity'], opened_at)


# ============================================================
# 1. Open-cost write skip
# ============================================================

class TestOpenCostWriteSkip:
    """record_channel_open_cost must not fire when nothing changed."""

    def test_second_pass_performs_zero_open_cost_writes(self):
        channels = {"800000x1x0": make_channel_seed("800000x1x0")}
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)

        # First pass: no cost row exists -> a write is expected
        results = analyzer.analyze_all_channels(force=True)
        assert "800000x1x0" in results
        assert db.calls['record_channel_open_cost'] >= 1

        # Second pass with unchanged data: zero writes
        before = db.calls['record_channel_open_cost']
        results2 = analyzer.analyze_all_channels(force=True)
        assert "800000x1x0" in results2
        assert db.calls['record_channel_open_cost'] == before, (
            "second analysis pass with unchanged data must perform zero "
            "record_channel_open_cost calls"
        )

    def test_steady_state_pass_writes_nothing(self):
        """Pre-seeded matching rows -> no writes at all on first pass."""
        channels = {"800000x1x0": make_channel_seed("800000x1x0")}
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)
        seed_steady_state_cost_rows(analyzer, db, channels)

        analyzer.analyze_all_channels(force=True)
        assert db.calls['record_channel_open_cost'] == 0

    def test_stored_opened_at_is_preserved_not_clobbered(self):
        """The stored opened_at is ground truth (written at actual open
        time); the analyzer's SCID-based estimate must never overwrite
        it — a differing stored value is trusted, so no write happens."""
        channels = {"800000x1x0": make_channel_seed("800000x1x0")}
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)
        seed = channels["800000x1x0"]
        opened_at = analyzer._get_channel_open_timestamp(
            "800000x1x0", seed['funding_txid'])
        stored = opened_at - 86400
        db.seed_cost_row("800000x1x0", seed['peer_id'], 2000,
                         seed['capacity'], stored)

        analyzer.analyze_all_channels(force=True)
        assert db.calls['record_channel_open_cost'] == 0
        assert db.cost_rows["800000x1x0"]["opened_at"] == stored

    def test_implausible_stored_opened_at_is_repaired(self):
        """Rows poisoned by the pre-fix rolling now-30d estimate (stored
        opened_at wildly off the tip-anchored SCID estimate) are repaired
        with the estimate instead of being frozen forever."""
        channels = {"800000x1x0": make_channel_seed("800000x1x0")}
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)
        seed = channels["800000x1x0"]
        opened_at = analyzer._get_channel_open_timestamp(
            "800000x1x0", seed['funding_txid'])
        poisoned = opened_at + 120 * 86400  # 4 months off — implausible
        db.seed_cost_row("800000x1x0", seed['peer_id'], 2000,
                         seed['capacity'], poisoned)

        analyzer.analyze_all_channels(force=True)
        assert db.calls['record_channel_open_cost'] >= 1
        assert db.cost_rows["800000x1x0"]["opened_at"] == opened_at

    def test_write_happens_when_capacity_differs(self):
        """A capacity change (e.g. splice) triggers the corrective write."""
        channels = {"800000x1x0": make_channel_seed("800000x1x0")}
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)
        seed = channels["800000x1x0"]
        opened_at = analyzer._get_channel_open_timestamp(
            "800000x1x0", seed['funding_txid'])
        db.seed_cost_row("800000x1x0", seed['peer_id'], 2000,
                         seed['capacity'] - 500_000, opened_at)

        analyzer.analyze_all_channels(force=True)
        assert db.calls['record_channel_open_cost'] >= 1
        assert db.cost_rows["800000x1x0"]["capacity_sats"] == seed['capacity']

    def test_db_without_full_row_falls_back_to_writing(self):
        """DB layers without get_channel_cost keep the legacy write behavior."""
        channels = {"800000x1x0": make_channel_seed("800000x1x0")}
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)
        seed = channels["800000x1x0"]
        opened_at = analyzer._get_channel_open_timestamp(
            "800000x1x0", seed['funding_txid'])
        db.seed_cost_row("800000x1x0", seed['peer_id'], 2000,
                         seed['capacity'], opened_at)
        # Remove the full-row accessor -> row state unknown -> parity write
        db.get_channel_cost = None

        analyzer.analyze_all_channels(force=True)
        assert db.calls['record_channel_open_cost'] >= 1


# ============================================================
# 2. Batch prefetch in analyze_all_channels
# ============================================================

class TestAnalyzeBatching:
    def _two_channel_seeds(self):
        low_variance_state = {
            'channel_id': "800000x1x0",
            'v2_state_json': json.dumps(
                {'thompson_state': {'posterior_variance': 100}}),
        }
        return {
            "800000x1x0": make_channel_seed(
                "800000x1x0", peer_id=PEER_A, fee_state=low_variance_state),
            "800100x2x0": make_channel_seed(
                "800100x2x0", peer_id=PEER_B, fees_earned_sats=3000,
                rebalance_costs=200),
        }

    def test_batch_methods_used_when_available(self):
        channels = self._two_channel_seeds()
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)
        seed_steady_state_cost_rows(analyzer, db, channels)

        results = analyzer.analyze_all_channels(force=True)
        assert len(results) == 2
        # Per-channel variants must NOT be called
        assert db.calls['get_channel_full_pnl'] == 0
        assert db.calls['get_fee_strategy_state'] == 0
        # Batch variants are called once per run
        assert db.calls['get_all_channels_full_pnl'] == 1
        assert db.calls['get_all_fee_strategy_states'] == 1

    def test_fallback_when_batch_absent(self):
        channels = self._two_channel_seeds()
        db = FakeDatabase(channels)  # no get_all_channels_full_pnl
        analyzer = make_analyzer(db, channels)
        seed_steady_state_cost_rows(analyzer, db, channels)

        results = analyzer.analyze_all_channels(force=True)
        assert len(results) == 2
        # Per-channel P&L fallback used once per channel
        assert db.calls['get_channel_full_pnl'] == 2

    def test_parity_batch_vs_fallback(self):
        """Classifications and full to_dict output identical either way."""
        channels_a = self._two_channel_seeds()
        channels_b = self._two_channel_seeds()
        db_batch = BatchFakeDatabase(channels_a)
        db_plain = FakeDatabase(channels_b)
        analyzer_batch = make_analyzer(db_batch, channels_a)
        analyzer_plain = make_analyzer(db_plain, channels_b)
        seed_steady_state_cost_rows(analyzer_batch, db_batch, channels_a)
        seed_steady_state_cost_rows(analyzer_plain, db_plain, channels_b)

        results_batch = analyzer_batch.analyze_all_channels(force=True)
        results_plain = analyzer_plain.analyze_all_channels(force=True)

        assert set(results_batch) == set(results_plain)
        for ch in results_batch:
            assert results_batch[ch].to_dict() == results_plain[ch].to_dict()
            assert (results_batch[ch].classification
                    == results_plain[ch].classification)

    def test_fee_state_threshold_widening_parity(self):
        """Prefetched fee states widen thresholds exactly like per-channel."""
        analyzer = make_analyzer(
            BatchFakeDatabase({}), {}, estimated_open_cost=1000)
        low_var_state = {
            'channel_id': "800000x1x0",
            'v2_state_json': json.dumps(
                {'thompson_state': {'posterior_variance': 100}}),
        }
        db = analyzer.database
        db.channels = {"800000x1x0": make_channel_seed(
            "800000x1x0", fee_state=low_var_state)}

        common = dict(
            roi=0.07, net_profit=140, last_routed=int(time.time()) - 3600,
            days_open=100, channel_id="800000x1x0", peer_id=PEER_A,
            forward_count=60,
        )
        # Per-channel path (fee_states=None) -> widened -> PROFITABLE
        via_db = analyzer._classify_channel(**common)
        # Prefetched path
        via_batch = analyzer._classify_channel(
            **common, fee_states={"800000x1x0": dict(low_var_state)})
        assert via_db == via_batch == ProfitabilityClass.PROFITABLE

        # Sanity: without any fee state the same ROI is BREAK_EVEN
        via_missing = analyzer._classify_channel(
            **common, fee_states={})
        db.channels["800000x1x0"]['fee_state'] = None
        via_db_default = analyzer._classify_channel(**common)
        assert via_missing == via_db_default == ProfitabilityClass.BREAK_EVEN


# ============================================================
# 3. Bleeder batching (v1 + v2)
# ============================================================

class TestBleederBatching:
    def _bleeder_seeds(self):
        """Hard, soft, recovering-soft and healthy channels."""
        return {
            # HARD: effective cost > 2x revenue, net < -1000 (both windows)
            "800000x1x0": make_channel_seed(
                "800000x1x0", peer_id=PEER_A,
                pnl_30d=make_pnl("800000x1x0", 30, contribution_sats=100,
                                 rebalance_cost_sats=5000,
                                 direct_forwards=10),
                pnl_7d=make_pnl("800000x1x0", 7, contribution_sats=0,
                                rebalance_cost_sats=100, direct_forwards=2),
            ),
            # SOFT: 7d negative, 30d positive
            "800100x2x0": make_channel_seed(
                "800100x2x0", peer_id=PEER_B,
                pnl_30d=make_pnl("800100x2x0", 30, contribution_sats=1000,
                                 rebalance_cost_sats=200,
                                 direct_forwards=30),
                pnl_7d=make_pnl("800100x2x0", 7, contribution_sats=0,
                                rebalance_cost_sats=50, direct_forwards=3),
            ),
            # SOFT (R1 recovering): 30d negative, 7d non-negative
            "800200x3x0": make_channel_seed(
                "800200x3x0", peer_id=PEER_A,
                pnl_30d=make_pnl("800200x3x0", 30, contribution_sats=100,
                                 rebalance_cost_sats=600,
                                 direct_forwards=12),
                pnl_7d=make_pnl("800200x3x0", 7, contribution_sats=20,
                                rebalance_cost_sats=10, direct_forwards=2),
            ),
            # NONE: profitable in both windows
            "800300x4x0": make_channel_seed(
                "800300x4x0", peer_id=PEER_B,
                pnl_30d=make_pnl("800300x4x0", 30, contribution_sats=2000,
                                 rebalance_cost_sats=100,
                                 direct_forwards=40),
                pnl_7d=make_pnl("800300x4x0", 7, contribution_sats=400,
                                rebalance_cost_sats=20, direct_forwards=8),
            ),
        }

    def test_v2_uses_batch_when_available(self):
        channels = self._bleeder_seeds()
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)

        classifications = analyzer.identify_bleeders_v2(window_days=30)
        assert len(classifications) == 4
        assert db.calls['get_channel_full_pnl'] == 0
        # One batch call per window (30d + 7d)
        assert db.calls['get_all_channels_full_pnl'] == 2

    def test_v2_fallback_without_batch(self):
        channels = self._bleeder_seeds()
        db = FakeDatabase(channels)
        analyzer = make_analyzer(db, channels)

        classifications = analyzer.identify_bleeders_v2(window_days=30)
        assert len(classifications) == 4
        # Two per-channel calls per channel (30d + 7d)
        assert db.calls['get_channel_full_pnl'] == 8

    def test_v2_verdict_parity_batch_vs_fallback(self):
        channels_a = self._bleeder_seeds()
        channels_b = self._bleeder_seeds()
        analyzer_batch = make_analyzer(BatchFakeDatabase(channels_a),
                                       channels_a)
        analyzer_plain = make_analyzer(FakeDatabase(channels_b), channels_b)

        batch = {c.channel_id: c.to_dict()
                 for c in analyzer_batch.identify_bleeders_v2(window_days=30)}
        plain = {c.channel_id: c.to_dict()
                 for c in analyzer_plain.identify_bleeders_v2(window_days=30)}
        assert batch == plain

        # And the expected verdicts hold
        assert batch["800000x1x0"]["classification"] == "hard"
        assert batch["800100x2x0"]["classification"] == "soft"
        assert batch["800200x3x0"]["classification"] == "soft"
        assert batch["800300x4x0"]["classification"] == "none"

    def test_v1_uses_batch_when_available(self):
        channels = self._bleeder_seeds()
        db = BatchFakeDatabase(channels)
        analyzer = make_analyzer(db, channels)

        bleeders = analyzer.identify_bleeders(window_days=30)
        assert db.calls['get_channel_full_pnl'] == 0
        assert db.calls['get_all_channels_full_pnl'] == 1
        # Hard and recovering channels have net < 0 -> reported
        ids = {b['channel_id'] for b in bleeders}
        assert ids == {"800000x1x0", "800200x3x0"}

    def test_v1_parity_batch_vs_fallback(self):
        channels_a = self._bleeder_seeds()
        channels_b = self._bleeder_seeds()
        analyzer_batch = make_analyzer(BatchFakeDatabase(channels_a),
                                       channels_a)
        analyzer_plain = make_analyzer(FakeDatabase(channels_b), channels_b)

        batch = analyzer_batch.identify_bleeders(window_days=30)
        plain = analyzer_plain.identify_bleeders(window_days=30)
        assert batch == plain
        assert analyzer_plain.database.calls['get_channel_full_pnl'] == 4

    def test_v1_falls_back_for_channel_missing_from_batch(self):
        """A channel absent from the batch dict uses the per-channel query."""
        channels = self._bleeder_seeds()
        db = BatchFakeDatabase(channels)

        full = db.get_all_channels_full_pnl

        def partial_batch(window_days):
            result = full(window_days)
            result.pop("800000x1x0", None)  # simulate missing row
            return result

        db.get_all_channels_full_pnl = partial_batch
        analyzer = make_analyzer(db, channels)

        bleeders = analyzer.identify_bleeders(window_days=30)
        ids = {b['channel_id'] for b in bleeders}
        assert ids == {"800000x1x0", "800200x3x0"}
        # Exactly one per-channel fallback (the missing channel)
        assert db.calls['get_channel_full_pnl'] == 1
