"""
Tests for flow analysis performance optimizations.

Covers:
1. TTL result cache on analyze_all_channels (cache hits skip RPC/DB/Kalman work,
   force/expiry trigger genuine refresh, stampede lock returns cache).
2. 24-hour raw-forward window for Kalman observation (was flow_window_days*24).
3. Batched DB writes: Kalman state saves and temporal profile updates use
   batch DB methods when available, falling back to per-channel methods.
4. Parity: analysis results identical with batch-capable and per-channel DBs.
"""

import json
import time
from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from modules.flow_analysis import FlowAnalyzer


PEER_A = "02" + "a" * 64
PEER_B = "02" + "b" * 64


def make_channel(scid, peer_id=PEER_A, spendable=5_000_000_000,
                 receivable=5_000_000_000):
    return {
        "short_channel_id": scid,
        "peer_id": peer_id,
        "capacity_msat": spendable + receivable,
        "spendable_msat": spendable,
        "receivable_msat": receivable,
        "state": "CHANNELD_NORMAL",
        "htlcs": [],
        "active_htlcs": 0,
        "max_htlcs": 483,
    }


def zero_histogram():
    return [{"out_sats": 0, "in_sats": 0, "count": 0} for _ in range(24)]


class FakeDatabase:
    """In-memory fake implementing the per-channel DB API (NO batch methods)."""

    def __init__(self, daily_buckets=None, net_flows=None, histograms=None):
        self.daily_buckets = daily_buckets or {}
        self.net_flows = net_flows or {}
        self.histograms = histograms or {}
        self.calls = defaultdict(int)
        self.kalman_states = {}
        self.channel_states = {}
        self.temporal_profiles = {}
        self.last_net_flow_window = None

    def kalman_purge_needed(self):
        return False

    def get_kalman_state(self, channel_id):
        self.calls["get_kalman_state"] += 1
        return self.kalman_states.get(channel_id)

    def save_kalman_state(self, channel_id, state):
        self.calls["save_kalman_state"] += 1
        self.kalman_states[channel_id] = dict(state, velocity_unit="per_hour")

    def get_daily_flow_buckets(self, window_days=7, channel_id=None):
        self.calls["get_daily_flow_buckets"] += 1
        if channel_id is not None:
            return {channel_id: self.daily_buckets.get(channel_id, [])}
        return self.daily_buckets

    def get_continuous_net_flow_all(self, window_hours=168):
        self.calls["get_continuous_net_flow_all"] += 1
        self.last_net_flow_window = window_hours
        return self.net_flows

    def get_continuous_net_flow_channel(self, channel_id, window_hours=168):
        self.calls["get_continuous_net_flow_channel"] += 1
        self.last_net_flow_window = window_hours
        return self.net_flows.get(channel_id, [])

    def get_all_channel_states(self):
        self.calls["get_all_channel_states"] += 1
        return [dict(v) for v in self.channel_states.values()]

    def get_channel_state(self, channel_id):
        self.calls["get_channel_state"] += 1
        return self.channel_states.get(channel_id)

    def update_channel_states_batch(self, rows):
        self.calls["update_channel_states_batch"] += 1
        for row in rows:
            self.channel_states[row["channel_id"]] = dict(
                row, updated_at=int(time.time()))

    def update_channel_state(self, **kwargs):
        self.calls["update_channel_state"] += 1
        self.channel_states[kwargs["channel_id"]] = dict(
            kwargs, updated_at=int(time.time()))

    def remove_closed_channel_data(self, channel_id, peer_id=None):
        self.calls["remove_closed_channel_data"] += 1
        self.channel_states.pop(channel_id, None)

    def get_hourly_forward_histogram(self, channel_id, window_days=7):
        self.calls["get_hourly_forward_histogram"] += 1
        return self.histograms.get(channel_id, zero_histogram())

    def load_temporal_profile(self, channel_id):
        self.calls["load_temporal_profile"] += 1
        return self.temporal_profiles.get(channel_id)

    def save_temporal_profile(self, channel_id, profile_json):
        self.calls["save_temporal_profile"] += 1
        self.temporal_profiles[channel_id] = profile_json

    def get_fee_strategy_state(self, channel_id):
        self.calls["get_fee_strategy_state"] += 1
        return {}

    def get_all_fee_strategy_states(self):
        self.calls["get_all_fee_strategy_states"] += 1
        return []


class FakeDatabaseWithBatch(FakeDatabase):
    """Fake DB that ALSO provides the batch methods another agent is landing."""

    def save_kalman_states_batch(self, rows):
        self.calls["save_kalman_states_batch"] += 1
        self.last_kalman_batch = rows
        for row in rows:
            self.kalman_states[row["channel_id"]] = dict(
                row["state"], velocity_unit="per_hour")

    def get_hourly_forward_histograms_all(self, window_days=7):
        self.calls["get_hourly_forward_histograms_all"] += 1
        return dict(self.histograms)

    def save_temporal_profiles_batch(self, rows):
        self.calls["save_temporal_profiles_batch"] += 1
        self.last_profile_batch = rows
        for row in rows:
            self.temporal_profiles[row["channel_id"]] = row["profile_json"]


def make_analyzer(database, channels=None, flow_window_days=7, flow_interval=3600):
    plugin = MagicMock()
    config = MagicMock()
    config.source_threshold = 0.5
    config.sink_threshold = -0.5
    config.flow_window_days = flow_window_days
    config.flow_interval = flow_interval
    config.htlc_congestion_threshold = 0.8
    analyzer = FlowAnalyzer(plugin, config, database)
    if channels is not None:
        analyzer._get_channels = MagicMock(return_value=channels)
    return analyzer


def seeded_daily_buckets(scid, out_per_day=200_000, in_per_day=50_000, days=7):
    now = int(time.time())
    return {
        scid: [
            {"in": in_per_day, "out": out_per_day, "count": 12,
             "last_ts": now - age * 86400}
            for age in range(days)
        ]
    }


def seeded_net_flows(scid, net_msat=1_000_000, count=10):
    now = int(time.time())
    return {
        scid: [
            {"timestamp": now - i * 3600, "net_msat": net_msat}
            for i in range(count)
        ]
    }


# =============================================================================
# 1. TTL result cache
# =============================================================================

class TestAnalyzeAllChannelsCache:

    def _fresh(self, db_cls=FakeDatabase):
        scid = "100x1x0"
        db = db_cls(
            daily_buckets=seeded_daily_buckets(scid),
            net_flows=seeded_net_flows(scid),
        )
        analyzer = make_analyzer(db, channels=[make_channel(scid)])
        return analyzer, db, scid

    def test_second_call_within_ttl_is_cache_hit(self):
        analyzer, db, scid = self._fresh()

        first = analyzer.analyze_all_channels()
        assert scid in first
        assert db.calls["get_continuous_net_flow_all"] == 1
        assert analyzer._get_channels.call_count == 1

        second = analyzer.analyze_all_channels()
        assert second is first  # cached object returned
        # No additional RPC fetch or bulk DB reads
        assert analyzer._get_channels.call_count == 1
        assert db.calls["get_continuous_net_flow_all"] == 1
        assert db.calls["get_daily_flow_buckets"] == 1

    def test_cache_hit_skips_db_writes_and_kalman_updates(self):
        analyzer, db, scid = self._fresh()

        analyzer.analyze_all_channels()
        kalman_after_first = dict(db.kalman_states.get(scid) or {})
        obs_after_first = kalman_after_first.get("observation_count")
        writes_after_first = {
            k: db.calls[k] for k in (
                "save_kalman_state", "update_channel_states_batch",
                "save_temporal_profile")
        }

        analyzer.analyze_all_channels()
        # No new writes of any kind on cache hit
        for key, count in writes_after_first.items():
            assert db.calls[key] == count, f"{key} written on cache hit"
        # Kalman state untouched (no observation consumed twice)
        assert db.kalman_states[scid].get("observation_count") == obs_after_first

    def test_force_bypasses_cache(self):
        analyzer, db, scid = self._fresh()

        first = analyzer.analyze_all_channels()
        second = analyzer.analyze_all_channels(force=True)
        assert second is not first
        assert analyzer._get_channels.call_count == 2
        assert db.calls["get_continuous_net_flow_all"] == 2

    def test_ttl_expiry_refreshes(self):
        analyzer, db, scid = self._fresh()

        analyzer.analyze_all_channels()
        # Simulate cache older than TTL
        analyzer._flow_cache_timestamp = int(time.time()) - (
            analyzer._flow_cache_ttl + 1)
        analyzer.analyze_all_channels()
        assert analyzer._get_channels.call_count == 2
        assert db.calls["get_continuous_net_flow_all"] == 2

    def test_ttl_is_configurable_attribute(self):
        analyzer, db, scid = self._fresh()
        # >= the hourly loop cadence so the 900s rebalance cycle can't force
        # extra Kalman-consuming refreshes (was 300s).
        assert analyzer._flow_cache_ttl == 1800
        analyzer._flow_cache_ttl = 0
        analyzer.analyze_all_channels()
        time.sleep(1.1)
        analyzer.analyze_all_channels()
        assert analyzer._get_channels.call_count == 2

    def test_ttl_scales_below_short_configured_flow_interval(self):
        scid = "100x1x0"
        db = FakeDatabase(
            daily_buckets=seeded_daily_buckets(scid),
            net_flows=seeded_net_flows(scid),
        )

        analyzer = make_analyzer(
            db, channels=[make_channel(scid)], flow_interval=60
        )

        assert analyzer._flow_cache_ttl == 30

    def test_concurrent_caller_gets_cache_not_duplicate_run(self):
        analyzer, db, scid = self._fresh()
        first = analyzer.analyze_all_channels()

        # Simulate another thread holding the analysis lock
        analyzer._analysis_lock.acquire()
        try:
            result = analyzer.analyze_all_channels(force=True)
        finally:
            analyzer._analysis_lock.release()
        assert result is first  # served from cache, no second run
        assert analyzer._get_channels.call_count == 1

    def test_error_restores_timestamp_for_retry(self):
        analyzer, db, scid = self._fresh()
        analyzer._get_channels.side_effect = RuntimeError("rpc down")
        with pytest.raises(RuntimeError):
            analyzer.analyze_all_channels()
        # Timestamp restored so a later call retries instead of caching failure
        assert analyzer._flow_cache_timestamp == 0
        analyzer._get_channels.side_effect = None
        results = analyzer.analyze_all_channels()
        assert scid in results


# =============================================================================
# 2. 24-hour raw-forward window
# =============================================================================

class TestContinuousNetFlowWindow:

    def test_analyze_all_uses_24h_window(self):
        scid = "100x1x0"
        db = FakeDatabase(daily_buckets=seeded_daily_buckets(scid),
                          net_flows=seeded_net_flows(scid))
        analyzer = make_analyzer(db, channels=[make_channel(scid)],
                                 flow_window_days=7)
        analyzer.analyze_all_channels()
        assert db.last_net_flow_window == 24

    def test_analyze_channel_uses_24h_window(self):
        scid = "100x1x0"
        db = FakeDatabase(daily_buckets=seeded_daily_buckets(scid),
                          net_flows=seeded_net_flows(scid))
        analyzer = make_analyzer(db, flow_window_days=7)
        analyzer._get_channel = MagicMock(return_value=make_channel(scid))
        analyzer.analyze_channel(scid)
        assert db.last_net_flow_window == 24


# =============================================================================
# 3a. Batched Kalman state saves
# =============================================================================

class TestKalmanBatchSaves:

    def test_batch_method_used_once_when_available(self):
        scids = ["100x1x0", "101x1x0", "102x1x0"]
        buckets = {}
        flows = {}
        for s in scids:
            buckets.update(seeded_daily_buckets(s))
            flows.update(seeded_net_flows(s))
        db = FakeDatabaseWithBatch(daily_buckets=buckets, net_flows=flows)
        analyzer = make_analyzer(db, channels=[make_channel(s) for s in scids])

        analyzer.analyze_all_channels()

        assert db.calls["save_kalman_states_batch"] == 1
        assert db.calls["save_kalman_state"] == 0
        assert len(db.last_kalman_batch) == len(scids)
        batch_ids = {row["channel_id"] for row in db.last_kalman_batch}
        assert batch_ids == set(scids)
        # Row dicts carry the exact kwargs of save_kalman_state
        for row in db.last_kalman_batch:
            assert set(row.keys()) == {"channel_id", "state"}
            assert "flow_ratio" in row["state"]
            assert "observation_count" in row["state"]

    def test_fallback_per_channel_when_batch_absent(self):
        scids = ["100x1x0", "101x1x0"]
        buckets = {}
        flows = {}
        for s in scids:
            buckets.update(seeded_daily_buckets(s))
            flows.update(seeded_net_flows(s))
        db = FakeDatabase(daily_buckets=buckets, net_flows=flows)
        analyzer = make_analyzer(db, channels=[make_channel(s) for s in scids])

        analyzer.analyze_all_channels()

        assert db.calls["save_kalman_state"] == len(scids)
        for s in scids:
            assert s in db.kalman_states

    def test_predict_only_state_still_persisted_batched(self):
        # Channel with no raw flow entries -> predict-only Kalman path.
        # Covariance and last_update still change, so the state must persist.
        scid = "100x1x0"
        db = FakeDatabaseWithBatch()
        analyzer = make_analyzer(db, channels=[make_channel(scid)])

        analyzer.analyze_all_channels()

        assert db.calls["save_kalman_states_batch"] == 1
        assert db.calls["save_kalman_state"] == 0
        assert {r["channel_id"] for r in db.last_kalman_batch} == {scid}

    def test_single_channel_path_saves_immediately(self):
        scid = "100x1x0"
        db = FakeDatabaseWithBatch(daily_buckets=seeded_daily_buckets(scid),
                                   net_flows=seeded_net_flows(scid))
        analyzer = make_analyzer(db)
        analyzer._get_channel = MagicMock(return_value=make_channel(scid))

        analyzer.analyze_channel(scid)

        # Single-channel analysis is not a bulk cycle: per-channel save is fine
        assert db.calls["save_kalman_state"] == 1
        assert db.calls["save_kalman_states_batch"] == 0


# =============================================================================
# 3b. Temporal profile batching
# =============================================================================

class TestTemporalProfileBatching:

    def _seeded(self, db_cls, scids):
        buckets = {}
        flows = {}
        hists = {}
        for s in scids:
            buckets.update(seeded_daily_buckets(s))
            flows.update(seeded_net_flows(s))
            hist = zero_histogram()
            hist[3] = {"out_sats": 5000, "in_sats": 1000, "count": 4}
            hists[s] = hist
        db = db_cls(daily_buckets=buckets, net_flows=flows, histograms=hists)
        analyzer = make_analyzer(db, channels=[make_channel(s) for s in scids])
        return analyzer, db

    def test_batch_methods_used_when_available(self):
        scids = ["100x1x0", "101x1x0", "102x1x0"]
        analyzer, db = self._seeded(FakeDatabaseWithBatch, scids)

        analyzer.analyze_all_channels()

        # Histograms fetched in one bulk call, not per channel
        assert db.calls["get_hourly_forward_histograms_all"] == 1
        assert db.calls["get_hourly_forward_histogram"] == 0
        # Fee strategy states prefetched once, not per channel
        assert db.calls["get_all_fee_strategy_states"] == 1
        # Profiles written in one batch, not per channel
        assert db.calls["save_temporal_profiles_batch"] == 1
        assert db.calls["save_temporal_profile"] == 0
        assert len(db.last_profile_batch) == len(scids)
        for row in db.last_profile_batch:
            assert set(row.keys()) == {"channel_id", "profile_json"}
            profile = json.loads(row["profile_json"])
            assert len(profile["hourly_out"]) == 24

    def test_fallback_per_channel_when_batch_absent(self):
        scids = ["100x1x0", "101x1x0"]
        analyzer, db = self._seeded(FakeDatabase, scids)

        analyzer.analyze_all_channels()

        assert db.calls["get_hourly_forward_histogram"] == len(scids)
        assert db.calls["save_temporal_profile"] == len(scids)
        for s in scids:
            assert s in db.temporal_profiles
            profile = json.loads(db.temporal_profiles[s])
            assert profile["hourly_out"][3] > 0

    def test_profile_content_identical_batch_vs_fallback(self):
        scids = ["100x1x0", "101x1x0"]
        analyzer_batch, db_batch = self._seeded(FakeDatabaseWithBatch, scids)
        analyzer_fall, db_fall = self._seeded(FakeDatabase, scids)

        analyzer_batch.analyze_all_channels()
        analyzer_fall.analyze_all_channels()

        for s in scids:
            p_batch = json.loads(db_batch.temporal_profiles[s])
            p_fall = json.loads(db_fall.temporal_profiles[s])
            # last_updated is a timestamp; compare everything else
            p_batch.pop("last_updated")
            p_fall.pop("last_updated")
            assert p_batch == p_fall


# =============================================================================
# 4. Result parity: batch-capable vs per-channel DB
# =============================================================================

class TestAnalysisParity:

    def test_metrics_identical_with_and_without_batch_db(self):
        scids = ["100x1x0", "101x1x0"]
        def build(db_cls):
            buckets = {}
            flows = {}
            for s in scids:
                buckets.update(seeded_daily_buckets(s))
                flows.update(seeded_net_flows(s))
            db = db_cls(daily_buckets=buckets, net_flows=flows)
            return make_analyzer(db, channels=[make_channel(s) for s in scids])

        results_batch = build(FakeDatabaseWithBatch).analyze_all_channels()
        results_fall = build(FakeDatabase).analyze_all_channels()

        assert set(results_batch) == set(results_fall) == set(scids)
        for s in scids:
            assert results_batch[s].to_dict() == results_fall[s].to_dict()

    def test_cached_result_equals_fresh_result(self):
        scid = "100x1x0"
        db = FakeDatabase(daily_buckets=seeded_daily_buckets(scid),
                          net_flows=seeded_net_flows(scid))
        analyzer = make_analyzer(db, channels=[make_channel(scid)])
        fresh = analyzer.analyze_all_channels()
        cached = analyzer.analyze_all_channels()
        assert cached[scid].to_dict() == fresh[scid].to_dict()
