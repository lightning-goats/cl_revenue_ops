"""
Tests for fee-cycle performance optimizations.

Covers:
1. Batched fee-strategy persistence: rows are deferred during
   _adjust_all_fees_inner and flushed once at cycle end via
   update_fee_strategy_states_batch (per-row fallback when absent).
   Saves made OUTSIDE a cycle (manual set_channel_fee) persist immediately.
2. Duplicate-query removal: get_forward_count_since runs once per channel
   per cycle, get_channel_state is not re-queried (the in-hand
   channel_states row is reused).
3. _build_merged_fee_strategy_row skips the full-row DB re-read when both
   in-memory states are warm (shared-field memo) without changing the
   persisted row contents.
4. _prune_stale_states prefers the cheap id-only query.
5. Neighbor-fee helpers accept a cfg pass-through (no per-call snapshot)
   and the gossip cache TTL covers ~2 fee cycles; cached entries are
   trimmed to consumed fields.
6. The profitability cache warm-up runs BEFORE _state_lock is acquired.
"""

import json
import threading
import time
import sys
import os
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import FeeController
from modules.config import Config


PEER_IDS = ["02" + c * 64 for c in "abc"]
CHANNEL_IDS = ["100x1x0", "100x2x0", "100x3x0"]


def _make_config_snapshot(**overrides):
    defaults = {
        "paused": False,
        "min_fee_ppm": 10,
        "max_fee_ppm": 5000,
        "fee_interval": 1800,
        "fee_profile": "active",
        "enable_vegas_reflex": False,
        "enable_dynamic_htlcmax": False,
        "market_fee_mode": "undercut",
        "neighbor_median_min_competitors": 3,
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 1,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "inbound_fee_estimate_ppm": 200,
        "thompson_prior_std_fee": 200.0,
        "routing_intelligence_enabled": False,
    }
    defaults.update(overrides)
    snap = MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _setup_db(mock_database, now=None):
    """Configure the MagicMock database for a full cycle run."""
    now = now or int(time.time())
    mock_database.get_channel_probe.return_value = None
    mock_database.get_last_rebalance_cost.return_value = None
    mock_database.get_volume_since.return_value = 50_000
    mock_database.get_forward_count_since.return_value = 10
    mock_database.get_peer_uptime_percent.return_value = 99.5
    mock_database.get_fee_strategy_state.return_value = {
        "last_revenue_rate": 5.0,
        "last_fee_ppm": 150,
        "trend_direction": 1,
        "step_ppm": 50,
        "last_update": now - 7200,
        "consecutive_same_direction": 0,
        "is_sleeping": 0,
        "sleep_until": 0,
        "stable_cycles": 0,
        "forward_count_since_update": 10,
        "last_volume_sats": 50_000,
        "v2_state_json": None,
    }
    mock_database.get_last_forward_time.return_value = now - 1800
    mock_database.get_failure_count.return_value = (0, 0)
    mock_database.get_channel_cost_history.return_value = []
    mock_database.get_channel_rebalance_success_rate.return_value = None
    mock_database.get_channel_age.return_value = 30
    mock_database.get_historical_inbound_fee_ppm.return_value = None
    mock_database.get_peer_latency_stats.return_value = {
        "avg": 0.0, "std": 0.0, "count": 0,
    }
    mock_database.get_all_channel_states.return_value = [
        {
            "channel_id": cid,
            "peer_id": pid,
            "state": "balanced",
            "flow_ratio": 0.5,
            "sats_in": 100_000,
            "sats_out": 100_000,
            "capacity": 2_000_000,
            "updated_at": now,
            "kalman_flow_ratio": 0.3,
            "kalman_velocity": 0.01,
        }
        for cid, pid in zip(CHANNEL_IDS, PEER_IDS)
    ]


def _make_cycle_fc(mock_plugin, mock_database):
    """Build a FeeController wired up for an end-to-end adjust_all_fees run."""
    config = MagicMock(spec=Config)
    fc = FeeController(mock_plugin, config, mock_database)
    cfg = _make_config_snapshot()
    fc.config.snapshot.return_value = cfg
    fc.config.dry_run = True  # set_channel_fee succeeds without RPC

    _setup_db(mock_database)

    channels_info = {
        cid: {
            "channel_id": cid,
            "short_channel_id": cid,
            "peer_id": pid,
            "capacity": 2_000_000,
            "spendable_msat": 1_000_000_000,
            "receivable_msat": 1_000_000_000,
            "fee_base_msat": 0,
            "fee_proportional_millionths": 150,
            "htlc_minimum_msat": 0,
            "htlc_min_msat": 0,
            "htlc_maximum_msat": 0,
            "htlc_max_msat": 0,
            "opener": "local",
        }
        for cid, pid in zip(CHANNEL_IDS, PEER_IDS)
    }
    fc._get_channels_info = lambda: channels_info
    fc._get_dynamic_chain_costs = lambda: None
    mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}
    return fc, cfg


# =============================================================================
# 1. Batched persistence
# =============================================================================

class TestCycleBatchPersistence:

    def test_cycle_flushes_once_via_batch_method(self, mock_plugin, mock_database):
        """A full cycle issues exactly ONE batch write and ZERO per-row writes."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)

        adjustments = fc.adjust_all_fees()

        assert mock_database.update_fee_strategy_states_batch.call_count == 1, \
            "Cycle should flush deferred rows in a single batch"
        assert mock_database.update_fee_strategy_state.call_count == 0, \
            "No per-row writes should happen during a batched cycle"

        rows = mock_database.update_fee_strategy_states_batch.call_args[0][0]
        row_ids = {row["channel_id"] for row in rows}
        assert row_ids == set(CHANNEL_IDS), \
            "Every touched channel must appear exactly once in the batch"

        # Rows carry the exact update_fee_strategy_state kwargs
        expected_keys = {
            "channel_id", "last_revenue_rate", "last_fee_ppm", "trend_direction",
            "step_ppm", "consecutive_same_direction", "last_broadcast_fee_ppm",
            "last_state", "is_sleeping", "sleep_until", "stable_cycles",
            "forward_count_since_update", "last_volume_sats", "v2_state_json",
            "last_update",
        }
        for row in rows:
            assert set(row.keys()) == expected_keys

        # State was actually processed (sanity that the cycle ran)
        assert isinstance(adjustments, list)
        assert fc._pending_fee_strategy_rows == {}
        assert fc._cycle_batch_active is False

    def test_cycle_write_count_reduced_vs_per_exit_saves(self, mock_plugin, mock_database):
        """Fallback path: one merged row per channel instead of ~2+ full-row
        writes per channel (the old behavior wrote cycle AND fee state on
        every exit path)."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        del mock_database.update_fee_strategy_states_batch  # method absent

        fc.adjust_all_fees()

        assert mock_database.update_fee_strategy_state.call_count == len(CHANNEL_IDS), \
            "Fallback flush should write exactly one merged row per channel"

    def test_batch_failure_falls_back_to_per_row_writes(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        mock_database.update_fee_strategy_states_batch.side_effect = RuntimeError("locked")

        fc.adjust_all_fees()

        assert mock_database.update_fee_strategy_state.call_count == len(CHANNEL_IDS), \
            "Batch failure must fall back to durable per-row writes"

    def test_manual_set_channel_fee_persists_immediately(self, mock_plugin, mock_database):
        """set_channel_fee OUTSIDE a cycle must not defer persistence."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        fc.config.dry_run = False
        fc.data_service = MagicMock()
        fc.data_service.set_channel.return_value = {}

        channel_info = {
            "channel_id": CHANNEL_IDS[0],
            "short_channel_id": CHANNEL_IDS[0],
            "peer_id": PEER_IDS[0],
            "fee_proportional_millionths": 150,
            "capacity": 2_000_000,
        }
        result = fc.set_channel_fee(
            CHANNEL_IDS[0], 500, reason="manual", manual=True,
            channel_info=channel_info,
        )

        assert result["success"] is True
        assert mock_database.update_fee_strategy_state.call_count >= 1, \
            "Manual fee changes must persist immediately"
        assert fc._pending_fee_strategy_rows == {}
        assert mock_database.update_fee_strategy_states_batch.call_count == 0

    def test_sleep_entry_round_trips_through_batch_row(self, mock_plugin, mock_database):
        """A channel that ENTERS sleep during a batched cycle must have
        is_sleeping=1 in the flushed batch row (restart-safe)."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)

        # Warm states, then stage every channel one stable cycle from sleep
        # with a revenue rate that exactly matches what the next cycle will
        # observe (50_000 sats * 150 ppm over 2h = 3.75 sats/hr).
        fc.adjust_all_fees()
        mock_database.update_fee_strategy_states_batch.reset_mock()

        now = int(time.time())
        for cid in CHANNEL_IDS:
            ts = fc._channel_fee_states[cid]
            ts.is_sleeping = False
            ts.sleep_until = 0
            ts.stable_cycles = fc.STABLE_CYCLES_REQUIRED - 1
            ts.last_revenue_rate = 3.75
            ts.last_update = now - 7200
            cyc = fc._cycle_states[cid]
            cyc.is_sleeping = False
            cyc.sleep_until = 0
            cyc.last_update = now - 7200

        fc.adjust_all_fees()

        assert mock_database.update_fee_strategy_states_batch.call_count == 1
        rows = {
            row["channel_id"]: row
            for row in mock_database.update_fee_strategy_states_batch.call_args[0][0]
        }
        for cid in CHANNEL_IDS:
            assert fc._channel_fee_states[cid].is_sleeping is True, \
                "Test premise: channel should have entered sleep"
            assert rows[cid]["is_sleeping"] == 1, \
                "Batched row lost is_sleeping — restart would wake the channel"
            assert rows[cid]["sleep_until"] > now


# =============================================================================
# 2. Duplicate query removal
# =============================================================================

class TestDuplicateQueryRemoval:

    def test_forward_count_queried_once_per_channel(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)

        fc.adjust_all_fees()

        assert mock_database.get_forward_count_since.call_count == len(CHANNEL_IDS), \
            (f"get_forward_count_since should run exactly once per channel, got "
             f"{mock_database.get_forward_count_since.call_count} calls for "
             f"{len(CHANNEL_IDS)} channels")

    def test_channel_state_not_requeried_per_channel(self, mock_plugin, mock_database):
        """The cycle already holds each row from get_all_channel_states —
        get_channel_state must not be called again (was 2x per channel)."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)

        fc.adjust_all_fees()

        assert mock_database.get_channel_state.call_count == 0

    def test_warm_second_cycle_reads_no_full_rows(self, mock_plugin, mock_database):
        """After one cycle the in-memory states are warm; a second cycle must
        not re-read fee_strategy_state rows at all."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)

        fc.adjust_all_fees()
        mock_database.get_fee_strategy_state.reset_mock()

        fc.adjust_all_fees()

        assert mock_database.get_fee_strategy_state.call_count == 0, \
            "Warm cycle must not re-read full fee_strategy_state rows"


# =============================================================================
# 3. Merged-row memo correctness
# =============================================================================

class TestMergedRowMemo:

    def _warm_states(self, fc, cid, peer_id):
        ts = fc._get_channel_fee_state(cid, peer_id)
        cyc = fc._get_cycle_state(cid)
        return ts, cyc

    def test_save_with_warm_states_skips_db_reread(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        cid, peer_id = CHANNEL_IDS[0], PEER_IDS[0]
        ts, cyc = self._warm_states(fc, cid, peer_id)

        mock_database.get_fee_strategy_state.reset_mock()
        fc._save_cycle_state(cid, cyc)
        fc._save_channel_fee_state(cid, ts)

        assert mock_database.get_fee_strategy_state.call_count == 0, \
            "Warm saves must not re-read the persisted row"
        # Out-of-cycle saves still persist immediately
        assert mock_database.update_fee_strategy_state.call_count == 2

    def test_memoized_save_row_identical_to_db_read_path(self, mock_plugin, mock_database):
        """The memo path must produce byte-identical rows to the re-read path."""
        rows = {}
        for clear_memo in (False, True):
            db = MagicMock()
            _setup_db(db)
            plugin = MagicMock()
            fc, _ = _make_cycle_fc(plugin, db)
            cid, peer_id = CHANNEL_IDS[0], PEER_IDS[0]
            ts, cyc = self._warm_states(fc, cid, peer_id)
            if clear_memo:
                fc._persisted_shared_fields.clear()  # force the DB re-read path
            fc._save_cycle_state(cid, cyc)
            kwargs = db.update_fee_strategy_state.call_args.kwargs
            rows[clear_memo] = {
                k: (json.loads(v) if k == "v2_state_json" else v)
                for k, v in kwargs.items()
            }

        assert rows[False] == rows[True], \
            "Memoized shared fields changed the persisted row contents"

    def test_gossip_refresh_timestamp_survives_warm_saves(self, mock_plugin, mock_database):
        """A shared field written by one save must survive a later save of the
        counterpart state that doesn't touch it."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        cid, peer_id = CHANNEL_IDS[0], PEER_IDS[0]
        ts, cyc = self._warm_states(fc, cid, peer_id)

        refresh_ts = int(time.time()) - 11
        ts.last_gossip_refresh = refresh_ts
        fc._save_channel_fee_state(cid, ts)

        # Now save the cycle state without touching last_gossip_refresh
        fc._save_cycle_state(cid, cyc)

        kwargs = mock_database.update_fee_strategy_state.call_args.kwargs
        v2 = json.loads(kwargs["v2_state_json"])
        assert v2["last_gossip_refresh"] == refresh_ts, \
            "Shared field lost across warm saves (memo out of sync)"


# =============================================================================
# 4. Prune uses id-only query
# =============================================================================

class TestPruneIdOnlyQuery:

    def test_prune_prefers_channel_id_query(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        mock_database.get_all_fee_strategy_channel_ids.return_value = [
            CHANNEL_IDS[0], "999x9x9",
        ]

        fc._prune_stale_states({CHANNEL_IDS[0]})

        mock_database.get_all_fee_strategy_states.assert_not_called()
        mock_database.reset_fee_strategy_state.assert_called_once_with("999x9x9")

    def test_prune_falls_back_to_full_rows_when_method_absent(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        del mock_database.get_all_fee_strategy_channel_ids
        mock_database.get_all_fee_strategy_states.return_value = [
            {"channel_id": CHANNEL_IDS[0]},
            {"channel_id": "999x9x9"},
        ]

        fc._prune_stale_states({CHANNEL_IDS[0]})

        mock_database.reset_fee_strategy_state.assert_called_once_with("999x9x9")


# =============================================================================
# 5. Neighbor-fee helpers: cfg pass-through + TTL + trimmed cache
# =============================================================================

class TestNeighborHelpers:

    def _make_fc_with_gossip(self, mock_plugin, mock_database, channels):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node_id"}
        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database)
        fc.data_service = MagicMock()
        fc.data_service.get_channels.return_value = {"channels": channels}
        fc.data_service.get_node_id.return_value = "02our_node_id"
        return fc

    def _gossip_channels(self, n=5):
        return [
            {
                "source": f"02node{i}",
                "active": True,
                "fee_per_millionth": 100 + i * 50,
                "satoshis": 1_000_000,
                "last_update": int(time.time()) - 60,
                "features": "8a", "channel_flags": 1, "htlc_maximum_msat": 1,
            }
            for i in range(n)
        ]

    def test_helpers_do_not_snapshot_when_cfg_passed(self, mock_plugin, mock_database):
        fc = self._make_fc_with_gossip(
            mock_plugin, mock_database, self._gossip_channels()
        )
        cfg = _make_config_snapshot()

        fc._get_neighbor_fee_median("02peer", cfg=cfg)
        fc._get_neighbor_fee_percentile("02peer", 0.25, cfg=cfg)
        fc._get_neighbor_fee_min("02peer", cfg=cfg)
        fc._get_competitive_undercut_pct("02peer", "chan1", 200, cfg=cfg)

        fc.config.snapshot.assert_not_called()

    def test_gossip_ttl_covers_two_fee_cycles(self, mock_plugin, mock_database):
        fc = self._make_fc_with_gossip(mock_plugin, mock_database, [])

        assert fc._gossip_cache_ttl_seconds(_make_config_snapshot(fee_interval=1800)) == 3900
        assert fc._gossip_cache_ttl_seconds(_make_config_snapshot(fee_interval=3600)) == 7200
        assert fc._gossip_cache_ttl_seconds(MagicMock()) == 3900  # unreadable cfg

    def test_gossip_cache_survives_next_cycle(self, mock_plugin, mock_database):
        """An entry older than the old 1800s TTL but younger than the new TTL
        is still served from cache (no second listchannels RPC)."""
        fc = self._make_fc_with_gossip(
            mock_plugin, mock_database, self._gossip_channels()
        )
        cfg = _make_config_snapshot(fee_interval=1800)

        fc._get_neighbor_fee_median("02peer", cfg=cfg)
        assert fc.data_service.get_channels.call_count == 1

        # Age the gossip entry past the old per-cycle TTL
        fc._neighbor_fee_cache["gossip_channels_02peer"]["ts"] = time.time() - 2000
        # Expire the derived-median cache so the gossip pool is consulted again
        fc._neighbor_fee_cache.pop("neighbor_fee_02peer", None)

        fc._get_neighbor_fee_median("02peer", cfg=cfg)
        assert fc.data_service.get_channels.call_count == 1, \
            "Gossip entry should survive into the next fee cycle"

    def test_cached_gossip_channels_are_trimmed(self, mock_plugin, mock_database):
        fc = self._make_fc_with_gossip(
            mock_plugin, mock_database, self._gossip_channels()
        )
        result = fc._get_peer_inbound_channels("02peer")

        allowed = set(FeeController._GOSSIP_CHANNEL_FIELDS)
        assert result, "Expected trimmed channels in the cache"
        for ch in result:
            assert set(ch.keys()) <= allowed, f"Unexpected cached fields: {set(ch) - allowed}"
            # Fields the consumers read must be preserved
            assert "source" in ch and "fee_per_millionth" in ch and "active" in ch


# =============================================================================
# 6. Profitability warm-up happens before the lock
# =============================================================================

class _RecordingLock:
    """RLock proxy that records successful acquisitions."""

    def __init__(self, events):
        self._lock = threading.RLock()
        self._events = events

    def acquire(self, blocking=True):
        ok = self._lock.acquire(blocking)
        if ok:
            self._events.append("lock_acquired")
        return ok

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
        return False


class TestGossipPrefetch:
    """Item: gossip prefetch pre-lock.

    The per-channel loop used to issue listchannels RPCs serially INSIDE
    _state_lock whenever a peer's gossip cache entry had expired (~46% of
    peers per cycle at the 3900s TTL = 18-41 lock-held RPCs). The prefetch
    warms the gossip cache for every peer about to be processed BEFORE the
    lock is acquired, so in-loop lookups always hit warm cache.
    """

    def _wire_gossip(self, fc, events):
        fc.data_service = MagicMock()

        def _get_channels(destination=None, **kwargs):
            events.append(("gossip_fetch", destination))
            return {"channels": []}

        fc.data_service.get_channels.side_effect = _get_channels
        fc.data_service.get_node_id.return_value = "02our_node_id"
        return fc

    def test_cold_cache_gossip_fetches_all_happen_before_lock(
        self, mock_plugin, mock_database
    ):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        events = []
        fc._state_lock = _RecordingLock(events)
        self._wire_gossip(fc, events)

        fc.adjust_all_fees()

        fetch_indices = [
            i for i, e in enumerate(events)
            if isinstance(e, tuple) and e[0] == "gossip_fetch"
        ]
        assert fetch_indices, "Cold cache must trigger gossip prefetch"
        lock_idx = events.index("lock_acquired")
        late = [events[i] for i in fetch_indices if i >= lock_idx]
        assert not late, \
            f"Gossip RPCs ran under _state_lock: {late}"

        prefetched_peers = {
            e[1] for e in events if isinstance(e, tuple) and e[0] == "gossip_fetch"
        }
        assert prefetched_peers == set(PEER_IDS), \
            "Every peer about to be processed must be prefetched"

    def test_warm_cache_skips_prefetch_rpcs(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        events = []
        self._wire_gossip(fc, events)

        fc.adjust_all_fees()
        first_cycle_fetches = len(events)
        assert first_cycle_fetches > 0

        # Second cycle inside the TTL: cache is warm, no new RPCs.
        fc.adjust_all_fees()
        assert len(events) == first_cycle_fetches, \
            "Warm gossip cache must not be refetched by the prefetch"

    def test_expired_entries_are_refreshed_by_prefetch(
        self, mock_plugin, mock_database
    ):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        events = []
        self._wire_gossip(fc, events)

        fc.adjust_all_fees()
        # Age one peer's gossip entry past the TTL (3900s at 1800s interval)
        key = f"gossip_channels_{PEER_IDS[0]}"
        fc._neighbor_fee_cache[key]["ts"] = time.time() - 4000
        before = len(events)

        fc.adjust_all_fees()

        new_fetches = [e for e in events[before:] if e[0] == "gossip_fetch"]
        assert [e[1] for e in new_fetches] == [PEER_IDS[0]], \
            "Only the expired peer should be refetched"

    def test_prefetch_skips_passive_peers(self, mock_plugin, mock_database):
        from modules.policy_manager import FeeStrategy

        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        events = []
        self._wire_gossip(fc, events)

        passive_policy = MagicMock()
        passive_policy.strategy = FeeStrategy.PASSIVE
        dynamic_policy = MagicMock()
        dynamic_policy.strategy = FeeStrategy.DYNAMIC
        dynamic_policy.fee_ppm_target = None
        fc.policy_manager = MagicMock()
        fc.policy_manager.get_policy.side_effect = (
            lambda pid: passive_policy if pid == PEER_IDS[0] else dynamic_policy
        )

        fc.adjust_all_fees()

        fetched = {e[1] for e in events if e[0] == "gossip_fetch"}
        assert PEER_IDS[0] not in fetched, \
            "PASSIVE peers never reach the gossip lookup; don't prefetch them"
        assert fetched == set(PEER_IDS[1:])

    def test_prefetch_errors_do_not_break_cycle(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        fc.data_service = MagicMock()
        fc.data_service.get_channels.side_effect = RuntimeError("rpc down")
        fc.data_service.get_node_id.return_value = "02our_node_id"

        result = fc.adjust_all_fees()  # must not raise
        assert isinstance(result, list)

    def test_channel_info_fetched_once_before_lock(self, mock_plugin, mock_database):
        """_get_channels_info is itself an RPC — it must run pre-lock and the
        cycle must reuse the result instead of fetching twice."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        events = []
        fc._state_lock = _RecordingLock(events)
        base_channels_info = fc._get_channels_info()

        def _recording_get_channels_info():
            events.append("channels_info_fetch")
            return base_channels_info

        fc._get_channels_info = _recording_get_channels_info

        fc.adjust_all_fees()

        assert events.count("channels_info_fetch") == 1, \
            "Channel info must be fetched exactly once per cycle"
        assert events.index("channels_info_fetch") < events.index("lock_acquired"), \
            "Channel info fetch must happen before _state_lock is acquired"


class TestSingleSerializationPerChannel:
    """Item: wasted json.dumps removal.

    Every terminal path saves cycle state then fee state back-to-back.
    Each save used to json.dumps the full ~50KB merged row even though the
    batch pending dict is last-write-wins, so the first serialization was
    always discarded. Batched rows are now enqueued UNSERIALIZED and dumped
    once at flush.
    """

    def _count_dumps(self, monkeypatch):
        import modules.fee_controller as fc_mod
        real_dumps = fc_mod.json.dumps
        calls = []

        def counting_dumps(*args, **kwargs):
            calls.append(args[0] if args else None)
            return real_dumps(*args, **kwargs)

        monkeypatch.setattr(fc_mod.json, "dumps", counting_dumps)
        return calls

    def test_one_dumps_per_channel_per_cycle(
        self, mock_plugin, mock_database, monkeypatch
    ):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        fc.adjust_all_fees()  # warm states (loads may dump during migration)

        calls = self._count_dumps(monkeypatch)
        fc.adjust_all_fees()

        assert len(calls) == len(CHANNEL_IDS), \
            (f"Expected exactly 1 json.dumps per channel per cycle "
             f"(was 2), got {len(calls)} for {len(CHANNEL_IDS)} channels")

    def test_batch_rows_carry_serialized_json(self, mock_plugin, mock_database):
        """The flush must hand the database layer fully serialized rows."""
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)

        fc.adjust_all_fees()

        rows = mock_database.update_fee_strategy_states_batch.call_args[0][0]
        assert rows
        for row in rows:
            assert isinstance(row["v2_state_json"], str)
            json.loads(row["v2_state_json"])  # valid JSON

    def test_fallback_per_row_writes_carry_serialized_json(
        self, mock_plugin, mock_database
    ):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        del mock_database.update_fee_strategy_states_batch

        fc.adjust_all_fees()

        assert mock_database.update_fee_strategy_state.call_count == len(CHANNEL_IDS)
        for call in mock_database.update_fee_strategy_state.call_args_list:
            assert isinstance(call.kwargs["v2_state_json"], str)

    def test_out_of_cycle_save_passes_serialized_json(
        self, mock_plugin, mock_database
    ):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        ts = fc._get_channel_fee_state(CHANNEL_IDS[0], PEER_IDS[0])

        mock_database.update_fee_strategy_state.reset_mock()
        fc._save_channel_fee_state(CHANNEL_IDS[0], ts)

        kwargs = mock_database.update_fee_strategy_state.call_args.kwargs
        assert isinstance(kwargs["v2_state_json"], str)
        json.loads(kwargs["v2_state_json"])


class TestPeerLatencyMemo:
    """Item: per-cycle per-peer latency memo.

    _calculate_floor fetches all 24h forward rows per CHANNEL per cycle
    (~0.8ms on busy peers). Parallel channels to the same peer should pay
    once per cycle; the memo is cleared at cycle start so data never goes
    stale across cycles, and out-of-cycle calls keep querying directly.
    """

    def _make_fc_two_channels_one_peer(self, mock_plugin, mock_database):
        fc, cfg = _make_cycle_fc(mock_plugin, mock_database)
        shared_peer = PEER_IDS[0]
        now = int(time.time())
        mock_database.get_all_channel_states.return_value = [
            {
                "channel_id": cid,
                "peer_id": shared_peer,
                "state": "balanced",
                "flow_ratio": 0.5,
                "sats_in": 100_000,
                "sats_out": 100_000,
                "capacity": 2_000_000,
                "updated_at": now,
                "kalman_flow_ratio": 0.3,
                "kalman_velocity": 0.01,
            }
            for cid in CHANNEL_IDS[:2]
        ]
        channels_info = {
            cid: {
                "channel_id": cid,
                "short_channel_id": cid,
                "peer_id": shared_peer,
                "capacity": 2_000_000,
                "spendable_msat": 1_000_000_000,
                "receivable_msat": 1_000_000_000,
                "fee_base_msat": 0,
                "fee_proportional_millionths": 150,
                "htlc_minimum_msat": 0,
                "htlc_min_msat": 0,
                "htlc_maximum_msat": 0,
                "htlc_max_msat": 0,
                "opener": "local",
            }
            for cid in CHANNEL_IDS[:2]
        }
        fc._get_channels_info = lambda: channels_info
        return fc

    def test_latency_queried_once_per_peer_per_cycle(
        self, mock_plugin, mock_database
    ):
        fc = self._make_fc_two_channels_one_peer(mock_plugin, mock_database)

        fc.adjust_all_fees()

        assert mock_database.get_peer_latency_stats.call_count == 1, \
            (f"2 channels to one peer must share a single latency query, got "
             f"{mock_database.get_peer_latency_stats.call_count}")

    def test_memo_cleared_between_cycles(self, mock_plugin, mock_database):
        fc = self._make_fc_two_channels_one_peer(mock_plugin, mock_database)

        fc.adjust_all_fees()
        mock_database.get_peer_latency_stats.reset_mock()
        # Make both channels eligible again next cycle
        now = int(time.time())
        for cid in CHANNEL_IDS[:2]:
            if cid in fc._cycle_states:
                fc._cycle_states[cid].last_update = now - 7200
            if cid in fc._channel_fee_states:
                fc._channel_fee_states[cid].last_update = now - 7200

        fc.adjust_all_fees()

        assert mock_database.get_peer_latency_stats.call_count == 1, \
            "Memo must be refreshed (and used) again on the next cycle"

    def test_out_of_cycle_floor_query_not_memoized(
        self, mock_plugin, mock_database
    ):
        fc = self._make_fc_two_channels_one_peer(mock_plugin, mock_database)

        fc._calculate_floor(2_000_000, peer_id=PEER_IDS[0])
        fc._calculate_floor(2_000_000, peer_id=PEER_IDS[0])

        assert mock_database.get_peer_latency_stats.call_count == 2, \
            "Out-of-cycle floor calculations must keep querying directly"


class TestProfitabilityWarmup:

    def test_warmup_runs_before_lock_acquisition(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        events = []
        fc._state_lock = _RecordingLock(events)
        fc.profitability = MagicMock()
        fc.profitability.analyze_all_channels.side_effect = lambda *a, **k: events.append("warm")
        fc.profitability.get_profitability.return_value = None

        fc.adjust_all_fees()

        assert "warm" in events, "Profitability cache must be warmed during the cycle"
        assert "lock_acquired" in events
        assert events.index("warm") < events.index("lock_acquired"), \
            "Warm-up must happen BEFORE _state_lock is acquired"

    def test_warmup_skipped_when_paused(self, mock_plugin, mock_database):
        fc, cfg = _make_cycle_fc(mock_plugin, mock_database)
        cfg.paused = True
        fc.profitability = MagicMock()

        result = fc.adjust_all_fees()

        assert result == []
        fc.profitability.analyze_all_channels.assert_not_called()

    def test_no_profitability_module_is_fine(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        fc.profitability = None

        result = fc.adjust_all_fees()  # must not raise
        assert isinstance(result, list)

    def test_warmup_errors_do_not_break_cycle(self, mock_plugin, mock_database):
        fc, _ = _make_cycle_fc(mock_plugin, mock_database)
        fc.profitability = MagicMock()
        fc.profitability.analyze_all_channels.side_effect = RuntimeError("boom")
        fc.profitability.get_profitability.return_value = None

        result = fc.adjust_all_fees()  # must not raise
        assert isinstance(result, list)
