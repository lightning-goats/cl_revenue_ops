"""
Regression tests for the fee_controller medium-severity audit batch (2026-06).

Covers four audited bugs:
1. Out-of-band DTS posterior nudges (failed forwards, neighbor seeding,
   undercut/boundary bias) were erased by the next _recompute_posterior,
   which rebuilds posterior_mean/std entirely from observations + the
   fixed prior. Nudges must now survive update_posterior.
2. DTS hysteresis sleep entry only set ts_state.is_sleeping; the persisted
   row sources is_sleeping/sleep_until/stable_cycles from the cycle
   payload, so a restart woke every sleeping channel.
3. set_channel_fee reported success=False when post-RPC bookkeeping
   (database.record_fee_change) failed, even though the fee WAS applied
   on-chain — leaving last_broadcast_fee_ppm stale in callers.
4. The forward_event handler blocking-acquired fee_controller._state_lock,
   which adjust_all_fees holds across the whole fee cycle, freezing the
   single pyln dispatch thread mid-cycle.

Plus: _get_rebalance_cost_floor passes since_timestamp to
database.get_channel_cost_history (with old-signature fallback).
"""

import copy
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
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import (
    FeeController,
    GaussianThompsonState,
    ChannelFeeState,
)
from modules.config import Config

from tests.plugin_test_utils import load_plugin_module


from modules.fee_authority import FeeAuthorityGate

def _make_fc(mock_plugin, mock_database):
    config = MagicMock(spec=Config)
    return FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())


def _make_config_snapshot(**overrides):
    defaults = {
        'min_fee_ppm': 10,
        'max_fee_ppm': 5000,
        'fee_interval': 1800,
        'inbound_fee_estimate_ppm': 200,
        'thompson_prior_std_fee': 200.0,
        'routing_intelligence_enabled': False,
    }
    defaults.update(overrides)

    class ConfigSnap:
        pass

    snap = ConfigSnap()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_database():
    return MagicMock()


def _seeded_thompson(fee=500, n=10, revenue=5.0):
    """Build a Thompson state with observations clustered at one fee."""
    state = GaussianThompsonState()
    now = int(time.time())
    for _ in range(n):
        state.observations.append((fee, revenue, 1.0, now, "normal"))
    state._recompute_posterior()
    return state


# =============================================================================
# Bug 1: Out-of-band posterior nudges must survive _recompute_posterior
# =============================================================================

class TestDurablePosteriorNudges:

    def test_nudge_survives_recompute_posterior(self):
        """The audit reproduction: nudge, then recompute — nudge must persist."""
        state = _seeded_thompson(fee=500)
        control = copy.deepcopy(state)

        state.record_posterior_nudge(400.0, 0.3)
        assert state.posterior_mean < control.posterior_mean, \
            "Nudge should have an immediate effect"

        # The recompute that previously wiped the nudge entirely
        state._recompute_posterior()
        control._recompute_posterior()

        assert state.posterior_mean < control.posterior_mean - 1.0, \
            "Nudge must still bias the posterior after a full recompute"

    def test_failed_forward_nudge_survives_update_posterior(
        self, mock_plugin, mock_database
    ):
        """record_failed_forward must shift the NEXT cycle's posterior even
        after update_posterior rebuilds it from observations."""
        fc = _make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"

        state = _seeded_thompson(fee=500)
        control = copy.deepcopy(state)
        fc._channel_fee_states[channel_id] = ChannelFeeState(thompson=state)

        # Large failed forward (5M sats) => strong advisory negative signal.
        # DTS-4: must carry a fee-relevant failcode or the nudge is dropped.
        fc.record_failed_forward(
            channel_id, 500, amount_msat=5_000_000_000,
            failcode=0x1000 | 12, failreason="WIRE_FEE_INSUFFICIENT",
        )
        assert state.posterior_mean < control.posterior_mean

        # Next fee cycle: update_posterior runs BEFORE sampling
        state.update_posterior(fee=500, revenue_rate=5.0, hours=2.0)
        control.update_posterior(fee=500, revenue_rate=5.0, hours=2.0)

        assert state.posterior_mean < control.posterior_mean - 1.0, \
            "Failed-forward nudge was erased by update_posterior"

    def test_nudge_survives_serialization_roundtrip(self):
        """Persisted state must retain pending nudges across restart."""
        state = _seeded_thompson(fee=500)
        control = copy.deepcopy(state)
        state.record_posterior_nudge(400.0, 0.3)

        # JSON round trip (as the DB layer does)
        restored = GaussianThompsonState.from_dict(
            json.loads(json.dumps(state.to_dict()))
        )
        restored.update_posterior(fee=500, revenue_rate=5.0, hours=2.0)
        control.update_posterior(fee=500, revenue_rate=5.0, hours=2.0)

        assert restored.posterior_mean < control.posterior_mean - 1.0, \
            "Nudge lost through to_dict/from_dict round trip"

    def test_stale_nudges_decay_and_prune(self):
        """Old nudges decay to nothing and get pruned from the bias list."""
        state = _seeded_thompson(fee=500)
        control = copy.deepcopy(state)

        # Inject a nudge dated far in the past (directly, to control timestamp)
        state.posterior_bias.append((400.0, 0.3, int(time.time()) - 30 * 86400))
        state._recompute_posterior()
        control._recompute_posterior()

        assert state.posterior_mean == pytest.approx(control.posterior_mean, abs=0.5), \
            "A weeks-old nudge should no longer move the posterior"
        assert state.posterior_bias == [], "Expired nudges should be pruned"

    def test_nudge_list_is_bounded(self):
        state = _seeded_thompson(fee=500)
        for _ in range(state.MAX_BIAS_NUDGES * 2):
            state.record_posterior_nudge(400.0, 0.01)
        assert len(state.posterior_bias) <= state.MAX_BIAS_NUDGES

    def test_invalid_nudges_rejected(self):
        state = _seeded_thompson(fee=500)
        state.record_posterior_nudge(float("nan"), 0.3)
        state.record_posterior_nudge(400.0, float("inf"))
        state.record_posterior_nudge(400.0, -1.0)
        state.record_posterior_nudge(-50.0, 0.3)
        assert state.posterior_bias == []


# =============================================================================
# Bug 2: Sleep state must round-trip through save/load
# =============================================================================

class TestSleepStatePersistence:

    def _make_fc_full(self, mock_plugin, mock_database, now):
        fc = _make_fc(mock_plugin, mock_database)
        cfg = _make_config_snapshot()
        fc.config.snapshot.return_value = cfg
        fc.config.dry_run = False

        volume = 50_000
        fee = 150
        hours = 2.0
        rate = (volume * fee / 1_000_000) / hours  # 3.75 sats/hr

        mock_database.get_channel_probe.return_value = None
        mock_database.get_last_rebalance_cost.return_value = None
        mock_database.get_volume_since.return_value = volume
        mock_database.get_forward_count_since.return_value = 10
        mock_database.get_peer_uptime_percent.return_value = 99.5
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.3,
            "kalman_velocity": 0.01,
        }
        # Channel is one stable cycle away from sleep; revenue rate is flat.
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": rate,
            "last_fee_ppm": fee,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": now - int(hours * 3600),
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 2,
            "forward_count_since_update": 10,
            "last_volume_sats": volume,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = now - 1800
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_channel_age.return_value = 30
        mock_database.get_peer_latency_stats.return_value = {
            'avg': 0.0, 'std': 0.0, 'count': 0,
        }
        mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

        return fc, cfg

    def test_sleep_entry_round_trips_through_save_load(
        self, mock_plugin, mock_database
    ):
        """A channel that enters sleep must still be asleep after a restart."""
        now = int(time.time())
        fc, cfg = self._make_fc_full(mock_plugin, mock_database, now)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 150,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "balanced", "forward_count": 10, "sats_out": 10000}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is None, "Sleep entry should not produce an adjustment"
        ts_state = fc._channel_fee_states[channel_id]
        assert ts_state.is_sleeping is True, "Channel should have entered sleep"
        assert ts_state.sleep_until > now

        # The WRITE path: the persisted row must carry the sleep state
        assert mock_database.update_fee_strategy_state.called
        last_call = mock_database.update_fee_strategy_state.call_args_list[-1]
        row = dict(last_call.kwargs)
        assert row["is_sleeping"] == 1, \
            "Persisted row lost is_sleeping — restart would wake the channel"
        assert row["sleep_until"] == ts_state.sleep_until
        assert row["stable_cycles"] == ts_state.stable_cycles

        # Full round trip: reload from the persisted row in a fresh controller
        fresh_db = MagicMock()
        persisted_row = {k: v for k, v in row.items()}
        persisted_row["channel_id"] = channel_id
        fresh_db.get_fee_strategy_state.return_value = persisted_row
        fc2 = _make_fc(mock_plugin, fresh_db)
        restored = fc2._get_channel_fee_state(channel_id, peer_id)
        assert restored.is_sleeping is True, "Restart woke a sleeping channel"
        assert restored.sleep_until == ts_state.sleep_until


# =============================================================================
# Bug 3: DB failure after a successful setchannel must not report failure
# =============================================================================

class TestSetChannelFeePostRpcBookkeeping:

    def _make_fc(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        cfg = _make_config_snapshot()
        fc.config.snapshot.return_value = cfg
        fc.config.dry_run = False
        fc.data_service = MagicMock()
        fc.data_service.set_channel.return_value = {}
        mock_database.get_fee_strategy_state.return_value = {}
        return fc

    def test_record_fee_change_failure_still_reports_success(
        self, mock_plugin, mock_database
    ):
        fc = self._make_fc(mock_plugin, mock_database)
        mock_database.record_fee_change.side_effect = Exception("disk full")

        channel_id = "123x456x0"
        channel_info = {
            "peer_id": "02" + "a" * 64,
            "short_channel_id": channel_id,
            "fee_proportional_millionths": 100,
        }
        result = fc.set_channel_fee(
            channel_id, 250, reason="test", manual=True,
            channel_info=channel_info,
        )

        assert fc.data_service.set_channel.called, "setchannel RPC should run"
        assert result["success"] is True, \
            "Fee WAS applied on-chain; bookkeeping failure must not report failure"
        assert result.get("warnings"), \
            "Bookkeeping failure should surface as a warning"

        # Broadcast state must be updated so the controller doesn't re-fight
        # the same change next cycle.
        ts_state = fc._channel_fee_states.get(channel_id)
        assert ts_state is not None
        assert ts_state.last_broadcast_fee_ppm == 250

    def test_rpc_failure_still_reports_failure(self, mock_plugin, mock_database):
        """Pre-RPC/RPC failures must still report success=False."""
        fc = self._make_fc(mock_plugin, mock_database)
        fc.data_service.set_channel.side_effect = Exception("RPC down")

        channel_id = "123x456x0"
        channel_info = {
            "peer_id": "02" + "a" * 64,
            "short_channel_id": channel_id,
            "fee_proportional_millionths": 100,
        }
        result = fc.set_channel_fee(
            channel_id, 250, reason="test", channel_info=channel_info,
        )
        assert result["success"] is False


# =============================================================================
# Bug 4: forward_event handler must not block on a busy fee cycle
# =============================================================================

class TestForwardEventLockContention:

    def _setup_module(self, mock_plugin, mock_database):
        mod = load_plugin_module()
        fc = _make_fc(mock_plugin, mock_database)
        mod.database = MagicMock()
        mod.fee_controller = fc
        mod._resolve_scid_to_peer = lambda scid: None
        return mod, fc

    def test_handler_skips_nudge_when_fee_cycle_holds_lock(
        self, mock_plugin, mock_database
    ):
        mod, fc = self._setup_module(mock_plugin, mock_database)
        channel_id = "123x456x0"
        state = GaussianThompsonState()
        state.posterior_mean = 500.0
        state.posterior_std = 50.0
        fc._channel_fee_states[channel_id] = ChannelFeeState(
            thompson=state, last_fee_ppm=500,
        )
        original_mean = state.posterior_mean

        # Keep the test fast: shrink the bounded-acquire timeout
        assert hasattr(mod, "FORWARD_EVENT_LOCK_TIMEOUT_SECS"), \
            "Handler should use a bounded lock acquire"
        mod.FORWARD_EVENT_LOCK_TIMEOUT_SECS = 0.1

        held = threading.Event()
        release = threading.Event()

        def fee_cycle():
            with fc._state_lock:
                held.set()
                release.wait(10)

        holder = threading.Thread(target=fee_cycle, daemon=True)
        holder.start()
        assert held.wait(5)

        done = threading.Event()

        def dispatch():
            # DTS-4: the nudge is keyed to out_channel and requires a
            # fee-relevant failcode (only present on local_failed).
            mod._on_forward_event_impl(
                {"status": "local_failed", "in_channel": "999x9x9",
                 "out_channel": channel_id,
                 "failcode": 0x1000 | 12,
                 "failreason": "WIRE_FEE_INSUFFICIENT",
                 "in_msat": "1000000msat"},
                MagicMock(),
            )
            done.set()

        worker = threading.Thread(target=dispatch, daemon=True)
        worker.start()
        finished = done.wait(2.0)
        release.set()
        holder.join(5)
        worker.join(5)

        assert finished, \
            "forward_event handler froze on _state_lock held by the fee cycle"
        assert state.posterior_mean == original_mean, \
            "Nudge should be skipped (not applied late) under contention"

    def test_handler_applies_nudge_when_lock_free(
        self, mock_plugin, mock_database
    ):
        mod, fc = self._setup_module(mock_plugin, mock_database)
        channel_id = "123x456x0"
        state = GaussianThompsonState()
        state.posterior_mean = 500.0
        state.posterior_std = 50.0
        fc._channel_fee_states[channel_id] = ChannelFeeState(
            thompson=state, last_fee_ppm=500,
        )
        original_mean = state.posterior_mean

        mod._on_forward_event_impl(
            {"status": "local_failed", "in_channel": "999x9x9",
             "out_channel": channel_id,
             "failcode": 0x1000 | 12,
             "failreason": "WIRE_FEE_INSUFFICIENT",
             "in_msat": "1000000msat"},
            MagicMock(),
        )

        assert state.posterior_mean < original_mean, \
            "Nudge should apply normally when the lock is uncontended"


# =============================================================================
# Audit DTS-4: failed-forward nudge must target the OUT channel and fire
# only on fee-relevant failures
# =============================================================================

class TestFailedForwardRekeying:
    """The fee a sender pays for traversing our node is set by OUR policy
    on the OUT channel (BOLT 7), so a fee-related failure is evidence about
    out_channel — the old handler nudged in_channel. And CLN's forward_event
    only carries failcode/failreason for status=local_failed; a plain
    'failed' (downstream onion error) has no usable failure reason, so the
    nudge must be dropped entirely rather than fed in misdirected."""

    IN_CHANNEL = "111x1x0"
    OUT_CHANNEL = "222x2x0"
    WIRE_FEE_INSUFFICIENT = 0x1000 | 12   # 4108
    WIRE_TEMPORARY_CHANNEL_FAILURE = 0x1000 | 7  # 4103

    def _setup_module(self, mock_plugin, mock_database):
        mod = load_plugin_module()
        fc = _make_fc(mock_plugin, mock_database)
        mod.database = MagicMock()
        mod.fee_controller = fc
        mod._resolve_scid_to_peer = lambda scid: None

        states = {}
        for channel_id in (self.IN_CHANNEL, self.OUT_CHANNEL):
            state = GaussianThompsonState()
            state.posterior_mean = 500.0
            state.posterior_std = 50.0
            fc._channel_fee_states[channel_id] = ChannelFeeState(
                thompson=state, last_fee_ppm=500,
            )
            states[channel_id] = state
        return mod, fc, states

    def _dispatch(self, mod, status, failcode=None, failreason=None):
        payload = {
            "status": status,
            "in_channel": self.IN_CHANNEL,
            "out_channel": self.OUT_CHANNEL,
            "in_msat": "1000000msat",
        }
        if failcode is not None:
            payload["failcode"] = failcode
        if failreason is not None:
            payload["failreason"] = failreason
        mod._on_forward_event_impl(payload, MagicMock())

    def test_fee_failure_nudges_out_channel_not_in_channel(
        self, mock_plugin, mock_database
    ):
        mod, fc, states = self._setup_module(mock_plugin, mock_database)

        self._dispatch(
            mod, "local_failed",
            failcode=self.WIRE_FEE_INSUFFICIENT,
            failreason="WIRE_FEE_INSUFFICIENT",
        )

        assert states[self.OUT_CHANNEL].posterior_mean < 500.0, \
            "Fee-relevant failure must nudge the OUT channel's posterior"
        assert states[self.IN_CHANNEL].posterior_mean == 500.0, \
            "The IN channel's fee is our peer's policy, not ours — no nudge"

    def test_liquidity_failcode_produces_no_nudge(
        self, mock_plugin, mock_database
    ):
        mod, fc, states = self._setup_module(mock_plugin, mock_database)

        self._dispatch(
            mod, "local_failed",
            failcode=self.WIRE_TEMPORARY_CHANNEL_FAILURE,
            failreason="WIRE_TEMPORARY_CHANNEL_FAILURE",
        )

        assert states[self.OUT_CHANNEL].posterior_mean == 500.0
        assert states[self.IN_CHANNEL].posterior_mean == 500.0

    def test_downstream_failed_without_failcode_produces_no_nudge(
        self, mock_plugin, mock_database
    ):
        """CLN's 'failed' status carries no failcode: drop the nudge."""
        mod, fc, states = self._setup_module(mock_plugin, mock_database)

        self._dispatch(mod, "failed")

        assert states[self.OUT_CHANNEL].posterior_mean == 500.0
        assert states[self.IN_CHANNEL].posterior_mean == 500.0


# =============================================================================
# Cost history call site: since_timestamp pushdown with fallback
# =============================================================================

class TestCostHistorySinceTimestamp:

    def test_passes_since_timestamp(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        captured = {}

        def cost_history(channel_id, since_timestamp=None):
            captured["channel_id"] = channel_id
            captured["since_timestamp"] = since_timestamp
            return []

        mock_database.get_channel_cost_history.side_effect = cost_history
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        fc._get_rebalance_cost_floor("123x456x0", "02" + "a" * 64, "source")

        assert captured.get("since_timestamp") is not None
        expected_cutoff = int(time.time()) - fc.REBALANCE_FLOOR_WINDOW_DAYS * 86400
        assert abs(captured["since_timestamp"] - expected_cutoff) <= 5

    def test_falls_back_to_old_signature(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        calls = []

        def old_signature(channel_id):
            calls.append(channel_id)
            return []

        mock_database.get_channel_cost_history = old_signature
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        # Must not raise despite the old single-arg signature
        fc._get_rebalance_cost_floor("123x456x0", "02" + "a" * 64, "source")
        assert calls == ["123x456x0"]
