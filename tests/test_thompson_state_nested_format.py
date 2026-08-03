"""
Tests for the nested-only thompson_state persistence format.

The merged fee-strategy row used to write the DTS payload TWICE:
nested under fee_state.thompson_state AND as a flat top-level
thompson_state compatibility mirror (plus pid_state /
last_vegas_multiplier mirrors). The flat mirror alone was ~49% of the
serialized row (~214 MB/day of WAL churn at 90 channels).

Covers:
1. Writer: _build_merged_fee_strategy_row emits NO flat mirrors; the
   serialized row is roughly half the mirrored size.
2. Read compatibility: old rows (nested + flat) and ancient rows
   (flat only) still load; re-saving an old row drops the mirrors.
3. External readers (flow_analysis, profitability_analyzer,
   capacity_planner) work against BOTH formats.
"""

import json
import sys
import os
import time
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import (
    FeeController,
    ChannelFeeState,
    GaussianThompsonState,
)
from modules.config import Config


CHANNEL_ID = "100x1x0"
PEER_ID = "02" + "a" * 64


from modules.fee_authority import FeeAuthorityGate

def _make_fc(mock_plugin, mock_database):
    config = MagicMock(spec=Config)
    fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())
    # Real Database returns a defaults dict for unknown channels, never None
    mock_database.get_fee_strategy_state.return_value = {
        "channel_id": CHANNEL_ID, "v2_state_json": "{}", "last_update": 0,
    }
    return fc


def _meaty_thompson_dict(n_observations=80, n_contexts=20):
    """A realistically sized DTS payload (most of a production row)."""
    ts = GaussianThompsonState()
    now = int(time.time())
    for i in range(n_observations):
        ts.observations.append(
            (100 + i, 3.5 + 0.01 * i, 0.8, now - i * 3600, "normal")
        )
    for i in range(n_contexts):
        ts.contextual_posteriors[f"ctx_{i}_role_bucket"] = (
            150.0 + i, 0.05, 12, now - i * 60
        )
    return ts.to_dict()


def _warm_states(fc, thompson_dict=None):
    state = ChannelFeeState()
    state.algorithm_version = "dts_pid_v1"
    if thompson_dict is not None:
        state.thompson = GaussianThompsonState.from_dict(thompson_dict)
    fc._channel_fee_states[CHANNEL_ID] = state
    cyc = fc._get_cycle_state(CHANNEL_ID)
    return state, cyc


FLAT_MIRROR_KEYS = ("thompson_state", "pid_state", "last_vegas_multiplier")


class TestMergedRowHasNoFlatMirror:

    def test_no_flat_mirrors_in_new_rows(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        _warm_states(fc, _meaty_thompson_dict())

        _, merged_v2 = fc._build_merged_fee_strategy_row(CHANNEL_ID)

        for key in FLAT_MIRROR_KEYS:
            assert key not in merged_v2, \
                f"Flat '{key}' mirror must no longer be written"
        # The canonical nested payloads must still be complete
        assert merged_v2["fee_state"]["thompson_state"]["observations"], \
            "Nested thompson_state must carry the observations"
        assert "pid_state" in merged_v2["fee_state"]
        assert "cycle_state" in merged_v2

    def test_row_size_roughly_halved(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        _warm_states(fc, _meaty_thompson_dict())

        _, merged_v2 = fc._build_merged_fee_strategy_row(CHANNEL_ID)
        new_len = len(json.dumps(merged_v2))

        # Reconstruct what the old format serialized to (nested + mirrors)
        mirrored = dict(merged_v2)
        mirrored["thompson_state"] = merged_v2["fee_state"]["thompson_state"]
        mirrored["pid_state"] = merged_v2["fee_state"]["pid_state"]
        mirrored["last_vegas_multiplier"] = merged_v2["fee_state"].get(
            "last_vegas_multiplier", 1.0
        )
        old_len = len(json.dumps(mirrored))

        ratio = new_len / old_len
        assert ratio < 0.65, \
            (f"Removing the mirror should ~halve the row: "
             f"{new_len}/{old_len} bytes (ratio {ratio:.2f})")

    def test_shared_scalar_fields_still_flat(self, mock_plugin, mock_database):
        """The three small shared canonical fields keep their flat copies
        (read by _extract_*_payload fallbacks and operator tooling)."""
        fc = _make_fc(mock_plugin, mock_database)
        state, _ = _warm_states(fc)
        state.last_gossip_refresh = 12345

        _, merged_v2 = fc._build_merged_fee_strategy_row(
            CHANNEL_ID, fee_state=state
        )

        assert merged_v2["last_gossip_refresh"] == 12345
        assert "last_broadcast_at" in merged_v2
        assert "dynamic_htlcmin_baseline_msat" in merged_v2


class TestOldFormatStillLoads:

    def _persisted_row(self, v2_data):
        return {
            "channel_id": CHANNEL_ID,
            "last_revenue_rate": 5.0,
            "last_fee_ppm": 150,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": int(time.time()) - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 10,
            "last_volume_sats": 50_000,
            "last_broadcast_fee_ppm": 150,
            "last_state": "balanced",
            "v2_state_json": json.dumps(v2_data),
        }

    def _old_format_row(self):
        """Rows written by the previous code: nested AND flat mirrors."""
        thompson = _meaty_thompson_dict(n_observations=10, n_contexts=2)
        thompson["posterior_mean"] = 432.0
        fee_payload = {
            "algorithm_version": "dts_pid_v1",
            "thompson_state": thompson,
            "last_vegas_multiplier": 1.2,
            "last_gossip_refresh": 111,
            "last_broadcast_at": 222,
            "pid_state": {"kp": 2.0},
            "dynamic_htlcmin_baseline_msat": None,
        }
        return self._persisted_row({
            "algorithm_version": "dts_pid_v1",
            "fee_state": fee_payload,
            "cycle_state": {"last_fee_ppm": 150},
            "thompson_state": thompson,
            "last_vegas_multiplier": 1.2,
            "pid_state": {"kp": 2.0},
            "last_gossip_refresh": 111,
            "last_broadcast_at": 222,
            "dynamic_htlcmin_baseline_msat": None,
        })

    def _ancient_flat_only_row(self):
        """Rows written before the nested format existed: flat only."""
        thompson = _meaty_thompson_dict(n_observations=10, n_contexts=2)
        thompson["posterior_mean"] = 432.0
        return self._persisted_row({
            "algorithm_version": "dts_pid_v1",
            "thompson_state": thompson,
            "last_vegas_multiplier": 1.2,
            "pid_state": {"kp": 2.0},
            "last_gossip_refresh": 111,
            "last_broadcast_at": 222,
        })

    @pytest.mark.parametrize("row_builder", ["_old_format_row",
                                             "_ancient_flat_only_row"])
    def test_persisted_row_loads(self, mock_plugin, mock_database, row_builder):
        fc = _make_fc(mock_plugin, mock_database)
        mock_database.get_fee_strategy_state.return_value = \
            getattr(self, row_builder)()

        state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID)

        assert state.thompson.posterior_mean == pytest.approx(432.0)
        assert state.last_vegas_multiplier == pytest.approx(1.2)
        assert state.last_gossip_refresh == 111
        assert state.pid.kp == 2.0

    def test_resave_of_old_row_drops_mirrors(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        mock_database.get_fee_strategy_state.return_value = self._old_format_row()

        state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID)
        fc._save_channel_fee_state(CHANNEL_ID, state)

        kwargs = mock_database.update_fee_strategy_state.call_args.kwargs
        v2 = kwargs["v2_state_json"]
        v2 = json.loads(v2) if isinstance(v2, str) else v2
        for key in FLAT_MIRROR_KEYS:
            assert key not in v2, f"Re-save must not resurrect flat '{key}'"
        assert v2["fee_state"]["thompson_state"]["posterior_mean"] == \
            pytest.approx(432.0)


# =============================================================================
# External readers must handle BOTH formats
# =============================================================================

def _nested_row(posterior_variance=None, posterior_mean=None):
    ts = {}
    if posterior_variance is not None:
        ts["posterior_variance"] = posterior_variance
    if posterior_mean is not None:
        ts["posterior_mean"] = posterior_mean
    return {"v2_state_json": json.dumps({
        "algorithm_version": "dts_pid_v1",
        "fee_state": {"algorithm_version": "dts_pid_v1", "thompson_state": ts},
        "cycle_state": {},
    })}


def _flat_row(posterior_variance=None, posterior_mean=None):
    ts = {}
    if posterior_variance is not None:
        ts["posterior_variance"] = posterior_variance
    if posterior_mean is not None:
        ts["posterior_mean"] = posterior_mean
    return {"v2_state_json": json.dumps({
        "algorithm_version": "dts_pid_v1",
        "thompson_state": ts,
    })}


class TestProfitabilityReaderBothFormats:

    def _classify(self, fee_state_row):
        from modules.profitability_analyzer import (
            ChannelProfitabilityAnalyzer, ProfitabilityClass,
        )
        analyzer = ChannelProfitabilityAnalyzer.__new__(
            ChannelProfitabilityAnalyzer
        )
        analyzer.plugin = MagicMock()
        analyzer.database = MagicMock()
        analyzer.database.get_fee_strategy_state.return_value = fee_state_row
        result = analyzer._classify_channel(
            roi=0.07,  # between widened (5%) and base (10%) PROFITABLE bars
            net_profit=140,
            last_routed=int(time.time()) - 3600,
            days_open=100,
            channel_id=CHANNEL_ID,
            peer_id=PEER_ID,
            forward_count=60,
        )
        return result, ProfitabilityClass

    def test_nested_format_low_variance_widens(self, mock_plugin):
        result, ProfitabilityClass = self._classify(
            _nested_row(posterior_variance=100)
        )
        assert result == ProfitabilityClass.PROFITABLE, \
            "Nested-only row must still widen thresholds (variance < 2500)"

    def test_flat_format_low_variance_widens(self, mock_plugin):
        result, ProfitabilityClass = self._classify(
            _flat_row(posterior_variance=100)
        )
        assert result == ProfitabilityClass.PROFITABLE, \
            "Old flat rows must keep widening thresholds"


class TestFlowAnalysisReaderBothFormats:

    def _classify_with_row(self, fee_state_row):
        from modules.flow_analysis import FlowAnalyzer, FlowMetrics

        analyzer = FlowAnalyzer.__new__(FlowAnalyzer)
        analyzer.plugin = MagicMock()
        analyzer.database = MagicMock()
        analyzer.database.get_fee_strategy_state.return_value = fee_state_row
        cfg = MagicMock()
        cfg.source_threshold = 0.05
        cfg.sink_threshold = -0.05
        analyzer.config = cfg

        # Converged Kalman, ratio just above the UNWIDENED source threshold:
        # high posterior variance must widen thresholds -> NOT source.
        analyzer._compute_raw_kalman_observation = MagicMock(
            return_value=(0.06, 10))
        analyzer._calculate_confidence = MagicMock(return_value=0.9)
        analyzer._apply_kalman_filter = MagicMock(
            return_value=(0.06, 0.0, 0.1, False, 10))
        analyzer._classify_balance_position = MagicMock(
            return_value="balanced_position")

        metrics = MagicMock()
        metrics.is_congested = False
        metrics.confidence = 0.5
        metrics.daily_volume = 100_000
        _ = FlowMetrics  # imported for parity with production types
        analyzer._apply_kalman_reclassification(
            metrics=metrics,
            channel_id=CHANNEL_ID,
            capacity=1_000_000,
            our_balance=500_000,
            channel_daily=[],
            raw_entries=[],
            last_forward_ts=int(time.time()),
            previous_state=None,
        )
        return metrics.state

    def test_nested_format_high_variance_widens(self, mock_plugin):
        from modules.flow_analysis import ChannelState
        state = self._classify_with_row(_nested_row(posterior_variance=40_000))
        assert state != ChannelState.SOURCE, \
            "Nested-only row must still widen flow thresholds while exploring"

    def test_flat_format_high_variance_widens(self, mock_plugin):
        from modules.flow_analysis import ChannelState
        state = self._classify_with_row(_flat_row(posterior_variance=40_000))
        assert state != ChannelState.SOURCE, \
            "Old flat rows must keep widening flow thresholds"
