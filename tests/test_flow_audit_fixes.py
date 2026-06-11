"""
Tests for the 2026-06 quantitative audit fixes in flow analysis.

F1: Classification flapping — dead ±0.5 flow thresholds lowered to ±0.05,
    hysteresis on the balance-position fallback, Kalman direction veto.
F2: estimate_depletion_hours() pure helper (correct units).
F6: 'dormant' vocabulary emitted by the classifier.
F7: No-flow fallback unified with hysteresis bands; no synthetic ±0.6
    flow_ratio contaminating EMA velocity.
F5: Temporal profile graduation accounting honesty.
"""
import math
import time
import pytest
from unittest.mock import MagicMock


def _make_analyzer(source_threshold=0.05, sink_threshold=-0.05):
    from modules.flow_analysis import FlowAnalyzer

    plugin = MagicMock()
    config = MagicMock()
    config.source_threshold = source_threshold
    config.sink_threshold = sink_threshold
    config.htlc_congestion_threshold = 0.8
    config.flow_window_days = 7
    db = MagicMock()
    db.get_kalman_state.return_value = None
    db.get_fee_strategy_state.return_value = None
    db.kalman_purge_needed.return_value = False
    analyzer = FlowAnalyzer(plugin, config, db)
    return analyzer, db


# =========================================================================
# F1a: Config flow thresholds actually reachable
# =========================================================================

class TestF1ConfigThresholds:
    def test_default_thresholds_are_plus_minus_0_05(self):
        """±0.5/day required half-capacity net flow; typical is 0.01-0.10.

        Defaults must be ±0.05 so the Kalman flow estimate classifies first.
        """
        from modules.config import Config

        cfg = Config()
        assert cfg.source_threshold == pytest.approx(0.05)
        assert cfg.sink_threshold == pytest.approx(-0.05)


# =========================================================================
# F1b: Hysteresis on the balance-position fallback
# =========================================================================

class TestF1BalanceHysteresis:
    def test_named_band_constants(self):
        from modules import flow_analysis as fa

        assert fa.SINK_ENTER_OUTBOUND_RATIO == pytest.approx(0.78)
        assert fa.SINK_EXIT_OUTBOUND_RATIO == pytest.approx(0.72)
        assert fa.SOURCE_ENTER_OUTBOUND_RATIO == pytest.approx(0.22)
        assert fa.SOURCE_EXIT_OUTBOUND_RATIO == pytest.approx(0.28)

    def _run_cycles(self, analyzer, balance_ratios, initial_state,
                    capacity=10_000_000, kalman_ratio=0.0):
        """Run _calculate_metrics over a series of balance positions,
        threading the previous class like the analysis loop does."""
        states = []
        prev = initial_state
        now = int(time.time())
        for ratio in balance_ratios:
            metrics = analyzer._calculate_metrics(
                channel_id="100x1x0",
                peer_id="peer1",
                sats_in=50_000,
                sats_out=50_000,
                capacity=capacity,
                ema_in=50_000 / 7.0,
                ema_out=50_000 / 7.0,
                our_balance=int(capacity * ratio),
                forward_count=10,
                last_forward_ts=now,
                previous_state=prev,
                previous_kalman_ratio=kalman_ratio,
            )
            states.append(metrics.state)
            prev = metrics.state.value
        return states

    def test_boundary_hover_holds_class_20_cycles_from_balanced(self):
        """A channel hovering at 0.75±0.02 with weak flow never flaps."""
        analyzer, _ = _make_analyzer()
        ratios = [0.75 + (0.02 if i % 2 else -0.02) for i in range(20)]
        states = self._run_cycles(analyzer, ratios, initial_state="balanced")
        assert len(set(states)) == 1, f"class flapped: {[s.value for s in states]}"

    def test_boundary_hover_holds_sink_class_20_cycles(self):
        """A SINK hovering at 0.75±0.02 stays SINK (exit band is 0.72)."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        ratios = [0.75 + (0.02 if i % 2 else -0.02) for i in range(20)]
        states = self._run_cycles(analyzer, ratios, initial_state="sink")
        assert all(s == ChannelState.SINK for s in states)

    def test_source_boundary_hover_holds_20_cycles(self):
        """A SOURCE hovering at 0.25±0.02 stays SOURCE (exit band is 0.28)."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        ratios = [0.25 + (0.02 if i % 2 else -0.02) for i in range(20)]
        states = self._run_cycles(analyzer, ratios, initial_state="source")
        assert all(s == ChannelState.SOURCE for s in states)

    def test_decisive_moves_still_transition(self):
        """Entering and exiting the bands decisively still reclassifies."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        # balanced -> fills past 0.78 -> SINK; drains below 0.72 -> not SINK
        states = self._run_cycles(
            analyzer, [0.50, 0.80, 0.75, 0.70], initial_state="balanced"
        )
        assert states[0].is_balanced
        assert states[1] == ChannelState.SINK
        assert states[2] == ChannelState.SINK  # hysteresis hold
        assert states[3] != ChannelState.SINK

    def test_balanced_channel_at_076_not_sink_without_hysteresis_entry(self):
        """0.76 is above the old 0.75 cutoff but below the 0.78 entry band."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        states = self._run_cycles(analyzer, [0.76], initial_state="balanced")
        assert states[0] != ChannelState.SINK


# =========================================================================
# F1c: Kalman direction veto
# =========================================================================

class TestF1KalmanVeto:
    def test_draining_full_channel_is_not_sink(self):
        """outbound 0.85 would be SINK, but kalman says draining (+0.06)."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        state = analyzer._classify_balance_position(
            outbound_ratio=0.85, previous_state="balanced",
            kalman_ratio=0.06, turnover=0.0,
        )
        assert state != ChannelState.SINK

    def test_filling_empty_channel_is_not_source(self):
        """outbound 0.15 would be SOURCE, but kalman says filling (-0.06)."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        state = analyzer._classify_balance_position(
            outbound_ratio=0.15, previous_state="balanced",
            kalman_ratio=-0.06, turnover=0.0,
        )
        assert state != ChannelState.SOURCE

    def test_weak_kalman_does_not_veto(self):
        """|kalman_ratio| below the veto threshold leaves the label intact."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        assert analyzer._classify_balance_position(
            outbound_ratio=0.85, previous_state="balanced",
            kalman_ratio=0.03, turnover=0.0,
        ) == ChannelState.SINK
        assert analyzer._classify_balance_position(
            outbound_ratio=0.15, previous_state="balanced",
            kalman_ratio=-0.03, turnover=0.0,
        ) == ChannelState.SOURCE

    def test_veto_applies_in_calculate_metrics(self):
        """The EMA-path balance fallback uses the previous Kalman ratio."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        capacity = 10_000_000
        metrics = analyzer._calculate_metrics(
            channel_id="100x1x0",
            peer_id="peer1",
            sats_in=500_000,
            sats_out=200_000,
            capacity=capacity,
            ema_in=500_000 / 7.0,
            ema_out=200_000 / 7.0,
            our_balance=8_500_000,  # 85% full — would be SINK
            forward_count=50,
            last_forward_ts=int(time.time()),
            previous_state="balanced",
            previous_kalman_ratio=0.08,  # but it's draining
        )
        assert metrics.state != ChannelState.SINK


# =========================================================================
# F1: Genuine regime change still transitions promptly via Kalman path
# =========================================================================

class TestF1RegimeChangeResponsiveness:
    def test_regime_change_transitions_within_3_cycles(self):
        """A converged near-zero filter hit by a strong sustained drain
        crosses the +0.05 source threshold within ~3 cycles."""
        from modules.flow_analysis import (
            FlowMetrics, ChannelState, KalmanFlowFilter, KalmanFlowState,
        )

        analyzer, _ = _make_analyzer()
        capacity = 10_000_000
        now = int(time.time())

        # Seed a converged, stable filter at flow_ratio 0.0
        kf = KalmanFlowFilter(KalmanFlowState(
            flow_ratio=0.0, flow_velocity=0.0,
            variance_ratio=0.01, variance_velocity=0.01,
            covariance=0.0, last_update=now - 3600,
            innovation_variance=0.01, observation_count=20,
        ))
        analyzer._kalman_filters["100x1x0"] = kf

        # Strong genuine drain: 30% of capacity per day net outflow
        raw_entries = [
            {"timestamp": now - i * 60, "net_msat": 0.3 * capacity * 1000 / 30}
            for i in range(30)
        ]

        state_after = None
        for cycle in range(3):
            metrics = FlowMetrics(
                channel_id="100x1x0", peer_id="peer1",
                sats_in=0, sats_out=3_000_000, capacity=capacity,
                flow_ratio=0.0, state=ChannelState.BALANCED,
                daily_volume=3_000_000, forward_count=30,
            )
            analyzer._apply_kalman_reclassification(
                metrics=metrics,
                channel_id="100x1x0",
                capacity=capacity,
                our_balance=5_000_000,
                channel_daily=[],
                raw_entries=raw_entries,
                last_forward_ts=now,
                previous_state="balanced",
            )
            state_after = metrics.state
            if state_after == ChannelState.SOURCE:
                break
            # advance the clock one cycle
            kf.state.last_update = int(time.time()) - 3600

        assert state_after == ChannelState.SOURCE, (
            f"regime change not detected within 3 cycles "
            f"(kalman_ratio={kf.state.flow_ratio:.4f})"
        )

    def test_kalman_path_holds_class_while_hovering(self):
        """Kalman-converged path: SINK hovering 0.73-0.77 with ~zero flow
        holds its class across 20 cycles."""
        from modules.flow_analysis import (
            FlowMetrics, ChannelState, KalmanFlowFilter, KalmanFlowState,
        )

        analyzer, _ = _make_analyzer()
        capacity = 10_000_000
        now = int(time.time())
        kf = KalmanFlowFilter(KalmanFlowState(
            flow_ratio=0.0, flow_velocity=0.0,
            variance_ratio=0.01, variance_velocity=0.001,
            covariance=0.0, last_update=now - 3600,
            innovation_variance=0.01, observation_count=20,
        ))
        analyzer._kalman_filters["100x1x0"] = kf

        prev = "sink"
        states = []
        for i in range(20):
            balance_ratio = 0.75 + (0.02 if i % 2 else -0.02)
            metrics = FlowMetrics(
                channel_id="100x1x0", peer_id="peer1",
                sats_in=25_000, sats_out=25_000, capacity=capacity,
                flow_ratio=0.0, state=ChannelState.BALANCED,
                daily_volume=7_000, forward_count=10,
            )
            analyzer._apply_kalman_reclassification(
                metrics=metrics,
                channel_id="100x1x0",
                capacity=capacity,
                our_balance=int(capacity * balance_ratio),
                channel_daily=[],
                raw_entries=[{"timestamp": now, "net_msat": 0}],
                last_forward_ts=now,
                previous_state=prev,
            )
            states.append(metrics.state)
            prev = metrics.state.value

        assert all(s == ChannelState.SINK for s in states), (
            f"class flapped: {[s.value for s in states]}"
        )


# =========================================================================
# F2: estimate_depletion_hours — correct units
# =========================================================================

class TestF2DepletionHours:
    """The audit found predicted_depletion_hours off by ~6.7x (unit error).

    kalman_ratio is net drain per DAY as a fraction of capacity, so
    drain_sats_per_day = ratio * capacity and hours = local/drain * 24.
    """

    def test_audit_table_case(self):
        """10M cap, ratio 0.12 → 1.2M sats/day drain; 5M local → 100h."""
        from modules.flow_analysis import estimate_depletion_hours

        hours = estimate_depletion_hours(
            local_sats=5_000_000,
            capacity_sats=10_000_000,
            kalman_ratio=0.12,
            kalman_velocity=0.0,
        )
        assert hours == pytest.approx(100.0)
        # Regression guard against the old 667h figure
        assert hours < 200.0

    def test_full_local_balance(self):
        """10M local at 1.2M/day → 200h."""
        from modules.flow_analysis import estimate_depletion_hours

        hours = estimate_depletion_hours(10_000_000, 10_000_000, 0.12, 0.0)
        assert hours == pytest.approx(200.0)

    def test_no_drain_returns_none(self):
        from modules.flow_analysis import estimate_depletion_hours

        assert estimate_depletion_hours(5_000_000, 10_000_000, 0.0, 0.0) is None

    def test_filling_channel_returns_none(self):
        """Negative ratio = filling; no depletion forecast."""
        from modules.flow_analysis import estimate_depletion_hours

        assert estimate_depletion_hours(5_000_000, 10_000_000, -0.2, 0.0) is None

    def test_velocity_term_accelerates_depletion(self):
        """Positive velocity (drain accelerating) shortens the estimate."""
        from modules.flow_analysis import estimate_depletion_hours

        base = estimate_depletion_hours(5_000_000, 10_000_000, 0.10, 0.0)
        accel = estimate_depletion_hours(5_000_000, 10_000_000, 0.10, 0.01)
        # drain = 1M + 0.01*24*10M/2 = 1M + 1.2M = 2.2M/day → 5/2.2*24 ≈ 54.5h
        assert accel == pytest.approx(5_000_000 / 2_200_000 * 24.0)
        assert accel < base

    def test_negative_velocity_clamps_drain_at_zero(self):
        """Strong deceleration cannot produce a negative drain (→ None)."""
        from modules.flow_analysis import estimate_depletion_hours

        assert estimate_depletion_hours(
            5_000_000, 10_000_000, 0.01, -0.05
        ) is None

    def test_invalid_inputs_return_none(self):
        from modules.flow_analysis import estimate_depletion_hours

        assert estimate_depletion_hours(5_000_000, 0, 0.1, 0.0) is None
        assert estimate_depletion_hours(-1, 10_000_000, 0.1, 0.0) is None
        assert estimate_depletion_hours(
            5_000_000, 10_000_000, float("nan"), 0.0
        ) is None

    def test_zero_local_depletes_immediately(self):
        from modules.flow_analysis import estimate_depletion_hours

        hours = estimate_depletion_hours(0, 10_000_000, 0.12, 0.0)
        assert hours == pytest.approx(0.0)
