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


from modules.fee_authority import FeeAuthorityGate

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
        # (DORMANT is acceptable here — weak flow at 50% balance; F6)
        assert states[0] not in (ChannelState.SINK, ChannelState.SOURCE)
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


# =========================================================================
# F6: 'dormant' vocabulary actually emitted by the classifier
# =========================================================================

class TestF6DormantVocabulary:
    """fee_controller branches on flow_state 'dormant' (rebalance-cost floor
    exemption, context role) but the producer never emitted it — such
    channels landed BALANCED and the branches were dead."""

    def test_dormant_enum_roundtrip(self):
        from modules.flow_analysis import ChannelState

        assert ChannelState.DORMANT.value == "dormant"
        assert ChannelState("dormant") == ChannelState.DORMANT
        assert ChannelState.DORMANT.is_balanced is False

    def test_classifier_emits_dormant(self):
        """turnover < 1%/day AND |kalman_ratio| < 0.01 -> DORMANT."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        state = analyzer._classify_balance_position(
            outbound_ratio=0.5, previous_state="balanced",
            kalman_ratio=0.005, turnover=0.005,
        )
        assert state == ChannelState.DORMANT

    def test_net_flow_blocks_dormant(self):
        """|kalman_ratio| >= 0.01 keeps the channel BALANCED."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        state = analyzer._classify_balance_position(
            outbound_ratio=0.5, previous_state="balanced",
            kalman_ratio=0.02, turnover=0.005,
        )
        assert state == ChannelState.BALANCED

    def test_turnover_blocks_dormant(self):
        """Busy two-way channels stay BALANCED_ACTIVE."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        state = analyzer._classify_balance_position(
            outbound_ratio=0.5, previous_state="balanced",
            kalman_ratio=0.0, turnover=0.02,
        )
        assert state == ChannelState.BALANCED_ACTIVE

    def test_calculate_metrics_emits_dormant(self):
        """A 50%-balanced channel with negligible turnover is DORMANT."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        capacity = 10_000_000
        metrics = analyzer._calculate_metrics(
            channel_id="100x1x0",
            peer_id="peer1",
            sats_in=10_000,
            sats_out=10_000,
            capacity=capacity,
            ema_in=10_000 / 7.0,
            ema_out=10_000 / 7.0,
            our_balance=5_000_000,
            forward_count=3,
            last_forward_ts=int(time.time()),
            previous_state="balanced",
            previous_kalman_ratio=0.0,
        )
        assert metrics.state == ChannelState.DORMANT

    def test_fee_side_rebalance_floor_exemption_activates(self):
        """The dormant rebalance-cost floor exemption short-circuits before
        any cost-history lookup (no flow to amortize costs against)."""
        from modules.fee_controller import FeeController
        from modules.config import Config

        plugin = MagicMock()
        config = MagicMock(spec=Config)
        db = MagicMock()
        fc = FeeController(plugin, config, db, fee_authority_gate=FeeAuthorityGate())

        assert fc._get_rebalance_cost_floor("100x1x0", "peer", "dormant") is None
        db.get_channel_cost_history.assert_not_called()

    def test_pid_target_ratio_defined_for_dormant(self):
        from modules.fee_controller import _PID_TARGET_RATIOS

        assert _PID_TARGET_RATIOS["dormant"] == pytest.approx(0.5)


# =========================================================================
# F7: no-flow fallback unified with hysteresis bands, no synthetic ratio
# =========================================================================

class TestF7NoFlowFallback:
    """The no-flow branch used 0.30/0.70 cutoffs (vs 0.25/0.75 with-flow)
    and wrote synthetic flow_ratio ±0.6 that contaminated EMA velocity."""

    def _metrics(self, analyzer, balance_ratio, previous_state="balanced",
                 capacity=10_000_000):
        return analyzer._calculate_metrics(
            channel_id="100x1x0",
            peer_id="peer1",
            sats_in=0,
            sats_out=0,
            capacity=capacity,
            ema_in=0.0,
            ema_out=0.0,
            our_balance=int(capacity * balance_ratio),
            forward_count=0,
            last_forward_ts=0,
            previous_state=previous_state,
            previous_kalman_ratio=0.0,
        )

    def test_no_flow_uses_unified_hysteresis_bands(self):
        """0.75 outbound with no flow was SINK under the old 0.70 cutoff;
        the unified bands require >0.78 to enter SINK."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        assert self._metrics(analyzer, 0.75).state != ChannelState.SINK
        assert self._metrics(analyzer, 0.80).state == ChannelState.SINK
        assert self._metrics(analyzer, 0.25).state != ChannelState.SOURCE
        assert self._metrics(analyzer, 0.20).state == ChannelState.SOURCE

    def test_no_flow_does_not_write_synthetic_flow_ratio(self):
        """flow_ratio stays 0.0 — synthetic ±0.6 polluted the persisted
        history and the EMA velocity calculation."""
        analyzer, _ = _make_analyzer()
        for balance_ratio in (0.10, 0.50, 0.90):
            metrics = self._metrics(analyzer, balance_ratio)
            assert metrics.flow_ratio == pytest.approx(0.0), (
                f"synthetic flow_ratio at balance {balance_ratio}"
            )

    def test_no_flow_hysteresis_holds_class(self):
        """No-flow channel hovering around 0.75 holds its class."""
        analyzer, _ = _make_analyzer()
        prev = "sink"
        states = []
        for i in range(20):
            m = self._metrics(analyzer, 0.75 + (0.02 if i % 2 else -0.02),
                              previous_state=prev)
            states.append(m.state)
            prev = m.state.value
        assert len(set(states)) == 1

    def test_no_flow_midrange_is_dormant(self):
        """An idle 50% channel with no flow at all is DORMANT (F6)."""
        from modules.flow_analysis import ChannelState

        analyzer, _ = _make_analyzer()
        assert self._metrics(analyzer, 0.50).state == ChannelState.DORMANT


# =========================================================================
# F5: temporal profile graduation accounting honesty
# =========================================================================

def _histogram(total_count, total_out=100_000):
    """24-bucket histogram spreading counts/sats evenly."""
    return [
        {"out_sats": total_out / 24.0, "in_sats": 0, "count": total_count / 24.0}
        for _ in range(24)
    ]


class TestF5TemporalAccounting:
    """avg_daily_forwards was the 7-day TOTAL compared against a daily
    threshold, and observation_days incremented once per HOURLY cycle —
    profiles 'graduated' ~24x too fast on ~7x too little traffic."""

    def test_avg_daily_forwards_is_daily_not_window_total(self):
        """35 forwards over the 7-day window = 5/day < 10 threshold:
        observation_days must NOT advance (the old total of 35 passed)."""
        import json

        analyzer, db = _make_analyzer()
        db.load_temporal_profile.return_value = None
        row = analyzer._update_temporal_profile(
            "100x1x0", histogram=_histogram(35), fee_state={}, defer_save=True
        )
        profile = json.loads(row["profile_json"])
        assert profile["observation_days"] == 0

    def test_sufficient_daily_traffic_advances_observation_days(self):
        """105 forwards over 7 days = 15/day >= 10 threshold."""
        import json

        analyzer, db = _make_analyzer()
        db.load_temporal_profile.return_value = None
        row = analyzer._update_temporal_profile(
            "100x1x0", histogram=_histogram(105), fee_state={}, defer_save=True
        )
        profile = json.loads(row["profile_json"])
        assert profile["observation_days"] == 1

    def test_observation_days_increment_once_per_day(self):
        """24 hourly cycles within the same day add at most ONE observation
        day (the old code added 24)."""
        from modules.flow_analysis import (
            TemporalProfile, update_temporal_profile,
        )

        profile = TemporalProfile()
        for _ in range(24):
            profile = update_temporal_profile(
                profile, _histogram(105), daily_forwards=15
            )
        assert profile.observation_days == 1

    def test_observation_days_advance_on_new_day(self):
        from modules.flow_analysis import (
            TemporalProfile, update_temporal_profile,
        )

        profile = TemporalProfile()
        profile = update_temporal_profile(
            profile, _histogram(105), daily_forwards=15
        )
        assert profile.observation_days == 1
        # Pretend the last qualifying observation was yesterday
        profile.last_observation_day -= 1
        profile = update_temporal_profile(
            profile, _histogram(105), daily_forwards=15
        )
        assert profile.observation_days == 2

    def test_last_observation_day_survives_roundtrip(self):
        from modules.flow_analysis import TemporalProfile

        profile = TemporalProfile()
        profile.observation_days = 3
        profile.last_observation_day = 20_000
        restored = TemporalProfile.from_dict(profile.to_dict())
        assert restored.observation_days == 3
        assert restored.last_observation_day == 20_000

    def test_graduation_still_requires_seven_days(self):
        """End-to-end: 7 qualifying days graduate the profile; the hourly
        cycle alone cannot."""
        from modules.flow_analysis import (
            TemporalProfile, update_temporal_profile,
        )

        profile = TemporalProfile()
        for day in range(7):
            # several cycles within the same day
            for _ in range(3):
                profile = update_temporal_profile(
                    profile, _histogram(105), daily_forwards=15
                )
            assert profile.observation_days == day + 1
            assert profile.graduated == (day + 1 >= 7)
            profile.last_observation_day -= 1  # advance to "next day"
        assert profile.graduated
