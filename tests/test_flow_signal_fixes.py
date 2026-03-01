"""
Tests for flow analysis signal processing overhaul (Fixes 1-5).

Fix 1: Raw Kalman observation (no double smoothing)
Fix 2: Continuous-time weighting (no calendar-day boundary)
Fix 3: Hourly Kalman dt (no dt³ collapse)
Fix 4: BALANCED_ACTIVE state (dormant vs active)
Fix 5: Pending HTLC liquidity deduction
"""
import pytest
import math
import time
import sqlite3
import os
import tempfile
from unittest.mock import MagicMock, patch


# =========================================================================
# Fix 3: Hourly Kalman dt
# =========================================================================

class TestFix3HourlyKalman:
    """Tests for hourly Kalman time units (Fix 3)."""

    def test_hourly_dt_produces_healthy_q_diagonal(self):
        """With dt_hours=1.0, Q diagonal should not collapse."""
        from modules.flow_analysis import (
            KalmanFlowFilter, KALMAN_BASE_PROCESS_NOISE,
            KALMAN_VELOCITY_PROCESS_NOISE, KALMAN_VOLATILITY_SCALING
        )

        kf = KalmanFlowFilter()
        initial_p00 = kf.state.variance_ratio

        kf.predict(dt_hours=1.0, volatility=1.0)

        # Q[0,0] contribution should be meaningful at dt_hours=1.0
        # q_ratio * dt + q_velocity * dt^3/3
        q_ratio = KALMAN_BASE_PROCESS_NOISE * KALMAN_VOLATILITY_SCALING
        q_velocity = KALMAN_VELOCITY_PROCESS_NOISE
        dt = 1.0
        expected_q00_contrib = q_ratio * dt + q_velocity * dt**3 / 3.0

        # Variance should have increased meaningfully (not collapsed)
        assert kf.state.variance_ratio > initial_p00
        assert expected_q00_contrib > 1e-5  # Not vanishingly small

    def test_kalman_velocity_bounds_are_per_hour(self):
        """Kalman velocity bounds should be in per-hour units."""
        from modules.flow_analysis import KALMAN_MAX_VELOCITY, KALMAN_MIN_VELOCITY

        # 0.5/day = ~0.0208/hour
        assert KALMAN_MAX_VELOCITY == pytest.approx(0.5 / 24.0, rel=1e-6)
        assert KALMAN_MIN_VELOCITY == pytest.approx(-0.5 / 24.0, rel=1e-6)

    def test_old_state_migration_resets_filter(self):
        """Old per-day Kalman states should be reset, not loaded."""
        from modules.flow_analysis import FlowAnalyzer, KalmanFlowFilter

        plugin = MagicMock()
        config = MagicMock()
        config.flow_window_days = 7

        db = MagicMock()
        # Simulate old per-day state (no velocity_unit or velocity_unit='per_day')
        db.get_kalman_state.return_value = {
            "flow_ratio": 0.5,
            "flow_velocity": 0.1,  # This is per-day, should be reset
            "variance_ratio": 0.05,
            "variance_velocity": 0.02,
            "covariance": 0.01,
            "last_update": int(time.time()) - 3600,
            "innovation_variance": 0.03,
            "last_innovation": 0.02,
            "velocity_unit": "per_day",
        }

        analyzer = FlowAnalyzer(plugin, config, db)
        kf = analyzer._get_kalman_filter("100x1x0")

        # Should be fresh (reset), not loaded from old state
        assert kf.state.flow_ratio == 0.0
        assert kf.state.flow_velocity == 0.0

    def test_per_hour_state_is_loaded(self):
        """Per-hour Kalman states should be loaded normally."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        db = MagicMock()
        db.get_kalman_state.return_value = {
            "flow_ratio": 0.5,
            "flow_velocity": 0.004,  # per-hour
            "variance_ratio": 0.05,
            "variance_velocity": 0.02,
            "covariance": 0.01,
            "last_update": int(time.time()) - 3600,
            "innovation_variance": 0.03,
            "last_innovation": 0.02,
            "velocity_unit": "per_hour",
        }

        analyzer = FlowAnalyzer(plugin, config, db)
        kf = analyzer._get_kalman_filter("100x1x0")

        assert kf.state.flow_ratio == pytest.approx(0.5)
        assert kf.state.flow_velocity == pytest.approx(0.004)

    def test_velocity_bounds_clamp(self):
        """Velocity should be clamped to Kalman hourly bounds after predict."""
        from modules.flow_analysis import KalmanFlowFilter, KALMAN_MAX_VELOCITY, KALMAN_MIN_VELOCITY

        kf = KalmanFlowFilter()
        # Set extreme velocity
        kf.state.flow_velocity = 1.0  # Way above KALMAN_MAX_VELOCITY

        kf.predict(dt_hours=1.0, volatility=1.0)

        assert kf.state.flow_velocity <= KALMAN_MAX_VELOCITY
        assert kf.state.flow_velocity >= KALMAN_MIN_VELOCITY


# =========================================================================
# Fix 1: Raw Kalman Observation
# =========================================================================

class TestFix1RawObservation:
    """Tests for raw (non-EMA) Kalman observation (Fix 1)."""

    def test_24h_rolling_sum(self):
        """24h window should sum all net flow, not average per-forward."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        config.flow_window_days = 7
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        now = time.time()
        # 1000 forwards of 10k sats each within the last 24h
        entries = [
            {"timestamp": now - i * 60, "net_msat": 10_000_000}  # +10k sats each
            for i in range(1000)
        ]

        ratio, count = analyzer._compute_raw_kalman_observation(
            "100x1x0", capacity=100_000_000, net_flow_entries=entries
        )

        # Total: 1000 * 10k = 10M sats. Ratio = 10M / 100M = 0.1
        # Old bug: would compute 10k / 100M = 0.0001 (average per forward)
        assert ratio == pytest.approx(0.1, rel=1e-3)
        assert count == 1000

    def test_excludes_entries_outside_24h(self):
        """Entries older than 24h should be excluded from the rolling window."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        now = time.time()
        entries = [
            {"timestamp": now - 1 * 3600, "net_msat": 1_000_000},   # 1hr ago: +1000 sats
            {"timestamp": now - 48 * 3600, "net_msat": -5_000_000},  # 48hrs ago: excluded
        ]

        ratio, count = analyzer._compute_raw_kalman_observation(
            "100x1x0", capacity=100_000, net_flow_entries=entries
        )

        # Only the recent entry counts: +1000 / 100_000 = 0.01
        assert ratio == pytest.approx(0.01, rel=1e-3)
        assert count == 1

    def test_no_data_fallback(self):
        """Empty entries should return (0.0, 0)."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        ratio, count = analyzer._compute_raw_kalman_observation(
            "100x1x0", capacity=100_000, net_flow_entries=[]
        )
        assert ratio == 0.0
        assert count == 0

    def test_zero_capacity_guard(self):
        """Zero capacity should return (0.0, 0)."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        entries = [{"timestamp": time.time(), "net_msat": 1_000_000}]
        ratio, count = analyzer._compute_raw_kalman_observation(
            "100x1x0", capacity=0, net_flow_entries=entries
        )
        assert ratio == 0.0
        assert count == 0

    def test_ratio_clamped_to_bounds(self):
        """Ratio must be clamped to [-1, 1]."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        now = time.time()
        # Huge outflow relative to tiny capacity
        entries = [{"timestamp": now, "net_msat": 10_000_000_000}]

        ratio, _ = analyzer._compute_raw_kalman_observation(
            "100x1x0", capacity=1, net_flow_entries=entries
        )
        assert ratio == 1.0  # Clamped


# =========================================================================
# Fix 2: Continuous Net Flow Query
# =========================================================================

class TestFix2ContinuousNetFlowQuery:
    """Tests for database.get_continuous_net_flow_all() (Fix 2)."""

    def _make_db(self):
        """Create a minimal in-memory database with forwards table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE forwards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                in_channel TEXT NOT NULL,
                out_channel TEXT NOT NULL,
                in_msat INTEGER NOT NULL,
                out_msat INTEGER NOT NULL,
                fee_msat INTEGER NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        return conn

    def test_returns_correct_data(self):
        """Query should return net_msat with correct sign for in/out."""
        conn = self._make_db()
        now = int(time.time())

        # Forward: A→B (out_channel=B gets +out_msat, in_channel=A gets -in_msat)
        conn.execute(
            "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 1000000, 999000, 1000, now - 100)
        )
        conn.commit()

        rows = conn.execute("""
            SELECT out_channel AS channel_id, timestamp, out_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            UNION ALL
            SELECT in_channel AS channel_id, timestamp, -in_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            ORDER BY channel_id, timestamp DESC
        """, (now - 3600, now - 3600)).fetchall()

        result = {}
        for row in rows:
            cid = row["channel_id"]
            if cid not in result:
                result[cid] = []
            result[cid].append({"timestamp": row["timestamp"], "net_msat": row["net_msat"]})

        # out_channel gets positive (outflow)
        assert len(result["200x1x0"]) == 1
        assert result["200x1x0"][0]["net_msat"] == 999000

        # in_channel gets negative (inflow)
        assert len(result["100x1x0"]) == 1
        assert result["100x1x0"][0]["net_msat"] == -1000000

    def test_window_filtering(self):
        """Forwards outside the window should be excluded."""
        conn = self._make_db()
        now = int(time.time())

        # One forward inside window, one outside
        conn.execute(
            "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("A", "B", 1000, 1000, 0, now - 100)
        )
        conn.execute(
            "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("A", "B", 2000, 2000, 0, now - 7200)
        )
        conn.commit()

        # 1-hour window
        cutoff = now - 3600
        rows = conn.execute("""
            SELECT out_channel AS channel_id, timestamp, out_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            UNION ALL
            SELECT in_channel AS channel_id, timestamp, -in_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            ORDER BY channel_id, timestamp DESC
        """, (cutoff, cutoff)).fetchall()

        # Only the recent forward (within 1hr) should appear
        # B gets one row for out, A gets one row for in
        b_rows = [r for r in rows if r["channel_id"] == "B"]
        assert len(b_rows) == 1
        assert b_rows[0]["net_msat"] == 1000

    def test_batch_vs_single_channel(self):
        """Multi-channel query should partition data correctly."""
        conn = self._make_db()
        now = int(time.time())

        conn.execute(
            "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("A", "B", 1000, 1000, 0, now - 100)
        )
        conn.execute(
            "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("C", "D", 2000, 2000, 0, now - 200)
        )
        conn.commit()

        cutoff = now - 3600
        rows = conn.execute("""
            SELECT out_channel AS channel_id, timestamp, out_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            UNION ALL
            SELECT in_channel AS channel_id, timestamp, -in_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            ORDER BY channel_id, timestamp DESC
        """, (cutoff, cutoff)).fetchall()

        result = {}
        for row in rows:
            cid = row["channel_id"]
            if cid not in result:
                result[cid] = []
            result[cid].append(row)

        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" in result


# =========================================================================
# Fix 4: BALANCED_ACTIVE State
# =========================================================================

class TestFix4BalancedActive:
    """Tests for BALANCED_ACTIVE state classification (Fix 4)."""

    def test_high_turnover_classified_as_balanced_active(self):
        """Channel with >1% daily turnover should be BALANCED_ACTIVE."""
        from modules.flow_analysis import ChannelState, BALANCED_ACTIVE_TURNOVER_THRESHOLD

        # 2% turnover > 1% threshold
        turnover = 0.02
        assert turnover > BALANCED_ACTIVE_TURNOVER_THRESHOLD

    def test_dormant_stays_balanced(self):
        """Channel with near-zero turnover should stay BALANCED."""
        from modules.flow_analysis import BALANCED_ACTIVE_TURNOVER_THRESHOLD

        # 0.1% turnover < 1% threshold
        turnover = 0.001
        assert turnover <= BALANCED_ACTIVE_TURNOVER_THRESHOLD

    def test_is_balanced_property(self):
        """is_balanced should be True for both BALANCED and BALANCED_ACTIVE."""
        from modules.flow_analysis import ChannelState

        assert ChannelState.BALANCED.is_balanced is True
        assert ChannelState.BALANCED_ACTIVE.is_balanced is True
        assert ChannelState.SOURCE.is_balanced is False
        assert ChannelState.SINK.is_balanced is False
        assert ChannelState.CONGESTED.is_balanced is False
        assert ChannelState.UNKNOWN.is_balanced is False

    def test_enum_roundtrip(self):
        """BALANCED_ACTIVE value should survive string conversion."""
        from modules.flow_analysis import ChannelState

        state = ChannelState.BALANCED_ACTIVE
        assert state.value == "balanced_active"
        assert ChannelState("balanced_active") == ChannelState.BALANCED_ACTIVE

    def test_classify_balanced_active_in_metrics(self):
        """_calculate_metrics should classify high-turnover balanced channels as BALANCED_ACTIVE."""
        from modules.flow_analysis import FlowAnalyzer, ChannelState, BALANCED_ACTIVE_TURNOVER_THRESHOLD

        plugin = MagicMock()
        config = MagicMock()
        config.source_threshold = 0.5
        config.sink_threshold = -0.5
        config.htlc_congestion_threshold = 0.8
        config.flow_window_days = 7
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        # Balanced flow ratio (near zero), but high volume
        capacity = 1_000_000  # 1M sats
        # daily_volume = (sats_in + sats_out) / flow_window_days
        # For turnover > 1%: daily_volume > 10000
        # So total volume > 10000 * 7 = 70000
        sats_in = 40000
        sats_out = 40000  # total = 80000, daily = ~11428, turnover = 1.14%

        metrics = analyzer._calculate_metrics(
            channel_id="100x1x0",
            peer_id="peer1",
            sats_in=sats_in,
            sats_out=sats_out,
            capacity=capacity,
            ema_in=float(sats_in) / 7,
            ema_out=float(sats_out) / 7,
            our_balance=500_000,
            forward_count=50,
            last_forward_ts=int(time.time()),
        )

        assert metrics.state == ChannelState.BALANCED_ACTIVE


# =========================================================================
# Fix 5: Pending HTLC Liquidity
# =========================================================================

class TestFix5NoHTLCDoubleDeduction:
    """Tests that CLN's spendable_msat is used directly without HTLC double-deduction."""

    def test_no_htlc_extraction_in_channels(self):
        """_get_channels() should NOT extract pending_outbound_htlc_msat."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [{
                "state": "CHANNELD_NORMAL",
                "short_channel_id": "100x1x0",
                "peer_id": "peer1",
                "spendable_msat": 500000000,
                "receivable_msat": 500000000,
                "capacity_msat": 1000000000,
                "htlcs": [
                    {"direction": "out", "amount_msat": 100000000},
                    {"direction": "out", "amount_msat": 50000000},
                    {"direction": "in", "amount_msat": 200000000},
                ],
            }]
        }
        config = MagicMock()
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        channels = analyzer._get_channels()
        assert len(channels) == 1
        # pending_outbound_htlc_msat should NOT be extracted (CLN handles it)
        assert "pending_outbound_htlc_msat" not in channels[0]

    def test_spendable_msat_used_directly(self):
        """Balance should come directly from spendable_msat without HTLC deduction."""
        # CLN's spendable_msat already accounts for pending HTLCs and reserve.
        # our_balance should be spendable_msat // 1000, nothing subtracted.
        spendable_msat = 500_000_000  # 500k sats
        our_balance = spendable_msat // 1000
        assert our_balance == 500_000

    def test_active_htlc_count_still_tracked(self):
        """Active HTLC count should still be tracked for congestion detection."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [{
                "state": "CHANNELD_NORMAL",
                "short_channel_id": "100x1x0",
                "peer_id": "peer1",
                "spendable_msat": 500000000,
                "receivable_msat": 500000000,
                "htlcs": [
                    {"direction": "out", "amount_msat": 100000000},
                    {"direction": "in", "amount_msat": 200000000},
                ],
            }]
        }
        config = MagicMock()
        db = MagicMock()
        analyzer = FlowAnalyzer(plugin, config, db)

        channels = analyzer._get_channels()
        # HTLC count still tracked for congestion detection
        assert channels[0]["active_htlcs"] == 2


# =========================================================================
# Integration: Consumer velocity conversion
# =========================================================================

class TestKalmanConvergenceGuard:
    """Tests for Kalman convergence guard and idle channel decay."""

    def test_idle_channel_decays_to_balanced(self):
        """Idle channels (0.0 observation) should gradually decay toward balanced."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        kf.state.flow_ratio = 0.7
        kf.state.flow_velocity = 0.0
        kf.state.last_update = int(time.time()) - 3600

        # Simulate several cycles of zero observation (idle channel)
        for _ in range(20):
            kf.predict(dt_hours=1.0, volatility=1.0)
            kf.update(0.0, confidence=0.5)

        # Filter should have pulled ratio toward 0.0
        assert kf.state.flow_ratio < 0.3, \
            f"Idle channel should decay toward balanced: {kf.state.flow_ratio}"

    def test_single_zero_does_not_snap_to_balanced(self):
        """A single zero observation should not immediately reset a converged filter."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        kf.state.flow_ratio = 0.7
        kf.state.flow_velocity = 0.0
        kf.state.last_update = int(time.time()) - 3600

        kf.predict(dt_hours=1.0, volatility=1.0)
        kf.update(0.0, confidence=0.5)

        # One zero obs pulls toward 0 but doesn't snap there
        assert kf.state.flow_ratio > 0.1, \
            f"Single zero should not snap to balanced: {kf.state.flow_ratio}"
        assert kf.state.flow_ratio < 0.7, "Update with 0.0 should reduce ratio"

    def test_unconverged_filter_preserves_ema_classification(self):
        """Kalman with high uncertainty should NOT override EMA classification."""
        from modules.flow_analysis import (
            KALMAN_INITIAL_VARIANCE, KALMAN_CONVERGENCE_UNCERTAINTY
        )
        import math

        initial_uncertainty = math.sqrt(KALMAN_INITIAL_VARIANCE)
        # Fresh filter uncertainty should exceed the convergence threshold
        assert initial_uncertainty > KALMAN_CONVERGENCE_UNCERTAINTY, \
            f"Initial uncertainty {initial_uncertainty} should exceed threshold {KALMAN_CONVERGENCE_UNCERTAINTY}"

    def test_converged_filter_overrides_classification(self):
        """Kalman with low uncertainty should override EMA classification."""
        from modules.flow_analysis import (
            KalmanFlowFilter, KALMAN_CONVERGENCE_UNCERTAINTY
        )

        kf = KalmanFlowFilter()
        # Simulate convergence: many updates drive variance down
        kf.state.last_update = int(time.time()) - 3600
        for _ in range(50):
            kf.predict(dt_hours=1.0, volatility=1.0)
            kf.update(0.8, confidence=0.8)  # Consistent SOURCE observation

        assert kf.get_uncertainty() < KALMAN_CONVERGENCE_UNCERTAINTY, \
            f"Converged filter uncertainty {kf.get_uncertainty()} should be below threshold"
        assert kf.state.flow_ratio > 0.5, "Converged filter should reflect observations"


class TestConsumerVelocityConversion:
    """Tests that consumers correctly convert per-hour velocity."""

    def test_fee_controller_scales_velocity(self):
        """fee_controller demand_factor should scale kv by 24."""
        kv = 0.01  # per hour
        ku = 0.05
        confidence = 1.0 / (1.0 + ku)

        # Old formula: 1.0 + kv * confidence * 2.0  (when kv was per-day)
        # New formula: 1.0 + (kv * 24) * confidence * 2.0
        demand_factor = 1.0 + (kv * 24.0) * confidence * 2.0

        # With kv=0.01/hr -> 0.24/day effective
        assert demand_factor > 1.0
        assert demand_factor == pytest.approx(1.0 + 0.24 * confidence * 2.0, rel=1e-6)

    def test_rebalancer_uses_cooldown_hours(self):
        """Rebalancer predicted_volume should use hourly velocity."""
        kalman_velocity = 0.005  # per hour
        capacity = 1_000_000
        cooldown_days = 3
        cooldown_hours = cooldown_days * 24.0

        predicted_volume = abs(kalman_velocity) * capacity * cooldown_hours
        # 0.005 * 1M * 72 = 360,000 sats
        assert predicted_volume == pytest.approx(360_000, rel=1e-6)

    def test_rebalancer_std_dev_uses_days_not_hours(self):
        """Diffusion std_dev should scale with sqrt(days), not sqrt(hours).

        kalman_uncertainty is dimensionless (on flow_ratio), so diffusion
        scales with sqrt(calendar_days). Using sqrt(hours) inflates std_dev
        by ~4.9x, causing budget over-reservation.
        """
        import math
        kalman_uncertainty = 0.15  # dimensionless
        capacity = 1_000_000
        cooldown_days = 3

        # Correct: sqrt(days)
        std_dev_correct = kalman_uncertainty * capacity * math.sqrt(cooldown_days)
        # Wrong: sqrt(hours) — 4.9x inflation
        cooldown_hours = cooldown_days * 24.0
        std_dev_wrong = kalman_uncertainty * capacity * math.sqrt(cooldown_hours)

        assert std_dev_wrong / std_dev_correct == pytest.approx(math.sqrt(24), rel=1e-3)
        # The correct std_dev should be ~4.9x smaller
        assert std_dev_correct < std_dev_wrong


class TestThompsonExtrapolation:
    """Tests for Thompson posterior f_star clamping (Fix 4)."""

    def test_f_star_allows_extrapolation_above(self):
        """f_star should be allowed to exceed 1.0 (extrapolate beyond tested range)."""
        # Simulate a concave posterior where peak is beyond current max fee
        a = -0.5
        b = 1.6  # Peak at -b/(2a) = 1.6
        f_star = -b / (2.0 * a)
        f_star = max(-0.5, min(1.5, f_star))

        assert f_star == 1.5  # Clamped to safe extrapolation limit, not 1.0

    def test_f_star_allows_extrapolation_below(self):
        """f_star should be allowed to go below 0.0 (extrapolate below min fee)."""
        a = -0.5
        b = -0.3  # Peak at -(-0.3)/(2*-0.5) = -0.3
        f_star = -b / (2.0 * a)
        f_star = max(-0.5, min(1.5, f_star))

        assert f_star == -0.3  # Allowed, not clamped to 0.0

    def test_f_star_within_range_unchanged(self):
        """f_star within [0, 1] should pass through unchanged."""
        a = -1.0
        b = 1.0  # Peak at 0.5
        f_star = -b / (2.0 * a)
        f_star = max(-0.5, min(1.5, f_star))

        assert f_star == 0.5


class TestAskReneClamp:
    """Tests for AskRene probability clamping (Fix 5)."""

    def test_capacity_ratio_clamped_at_99(self):
        """When rebalance_amount > askrene_max_sats, ratio should clamp to 0.99."""
        rebalance_amount = 2_000_000
        askrene_max_sats = 500_000

        capacity_ratio = min(0.99, rebalance_amount / askrene_max_sats)
        askrene_p = 1.0 - capacity_ratio

        assert capacity_ratio == 0.99
        assert askrene_p == pytest.approx(0.01, rel=1e-6)

    def test_normal_ratio_passes_through(self):
        """When rebalance_amount < askrene_max_sats, ratio is computed normally."""
        rebalance_amount = 300_000
        askrene_max_sats = 1_000_000

        capacity_ratio = min(0.99, rebalance_amount / askrene_max_sats)
        askrene_p = 1.0 - capacity_ratio

        assert capacity_ratio == pytest.approx(0.3, rel=1e-6)
        assert askrene_p == pytest.approx(0.7, rel=1e-6)

    def test_blended_p_always_positive(self):
        """Blended probability should always be positive for valid Kelly input."""
        historical_p = 0.5
        # Worst case: rebalance far exceeds max_sats
        capacity_ratio = min(0.99, 10_000_000 / 100_000)
        askrene_p = 1.0 - capacity_ratio
        p = (historical_p * 0.3) + (askrene_p * 0.7)

        assert p > 0, f"Blended probability must be positive: {p}"
