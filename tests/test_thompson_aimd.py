"""
Unit tests for Thompson Sampling + AIMD fee optimization algorithm.

Tests the new GaussianThompsonState, AIMDDefenseState, and ThompsonAIMDState
classes that replace Hill Climbing as the primary fee optimization algorithm.
"""
import pytest
import time


# Test data fixtures
@pytest.fixture
def gaussian_thompson_state():
    """Create a GaussianThompsonState for testing."""
    # Import here to avoid import errors if modules aren't installed
    from modules.fee_controller import GaussianThompsonState
    return GaussianThompsonState()


@pytest.fixture
def aimd_defense_state():
    """Create an AIMDDefenseState for testing."""
    from modules.fee_controller import AIMDDefenseState
    return AIMDDefenseState()


@pytest.fixture
def thompson_aimd_state():
    """Create a ThompsonAIMDState for testing."""
    from modules.fee_controller import ThompsonAIMDState
    return ThompsonAIMDState()


class TestGaussianThompsonState:
    """Tests for GaussianThompsonState class."""

    def test_default_initialization(self, gaussian_thompson_state):
        """Test default state initialization."""
        state = gaussian_thompson_state

        assert state.prior_mean_fee == 200
        assert state.prior_std_fee == 100
        assert state.posterior_mean == 200.0
        assert state.posterior_std == 100.0
        assert len(state.observations) == 0
        assert len(state.contextual_posteriors) == 0

    def test_sample_fee_within_bounds(self, gaussian_thompson_state):
        """Test that sampled fee respects floor and ceiling."""
        state = gaussian_thompson_state

        for _ in range(100):
            fee = state.sample_fee(floor=50, ceiling=500)
            assert 50 <= fee <= 500

    def test_sample_fee_exploration_with_few_observations(self, gaussian_thompson_state):
        """Test that Thompson explores more when few observations."""
        state = gaussian_thompson_state

        # With no observations, should use prior with extra exploration
        fees = [state.sample_fee(floor=50, ceiling=500) for _ in range(50)]

        # Should have some variance in sampled fees (exploration)
        assert len(set(fees)) > 5

    def test_update_posterior_adds_observation(self, gaussian_thompson_state):
        """Test that updating posterior adds an observation."""
        state = gaussian_thompson_state

        assert len(state.observations) == 0

        state.update_posterior(fee=150, revenue_rate=50.0, hours=2.0)

        assert len(state.observations) == 1
        assert state.observations[0][0] == 150  # fee
        assert state.observations[0][1] == 50.0  # revenue_rate

    def test_update_posterior_recomputes_posterior(self, gaussian_thompson_state):
        """Test that posterior is recomputed after observations."""
        state = gaussian_thompson_state

        initial_mean = state.posterior_mean

        # Add observations clustered around 300 ppm
        for _ in range(10):
            state.update_posterior(fee=300, revenue_rate=100.0, hours=1.0)

        # Posterior mean should shift toward observations
        assert state.posterior_mean != initial_mean
        # With high-revenue observations at 300, mean should be closer to 300
        assert abs(state.posterior_mean - 300) < abs(initial_mean - 300)

    def test_observations_bounded(self, gaussian_thompson_state):
        """Test that observations are bounded to MAX_OBSERVATIONS."""
        state = gaussian_thompson_state

        # Add more than MAX_OBSERVATIONS
        for i in range(state.MAX_OBSERVATIONS + 50):
            state.update_posterior(fee=100 + i, revenue_rate=10.0, hours=1.0)

        assert len(state.observations) <= state.MAX_OBSERVATIONS

    def test_initialize_from_hive(self, gaussian_thompson_state):
        """Test initialization from hive intelligence."""
        state = gaussian_thompson_state

        # High confidence hive prior
        state.initialize_from_hive(
            optimal_fee=350,
            confidence=0.8,
            elasticity=-1.5
        )

        # Prior should shift toward hive estimate
        assert state.fleet_optimal_estimate == 350
        assert state.fleet_confidence == 0.8
        # With high confidence, prior_mean should be closer to hive estimate
        assert state.prior_mean_fee > 200  # shifted from default

    def test_contextual_posterior_isolation(self, gaussian_thompson_state):
        """Test that contextual posteriors are isolated from each other."""
        state = gaussian_thompson_state

        # Add observations to two different contexts
        state.update_contextual("low:strong:peak:P", fee=100, revenue_rate=50.0)
        state.update_contextual("high:none:normal:S", fee=400, revenue_rate=30.0)

        assert "low:strong:peak:P" in state.contextual_posteriors
        assert "high:none:normal:S" in state.contextual_posteriors

        # Contexts should have different means
        low_mean = state.contextual_posteriors["low:strong:peak:P"][0]
        high_mean = state.contextual_posteriors["high:none:normal:S"][0]
        assert low_mean != high_mean

    def test_serialization_roundtrip(self, gaussian_thompson_state):
        """Test that state serializes and deserializes correctly."""
        from modules.fee_controller import GaussianThompsonState

        state = gaussian_thompson_state

        # Add some data
        state.update_posterior(fee=200, revenue_rate=50.0, hours=2.0)
        state.update_contextual("balanced:medium:normal:P", fee=200, revenue_rate=50.0)
        state.initialize_from_hive(optimal_fee=250, confidence=0.7, elasticity=-1.0)

        # Serialize and deserialize
        state_dict = state.to_dict()
        restored = GaussianThompsonState.from_dict(state_dict)

        assert restored.prior_mean_fee == state.prior_mean_fee
        assert restored.posterior_mean == state.posterior_mean
        assert len(restored.observations) == len(state.observations)
        assert restored.fleet_optimal_estimate == state.fleet_optimal_estimate


class TestAIMDDefenseState:
    """Tests for AIMDDefenseState class."""

    def test_default_initialization(self, aimd_defense_state):
        """Test default state initialization."""
        state = aimd_defense_state

        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0
        assert state.aimd_modifier == 1.0
        assert state.is_active is False

    def test_success_increments_counter(self, aimd_defense_state):
        """Test that success increments success counter."""
        state = aimd_defense_state

        state.record_outcome(was_success=True)

        assert state.consecutive_successes == 1
        assert state.consecutive_failures == 0

    def test_failure_increments_counter(self, aimd_defense_state):
        """Test that failure increments failure counter."""
        state = aimd_defense_state

        state.record_outcome(was_success=False)

        assert state.consecutive_failures == 1
        assert state.consecutive_successes == 0

    def test_success_resets_failure_counter(self, aimd_defense_state):
        """Test that success resets failure counter."""
        state = aimd_defense_state

        # Record some failures
        state.record_outcome(was_success=False)
        state.record_outcome(was_success=False)
        assert state.consecutive_failures == 2

        # Then a success
        state.record_outcome(was_success=True)

        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 1

    def test_multiplicative_decrease_on_failure_streak(self, aimd_defense_state):
        """Test multiplicative decrease triggers on failure streak."""
        state = aimd_defense_state

        initial_modifier = state.aimd_modifier

        # Trigger failure threshold
        for _ in range(state.FAILURE_THRESHOLD):
            state.record_outcome(was_success=False)

        # Modifier should decrease
        assert state.aimd_modifier < initial_modifier
        assert state.aimd_modifier == initial_modifier * state.MULTIPLICATIVE_DECREASE
        assert state.is_active is True

    def test_additive_increase_on_success_streak(self, aimd_defense_state):
        """Test additive increase triggers on success streak."""
        state = aimd_defense_state

        initial_modifier = state.aimd_modifier

        # Trigger success threshold
        for _ in range(state.SUCCESS_THRESHOLD):
            state.record_outcome(was_success=True)

        # Modifier should increase slightly
        assert state.aimd_modifier > initial_modifier

    def test_apply_to_fee_when_inactive(self, aimd_defense_state):
        """Test that inactive AIMD passes fee through unchanged."""
        state = aimd_defense_state

        assert state.is_active is False

        thompson_fee = 200
        adjusted = state.apply_to_fee(thompson_fee, floor=50, ceiling=500)

        assert adjusted == thompson_fee

    def test_apply_to_fee_when_active(self, aimd_defense_state):
        """Test that active AIMD applies modifier."""
        state = aimd_defense_state

        # Trigger defense mode
        for _ in range(state.FAILURE_THRESHOLD):
            state.record_outcome(was_success=False)

        assert state.is_active is True

        thompson_fee = 200
        adjusted = state.apply_to_fee(thompson_fee, floor=50, ceiling=500)

        # Fee should be reduced
        assert adjusted < thompson_fee
        assert adjusted == int(thompson_fee * state.aimd_modifier)

    def test_apply_to_fee_respects_floor(self, aimd_defense_state):
        """Test that AIMD respects floor bound."""
        state = aimd_defense_state

        # Force very low modifier
        state.aimd_modifier = 0.5
        state.is_active = True

        floor = 100
        adjusted = state.apply_to_fee(50, floor=floor, ceiling=500)

        assert adjusted >= floor

    def test_reset_clears_state(self, aimd_defense_state):
        """Test that reset clears all state."""
        state = aimd_defense_state

        # Build up some state
        for _ in range(state.FAILURE_THRESHOLD):
            state.record_outcome(was_success=False)

        assert state.is_active is True

        state.reset()

        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0
        assert state.aimd_modifier == 1.0
        assert state.is_active is False

    def test_serialization_roundtrip(self, aimd_defense_state):
        """Test that state serializes and deserializes correctly."""
        from modules.fee_controller import AIMDDefenseState

        state = aimd_defense_state

        # Build up some state
        for _ in range(state.FAILURE_THRESHOLD):
            state.record_outcome(was_success=False)

        # Serialize and deserialize
        state_dict = state.to_dict()
        restored = AIMDDefenseState.from_dict(state_dict)

        assert restored.aimd_modifier == state.aimd_modifier
        assert restored.is_active == state.is_active
        assert restored.total_decreases == state.total_decreases


class TestThompsonAIMDState:
    """Tests for ThompsonAIMDState combined state."""

    def test_default_initialization(self, thompson_aimd_state):
        """Test default state initialization."""
        state = thompson_aimd_state

        assert state.thompson is not None
        assert state.aimd is not None
        assert state.last_revenue_rate == 0.0
        assert state.algorithm_version == "thompson_aimd_v1"

    def test_ema_update(self, thompson_aimd_state):
        """Test EMA revenue rate update."""
        state = thompson_aimd_state

        # First update initializes
        ema = state.update_ema_revenue_rate(100.0)
        assert ema == 100.0

        # Subsequent updates blend
        ema = state.update_ema_revenue_rate(200.0, alpha=0.3)
        # EMA = 0.3 * 200 + 0.7 * 100 = 60 + 70 = 130
        assert abs(ema - 130.0) < 0.01

    def test_historical_curve_persistence(self, thompson_aimd_state):
        """Test that historical curve data persists."""
        from modules.fee_controller import HistoricalResponseCurve

        state = thompson_aimd_state

        # Create and set a curve
        curve = HistoricalResponseCurve()
        curve.add_observation(fee_ppm=200, revenue_rate=50.0, forward_count=5)
        state.set_historical_curve(curve)

        # Retrieve and verify
        retrieved = state.get_historical_curve()
        assert len(retrieved.observations) == 1

    def test_v2_serialization_roundtrip(self, thompson_aimd_state):
        """Test v2 JSON serialization."""
        from modules.fee_controller import ThompsonAIMDState

        state = thompson_aimd_state

        # Add some data
        state.thompson.update_posterior(fee=200, revenue_rate=50.0, hours=2.0)
        state.aimd.record_outcome(was_success=True)
        state.last_revenue_rate = 42.5
        state.last_fee_ppm = 200

        # Serialize
        v2_dict = state.to_v2_dict()

        # Deserialize with legacy state
        legacy_state = {
            "last_revenue_rate": 42.5,
            "last_fee_ppm": 200,
            "last_broadcast_fee_ppm": 200,
            "last_update": 1000,
            "is_sleeping": 0,
            "stable_cycles": 0
        }

        restored = ThompsonAIMDState.from_v2_dict(v2_dict, legacy_state)

        assert restored.algorithm_version == "thompson_aimd_v1"
        assert restored.last_revenue_rate == 42.5
        assert len(restored.thompson.observations) == 1


class TestThompsonAIMDIntegration:
    """Integration tests for Thompson+AIMD algorithm."""

    def test_thompson_sample_then_aimd_defense(self):
        """Test full flow: Thompson samples, AIMD defends."""
        from modules.fee_controller import ThompsonAIMDState

        state = ThompsonAIMDState()

        # Simulate routing with some successes
        for _ in range(5):
            state.thompson.update_posterior(fee=200, revenue_rate=50.0, hours=1.0)
            state.aimd.record_outcome(was_success=True)

        # Sample fee
        thompson_fee = state.thompson.sample_fee(floor=50, ceiling=500)

        # AIMD should pass through (not in defense mode)
        final_fee = state.aimd.apply_to_fee(thompson_fee, floor=50, ceiling=500)
        assert final_fee == thompson_fee

        # Now simulate failure streak
        for _ in range(state.aimd.FAILURE_THRESHOLD):
            state.aimd.record_outcome(was_success=False)

        # AIMD should now reduce fee
        final_fee = state.aimd.apply_to_fee(thompson_fee, floor=50, ceiling=500)
        assert final_fee < thompson_fee

    def test_migration_from_empty_state(self):
        """Test migration from empty/legacy state."""
        from modules.fee_controller import ThompsonAIMDState

        # Simulate loading from empty v2_state_json
        v2_data = {}
        legacy_state = {
            "last_revenue_rate": 100.0,
            "last_fee_ppm": 250,
            "last_broadcast_fee_ppm": 250
        }

        state = ThompsonAIMDState.from_v2_dict(v2_data, legacy_state)

        # Should initialize fresh Thompson state
        assert state.thompson is not None
        assert state.aimd is not None
        assert state.last_revenue_rate == 100.0
        assert state.last_fee_ppm == 250

    def test_migration_preserves_historical_observations(self):
        """Test that migration preserves historical curve observations."""
        from modules.fee_controller import ThompsonAIMDState

        # Simulate legacy state with historical curve data
        v2_data = {
            "historical_curve": {
                "observations": [
                    {"fee_ppm": 200, "revenue_rate": 50.0, "forward_count": 5, "timestamp": 1000},
                    {"fee_ppm": 250, "revenue_rate": 60.0, "forward_count": 8, "timestamp": 2000},
                ]
            }
        }

        state = ThompsonAIMDState.from_v2_dict(v2_data, {})

        # Thompson should have observations from migration
        assert len(state.thompson.observations) == 2


class TestFleetInformedPriors:
    """Tests for fleet-informed Thompson priors (P0 integration)."""

    def test_initialize_from_hive_profile_full(self):
        """Test initialization from full hive profile."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        profile = {
            "optimal_fee_estimate": 180,
            "avg_fee_charged": 200,
            "min_fee": 100,
            "max_fee": 400,
            "fee_volatility": 0.2,
            "estimated_elasticity": -1.2,
            "confidence": 0.75,
            "hive_reporters": 4
        }

        state.initialize_from_hive_profile(profile)

        # Should store fleet data
        assert state.fleet_optimal_estimate == 180
        assert state.fleet_avg_fee == 200
        assert state.fleet_min_fee == 100
        assert state.fleet_max_fee == 400
        assert state.fleet_fee_volatility == 0.2
        assert state.fleet_reporters == 4

        # Prior mean should shift toward optimal estimate
        assert state.prior_mean_fee != 200  # Changed from default
        assert abs(state.prior_mean_fee - 180) < 50  # Closer to optimal

    def test_profile_with_low_confidence_ignored(self):
        """Test that low confidence profiles don't affect prior."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        original_mean = state.prior_mean_fee
        original_std = state.prior_std_fee

        profile = {
            "optimal_fee_estimate": 500,  # Very different
            "confidence": 0.1,  # Too low
            "hive_reporters": 1
        }

        state.initialize_from_hive_profile(profile)

        # Should not change priors
        assert state.prior_mean_fee == original_mean
        assert state.prior_std_fee == original_std

    def test_volatility_increases_uncertainty(self):
        """Test that high volatility increases prior std."""
        from modules.fee_controller import GaussianThompsonState

        # Low volatility profile
        low_vol = GaussianThompsonState()
        low_vol.initialize_from_hive_profile({
            "optimal_fee_estimate": 200,
            "fee_volatility": 0.1,
            "confidence": 0.7,
            "hive_reporters": 3
        })

        # High volatility profile
        high_vol = GaussianThompsonState()
        high_vol.initialize_from_hive_profile({
            "optimal_fee_estimate": 200,
            "fee_volatility": 0.8,
            "confidence": 0.7,
            "hive_reporters": 3
        })

        # Higher volatility should result in higher uncertainty
        assert high_vol.prior_std_fee > low_vol.prior_std_fee

    def test_multiple_reporters_boost_confidence(self):
        """Test that multiple reporters reduce uncertainty."""
        from modules.fee_controller import GaussianThompsonState

        # Single reporter
        single = GaussianThompsonState()
        single.initialize_from_hive_profile({
            "optimal_fee_estimate": 200,
            "fee_volatility": 0.3,
            "confidence": 0.6,
            "hive_reporters": 1
        })

        # Multiple reporters
        multi = GaussianThompsonState()
        multi.initialize_from_hive_profile({
            "optimal_fee_estimate": 200,
            "fee_volatility": 0.3,
            "confidence": 0.6,
            "hive_reporters": 5
        })

        # More reporters should reduce uncertainty
        assert multi.prior_std_fee < single.prior_std_fee

    def test_fleet_data_persistence(self):
        """Test that fleet data survives serialization."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        state.initialize_from_hive_profile({
            "optimal_fee_estimate": 180,
            "avg_fee_charged": 200,
            "min_fee": 100,
            "max_fee": 400,
            "fee_volatility": 0.2,
            "estimated_elasticity": -1.2,
            "confidence": 0.75,
            "hive_reporters": 4
        })

        # Serialize and deserialize
        state_dict = state.to_dict()
        restored = GaussianThompsonState.from_dict(state_dict)

        assert restored.fleet_optimal_estimate == 180
        assert restored.fleet_avg_fee == 200
        assert restored.fleet_fee_volatility == 0.2
        assert restored.fleet_reporters == 4


class TestFleetDefenseCoordination:
    """Tests for fleet defense coordination (P0 integration)."""

    def test_fleet_threat_update(self):
        """Test updating AIMD with fleet threat info."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()

        # Initially no threat
        assert state.fleet_threat_active is False

        # Update with threat info
        threat_info = {
            "is_threat": True,
            "threat_type": "drain",
            "severity": 0.8,
            "defensive_multiplier": 2.5,
            "expires_at": int(time.time()) + 3600
        }
        state.update_fleet_threat(threat_info)

        assert state.fleet_threat_active is True
        assert state.fleet_threat_type == "drain"
        assert state.fleet_threat_severity == 0.8
        assert state.fleet_defensive_multiplier == 2.5

    def test_fleet_threat_clears_on_none(self):
        """Test that passing None clears threat state."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()
        state.fleet_threat_active = True
        state.fleet_threat_type = "drain"
        state.fleet_threat_severity = 0.8

        state.update_fleet_threat(None)

        assert state.fleet_threat_active is False
        assert state.fleet_threat_type is None
        assert state.fleet_threat_severity == 0.0

    def test_expired_threat_clears(self):
        """Test that expired threats are cleared."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()

        # Update with expired threat
        expired_threat = {
            "is_threat": True,
            "threat_type": "drain",
            "severity": 0.8,
            "defensive_multiplier": 2.5,
            "expires_at": int(time.time()) - 100  # Already expired
        }
        state.update_fleet_threat(expired_threat)

        # Should be cleared
        assert state.fleet_threat_active is False

    def test_effective_modifier_combines_aimd_and_fleet(self):
        """Test that effective modifier combines both AIMD and fleet defense."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()

        # Set up AIMD defense (reduces fee)
        state.is_active = True
        state.aimd_modifier = 0.8

        # Set up fleet threat (increases fee)
        state.fleet_threat_active = True
        state.fleet_defensive_multiplier = 2.0

        # Effective should combine both: 0.8 * 2.0 = 1.6
        effective = state.get_effective_modifier()
        assert abs(effective - 1.6) < 0.01

    def test_apply_to_fee_with_fleet_defense(self):
        """Test fee application with fleet defense."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()

        # Set fleet threat
        state.fleet_threat_active = True
        state.fleet_defensive_multiplier = 1.5

        thompson_fee = 200
        adjusted = state.apply_to_fee(thompson_fee, floor=50, ceiling=500)

        # Should increase fee: 200 * 1.5 = 300
        assert adjusted == 300

    def test_drain_attack_triggers_local_aimd(self):
        """Test that severe drain attack triggers local AIMD defense."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()
        assert state.is_active is False

        # Severe drain attack
        threat_info = {
            "is_threat": True,
            "threat_type": "drain",
            "severity": 0.7,
            "defensive_multiplier": 2.5,
            "expires_at": int(time.time()) + 3600
        }
        state.update_fleet_threat(threat_info)

        # Should also activate local AIMD
        assert state.is_active is True

    def test_fleet_threat_persistence(self):
        """Test that fleet threat state survives serialization."""
        from modules.fee_controller import AIMDDefenseState

        state = AIMDDefenseState()
        state.update_fleet_threat({
            "is_threat": True,
            "threat_type": "drain",
            "severity": 0.8,
            "defensive_multiplier": 2.5,
            "expires_at": int(time.time()) + 3600
        })

        # Serialize and deserialize
        state_dict = state.to_dict()
        restored = AIMDDefenseState.from_dict(state_dict)

        assert restored.fleet_threat_active == state.fleet_threat_active
        assert restored.fleet_threat_type == state.fleet_threat_type
        assert restored.fleet_threat_severity == state.fleet_threat_severity
        assert restored.fleet_defensive_multiplier == state.fleet_defensive_multiplier


class TestWeightedAIMDSuccessMetric:
    """Tests for Improvement #2: Weighted AIMD Success Metric."""

    def test_high_score_counts_as_success(self):
        """Score >= 0.7 increments consecutive_successes."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        aimd.record_outcome(success_score=0.9)
        assert aimd.consecutive_successes == 1
        assert aimd.consecutive_failures == 0

    def test_low_score_counts_as_failure(self):
        """Score < 0.3 increments consecutive_failures."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        aimd.record_outcome(success_score=0.1)
        assert aimd.consecutive_failures == 1
        assert aimd.consecutive_successes == 0

    def test_neutral_score_no_change(self):
        """Score between 0.3 and 0.7 doesn't change counters."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()
        aimd.consecutive_successes = 3
        aimd.consecutive_failures = 2

        aimd.record_outcome(success_score=0.5)
        assert aimd.consecutive_successes == 3
        assert aimd.consecutive_failures == 2

    def test_backward_compat_was_success(self):
        """Legacy was_success=True/False still works."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        aimd.record_outcome(was_success=True)
        assert aimd.consecutive_successes == 1

        aimd.record_outcome(was_success=False)
        assert aimd.consecutive_failures == 1

    def test_exploration_suppresses_failure(self):
        """Failures during exploration (is_exploring=True) are ignored."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        aimd.record_outcome(success_score=0.1, is_exploring=True)
        assert aimd.consecutive_failures == 0  # Suppressed
        assert aimd.consecutive_successes == 0

    def test_exploration_does_not_suppress_success(self):
        """Successes during exploration are still counted."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        aimd.record_outcome(success_score=0.9, is_exploring=True)
        assert aimd.consecutive_successes == 1


class TestAIMDThompsonCoordination:
    """Tests for Improvement #3: AIMD-Thompson Coordination."""

    def test_exploring_detection(self):
        """is_exploring is True when thompson_fee is far from posterior mean."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0
        ts.posterior_std = 50.0

        # Fee 230 is 0.6 std from mean > 0.5 threshold
        thompson_fee = 230
        is_exploring = abs(thompson_fee - ts.posterior_mean) > ts.posterior_std * 0.5
        assert is_exploring is True

        # Fee 210 is 0.2 std from mean < 0.5 threshold
        thompson_fee = 210
        is_exploring = abs(thompson_fee - ts.posterior_mean) > ts.posterior_std * 0.5
        assert is_exploring is False

    def test_failure_suppression_during_exploration(self):
        """AIMD doesn't decrease modifier when Thompson is exploring."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        # Feed failures during exploration — should not trigger decrease
        for _ in range(10):
            aimd.record_outcome(success_score=0.0, is_exploring=True)

        assert aimd.consecutive_failures == 0
        assert aimd.aimd_modifier == 1.0

    def test_success_counted_during_exploration(self):
        """Successes during exploration still increment counters."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        for _ in range(5):
            aimd.record_outcome(success_score=1.0, is_exploring=True)

        assert aimd.consecutive_successes == 5

    def test_downward_spiral_prevention(self):
        """Mixing exploration failures with normal successes doesn't spiral."""
        from modules.fee_controller import AIMDDefenseState
        aimd = AIMDDefenseState()

        # Alternate: explore-fail, then normal-success
        for _ in range(5):
            aimd.record_outcome(success_score=0.0, is_exploring=True)  # suppressed
            aimd.record_outcome(success_score=0.8, is_exploring=False)  # counted

        assert aimd.consecutive_successes == 5
        assert aimd.aimd_modifier >= 1.0

    def test_non_exploring_failures_still_trigger_decrease(self):
        """Failures when NOT exploring still trigger normal AIMD decrease."""
        from modules.fee_controller import AIMDDefenseState
        import time as t
        aimd = AIMDDefenseState()
        aimd.last_decrease_time = 0  # Allow decrease

        # Feed enough non-exploring failures to trigger decrease
        for _ in range(aimd.FAILURE_THRESHOLD + 1):
            aimd.record_outcome(success_score=0.0, is_exploring=False)

        assert aimd.aimd_modifier < 1.0
        assert aimd.is_active is True


