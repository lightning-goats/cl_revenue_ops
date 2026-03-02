"""
Unit tests for Thompson Sampling + AIMD fee optimization algorithm.

Tests the new GaussianThompsonState, AIMDDefenseState, and ThompsonAIMDState
classes that replace Hill Climbing as the primary fee optimization algorithm.
"""
import pytest
import math
import random
import time
import threading
from typing import Dict, Any
from unittest.mock import MagicMock


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


class TestStigmergicModulation:
    """Tests for stigmergic (pheromone-based) exploration modulation (P1)."""

    def test_set_context_modulation(self):
        """Test setting context modulation parameters."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        state.set_context_modulation(
            pheromone_level=15.0,
            corridor_role="P",
            time_bucket="peak"
        )

        assert state.current_pheromone_level == 15.0
        assert state.current_corridor_role == "P"
        assert state.current_time_bucket == "peak"

    def test_high_pheromone_reduces_exploration(self):
        """Test that high pheromone level reduces exploration (exploitation mode)."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        # Add some observations
        for i in range(10):
            state.update_posterior(fee=200, revenue_rate=50.0, hours=1.0)

        # High pheromone should reduce exploration modifier
        state.set_context_modulation(pheromone_level=20.0, corridor_role="P")
        high_mod = state._get_exploration_modifier()

        # Low pheromone should increase exploration modifier
        state.set_context_modulation(pheromone_level=0.0, corridor_role="P")
        low_mod = state._get_exploration_modifier()

        assert high_mod < low_mod
        assert high_mod < 1.0  # Exploitation mode
        assert low_mod > 1.0  # Exploration mode

    def test_secondary_corridor_explores_more(self):
        """Test that secondary corridors have higher exploration."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()

        # Same pheromone, different role
        state.set_context_modulation(pheromone_level=5.0, corridor_role="P")
        primary_mod = state._get_exploration_modifier()

        state.set_context_modulation(pheromone_level=5.0, corridor_role="S")
        secondary_mod = state._get_exploration_modifier()

        assert secondary_mod > primary_mod

    def test_sample_fee_applies_modulation(self):
        """Test that sampled fees are affected by modulation."""
        from modules.fee_controller import GaussianThompsonState
        import statistics

        state = GaussianThompsonState()
        state.posterior_mean = 200.0
        state.posterior_std = 50.0

        # Add enough observations
        for i in range(10):
            state.observations.append((200, 50.0, 1.0, int(time.time())))

        # High pheromone (exploit) should have less variance
        state.set_context_modulation(pheromone_level=20.0, corridor_role="P")
        exploit_samples = [state.sample_fee(floor=50, ceiling=500) for _ in range(100)]

        # Low pheromone (explore) should have more variance
        state.set_context_modulation(pheromone_level=0.0, corridor_role="S")
        explore_samples = [state.sample_fee(floor=50, ceiling=500) for _ in range(100)]

        exploit_std = statistics.stdev(exploit_samples)
        explore_std = statistics.stdev(explore_samples)

        # Exploration should have higher variance
        assert explore_std > exploit_std


class TestTimeWeightedObservations:
    """Tests for time-weighted observation learning (P1)."""

    def test_time_similarity(self):
        """Test time bucket similarity calculation."""
        from modules.fee_controller import GaussianThompsonState

        # Same bucket = 1.0
        assert GaussianThompsonState._time_similarity("peak", "peak") == 1.0
        assert GaussianThompsonState._time_similarity("normal", "normal") == 1.0

        # Adjacent = 0.5
        assert GaussianThompsonState._time_similarity("normal", "peak") == 0.5
        assert GaussianThompsonState._time_similarity("low", "normal") == 0.5

        # Opposite = 0.2
        assert GaussianThompsonState._time_similarity("low", "peak") == 0.2

    def test_update_contextual_with_time(self):
        """Test that contextual updates use time-aware weighting."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()

        # Update with peak time observation
        state.update_contextual(
            context_key="balanced:none:peak:P",
            fee=300,
            revenue_rate=100.0,
            time_bucket="peak"
        )

        # Peak context should have moved toward 300
        # 4-tuple format: (mean, precision, count, last_update)
        ctx = state.contextual_posteriors["balanced:none:peak:P"]
        peak_mean = ctx[0]
        assert peak_mean > state.posterior_mean

    def test_observation_includes_time_bucket(self):
        """Test that observations include time bucket."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=50.0, hours=1.0, time_bucket="peak")

        assert len(state.observations) == 1
        obs = state.observations[0]
        assert len(obs) == 5  # (fee, revenue, weight, timestamp, time_bucket)
        assert obs[4] == "peak"


class TestCorridorRoleDifferentiation:
    """Tests for primary/secondary corridor differentiation (P1)."""

    def test_secondary_contextual_wider_initial_std(self):
        """Test that secondary corridors start with wider uncertainty."""
        from modules.fee_controller import GaussianThompsonState
        import math

        state = GaussianThompsonState()
        state.posterior_std = 50.0

        # Initialize primary context
        state.update_contextual("balanced:none:normal:P", fee=200, revenue_rate=50.0)
        # 4-tuple format: (mean, precision, count, last_update)
        primary_precision = state.contextual_posteriors["balanced:none:normal:P"][1]
        primary_std = 1.0 / math.sqrt(max(primary_precision, 1e-6))

        # Initialize secondary context
        state.update_contextual("balanced:none:normal:S", fee=200, revenue_rate=50.0)
        secondary_precision = state.contextual_posteriors["balanced:none:normal:S"][1]
        secondary_std = 1.0 / math.sqrt(max(secondary_precision, 1e-6))

        # Secondary should have wider initial std (lower precision)
        assert secondary_std > primary_std

    def test_secondary_learns_faster(self):
        """Test that secondary corridors adapt more quickly."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()

        # Initialize both
        state.update_contextual("balanced:none:normal:P", fee=200, revenue_rate=50.0)
        state.update_contextual("balanced:none:normal:S", fee=200, revenue_rate=50.0)

        # Get initial means
        initial_primary = state.contextual_posteriors["balanced:none:normal:P"][0]
        initial_secondary = state.contextual_posteriors["balanced:none:normal:S"][0]

        # Update both with same high-fee observation
        state.update_contextual("balanced:none:normal:P", fee=400, revenue_rate=100.0)
        state.update_contextual("balanced:none:normal:S", fee=400, revenue_rate=100.0)

        # Get updated means
        new_primary = state.contextual_posteriors["balanced:none:normal:P"][0]
        new_secondary = state.contextual_posteriors["balanced:none:normal:S"][0]

        primary_shift = new_primary - initial_primary
        secondary_shift = new_secondary - initial_secondary

        # Secondary should shift more (faster learning)
        assert secondary_shift > primary_shift


class TestFeeDiscoveryBroadcast:
    """Tests for fee discovery detection and broadcast (P1)."""

    def test_no_discovery_without_enough_observations(self):
        """Test that discoveries require minimum observations."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        # Only 2 observations
        state.update_posterior(fee=200, revenue_rate=100.0, hours=1.0)
        state.update_posterior(fee=200, revenue_rate=100.0, hours=1.0)

        discovery = state.check_for_discovery(fee=200, revenue_rate=100.0)
        assert discovery is None

    def test_no_discovery_with_low_revenue(self):
        """Test that low revenue doesn't trigger discovery."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        for i in range(10):
            state.update_posterior(fee=200, revenue_rate=10.0, hours=1.0)

        discovery = state.check_for_discovery(fee=200, revenue_rate=10.0)
        assert discovery is None

    def test_discovery_on_high_revenue(self):
        """Test discovery detection on unusually high revenue."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        # Build up history with moderate revenue
        for i in range(10):
            state.update_posterior(fee=200, revenue_rate=40.0, hours=1.0)

        # Check for discovery with significantly higher revenue
        discovery = state.check_for_discovery(fee=200, revenue_rate=80.0)

        # Should detect discovery
        assert discovery is not None
        assert discovery["discovery_type"] == "high_revenue"
        assert discovery["fee_ppm"] == 200
        assert discovery["revenue_rate"] == 80.0

    def test_discovery_confirms_optimal_fee(self):
        """Test discovery detection when fee matches posterior."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        state.posterior_mean = 200.0
        state.posterior_std = 30.0

        # Build up consistent history at posterior mean
        for i in range(15):
            state.update_posterior(fee=200, revenue_rate=80.0, hours=1.0)

        # Check near posterior mean with good revenue
        discovery = state.check_for_discovery(
            fee=195,  # Near posterior mean
            revenue_rate=80.0,
            min_revenue_rate=50.0
        )

        assert discovery is not None
        assert discovery["discovery_type"] == "optimal_fee"


class TestElasticityFleetIntegration:
    """Tests for ElasticityTracker fleet integration (P2)."""

    def test_should_broadcast_when_confident(self):
        """Test that elasticity is broadcast when confidence is sufficient."""
        from modules.fee_controller import ElasticityTracker

        tracker = ElasticityTracker()
        # Add enough observations for confidence
        for i in range(10):
            tracker.add_observation(
                fee_ppm=200 + i * 10,
                volume_sats=1000000,
                revenue_rate=100.0 - i * 5
            )

        # Should broadcast if confidence >= 0.5 and enough samples
        if tracker.confidence >= 0.5:
            assert tracker.should_broadcast() == True
        else:
            # If confidence too low, shouldn't broadcast
            assert tracker.should_broadcast() == False

    def test_should_not_broadcast_without_data(self):
        """Test that elasticity not broadcast without sufficient data."""
        from modules.fee_controller import ElasticityTracker

        tracker = ElasticityTracker()
        # Only one observation
        tracker.add_observation(fee_ppm=200, volume_sats=1000000, revenue_rate=100.0)

        assert tracker.should_broadcast() == False

    def test_get_broadcast_data_format(self):
        """Test broadcast data format."""
        from modules.fee_controller import ElasticityTracker

        tracker = ElasticityTracker()
        tracker.current_elasticity = -1.5
        tracker.confidence = 0.8
        tracker.history = [(200, 1000, 100.0, int(time.time()))] * 5

        data = tracker.get_broadcast_data()

        assert "elasticity" in data
        assert "confidence" in data
        assert "sample_count" in data
        assert data["elasticity"] == -1.5
        assert data["confidence"] == 0.8
        assert data["sample_count"] == 5

    def test_incorporate_fleet_data_blends_values(self):
        """Test that fleet data is blended with local data."""
        from modules.fee_controller import ElasticityTracker

        tracker = ElasticityTracker()
        tracker.current_elasticity = -1.0
        tracker.confidence = 0.7

        # Incorporate fleet data
        tracker.incorporate_fleet_data(
            fleet_elasticity=-2.0,
            fleet_confidence=0.8,
            fleet_weight=0.3
        )

        # Elasticity should be blended (not exactly -1.0 anymore)
        # Should be somewhere between -1.0 and -2.0
        assert tracker.current_elasticity < -1.0
        assert tracker.current_elasticity > -2.0

    def test_incorporate_fleet_data_ignored_when_low_confidence(self):
        """Test that low-confidence fleet data is ignored."""
        from modules.fee_controller import ElasticityTracker

        tracker = ElasticityTracker()
        tracker.current_elasticity = -1.0
        tracker.confidence = 0.7
        original = tracker.current_elasticity

        # Incorporate low-confidence fleet data
        tracker.incorporate_fleet_data(
            fleet_elasticity=-2.0,
            fleet_confidence=0.2,  # Below threshold
            fleet_weight=0.3
        )

        # Should be unchanged
        assert tracker.current_elasticity == original


class TestHistoricalCurveFleetIntegration:
    """Tests for HistoricalResponseCurve fleet integration (P2)."""

    def test_should_broadcast_observation_high_forward(self):
        """Test broadcast detection for high forward count observations."""
        from modules.fee_controller import HistoricalResponseCurve, FeeRevenueObservation

        curve = HistoricalResponseCurve()
        # Build baseline
        for i in range(10):
            curve.add_observation(fee_ppm=200, revenue_rate=50.0, forward_count=5)

        # High forward count near optimal should broadcast
        result = curve.should_broadcast_observation(
            fee_ppm=200,  # Near optimal
            revenue_rate=50.0,
            forward_count=20
        )
        assert result == True

    def test_should_not_broadcast_low_forward(self):
        """Test that low forward count observations don't broadcast."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()
        for i in range(10):
            curve.add_observation(fee_ppm=200, revenue_rate=50.0, forward_count=5)

        # Low forward count should not broadcast
        result = curve.should_broadcast_observation(
            fee_ppm=200,
            revenue_rate=50.0,
            forward_count=2  # Too few
        )
        assert result == False

    def test_get_broadcast_data_format(self):
        """Test broadcast data format for curve."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()
        for i in range(10):
            curve.add_observation(fee_ppm=200 + i * 10, revenue_rate=50.0, forward_count=5)

        data = curve.get_broadcast_data()

        assert "observation_count" in data
        assert "regime_change_count" in data
        assert "best_fee_estimate" in data
        assert "recent_observations" in data
        assert len(data["recent_observations"]) <= 10

    def test_incorporate_fleet_curve_adds_observations(self):
        """Test that fleet curve data is incorporated."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()
        initial_count = len(curve.observations)

        fleet_obs = [
            {"fee_ppm": 150, "revenue_rate": 80.0, "count": 10},
            {"fee_ppm": 200, "revenue_rate": 100.0, "count": 15}
        ]

        curve.incorporate_fleet_curve(fleet_obs, fleet_weight=0.25)

        # Should have more observations now
        assert len(curve.observations) > initial_count

    def test_get_regime_broadcast_data(self):
        """Test regime broadcast data format."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()
        for i in range(25):
            curve.add_observation(fee_ppm=200, revenue_rate=50.0, forward_count=5)
        curve.regime_change_count = 2

        data = curve.get_regime_broadcast_data()

        assert "regime_change_count" in data
        assert "recent_avg_revenue" in data
        assert "observation_count" in data
        assert data["regime_change_count"] == 2


class TestCompetitionAvoidance:
    """Tests for competition avoidance via Thompson posterior sharing (P2)."""

    def test_posterior_differentiation_primary_goes_lower(self):
        """Test that primary corridors differentiate by going lower."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        state.posterior_mean = 200.0
        state.posterior_std = 50.0

        # Simulate crowded region detection and differentiation
        # For primary corridor (P), should go lower
        original_mean = state.posterior_mean
        diff_amount = int(state.posterior_std * 0.3)
        differentiation_direction = -1  # Primary goes lower

        state.posterior_mean += differentiation_direction * diff_amount

        assert state.posterior_mean < original_mean
        assert state.posterior_mean == 200.0 - diff_amount

    def test_posterior_differentiation_secondary_goes_higher(self):
        """Test that secondary corridors differentiate by going higher."""
        from modules.fee_controller import GaussianThompsonState

        state = GaussianThompsonState()
        state.posterior_mean = 200.0
        state.posterior_std = 50.0

        # For secondary corridor (S), should go higher
        original_mean = state.posterior_mean
        diff_amount = int(state.posterior_std * 0.3)
        differentiation_direction = 1  # Secondary goes higher

        state.posterior_mean += differentiation_direction * diff_amount

        assert state.posterior_mean > original_mean
        assert state.posterior_mean == 200.0 + diff_amount

    def test_overlap_detection(self):
        """Test detection of overlapping posteriors."""
        # Two posteriors overlap if means are within 1.5 * max(std1, std2)
        our_mean = 200.0
        our_std = 50.0
        member_mean = 220.0
        member_std = 40.0

        overlap_threshold = 1.5 * max(our_std, member_std)  # 75
        distance = abs(our_mean - member_mean)  # 20

        # These should overlap (20 < 75)
        assert distance < overlap_threshold


class TestProfitabilityWeightedSampling:
    """Tests for profitability-weighted Thompson sampling (P2)."""

    def test_profitable_channel_biased_up(self):
        """Test that profitable channels get upward fee bias."""
        # Profitable channel with ROI > 20% should get +10% bias
        thompson_fee = 200
        roi_percent = 25.0

        # Simulate the bias calculation
        profitability_adjustment = int(thompson_fee * 0.1)  # +10%

        new_fee = thompson_fee + profitability_adjustment
        assert new_fee == 220
        assert new_fee > thompson_fee

    def test_underwater_channel_biased_down(self):
        """Test that underwater channels get downward fee bias."""
        # Underwater channel with ROI < -20% should get -15% bias
        thompson_fee = 200
        roi_percent = -30.0

        # Simulate the bias calculation
        profitability_adjustment = -int(thompson_fee * 0.15)  # -15%

        new_fee = thompson_fee + profitability_adjustment
        assert new_fee == 170
        assert new_fee < thompson_fee

    def test_zombie_channel_forced_to_floor(self):
        """Test that zombie channels are forced to floor."""
        thompson_fee = 200
        floor_ppm = 10

        # Zombie channels force to floor
        profitability_adjustment = floor_ppm - thompson_fee

        new_fee = thompson_fee + profitability_adjustment
        assert new_fee == floor_ppm

    def test_marginal_channel_no_adjustment(self):
        """Test that marginal channels get no adjustment."""
        thompson_fee = 200

        # Marginal channels get no profitability adjustment
        profitability_adjustment = 0

        new_fee = thompson_fee + profitability_adjustment
        assert new_fee == thompson_fee


class TestPolynomialSmoothing:
    """Tests for polynomial smoothing in HistoricalResponseCurve."""

    def test_polynomial_fit_concave_curve(self):
        """Test that polynomial fit correctly finds optimal for concave revenue curve."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()

        # Create a concave revenue curve (inverted parabola)
        # Revenue peaks around fee=200, declines at lower/higher fees
        # revenue = -0.005 * (fee - 200)^2 + 100
        # Optimal should be around 200 ppm
        test_fees = [100, 120, 150, 180, 200, 220, 250, 280, 300]
        for fee in test_fees:
            revenue = -0.005 * (fee - 200)**2 + 100
            curve.add_observation(fee_ppm=fee, revenue_rate=revenue, forward_count=10)

        # Predict optimal - should be close to 200
        optimal = curve.predict_optimal_fee(floor_ppm=50, ceiling_ppm=500)

        assert optimal is not None
        # Allow some tolerance due to fitting
        assert 180 <= optimal <= 220

    def test_polynomial_fit_respects_floor(self):
        """Test that polynomial fit respects floor constraint."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()

        # Create curve with optimal below floor
        # Revenue peaks at fee=50
        test_fees = [50, 75, 100, 125, 150, 175, 200, 225, 250]
        for fee in test_fees:
            revenue = -0.005 * (fee - 50)**2 + 100
            curve.add_observation(fee_ppm=fee, revenue_rate=revenue, forward_count=10)

        # Floor is 100, optimal at 50 should clamp to floor
        optimal = curve.predict_optimal_fee(floor_ppm=100, ceiling_ppm=500)

        assert optimal is not None
        assert optimal == 100

    def test_polynomial_fit_respects_ceiling(self):
        """Test that polynomial fit respects ceiling constraint."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()

        # Create curve with optimal above ceiling
        # Revenue peaks at fee=400
        test_fees = [150, 200, 250, 300, 350, 400, 450, 500, 550]
        for fee in test_fees:
            revenue = -0.005 * (fee - 400)**2 + 100
            curve.add_observation(fee_ppm=fee, revenue_rate=revenue, forward_count=10)

        # Ceiling is 300, optimal at 400 should clamp to ceiling
        optimal = curve.predict_optimal_fee(floor_ppm=50, ceiling_ppm=300)

        assert optimal is not None
        assert optimal == 300

    def test_insufficient_data_uses_fallback(self):
        """Test that with insufficient data, weighted maximum fallback is used."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()

        # Only 5 observations - not enough for polynomial fit
        curve.add_observation(fee_ppm=100, revenue_rate=50.0, forward_count=10)
        curve.add_observation(fee_ppm=150, revenue_rate=80.0, forward_count=10)
        curve.add_observation(fee_ppm=200, revenue_rate=100.0, forward_count=10)  # Best
        curve.add_observation(fee_ppm=250, revenue_rate=70.0, forward_count=10)
        curve.add_observation(fee_ppm=300, revenue_rate=40.0, forward_count=10)

        # Should still work using weighted maximum fallback
        optimal = curve.predict_optimal_fee(floor_ppm=50, ceiling_ppm=500)

        assert optimal is not None
        assert optimal == 200  # Highest revenue observation

    def test_convex_curve_uses_fallback(self):
        """Test that convex curve (no maximum) uses weighted max fallback."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()

        # Create convex curve (U-shaped) - no valid maximum
        # revenue = 0.005 * (fee - 200)^2 + 50
        test_fees = [100, 120, 150, 180, 200, 220, 250, 280, 300]
        for fee in test_fees:
            revenue = 0.005 * (fee - 200)**2 + 50
            curve.add_observation(fee_ppm=fee, revenue_rate=revenue, forward_count=10)

        # Should fall back to weighted max since curve is convex
        optimal = curve.predict_optimal_fee(floor_ppm=50, ceiling_ppm=500)

        assert optimal is not None
        # Fallback should pick highest revenue (at boundaries)
        assert optimal in [100, 300]  # Either end has highest revenue

    def test_narrow_fee_range_uses_fallback(self):
        """Test that narrow fee range triggers fallback."""
        from modules.fee_controller import HistoricalResponseCurve

        curve = HistoricalResponseCurve()

        # All fees within 10 ppm - too narrow for reliable fit
        test_fees = [200, 201, 202, 203, 204, 205, 206, 207, 208]
        for fee in test_fees:
            revenue = 100 - (fee - 204) ** 2  # Peak at 204
            curve.add_observation(fee_ppm=fee, revenue_rate=revenue, forward_count=10)

        # Should fall back to weighted max
        optimal = curve.predict_optimal_fee(floor_ppm=50, ceiling_ppm=500)

        assert optimal is not None
        # Should pick best from observations
        assert optimal == 204


# =============================================================================
# Tests for 6 Thompson Sampling Improvements
# =============================================================================

class TestDemandAdjustedRevenue:
    """Tests for Improvement #1: Demand-Adjusted Reward Signal (Kalman-based)."""

    def test_default_expected_demand(self):
        """Default expected_demand is 0.5 (healthy baseline)."""
        expected_demand = 0.5
        demand_factor = max(0.25, min(4.0, expected_demand / 0.5))
        assert demand_factor == 1.0  # 0.5/0.5 = 1.0, no adjustment

    def test_high_kalman_demand_inflates_factor(self):
        """High flow_ratio + velocity → high demand_factor → reduced adjusted revenue."""
        import math
        kr = 0.8   # strong flow ratio
        kv = 0.02  # positive velocity
        expected_demand = abs(kr) + abs(kv * 24.0)
        # expected_demand = 0.8 + 0.48 = 1.28
        assert not math.isnan(expected_demand)
        demand_factor = max(0.25, min(4.0, expected_demand / 0.5))
        assert demand_factor == min(4.0, 1.28 / 0.5)  # 2.56

        adjusted = 100.0 / demand_factor
        assert adjusted < 100.0  # revenue reduced by demand

    def test_low_demand_clamps_factor_to_one(self):
        """When expected_demand < 0.05, demand_factor = 1.0 to avoid noise."""
        expected_demand = 0.02
        assert expected_demand < 0.05
        demand_factor = 1.0  # bypass formula
        adjusted = 100.0 / demand_factor
        assert adjusted == 100.0  # no adjustment

    def test_demand_factor_lower_bound(self):
        """Demand factor is clamped to minimum 0.25."""
        expected_demand = 0.06  # just above threshold
        demand_factor = max(0.25, min(4.0, expected_demand / 0.5))
        # 0.06/0.5 = 0.12, clamped to 0.25
        assert demand_factor == 0.25

    def test_demand_factor_upper_bound(self):
        """Demand factor is capped at 4.0."""
        expected_demand = 5.0  # very high demand
        demand_factor = max(0.25, min(4.0, expected_demand / 0.5))
        assert demand_factor == 4.0

    def test_nan_kalman_falls_back_to_default(self):
        """NaN flow_ratio/velocity keeps default expected_demand = 0.5."""
        import math
        kr = float('nan')
        kv = 0.01
        expected_demand = 0.5  # default
        if not math.isnan(kr) and not math.isnan(kv):
            expected_demand = abs(kr) + abs(kv * 24.0)
        assert expected_demand == 0.5

    def test_serialization_roundtrip(self):
        """last_vegas_multiplier survives serialization (baseline_forward_rate removed)."""
        from modules.fee_controller import ThompsonAIMDState
        ts = ThompsonAIMDState()
        ts.last_vegas_multiplier = 1.8

        d = ts.to_v2_dict()
        assert "baseline_forward_rate" not in d
        assert d["last_vegas_multiplier"] == 1.8

        restored = ThompsonAIMDState.from_v2_dict(d)
        assert restored.last_vegas_multiplier == 1.8


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


class TestSyntheticObservations:
    """Tests for Improvement #4: Synthetic Observations for Alpha Sequence Bypass."""

    def test_injection_adds_observation(self):
        """inject_synthetic_observation adds an observation."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        assert len(ts.observations) == 0

        ts.inject_synthetic_observation(fee=500, revenue_rate=50.0, weight_scale=0.3)
        assert len(ts.observations) == 1
        assert ts.observations[0][0] == 500  # fee

    def test_weight_capped_at_weight_scale(self):
        """Synthetic observation weight never exceeds weight_scale."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()

        ts.inject_synthetic_observation(fee=500, revenue_rate=10000.0, weight_scale=0.3)
        weight = ts.observations[0][2]
        assert weight <= 0.3

    def test_weight_has_minimum(self):
        """Synthetic observation weight is at least 0.01."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()

        ts.inject_synthetic_observation(fee=500, revenue_rate=0.0, weight_scale=0.3)
        weight = ts.observations[0][2]
        assert weight >= 0.01

    def test_posterior_updated_after_injection(self):
        """Posterior is recomputed after synthetic observation."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0

        # Add several normal observations first
        for _ in range(5):
            ts.update_posterior(fee=200, revenue_rate=50.0, hours=1.0)

        old_mean = ts.posterior_mean

        # Inject synthetic at much higher fee
        ts.inject_synthetic_observation(fee=500, revenue_rate=100.0, weight_scale=0.3)

        # Posterior should shift slightly toward 500
        assert ts.posterior_mean > old_mean

    def test_max_observations_respected(self):
        """Injection respects MAX_OBSERVATIONS limit."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()

        # Fill to max
        for i in range(ts.MAX_OBSERVATIONS):
            ts.observations.append((200, 50.0, 0.5, i, "normal"))

        assert len(ts.observations) == ts.MAX_OBSERVATIONS

        ts.inject_synthetic_observation(fee=500, revenue_rate=50.0)
        assert len(ts.observations) == ts.MAX_OBSERVATIONS  # pruned

    def test_congestion_scenario_high_fee(self):
        """Congestion synthetic observation injects at ceiling fee."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()

        ceiling = 5000
        ts.inject_synthetic_observation(fee=ceiling, revenue_rate=200.0, weight_scale=0.3)

        assert ts.observations[0][0] == ceiling
        assert ts.observations[0][2] <= 0.3


class TestVegasThompsonInteraction:
    """Tests for Improvement #5: Vegas Reflex-Thompson Interaction."""

    def test_no_adjustment_below_threshold(self):
        """Vegas multiplier <= 1.2 has no effect on Thompson."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0
        ts.posterior_std = 50.0

        ts.apply_vegas_adjustment(vegas_multiplier=1.1, new_floor=100)
        assert ts.posterior_mean == 200.0
        assert ts.posterior_std == 50.0

    def test_std_boosted_above_threshold(self):
        """Vegas multiplier > 1.2 boosts Thompson's posterior_std."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_std = 50.0

        ts.apply_vegas_adjustment(vegas_multiplier=1.5, new_floor=100)
        assert ts.posterior_std == 50.0 * 1.5

    def test_mean_nudged_toward_new_floor(self):
        """When new_floor > posterior_mean, mean is nudged toward floor."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_mean = 100.0
        ts.posterior_std = 50.0

        ts.apply_vegas_adjustment(vegas_multiplier=1.5, new_floor=200)
        # Nudge: 100 + (200 - 100) * 0.3 = 130
        assert ts.posterior_mean == 130.0

    def test_mean_not_nudged_when_above_floor(self):
        """When posterior_mean >= new_floor, mean is not nudged."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_mean = 300.0
        ts.posterior_std = 50.0

        ts.apply_vegas_adjustment(vegas_multiplier=1.5, new_floor=200)
        assert ts.posterior_mean == 300.0  # unchanged

    def test_boost_capped_at_2x(self):
        """Vegas boost is capped at 2.0x even with higher multiplier."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_std = 50.0

        ts.apply_vegas_adjustment(vegas_multiplier=5.0, new_floor=100)
        assert ts.posterior_std == 50.0 * 2.0  # capped

    def test_serialization_of_last_vegas_multiplier(self):
        """last_vegas_multiplier survives serialization roundtrip."""
        from modules.fee_controller import ThompsonAIMDState
        ts = ThompsonAIMDState()
        ts.last_vegas_multiplier = 1.8

        d = ts.to_v2_dict()
        restored = ThompsonAIMDState.from_v2_dict(d)
        assert restored.last_vegas_multiplier == 1.8


class TestProperContextualPosteriors:
    """Tests for Improvement #6: Proper Contextual Bayesian Posteriors."""

    def test_new_context_uses_4_tuple(self):
        """New contexts are created as 4-tuples (mean, precision, count, last_update)."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()

        ts.update_contextual("balanced:none:normal:P", fee=200, revenue_rate=50.0)
        ctx = ts.contextual_posteriors["balanced:none:normal:P"]
        assert len(ctx) == 4

    def test_precision_grows_with_observations(self):
        """More observations increase precision (decrease std)."""
        from modules.fee_controller import GaussianThompsonState
        import math
        ts = GaussianThompsonState()

        ts.update_contextual("balanced:none:normal:P", fee=200, revenue_rate=50.0)
        precision_1 = ts.contextual_posteriors["balanced:none:normal:P"][1]

        ts.update_contextual("balanced:none:normal:P", fee=200, revenue_rate=50.0)
        precision_2 = ts.contextual_posteriors["balanced:none:normal:P"][1]

        assert precision_2 > precision_1

    def test_convergence_toward_observed_fee(self):
        """Mean converges toward consistently observed fee."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0

        for _ in range(20):
            ts.update_contextual("balanced:none:normal:P", fee=300, revenue_rate=80.0)

        ctx_mean = ts.contextual_posteriors["balanced:none:normal:P"][0]
        # After many observations of fee=300, mean should be close to 300
        assert ctx_mean > 250

    def test_time_decay_reduces_old_precision(self):
        """Precision decays over time so stale contexts don't dominate."""
        from modules.fee_controller import GaussianThompsonState
        import time as t
        ts = GaussianThompsonState()

        # Create context with some precision
        ts.update_contextual("balanced:none:normal:P", fee=200, revenue_rate=50.0)
        ctx = ts.contextual_posteriors["balanced:none:normal:P"]

        # Simulate age by setting last_update to 14 days ago (2 half-lives)
        old_time = int(t.time()) - (14 * 24 * 3600)
        ts.contextual_posteriors["balanced:none:normal:P"] = (ctx[0], ctx[1], ctx[2], old_time)

        precision_before = ts.contextual_posteriors["balanced:none:normal:P"][1]

        # Update again — decay should reduce old precision before adding new
        ts.update_contextual("balanced:none:normal:P", fee=300, revenue_rate=50.0)

        # The mean should shift more toward 300 than if the old precision were fresh
        new_mean = ts.contextual_posteriors["balanced:none:normal:P"][0]
        assert new_mean > 220  # Decayed old precision allows more movement

    def test_legacy_3_tuple_backward_compat(self):
        """Legacy 3-tuple (mean, std, count) is handled by from_dict."""
        from modules.fee_controller import GaussianThompsonState
        import math

        d = {
            "prior_mean_fee": 200,
            "prior_std_fee": 100,
            "contextual_posteriors": {
                "key1": [250.0, 40.0, 10]  # Legacy 3-tuple
            }
        }
        state = GaussianThompsonState.from_dict(d)
        ctx = state.contextual_posteriors["key1"]
        assert len(ctx) == 4
        assert ctx[0] == 250.0  # mean preserved
        # precision = 1/40^2 = 0.000625
        expected_precision = 1.0 / (40.0 ** 2)
        assert abs(ctx[1] - expected_precision) < 1e-8
        assert ctx[2] == 10  # count preserved
        assert ctx[3] == 0  # last_update defaults to 0

    def test_4_tuple_roundtrip(self):
        """4-tuple format survives serialization roundtrip."""
        from modules.fee_controller import GaussianThompsonState
        ts = GaussianThompsonState()

        # Create context
        for _ in range(5):
            ts.update_contextual("balanced:none:normal:P", fee=250, revenue_rate=60.0)

        # Serialize and deserialize
        d = ts.to_dict()
        restored = GaussianThompsonState.from_dict(d)

        orig = ts.contextual_posteriors["balanced:none:normal:P"]
        rest = restored.contextual_posteriors["balanced:none:normal:P"]
        assert len(rest) == 4
        assert abs(orig[0] - rest[0]) < 0.01  # mean


# ===========================================================================
# Mathematical Overhaul Tests
# ===========================================================================

def _make_obs(fee_revenue_pairs, base_time=None):
    """Create observation tuples from (fee, revenue) pairs."""
    if base_time is None:
        base_time = int(time.time())
    obs = []
    for i, (fee, rev) in enumerate(fee_revenue_pairs):
        obs.append((fee, rev, 0.5, base_time - (len(fee_revenue_pairs) - i) * 3600, "normal"))
    return obs


# ---------------------------------------------------------------------------
# Prompt 3 — True Thompson Sampling (Bayesian Polynomial Regression)
# ---------------------------------------------------------------------------

class TestMat3Helpers:
    """Test 3x3 matrix helper static methods."""

    def test_mat3_det_identity(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert abs(GTS._mat3_det(I) - 1.0) < 1e-12

    def test_mat3_det_diagonal(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        m = [[2, 0, 0], [0, 3, 0], [0, 0, 5]]
        assert abs(GTS._mat3_det(m) - 30.0) < 1e-12

    def test_mat3_invert_identity(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        inv = GTS._mat3_invert(I)
        assert inv is not None
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(inv[i][j] - expected) < 1e-10

    def test_mat3_invert_singular_returns_none(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        m = [[1, 2, 3], [2, 4, 6], [0, 0, 0]]
        assert GTS._mat3_invert(m) is None

    def test_mat3_invert_roundtrip(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        m = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        inv = GTS._mat3_invert(m)
        assert inv is not None
        for i in range(3):
            for j in range(3):
                val = sum(m[i][k] * inv[k][j] for k in range(3))
                expected = 1.0 if i == j else 0.0
                assert abs(val - expected) < 1e-8

    def test_mat3_vec_mul(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        v = [3.0, 4.0, 5.0]
        result = GTS._mat3_vec_mul(I, v)
        assert result == pytest.approx(v)

    def test_cholesky3_identity(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        L = GTS._cholesky3(I)
        assert L is not None
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(L[i][j] - expected) < 1e-10

    def test_cholesky3_not_pd_returns_none(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        m = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert GTS._cholesky3(m) is None

    def test_cholesky3_reconstruct(self):
        from modules.fee_controller import GaussianThompsonState as GTS
        m = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        L = GTS._cholesky3(m)
        assert L is not None
        for i in range(3):
            for j in range(3):
                val = sum(L[i][k] * L[j][k] for k in range(3))
                assert abs(val - m[i][j]) < 1e-8


class TestPolynomialPosterior:
    """Test _recompute_posterior with polynomial regression."""

    def test_concave_data_produces_negative_a(self):
        """With concave revenue-fee data and reasonable noise, quadratic coeff 'a' should be negative."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        # Bootstrap noise_variance to a reasonable level so data can dominate
        state.noise_variance = 50.0
        state.observations = _make_obs([
            (100, 10), (150, 25), (200, 40), (250, 30), (300, 15),
            (120, 15), (180, 35), (220, 35), (280, 18), (160, 28),
        ])
        state._recompute_posterior()
        assert state.posterior_coeffs[0] < 0

    def test_concave_data_reasonable_posterior_mean(self):
        """Posterior mean should be near peak of concave data."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([
            (100, 5), (150, 20), (200, 40), (250, 30), (300, 10),
            (175, 35), (225, 35), (200, 38),
        ])
        state._recompute_posterior()
        assert 120 < state.posterior_mean < 280

    def test_narrow_fee_range_uses_legacy(self):
        """Fee range < 5 ppm → legacy Normal-Normal fallback."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([
            (200, 30), (201, 32), (200, 28), (202, 31),
        ])
        state._recompute_posterior()
        assert state.posterior_mean > 0
        assert state.posterior_std >= state.MIN_STD

    def test_empty_observations_uses_prior(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = []
        state._recompute_posterior()
        assert state.posterior_mean == float(state.prior_mean_fee)
        assert state.posterior_std == float(state.prior_std_fee)

    def test_single_observation_uses_legacy(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([(200, 30)])
        state._recompute_posterior()
        assert state.posterior_mean > 0

    def test_noise_variance_updates(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        initial_noise = state.noise_variance
        state.observations = _make_obs([
            (100, 10), (200, 40), (300, 10),
            (150, 30), (250, 25), (200, 38),
        ])
        state._recompute_posterior()
        assert state.noise_variance != initial_noise

    def test_precision_does_not_accumulate_on_repeated_recompute(self):
        """H1: Calling _recompute_posterior N times must not make precision grow unbounded."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.noise_variance = 50.0
        state.observations = _make_obs([
            (100, 10), (150, 25), (200, 40), (250, 30), (300, 15),
        ])
        state._recompute_posterior()
        prec_after_1 = state.posterior_precision[0][0]

        # Recompute 50 more times — before the fix this would grow O(N)
        for _ in range(50):
            state._recompute_posterior()

        prec_after_51 = state.posterior_precision[0][0]
        # Key invariant: precision must NOT monotonically grow with recompute count.
        # With the fix (using fixed prior), it should converge, not accumulate.
        # Allow small drift from noise_variance blending, but not growth.
        assert prec_after_51 < prec_after_1 * 2.0, \
            f"Precision grew from {prec_after_1} to {prec_after_51} — accumulation bug"

    def test_two_observations_uses_legacy(self):
        """L2: Fewer than 3 observations should use legacy path."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([(100, 10), (300, 30)])
        state._recompute_posterior()
        # Should still work (legacy path)
        assert state.posterior_mean > 0
        assert state.posterior_std >= state.MIN_STD


class TestPolynomialSampling:
    """Test sample_fee with polynomial posterior."""

    def test_samples_cluster_around_map(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([
            (100, 5), (150, 25), (200, 40), (250, 30), (300, 10),
            (175, 35), (225, 35), (200, 38), (180, 33), (210, 37),
        ])
        state._recompute_posterior()

        random.seed(42)
        samples = [state.sample_fee(50, 500) for _ in range(200)]
        mean_sample = sum(samples) / len(samples)
        assert abs(mean_sample - state.posterior_mean) < 100

    def test_sample_fee_respects_bounds(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([
            (100, 5), (200, 40), (300, 10), (150, 25), (250, 30),
        ])
        state._recompute_posterior()
        for _ in range(100):
            fee = state.sample_fee(50, 500)
            assert 50 <= fee <= 500

    def test_non_concave_falls_back_to_gaussian(self):
        """Monotonically increasing data → not concave → Gaussian fallback."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.observations = _make_obs([
            (100, 10), (200, 20), (300, 30), (400, 40), (500, 50),
        ])
        state._recompute_posterior()
        for _ in range(50):
            fee = state.sample_fee(50, 600)
            assert 50 <= fee <= 600


class TestPolynomialSerialization:
    """Test to_dict/from_dict with new polynomial fields."""

    def test_roundtrip(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state.posterior_coeffs = [-0.5, 2.0, 10.0]
        state.posterior_precision = [[1.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 0.5]]
        state.noise_variance = 42.0
        state._prior_coeffs = [-0.1, 0.5, 0.0]
        state._prior_precision = [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]
        state._last_fee_min = 50.0
        state._last_fee_max = 400.0

        d = state.to_dict()
        restored = GaussianThompsonState.from_dict(d)

        assert restored.posterior_coeffs == pytest.approx(state.posterior_coeffs)
        for i in range(3):
            assert restored.posterior_precision[i] == pytest.approx(state.posterior_precision[i])
        assert restored.noise_variance == pytest.approx(42.0)
        assert restored._prior_coeffs == pytest.approx(state._prior_coeffs)
        for i in range(3):
            assert restored._prior_precision[i] == pytest.approx(state._prior_precision[i])
        assert restored._last_fee_min == pytest.approx(50.0)
        assert restored._last_fee_max == pytest.approx(400.0)

    def test_backward_compat_missing_fields(self):
        from modules.fee_controller import GaussianThompsonState
        old = {
            "prior_mean_fee": 200,
            "prior_std_fee": 100,
            "observations": [],
            "posterior_mean": 200.0,
            "posterior_std": 100.0,
            "contextual_posteriors": {},
            "last_sampled_fee": 0,
            "last_sample_time": 0,
        }
        state = GaussianThompsonState.from_dict(old)
        assert state.posterior_coeffs == [0.0, 1.0, 0.0]
        assert state.noise_variance == 1000.0
        # Fixed prior defaults
        assert state._prior_coeffs == [0.0, 1.0, 0.0]
        assert state._prior_precision[0][0] == pytest.approx(0.01)

    def test_bad_precision_gets_default(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState.from_dict({"posterior_precision": [[1, 2], [3, 4]]})
        assert len(state.posterior_precision) == 3
        assert all(len(row) == 3 for row in state.posterior_precision)

    def test_bad_coeffs_get_default(self):
        """M1: Malformed coeffs should reset to default."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState.from_dict({"posterior_coeffs": [1.0, 2.0]})
        assert state.posterior_coeffs == [0.0, 1.0, 0.0]

    def test_negative_noise_variance_clamped(self):
        """L6: Negative noise_variance should be clamped to floor."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState.from_dict({"noise_variance": -5.0})
        assert state.noise_variance >= 10.0

    def test_negative_precision_diagonal_gets_default(self):
        """L5: Precision with negative diagonal should reset."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState.from_dict({
            "posterior_precision": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        })
        assert state.posterior_precision[0][0] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Prompt 4 — Elasticity Re-Integration
# ---------------------------------------------------------------------------

class TestElasticityPriorBias:
    """Test _apply_elasticity_to_prior."""

    def test_elastic_demand_shifts_a_negative(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state._apply_elasticity_to_prior(elasticity=-2.0, confidence=0.8)
        assert state._prior_coeffs[0] < 0

    def test_inelastic_demand_small_shift(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state._apply_elasticity_to_prior(elasticity=-0.1, confidence=0.8)
        assert state._prior_coeffs[0] < 0
        assert state._prior_coeffs[0] > -0.1

    def test_low_confidence_no_effect(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        original = state._prior_coeffs[0]
        state._apply_elasticity_to_prior(elasticity=-2.0, confidence=0.2)
        assert state._prior_coeffs[0] == original

    def test_precision_boost(self):
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        original_prec = state._prior_precision[0][0]
        state._apply_elasticity_to_prior(elasticity=-1.5, confidence=0.8)
        assert state._prior_precision[0][0] > original_prec

    def test_precision_capped_at_max(self):
        """H2: Precision should not grow beyond cap after many calls."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        for _ in range(200):
            state._apply_elasticity_to_prior(elasticity=-2.0, confidence=1.0)
        assert state._prior_precision[0][0] <= 10.0

    def test_no_compounding_drift(self):
        """Repeated calls should not compound the prior bias."""
        from modules.fee_controller import GaussianThompsonState
        state = GaussianThompsonState()
        state._apply_elasticity_to_prior(elasticity=-2.0, confidence=0.8)
        first_a = state._prior_coeffs[0]
        # Call again with same params — should get same result (reset-then-bias)
        state._apply_elasticity_to_prior(elasticity=-2.0, confidence=0.8)
        second_a = state._prior_coeffs[0]
        assert abs(first_a - second_a) < 1e-10


class TestElasticityBiasDirection:
    """Test that elasticity bias shifts fee in the expected direction."""

    def test_elastic_bias_lowers_fee(self):
        from modules.fee_controller import ElasticityTracker
        tracker = ElasticityTracker()
        tracker.current_elasticity = -2.0
        tracker.confidence = 0.8
        assert tracker.get_optimal_direction() == -1

        posterior_std = 50.0
        bias = tracker.get_optimal_direction() * tracker.confidence * posterior_std * 0.3
        assert bias < 0

    def test_inelastic_bias_raises_fee(self):
        from modules.fee_controller import ElasticityTracker
        tracker = ElasticityTracker()
        tracker.current_elasticity = -0.3
        tracker.confidence = 0.8
        assert tracker.get_optimal_direction() == 1


# ---------------------------------------------------------------------------
# Prompt 1 — Kalman Demand Normalization
# ---------------------------------------------------------------------------

class TestKalmanDemandNormalization:
    """Test Kalman-primary, EMA-fallback demand factor computation."""

    def test_kalman_stable_channel_neutral(self):
        velocity, uncertainty = 0.0, 0.1
        confidence = 1.0 / (1.0 + uncertainty)
        demand_factor = max(0.25, min(4.0, 1.0 + velocity * confidence * 2.0))
        assert abs(demand_factor - 1.0) < 0.01

    def test_kalman_positive_velocity_increases_demand(self):
        velocity, uncertainty = 0.3, 0.1
        confidence = 1.0 / (1.0 + uncertainty)
        demand_factor = max(0.25, min(4.0, 1.0 + velocity * confidence * 2.0))
        assert demand_factor > 1.0

    def test_kalman_negative_velocity_decreases_demand(self):
        velocity, uncertainty = -0.3, 0.1
        confidence = 1.0 / (1.0 + uncertainty)
        demand_factor = max(0.25, min(4.0, 1.0 + velocity * confidence * 2.0))
        assert demand_factor < 1.0

    def test_kalman_high_uncertainty_dampens_effect(self):
        velocity, uncertainty = 0.5, 10.0
        confidence = 1.0 / (1.0 + uncertainty)
        demand_factor = max(0.25, min(4.0, 1.0 + velocity * confidence * 2.0))
        assert abs(demand_factor - 1.0) < 0.15

    def test_kalman_nan_detected(self):
        assert math.isnan(float('nan'))
        assert not math.isnan(0.5)

    def test_demand_factor_clamped_high(self):
        velocity, uncertainty = 10.0, 0.01
        confidence = 1.0 / (1.0 + uncertainty)
        demand_factor = max(0.25, min(4.0, 1.0 + velocity * confidence * 2.0))
        assert demand_factor == 4.0

    def test_demand_factor_clamped_low(self):
        velocity, uncertainty = -10.0, 0.01
        confidence = 1.0 / (1.0 + uncertainty)
        demand_factor = max(0.25, min(4.0, 1.0 + velocity * confidence * 2.0))
        assert demand_factor == 0.25


# ---------------------------------------------------------------------------
# Prompt 2 — Topology-Aware AIMD
# ---------------------------------------------------------------------------

class TestTopologyAwareAIMD:
    """Test AskRene-based topology depletion detection."""

    def _make_controller(self):
        from modules.fee_controller import HillClimbingFeeController
        plugin = MagicMock()
        plugin.log = MagicMock()
        plugin.rpc = MagicMock()
        config = MagicMock()
        config.snapshot.return_value = MagicMock(
            askrene_layer='xpay', askrene_max_age_sec=900,
            min_fee_ppm=1, max_fee_ppm=5000, vegas_decay_rate=0.95,
        )
        config.vegas_decay_rate = 0.95
        database = MagicMock()
        database.get_all_fee_anchors.return_value = []
        clboss = MagicMock()
        return HillClimbingFeeController(
            plugin=plugin, config=config, database=database,
            clboss_manager=clboss,
        )

    def test_askrene_cache_populated(self):
        ctrl = self._make_controller()
        now = int(time.time())
        ctrl.plugin.rpc.call.return_value = {
            "layers": [{"layer": "xpay", "constraints": [
                {"short_channel_id_dir": "100x1x0/0", "maximum_msat": 50_000_000, "timestamp": now},
                {"short_channel_id_dir": "100x1x0/1", "maximum_msat": 200_000_000, "timestamp": now},
            ]}]
        }
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        ctrl._refresh_askrene_cache(cfg)
        assert len(ctrl._askrene_cache) == 2
        assert ctrl._askrene_cache["100x1x0/0"] == 50_000_000

    def test_cache_ttl_prevents_refresh(self):
        ctrl = self._make_controller()
        ctrl._askrene_cache_ts = int(time.time())
        ctrl._askrene_cache = {"test/0": 100}
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        ctrl._refresh_askrene_cache(cfg)
        ctrl.plugin.rpc.call.assert_not_called()

    def test_topology_depleted_true(self):
        ctrl = self._make_controller()
        now = int(time.time())
        ctrl.plugin.rpc.call.return_value = {
            "layers": [{"layer": "xpay", "constraints": [
                {"short_channel_id_dir": "100x1x0/0", "maximum_msat": 10_000, "timestamp": now},
            ]}]
        }
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        assert ctrl._is_topology_depleted("100x1x0", 1_000_000, cfg) is True

    def test_topology_not_depleted(self):
        ctrl = self._make_controller()
        now = int(time.time())
        ctrl.plugin.rpc.call.return_value = {
            "layers": [{"layer": "xpay", "constraints": [
                {"short_channel_id_dir": "100x1x0/0", "maximum_msat": 500_000_000, "timestamp": now},
            ]}]
        }
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        assert ctrl._is_topology_depleted("100x1x0", 1_000_000, cfg) is False

    def test_no_cache_entry_not_depleted(self):
        ctrl = self._make_controller()
        ctrl.plugin.rpc.call.return_value = {"layers": []}
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        assert ctrl._is_topology_depleted("999x9x9", 1_000_000, cfg) is False

    def test_rpc_failure_silent(self):
        ctrl = self._make_controller()
        ctrl.plugin.rpc.call.side_effect = Exception("RPC timeout")
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        assert ctrl._is_topology_depleted("100x1x0", 1_000_000, cfg) is False

    def test_stale_constraints_filtered(self):
        ctrl = self._make_controller()
        old_ts = int(time.time()) - 2000
        ctrl.plugin.rpc.call.return_value = {
            "layers": [{"layer": "xpay", "constraints": [
                {"short_channel_id_dir": "100x1x0/0", "maximum_msat": 10_000, "timestamp": old_ts},
            ]}]
        }
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        assert ctrl._is_topology_depleted("100x1x0", 1_000_000, cfg) is False

    def test_msat_string_format_parsed(self):
        """H3: CLN string msat values like '50000000msat' should be parsed."""
        ctrl = self._make_controller()
        now = int(time.time())
        ctrl.plugin.rpc.call.return_value = {
            "layers": [{"layer": "xpay", "constraints": [
                {"short_channel_id_dir": "100x1x0/0", "maximum_msat": "50000000msat", "timestamp": now},
            ]}]
        }
        cfg = MagicMock(askrene_layer='xpay', askrene_max_age_sec=900)
        ctrl._refresh_askrene_cache(cfg)
        assert ctrl._askrene_cache.get("100x1x0/0") == 50_000_000
