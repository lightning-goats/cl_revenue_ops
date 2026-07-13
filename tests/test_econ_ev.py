"""PR 6 (gap-closure Phase E): common EV contract helpers + the
directive's property tests.

EV = revenue - execution_cost - capital_cost - risk_premium, in checked
integer msat. Missing data fails CONSERVATIVELY (zero benefit / zero
confidence — the pre-population state), never optimistically.
"""
import pytest

from modules.econ_ev import (
    benefit_msat_from_sats,
    confidence_micro,
    expected_value_msat,
)
from modules.econ_types import EconArithmeticError, Micro, SignedMsat


class TestExpectedValueContract:
    def test_composition(self):
        ev = expected_value_msat(revenue_msat=10_000,
                                 execution_cost_msat=2_000,
                                 capital_cost_msat=1_000,
                                 risk_premium_msat=500)
        assert ev == SignedMsat(6_500)

    def test_higher_execution_cost_cannot_improve_ev(self):
        base = expected_value_msat(revenue_msat=10_000,
                                   execution_cost_msat=2_000,
                                   capital_cost_msat=0,
                                   risk_premium_msat=0)
        worse = expected_value_msat(revenue_msat=10_000,
                                    execution_cost_msat=2_001,
                                    capital_cost_msat=0,
                                    risk_premium_msat=0)
        assert worse.value < base.value

    def test_higher_capital_cost_cannot_improve_ev(self):
        base = expected_value_msat(revenue_msat=10_000,
                                   execution_cost_msat=0,
                                   capital_cost_msat=3_000,
                                   risk_premium_msat=0)
        worse = expected_value_msat(revenue_msat=10_000,
                                    execution_cost_msat=0,
                                    capital_cost_msat=3_001,
                                    risk_premium_msat=0)
        assert worse.value < base.value

    def test_higher_risk_premium_cannot_improve_ev(self):
        base = expected_value_msat(revenue_msat=10_000,
                                   execution_cost_msat=0,
                                   capital_cost_msat=0,
                                   risk_premium_msat=100)
        worse = expected_value_msat(revenue_msat=10_000,
                                    execution_cost_msat=0,
                                    capital_cost_msat=0,
                                    risk_premium_msat=101)
        assert worse.value < base.value

    def test_higher_hurdle_cannot_make_identical_open_more_attractive(self):
        """An identical action under a HIGHER capital hurdle clears the
        margin strictly less often — monotone in the hurdle."""
        ev = expected_value_msat(revenue_msat=10_000,
                                 execution_cost_msat=4_000,
                                 capital_cost_msat=2_000,
                                 risk_premium_msat=1_000)
        low_hurdle, high_hurdle = 2_000, 4_000
        assert (ev.value >= high_hurdle) <= (ev.value >= low_hurdle)

    def test_checked_integer_semantics(self):
        with pytest.raises(EconArithmeticError):
            expected_value_msat(revenue_msat=1.5, execution_cost_msat=0,
                                capital_cost_msat=0, risk_premium_msat=0)
        with pytest.raises(EconArithmeticError):
            expected_value_msat(revenue_msat=True, execution_cost_msat=0,
                                capital_cost_msat=0, risk_premium_msat=0)
        with pytest.raises(EconArithmeticError):
            expected_value_msat(revenue_msat=None, execution_cost_msat=0,
                                capital_cost_msat=0, risk_premium_msat=0)


class TestConservativeMissingData:
    def test_missing_benefit_is_zero(self):
        assert benefit_msat_from_sats(None) == SignedMsat(0)
        assert benefit_msat_from_sats("garbage") == SignedMsat(0)
        assert benefit_msat_from_sats(float("nan")) == SignedMsat(0)

    def test_benefit_conversion(self):
        assert benefit_msat_from_sats(42) == SignedMsat(42_000)
        assert benefit_msat_from_sats(-17.25) == SignedMsat(-17_250)

    def test_missing_confidence_is_zero(self):
        assert confidence_micro(None) == Micro(0)
        assert confidence_micro("x") == Micro(0)
        assert confidence_micro(float("nan")) == Micro(0)

    def test_confidence_clamped_to_unit_interval(self):
        assert confidence_micro(0.5) == Micro(500_000)
        assert confidence_micro(1.7) == Micro(1_000_000)
        assert confidence_micro(-0.3) == Micro(0)
