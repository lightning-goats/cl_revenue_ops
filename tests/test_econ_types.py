"""Phase 1: checked economic domain types (wire-contract-spec.md J2).

Every violated bound raises EconArithmeticError (fail closed). Explicit
methods only — no silent mixing with plain ints.
"""
import dataclasses

import pytest

from modules.econ_types import (
    U63_MAX,
    I64_MIN,
    ChannelId,
    EconArithmeticError,
    IntentId,
    Micro,
    Msat,
    PeerId,
    Ppm,
    Sat,
    SignedMsat,
    UnixTime,
)


class TestMsatBounds:
    def test_zero_and_max_ok(self):
        assert Msat(0).value == 0
        assert Msat(U63_MAX).value == U63_MAX

    @pytest.mark.parametrize("bad", [-1, U63_MAX + 1, True, 1.5, "10"])
    def test_invalid_raises(self, bad):
        with pytest.raises(EconArithmeticError):
            Msat(bad)

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Msat(1).value = 2


class TestMsatArithmetic:
    def test_add(self):
        assert Msat(1000).add(Msat(500)) == Msat(1500)

    def test_add_overflow_fails_closed(self):
        with pytest.raises(EconArithmeticError):
            Msat(U63_MAX).add(Msat(1))

    def test_sub(self):
        assert Msat(1500).sub(Msat(500)) == Msat(1000)

    def test_sub_below_zero_fails_closed(self):
        with pytest.raises(EconArithmeticError):
            Msat(1).sub(Msat(2))

    def test_diff_signed(self):
        assert Msat(1000).diff(Msat(1500)) == SignedMsat(-500)
        assert Msat(1500).diff(Msat(1000)) == SignedMsat(500)

    def test_no_plain_int_mixing(self):
        with pytest.raises((EconArithmeticError, AttributeError, TypeError)):
            Msat(1).add(1)  # type: ignore[arg-type]


class TestConversions:
    """The three canonical rounding rules (wire-contract-spec.md J2)."""

    def test_ceil_for_costs_and_revenue(self):
        assert Msat(1500).to_sats_ceil() == Sat(2)
        assert Msat(1000).to_sats_ceil() == Sat(1)
        assert Msat(1).to_sats_ceil() == Sat(1)  # sub-sat stays visible

    def test_floor_for_balances(self):
        assert Msat(1500).to_sats_floor() == Sat(1)
        assert Msat(999).to_sats_floor() == Sat(0)

    def test_toward_zero_for_signed_deltas(self):
        assert SignedMsat(-1500).to_sats_toward_zero() == -1
        assert SignedMsat(1500).to_sats_toward_zero() == 1

    def test_sat_to_msat_exact(self):
        assert Sat(2).to_msat() == Msat(2000)
        with pytest.raises(EconArithmeticError):
            Sat(U63_MAX).to_msat()  # x1000 overflows

    def test_msat_from_sats(self):
        assert Msat.from_sats(3) == Msat(3000)


class TestPpm:
    def test_fee_ceil_exact(self):
        assert Ppm(250).fee_ceil(Msat(1_000_000)) == Msat(250)

    def test_fee_ceil_rounds_up(self):
        # 1 msat * 250ppm = 0.00025 -> ceil 1
        assert Ppm(250).fee_ceil(Msat(1)) == Msat(1)

    def test_fee_floor_rounds_down(self):
        assert Ppm(250).fee_floor(Msat(1)) == Msat(0)

    @pytest.mark.parametrize("bad", [-1, 10_000_001, 1.0])
    def test_bounds(self, bad):
        with pytest.raises(EconArithmeticError):
            Ppm(bad)


class TestMicro:
    def test_bounds(self):
        assert Micro(0).value == 0
        assert Micro(1_000_000).value == 1_000_000
        with pytest.raises(EconArithmeticError):
            Micro(1_000_001)
        with pytest.raises(EconArithmeticError):
            Micro(-1)

    def test_from_float_clamped(self):
        assert Micro.from_float_clamped(0.85) == Micro(850000)
        assert Micro.from_float_clamped(1.7) == Micro(1_000_000)
        assert Micro.from_float_clamped(-0.5) == Micro(0)


class TestIdentifiers:
    def test_channel_id(self):
        assert ChannelId("123x456x0").value == "123x456x0"
        with pytest.raises(EconArithmeticError):
            ChannelId("bogus")

    def test_peer_id(self):
        assert PeerId("02" + "a" * 64).value == "02" + "a" * 64
        with pytest.raises(EconArithmeticError):
            PeerId("02" + "a" * 63)
        with pytest.raises(EconArithmeticError):
            PeerId("04" + "a" * 64)

    def test_intent_id(self):
        assert IntentId("int-abc123").value == "int-abc123"
        with pytest.raises(EconArithmeticError):
            IntentId("")
        with pytest.raises(EconArithmeticError):
            IntentId("UPPER")


class TestUnixTime:
    def test_bounds_and_plus(self):
        assert UnixTime(0).plus_seconds(60) == UnixTime(60)
        with pytest.raises(EconArithmeticError):
            UnixTime(-1)
        with pytest.raises(EconArithmeticError):
            UnixTime(U63_MAX).plus_seconds(1)


class TestSignedMsat:
    def test_range(self):
        assert SignedMsat(I64_MIN).value == I64_MIN
        assert SignedMsat(U63_MAX).value == U63_MAX
        with pytest.raises(EconArithmeticError):
            SignedMsat(I64_MIN - 1)
