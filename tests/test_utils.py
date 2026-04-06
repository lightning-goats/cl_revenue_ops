"""Tests for unit conversion utilities."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import (
    BASE_UNITS_PER_SAT,
    BASE_UNIT_NAME,
    base_to_sats_ceil,
    base_to_sats_floor,
    sats_to_base,
    parse_base_unit,
    MSAT_PER_SAT,
    msat_to_sats_ceil,
    msat_to_sats_floor,
    sats_to_msat,
    parse_msat,
)


class TestConstants:
    def test_base_units_per_sat(self):
        assert BASE_UNITS_PER_SAT == 1000

    def test_base_unit_name(self):
        assert BASE_UNIT_NAME == "msat"

    def test_msat_alias(self):
        assert MSAT_PER_SAT is BASE_UNITS_PER_SAT


class TestBaseToSatsCeil:
    def test_zero(self):
        assert base_to_sats_ceil(0) == 0

    def test_exact_multiple(self):
        assert base_to_sats_ceil(1000) == 1

    def test_rounds_up_1_msat(self):
        assert base_to_sats_ceil(1) == 1

    def test_rounds_up_999_msat(self):
        assert base_to_sats_ceil(999) == 1

    def test_rounds_up_1001_msat(self):
        assert base_to_sats_ceil(1001) == 2

    def test_large_value(self):
        assert base_to_sats_ceil(500_000_000) == 500_000

    def test_large_value_plus_1(self):
        assert base_to_sats_ceil(500_000_001) == 500_001

    def test_alias_is_same_function(self):
        assert msat_to_sats_ceil is base_to_sats_ceil


class TestBaseToSatsFloor:
    def test_zero(self):
        assert base_to_sats_floor(0) == 0

    def test_exact_multiple(self):
        assert base_to_sats_floor(1000) == 1

    def test_truncates_999_msat(self):
        assert base_to_sats_floor(999) == 0

    def test_truncates_1999_msat(self):
        assert base_to_sats_floor(1999) == 1

    def test_large_value(self):
        assert base_to_sats_floor(500_000_999) == 500_000

    def test_alias_is_same_function(self):
        assert msat_to_sats_floor is base_to_sats_floor


class TestSatsToBase:
    def test_zero(self):
        assert sats_to_base(0) == 0

    def test_one_sat(self):
        assert sats_to_base(1) == 1000

    def test_large_value(self):
        assert sats_to_base(500_000) == 500_000_000

    def test_roundtrip_exact(self):
        assert base_to_sats_floor(sats_to_base(42)) == 42

    def test_roundtrip_ceil(self):
        assert base_to_sats_ceil(sats_to_base(42)) == 42

    def test_alias_is_same_function(self):
        assert sats_to_msat is sats_to_base


class TestParseBaseUnit:
    def test_integer(self):
        assert parse_base_unit(1000) == 1000

    def test_float(self):
        assert parse_base_unit(1000.0) == 1000

    def test_string_plain(self):
        assert parse_base_unit("1000") == 1000

    def test_string_msat_suffix(self):
        assert parse_base_unit("1000msat") == 1000

    def test_none_returns_zero(self):
        assert parse_base_unit(None) == 0

    def test_bool_returns_zero(self):
        assert parse_base_unit(True) == 0
        assert parse_base_unit(False) == 0

    def test_invalid_string_returns_zero(self):
        assert parse_base_unit("notanumber") == 0

    def test_alias_is_same_function(self):
        assert parse_msat is parse_base_unit
