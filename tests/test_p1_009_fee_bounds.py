"""P1-009: enforce min_fee_ppm <= max_fee_ppm at init.

An inverted config is corrected (swapped) with a warning, not silently
pinned to a low ceiling.
"""

import pytest

from tests.plugin_test_utils import load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def _val(mod, **kw):
    return mod._enforce_fee_bound_invariant(dict(kw), log=None)


def test_inverted_fee_bounds_swapped(mod):
    out = _val(mod, min_fee_ppm=500, max_fee_ppm=100)
    assert out["min_fee_ppm"] <= out["max_fee_ppm"]
    assert {out["min_fee_ppm"], out["max_fee_ppm"]} == {100, 500}


def test_ordered_fee_bounds_unchanged(mod):
    out = _val(mod, min_fee_ppm=10, max_fee_ppm=1000)
    assert out["min_fee_ppm"] == 10 and out["max_fee_ppm"] == 1000


def test_equal_fee_bounds_unchanged(mod):
    out = _val(mod, min_fee_ppm=200, max_fee_ppm=200)
    assert out["min_fee_ppm"] == 200 and out["max_fee_ppm"] == 200
