"""P1-026: validate enum-style string options at init.

market_fee_mode / base_fee_policy / fee_profile / preferred_currency are
checked against their valid sets; an unknown value warns and falls back to
the documented default instead of silently mis-behaving.
"""

import pytest

from tests.plugin_test_utils import load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def _val(mod, **kw):
    return mod._validate_enum_config_options(dict(kw), log=None)


def test_market_fee_mode_typo_defaulted(mod):
    assert _val(mod, market_fee_mode="undercutt")["market_fee_mode"] == "undercut"


def test_market_fee_mode_valid_unchanged(mod):
    assert _val(mod, market_fee_mode="competition_aware")["market_fee_mode"] == "competition_aware"


def test_base_fee_policy_typo_defaulted(mod):
    assert _val(mod, base_fee_policy="offf")["base_fee_policy"] == "off"


def test_base_fee_policy_adaptive_unchanged(mod):
    assert _val(mod, base_fee_policy="adaptive")["base_fee_policy"] == "adaptive"


def test_fee_profile_typo_defaulted(mod):
    assert _val(mod, fee_profile="activ")["fee_profile"] == "active"


def test_fee_profile_valid_unchanged(mod):
    assert _val(mod, fee_profile="conservative")["fee_profile"] == "conservative"


def test_preferred_currency_typo_defaulted(mod):
    assert _val(mod, expansion_treasury_preferred_currency="XYZ")["expansion_treasury_preferred_currency"] == "BTC"


def test_preferred_currency_lbtc_unchanged(mod):
    assert _val(mod, expansion_treasury_preferred_currency="LBTC")["expansion_treasury_preferred_currency"] == "LBTC"
