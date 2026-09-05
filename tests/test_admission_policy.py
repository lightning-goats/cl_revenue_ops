"""Phase 3A: admission-control policy extraction.

The Phase 0 golden fixtures (via the FeeController shims) are the
parity oracle; these tests exercise the extracted module directly and
pin the extraction pattern."""
from types import SimpleNamespace

import pytest

from modules import admission_policy


from modules.fee_authority import FeeAuthorityGate

def _cfg(**over):
    base = dict(
        enable_dynamic_htlcmax=True,
        htlcmax_source_pct=0.85,
        htlcmax_sink_pct=0.25,
        htlcmax_balanced_pct=0.50,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_module_is_pure_of_plugin_dependencies():
    import inspect
    source = inspect.getsource(admission_policy)
    for forbidden in ("self.plugin", ".rpc.", "from .database",
                      "import time", "import random", "Database("):
        assert forbidden not in source, forbidden
    import_lines = [line for line in source.splitlines()
                    if line.startswith(("import ", "from "))]
    allowed = {"from __future__ import annotations",
               "from typing import Any, Dict, Optional",
               "from .utils import sats_to_base"}
    assert set(import_lines) <= allowed, import_lines


def test_direct_matches_shim():
    """The FeeController shim and the module must be the same function."""
    from unittest.mock import MagicMock
    from modules.config import Config
    from modules.fee_controller import FeeController
    fc = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock(), fee_authority_gate=FeeAuthorityGate())
    chan = {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}
    for state in ("source", "sink", "balanced"):
        assert fc._compute_dynamic_htlcmax_msat(_cfg(), chan, state) == \
            admission_policy.compute_htlcmax_msat(_cfg(), chan, state)
    assert fc._htlcmax_delta_exceeds_deadband(900, 500) == \
        admission_policy.delta_exceeds_deadband(900, 500)


def test_constants_alias_intact():
    from unittest.mock import MagicMock
    from modules.config import Config
    from modules.fee_controller import FeeController
    fc = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock(), fee_authority_gate=FeeAuthorityGate())
    assert fc.HTLCMAX_FLOOR_MSAT == admission_policy.FLOOR_MSAT
    assert fc.HTLCMAX_DEPLETION_SPENDABLE_FRACTION == \
        admission_policy.DEPLETION_SPENDABLE_FRACTION
    assert fc.HTLCMAX_UPDATE_DEADBAND_FRAC == \
        admission_policy.UPDATE_DEADBAND_FRAC


def test_core_semantics_direct():
    assert admission_policy.compute_htlcmax_msat(
        _cfg(enable_dynamic_htlcmax=False),
        {"capacity": 2_000_000, "spendable_msat": 1}, "source") is None
    result = admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 2_000_000, "spendable_msat": 1_900_000_000},
        "sink")
    assert result == 500_000_000
    assert admission_policy.delta_exceeds_deadband(500, 500) is False
    assert admission_policy.delta_exceeds_deadband(500, 0) is True


def test_default_policy_advertises_truthful_live_spendable_capacity_for_all_classes():
    from modules.config import Config

    channel = {"capacity": 1_000_000, "spendable_msat": "970000000msat"}
    expected = int(970_000_000 * admission_policy.DEPLETION_SPENDABLE_FRACTION)
    for state in ("source", "sink", "balanced"):
        assert admission_policy.compute_htlcmax_msat(Config(), channel, state) == expected


def test_missing_or_malformed_admission_evidence_is_neutral():
    good = {"capacity": 1_000_000, "spendable_msat": "970000000msat"}
    assert admission_policy.compute_htlcmax_msat(_cfg(), {}, "sink") is None
    assert admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 1_000_000}, "sink"
    ) is None
    for bad in (None, True, -1, 1.5, "bad"):
        assert admission_policy.compute_htlcmax_msat(
            _cfg(), {**good, "capacity": bad}, "sink"
        ) is None
        assert admission_policy.compute_htlcmax_msat(
            _cfg(), {**good, "spendable_msat": bad}, "sink"
        ) is None
    for bad in (None, True, 0, -1, 1.01, float("nan"), "bad"):
        assert admission_policy.compute_htlcmax_msat(
            _cfg(htlcmax_sink_pct=bad), good, "sink"
        ) is None


@pytest.mark.parametrize("state", ["source", "sink", "balanced"])
@pytest.mark.parametrize("spendable_msat", [0, 1, 1_000, 9_000_000, 10_000_000, 11_764_706])
def test_preferred_floor_never_overrides_executable_liquidity(state, spendable_msat):
    result = admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 15_000_000, "spendable_msat": spendable_msat}, state
    )
    assert result == int(spendable_msat * admission_policy.DEPLETION_SPENDABLE_FRACTION)
    assert 0 <= result <= spendable_msat


def test_admission_reopens_from_zero_when_liquidity_returns():
    empty = admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 2_000_000, "spendable_msat": 0}, "source"
    )
    refilled = admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 2_000_000, "spendable_msat": 100_000_000}, "source"
    )
    assert empty == 0
    assert refilled == 85_000_000
    assert admission_policy.delta_exceeds_deadband(refilled, empty)


def test_preferred_floor_never_exceeds_channel_capacity():
    result = admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 5_000, "spendable_msat": 5_000_000}, "source"
    )
    assert result == 4_250_000


@pytest.mark.parametrize("minimum_key", ["htlc_minimum_msat", "htlc_min_msat", "minimum_htlc_out_msat"])
def test_depletion_preserves_protocol_minimum_maximum_order(minimum_key):
    result = admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 15_000_000, "spendable_msat": 0, minimum_key: "1msat"}, "sink"
    )
    assert result == 1  # Protocol minimum, not the old 10k-sat preferred floor.


@pytest.mark.parametrize("minimum", [None, True, -1, "bad", 2_000_000_001])
def test_invalid_protocol_minimum_is_neutral(minimum):
    assert admission_policy.compute_htlcmax_msat(
        _cfg(), {"capacity": 2_000_000, "spendable_msat": 0, "htlc_minimum_msat": minimum}, "sink"
    ) is None
