"""Trust-boundary hardening tests for non-finite (Infinity/NaN) hint input.

The hint producer and the CLN datastore contents are untrusted. A poisoned
snapshot containing JSON Infinity/NaN literals, non-finite numeric hint
fields, an unbounded fleet_fee_prior, or an unbounded priority_score must
neutralize safely instead of crashing poll()/get_status() or pinning a
maximum bias.
"""

import math
import time

import pytest
from unittest.mock import MagicMock

from modules.rebalance_coordination_overlay import (
    _priority_score as overlay_priority_score,
)
from modules.rebalance_route_policy import (
    _priority_score as route_priority_score,
)


class TestPriorityScoreClamped:
    @pytest.mark.parametrize(
        "score_fn", [route_priority_score, overlay_priority_score]
    )
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_priority_score_neutralizes(self, score_fn, bad):
        assert score_fn({"priority_score": bad}) == 0.0

    @pytest.mark.parametrize(
        "score_fn", [route_priority_score, overlay_priority_score]
    )
    def test_huge_priority_score_clamped_to_upper_bound(self, score_fn):
        assert score_fn({"priority_score": 1e9}) == 100.0

    @pytest.mark.parametrize(
        "score_fn", [route_priority_score, overlay_priority_score]
    )
    def test_sane_priority_score_passes_through(self, score_fn):
        assert score_fn({"priority_score": 50.0}) == 50.0
        assert score_fn({"priority_score": -5.0}) == 0.0
        assert score_fn({}) == 0.0
        assert score_fn({"priority_score": "junk"}) == 0.0
