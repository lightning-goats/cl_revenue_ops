"""P1-005: revenue-planner-candidates / revenue-planner-history clamp `limit`.

A negative limit must not reach SQLite (LIMIT -1 == unbounded whole-table
return); a non-int limit must return a clean error dict, not crash.
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def _wire(mod):
    mod.capacity_planner = MagicMock()
    mod.capacity_planner.get_metabolic_planner_influence_debug = lambda: {}
    mod.database = MagicMock()
    mod.database.get_planner_candidates = MagicMock(return_value=[])
    mod.database.get_planner_actions = MagicMock(return_value=[])


def test_candidates_negative_limit_not_unbounded(mod):
    _wire(mod)
    res = mod.revenue_planner_candidates(mod.plugin, limit=-1)
    assert "error" not in res
    assert mod.database.get_planner_candidates.call_args.kwargs["limit"] >= 1


def test_candidates_huge_limit_clamped(mod):
    _wire(mod)
    mod.revenue_planner_candidates(mod.plugin, limit=10 ** 9)
    assert mod.database.get_planner_candidates.call_args.kwargs["limit"] <= 1000


def test_candidates_non_int_limit_clean_error(mod):
    _wire(mod)
    res = mod.revenue_planner_candidates(mod.plugin, limit="x")
    assert "error" in res
    mod.database.get_planner_candidates.assert_not_called()


def test_candidates_valid_limit_unchanged(mod):
    _wire(mod)
    mod.revenue_planner_candidates(mod.plugin, limit=20)
    assert mod.database.get_planner_candidates.call_args.kwargs["limit"] == 20


def test_history_negative_limit_not_unbounded(mod):
    _wire(mod)
    res = mod.revenue_planner_history(mod.plugin, limit=-1)
    assert "error" not in res
    assert mod.database.get_planner_actions.call_args.kwargs["limit"] >= 1


def test_history_non_int_limit_clean_error(mod):
    _wire(mod)
    res = mod.revenue_planner_history(mod.plugin, limit="x")
    assert "error" in res
    mod.database.get_planner_actions.assert_not_called()
