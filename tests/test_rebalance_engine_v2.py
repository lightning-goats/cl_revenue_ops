"""Tests for the rebalance engine delegation and integration."""

from unittest.mock import MagicMock

import pytest


def _make_cycle_result(candidates=None, executions=None):
    from modules.rebalance_engine_v2 import CycleResult
    cr = CycleResult()
    cr.candidates = candidates or []
    cr.executions = executions or []
    return cr


def test_rebalancer_calls_engine_run_cycle(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.run_cycle.assert_called_once()


def test_rebalancer_without_engine_suppresses(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = None

    result = r.find_rebalance_candidates()

    assert result == []
    summary = r.get_last_decision_summary()
    assert summary["action"] == "suppressed"
    assert "unavailable" in summary["reason"]


def test_rebalancer_reports_no_candidates(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "hold"
    assert summary["safety_block"] is False


def test_rebalancer_reports_successful_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result(
        candidates=["c1"],
        executions=[ExecutionResult(success=True)],
    )

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "rebalance"
    assert "succeeded" in summary["reason"]


def test_rebalancer_reports_failed_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result(
        candidates=["c1"],
        executions=[ExecutionResult(success=False, error="no_route")],
    )

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "rebalance"
    assert "0/1" in summary["reason"]
