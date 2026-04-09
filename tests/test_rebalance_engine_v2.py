from unittest.mock import MagicMock

import pytest


def test_rebalance_engine_normalizes_to_lowercase():
    from modules.config import Config

    cfg = Config(dry_run=True, rebalance_engine="V2")

    assert cfg.rebalance_engine == "v2"


def test_rebalance_engine_rejects_invalid_value():
    from modules.config import Config

    with pytest.raises(ValueError, match="rebalance_engine must be one of"):
        Config(dry_run=True, rebalance_engine="v3")


def test_rebalancer_delegates_to_v2_after_shared_preflight(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    calls = []

    cfg = Config(dry_run=True)
    cfg.rebalance_engine = "v2"
    mock_database.cleanup_stale_reservations.side_effect = lambda timeout: calls.append(("cleanup", timeout)) or 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._capex_engine = MagicMock()
    r._capex_engine.compute_allocations.side_effect = lambda: calls.append(("capex", None))
    r.job_manager.slots_available = MagicMock(side_effect=lambda: calls.append(("slots", None)) or 1)
    r._check_capital_controls = MagicMock(side_effect=lambda snapshot: calls.append(("capital", snapshot.rebalance_engine)) or True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.find_candidates.side_effect = lambda: calls.append(("delegate", None)) or []

    result = r.find_rebalance_candidates()

    assert result == []
    assert calls == [
        ("capex", None),
        ("cleanup", cfg.reservation_timeout_hours * 3600),
        ("slots", None),
        ("capital", "v2"),
        ("delegate", None),
    ]
    r.rebalance_engine_v2.find_candidates.assert_called_once()


def test_rebalancer_v2_without_engine_fails_closed(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    cfg.rebalance_engine = "v2"
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = None

    result = r.find_rebalance_candidates()

    assert result == []
    assert r.get_last_decision_summary() == {
        "action": "suppressed",
        "reason": "rebalance_engine_v2_unavailable",
        "dominant_input": "rebalance_engine",
        "safety_block": True,
        "budget_blocked": False,
    }
    mock_plugin.log.assert_any_call(
        "rebalance_engine=v2 requested but no v2 engine is injected; suppressing rebalance candidate search",
        level="warn",
    )


def test_rebalancer_delegates_to_v2_when_flag_enabled(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    cfg.rebalance_engine = "v2"
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.find_candidates.return_value = []

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.find_candidates.assert_called_once()
