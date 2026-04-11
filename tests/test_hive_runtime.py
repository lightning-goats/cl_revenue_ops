from unittest.mock import MagicMock

from modules.hive_runtime import refresh_hive_runtime


def test_refresh_hive_runtime_polls_hints_then_refreshes_shared_router():
    hive_hints = MagicMock()
    hive_router = MagicMock()
    log = MagicMock()

    refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router, log=log)

    hive_hints.poll.assert_called_once_with()
    hive_router.refresh_layer.assert_called_once_with()
    hive_router.refresh_fleet_balances.assert_called_once_with()
    hive_router.clear_route_cache.assert_called_once_with()


def test_refresh_hive_runtime_fail_opens_when_router_refresh_errors():
    hive_hints = MagicMock()
    hive_router = MagicMock()
    hive_router.refresh_layer.side_effect = RuntimeError("askrene down")
    log = MagicMock()

    refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router, log=log)

    hive_hints.poll.assert_called_once_with()
    hive_router.refresh_layer.assert_called_once_with()
    hive_router.refresh_fleet_balances.assert_not_called()
    hive_router.clear_route_cache.assert_not_called()
    log.assert_called()
