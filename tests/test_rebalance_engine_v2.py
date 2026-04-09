from unittest.mock import MagicMock


def test_rebalancer_delegates_to_v2_when_flag_enabled(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    cfg.rebalance_engine = "v2"
    mock_database.cleanup_stale_reservations.return_value = 0
    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.find_candidates.return_value = []

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.find_candidates.assert_called_once()
