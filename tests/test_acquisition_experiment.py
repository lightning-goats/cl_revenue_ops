"""Bounded market-acquisition experiment safety and lifecycle tests."""

import time
from unittest.mock import MagicMock

from modules.config import Config
from modules.database import Database
from modules.fee_controller import FeeReasonCode
from tests.test_fee_pipeline_composition import (
    CHANNEL_ID,
    PEER_ID,
    _channel_info,
    _make_fc,
)


def _episode(*, started_at: int, baseline: int = 100):
    return {
        "id": 7,
        "channel_id": CHANNEL_ID,
        "peer_id": PEER_ID,
        "state": "active",
        "started_at": started_at,
        "baseline_fee_ppm": baseline,
        "target_fee_ppm": 0,
        "starting_outbound_ratio": 0.9,
        "competitor_floor_ppm": 1,
    }


def test_database_enforces_one_active_and_persists_exact_restore(tmp_path):
    db_path = str(tmp_path / "acquisition.db")
    db = Database(db_path, MagicMock())
    db.initialize()
    first = db.start_acquisition_experiment(
        channel_id="1x1x0",
        peer_id="02a",
        baseline_fee_ppm=37,
        target_fee_ppm=0,
        starting_outbound_ratio=0.91,
        competitor_floor_ppm=1,
        started_at=1_000,
    )
    assert first is not None
    assert db.start_acquisition_experiment(
        channel_id="2x1x0",
        peer_id="02b",
        baseline_fee_ppm=44,
        target_fee_ppm=0,
        starting_outbound_ratio=0.92,
        competitor_floor_ppm=1,
        started_at=1_001,
    ) is None

    # A new Database instance sees the active row: restart does not orphan
    # a sub-economic quote.
    restarted = Database(db_path, MagicMock())
    restarted.initialize()
    assert restarted.get_active_acquisition_experiments()[0]["id"] == first["id"]
    retained = restarted.transition_acquisition_to_retention(
        first["id"],
        retention_fee_ppm=2,
        phase_start_volume_sats=50_000,
        transitioned_at=1_500,
    )
    assert retained["phase"] == "retention"
    assert retained["target_fee_ppm"] == 0
    assert retained["retention_fee_ppm"] == 2
    assert retained["phase_start_volume_sats"] == 50_000
    assert restarted.complete_acquisition_experiment(
        first["id"],
        exit_reason="duration_cap",
        observed_volume_sats=123_456,
        opportunity_cost_sats=4.567,
        ending_outbound_ratio=0.79,
        restored_fee_ppm=37,
        completed_at=2_000,
    )
    completed = restarted.get_acquisition_experiment(first["id"])
    assert completed["state"] == "completed"
    assert completed["restored_fee_ppm"] == 37
    assert restarted.get_recent_acquisition_experiments(limit=1)[0]["id"] == first["id"]
    assert restarted.channel_acquisition_on_cooldown(
        "1x1x0", now=2_001, cooldown_seconds=100
    )
    assert not restarted.channel_acquisition_on_cooldown(
        "1x1x0", now=2_101, cooldown_seconds=100
    )


def test_positive_acquisition_transitions_to_bounded_paid_retention(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    now = int(time.time())
    episode = _episode(started_at=now - 10)
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=2)
    fc._calculate_floor = MagicMock(return_value=10)
    mock_database.get_volume_since.return_value = (
        fc.ACQUISITION_RETENTION_MIN_VOLUME_SATS
    )
    mock_database.transition_acquisition_to_retention.return_value = {
        **episode,
        "phase": "retention",
        "phase_started_at": now,
        "phase_start_volume_sats": fc.ACQUISITION_RETENTION_MIN_VOLUME_SATS,
        "retention_fee_ppm": 2,
    }
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 2})
    info = _channel_info(0)
    info["spendable_msat"] = "1900000000msat"

    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )

    assert result is not None
    assert result.new_fee_ppm == 2
    assert result.reason_code == FeeReasonCode.ACQUISITION_RETENTION.value
    assert result.algorithm_values["acquisition_phase"] == "retention"
    assert result.algorithm_values["acquisition_phase_volume_sats"] == 0
    assert result.algorithm_values["acquisition_retention_fee_ppm"] == 2
    transition = mock_database.transition_acquisition_to_retention.call_args.kwargs
    assert transition["phase_start_volume_sats"] == 50_000
    mock_database.complete_acquisition_experiment.assert_not_called()


def test_retention_transition_persists_when_paid_fee_is_already_live(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    now = int(time.time())
    episode = _episode(started_at=now - 10)
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=2)
    fc._calculate_floor = MagicMock(return_value=10)
    mock_database.get_volume_since.return_value = 50_000
    mock_database.transition_acquisition_to_retention.return_value = {
        **episode,
        "phase": "retention",
        "phase_started_at": now,
        "phase_start_volume_sats": 50_000,
        "retention_fee_ppm": 2,
    }
    fc.set_channel_fee = MagicMock()
    info = _channel_info(2)
    info["spendable_msat"] = "1900000000msat"

    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )

    assert result is None
    fc.set_channel_fee.assert_not_called()
    mock_database.transition_acquisition_to_retention.assert_called_once()
    mock_database.complete_acquisition_experiment.assert_not_called()


def test_paid_retention_duration_restores_baseline_with_exact_cost(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    now = int(time.time())
    episode = {
        **_episode(started_at=now - 4_000, baseline=100),
        "phase": "retention",
        "phase_started_at": now - fc.ACQUISITION_RETENTION_DURATION_SECONDS,
        "phase_start_volume_sats": 50_000,
        "retention_fee_ppm": 1,
    }
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    fc._calculate_floor = MagicMock(return_value=10)
    mock_database.get_volume_since.return_value = 70_000
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 100})
    info = _channel_info(1)
    info["spendable_msat"] = "1900000000msat"

    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )

    assert result.new_fee_ppm == 100
    assert result.reason_code == FeeReasonCode.ACQUISITION_EXIT.value
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "retention_duration_cap"
    assert complete["observed_volume_sats"] == 70_000
    assert complete["opportunity_cost_sats"] == 6.98
    assert complete["restored_fee_ppm"] == 100


def test_retention_transition_without_persistence_restores_baseline(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    episode = _episode(started_at=int(time.time()) - 10)
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    fc._calculate_floor = MagicMock(return_value=10)
    mock_database.get_volume_since.return_value = 50_000
    mock_database.transition_acquisition_to_retention = None
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 100})
    info = _channel_info(0)
    info["spendable_msat"] = "1900000000msat"

    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )

    assert result.new_fee_ppm == 100
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "retention_persistence_unavailable"


def test_malformed_retention_state_fails_closed_without_crashing(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    episode = {
        **_episode(started_at=int(time.time()) - 10, baseline=100),
        "phase": "retention",
        "phase_started_at": "bad-time",
        "phase_start_volume_sats": 50_000,
        "retention_fee_ppm": "bad-fee",
    }
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._calculate_floor = MagicMock(return_value=10)
    mock_database.get_volume_since.return_value = 50_000
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 100})
    info = _channel_info(1)
    info["spendable_msat"] = "1900000000msat"

    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )

    assert result.new_fee_ppm == 100
    assert result.reason_code == FeeReasonCode.ACQUISITION_EXIT.value
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "evidence_unavailable"
    assert complete["restored_fee_ppm"] == 100


def test_prepare_is_default_off_and_selects_only_one_best_lane(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(mock_plugin, mock_database)
    now = int(time.time())
    states = [
        {"channel_id": "1x1x0", "peer_id": "02a", "state": "source"},
        {"channel_id": "2x1x0", "peer_id": "02b", "state": "source"},
    ]
    channels = {
        "1x1x0": {
            "capacity": 1_000_000,
            "spendable_msat": 900_000_000,
            "fee_proportional_millionths": 50,
        },
        "2x1x0": {
            "capacity": 1_000_000,
            "spendable_msat": 950_000_000,
            "fee_proportional_millionths": 75,
        },
    }
    mock_database.get_active_acquisition_experiments.return_value = []
    mock_database.channel_acquisition_on_cooldown.return_value = False
    mock_database.get_channel_probe.return_value = None
    mock_database.get_forward_count_since.return_value = 0
    fc._effective_min_fee_ppm = MagicMock(return_value=0)
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)

    cfg.acquisition_experiment_enabled = False
    assert fc._prepare_acquisition_experiments(states, channels, cfg, now) == {}
    mock_database.start_acquisition_experiment.assert_not_called()

    cfg.acquisition_experiment_enabled = True
    cfg.min_fee_ppm_saturated = 0
    mock_database.start_acquisition_experiment.return_value = {
        "id": 1,
        "channel_id": "2x1x0",
    }
    active = fc._prepare_acquisition_experiments(states, channels, cfg, now)
    assert set(active) == {"2x1x0"}
    assert mock_database.start_acquisition_experiment.call_count == 1
    assert (
        mock_database.start_acquisition_experiment.call_args.kwargs["baseline_fee_ppm"]
        == 75
    )


def test_prepare_accepts_two_ppm_competitor_but_rejects_outside_low_fee_band(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    now = int(time.time())
    states = [{"channel_id": CHANNEL_ID, "peer_id": PEER_ID, "state": "source"}]
    channels = {
        CHANNEL_ID: {
            "capacity": 1_000_000,
            "spendable_msat": 900_000_000,
            "fee_proportional_millionths": 50,
        }
    }
    mock_database.get_active_acquisition_experiments.return_value = []
    mock_database.channel_acquisition_on_cooldown.return_value = False
    mock_database.get_channel_probe.return_value = None
    mock_database.get_forward_count_since.return_value = 0
    mock_database.start_acquisition_experiment.return_value = {
        "id": 1,
        "channel_id": CHANNEL_ID,
    }
    fc._effective_min_fee_ppm = MagicMock(return_value=0)
    fc._get_neighbor_fee_percentile = MagicMock(return_value=2)

    active = fc._prepare_acquisition_experiments(states, channels, cfg, now)
    assert set(active) == {CHANNEL_ID}
    assert (
        mock_database.start_acquisition_experiment.call_args.kwargs[
            "competitor_floor_ppm"
        ]
        == 2
    )

    for outside in (0, fc.ACQUISITION_MAX_COMPETITOR_FLOOR_PPM + 1, None, "bad"):
        mock_database.reset_mock()
        mock_database.get_active_acquisition_experiments.return_value = []
        mock_database.channel_acquisition_on_cooldown.return_value = False
        mock_database.get_channel_probe.return_value = None
        mock_database.get_forward_count_since.return_value = 0
        fc._get_neighbor_fee_percentile.return_value = outside
        assert fc._prepare_acquisition_experiments(states, channels, cfg, now) == {}
        mock_database.start_acquisition_experiment.assert_not_called()


def test_active_episode_quotes_zero_then_restores_baseline_on_disable(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    now = int(time.time())
    episode = _episode(started_at=now - 10)
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    fc._calculate_floor = MagicMock(return_value=10)
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 0})
    mock_database.get_volume_since.return_value = 0

    info = _channel_info(100)
    info["spendable_msat"] = "1900000000msat"
    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )
    assert result is not None
    assert result.new_fee_ppm == 0
    assert result.reason_code == FeeReasonCode.ACQUISITION_EXPERIMENT.value
    mock_database.complete_acquisition_experiment.assert_not_called()

    cfg.acquisition_experiment_enabled = False
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 100})
    info["fee_proportional_millionths"] = 0
    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )
    assert result is not None
    assert result.new_fee_ppm == 100
    assert result.reason_code == FeeReasonCode.ACQUISITION_EXIT.value
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "disabled"
    assert complete["restored_fee_ppm"] == 100


def test_active_episode_survives_low_fee_drift_and_exits_above_band(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    episode = _episode(started_at=int(time.time()) - 10)
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=2)
    fc._calculate_floor = MagicMock(return_value=10)
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 0})
    mock_database.get_volume_since.return_value = 0
    info = _channel_info(100)
    info["spendable_msat"] = "1900000000msat"

    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )
    assert result.new_fee_ppm == 0
    mock_database.complete_acquisition_experiment.assert_not_called()

    fc._get_neighbor_fee_percentile.return_value = (
        fc.ACQUISITION_MAX_COMPETITOR_FLOOR_PPM + 1
    )
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 100})
    info["fee_proportional_millionths"] = 0
    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )
    assert result.new_fee_ppm == 100
    assert (
        mock_database.complete_acquisition_experiment.call_args.kwargs["exit_reason"]
        == "competitor_evidence_changed"
    )


def test_volume_cap_exit_retries_until_baseline_rpc_succeeds(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    episode = _episode(started_at=int(time.time()) - 10)
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    fc._calculate_floor = MagicMock(return_value=10)
    mock_database.get_volume_since.return_value = fc.ACQUISITION_VOLUME_CAP_SATS
    info = _channel_info(0)
    info["spendable_msat"] = "1900000000msat"

    fc.set_channel_fee = MagicMock(
        return_value={"success": False, "message": "setchannel failed"}
    )
    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )
    assert result is None
    mock_database.complete_acquisition_experiment.assert_not_called()

    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 100})
    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )
    assert result.new_fee_ppm == 100
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "volume_cap"


def test_malformed_persisted_episode_fails_closed_without_crashing(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm=10,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    episode = _episode(started_at=int(time.time()) - 10)
    episode["started_at"] = "not-a-timestamp"
    episode["baseline_fee_ppm"] = "not-a-fee"
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._calculate_floor = MagicMock(return_value=10)
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 10})

    info = _channel_info(0)
    info["spendable_msat"] = "1900000000msat"
    result = fc._adjust_channel_fee(
        CHANNEL_ID,
        PEER_ID,
        {"state": "source"},
        info,
        cfg=cfg,
        force_reprice_reason="acquisition_experiment",
    )

    assert result is not None
    assert result.new_fee_ppm == 10
    assert result.reason_code == FeeReasonCode.ACQUISITION_EXIT.value
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "evidence_unavailable"
    assert complete["restored_fee_ppm"] == 10


def test_acquisition_plugin_option_is_dynamic_default_off():
    from tests.plugin_test_utils import load_plugin_module

    mod = load_plugin_module()
    option = mod.plugin.options["revenue-ops-acquisition-experiment-enabled"]
    assert option["default"] is False
    assert option["opt_type"] == "bool"
    assert option["dynamic"] is True
    assert Config().acquisition_experiment_enabled is False


def test_setconfig_refresh_updates_acquisition_gate():
    from tests.plugin_test_utils import load_plugin_module

    mod = load_plugin_module()
    mod.config = Config(acquisition_experiment_enabled=False)
    mod.database = MagicMock()
    mod.database.get_config_override.return_value = None
    mod.safe_plugin = MagicMock()
    mod.safe_plugin.rpc.listconfigs.return_value = {
        "configs": {
            "revenue-ops-acquisition-experiment-enabled": {
                "value_str": "true"
            }
        }
    }
    mod._refresh_dynamic_config()
    assert mod.config.acquisition_experiment_enabled is True
