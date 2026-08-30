"""Bounded market-acquisition experiment safety and lifecycle tests."""

import sqlite3
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
        "baseline_base_fee_msat": 0,
        "target_base_fee_msat": 0,
        "starting_outbound_ratio": 0.9,
        "competitor_floor_ppm": 1,
    }


def _set_acquisition_evidence(
    database, *, volume_sats: int, forward_count: int, min_out_msat=None
):
    database.get_volume_since.return_value = volume_sats
    database.get_forward_count_since.return_value = forward_count
    database.get_acquisition_forward_evidence_since.return_value = {
        "volume_sats": volume_sats,
        "forward_count": forward_count,
        "min_out_msat": min_out_msat,
    }


def test_acquisition_monitor_wakes_on_bounded_opportunity_cost_step(
    mock_plugin, mock_database
):
    fc, _ = _make_fc(mock_plugin, mock_database)
    episode = {
        **_episode(started_at=int(time.time())),
        "baseline_fee_ppm": 150,
        "baseline_base_fee_msat": 500,
    }
    fc._sync_acquisition_monitor_episodes(
        {CHANNEL_ID: episode}, reset_pending=True
    )

    # Each 5,000-sat HTLC forgoes 1.25 sats (500msat base + 750msat ppm).
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 5_000_000)
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 5_000_000)
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 5_000_000)
    assert fc.should_wake_acquisition_cycle(CHANNEL_ID, 5_000_000)
    # The coalescer resets after a wake rather than storming each forward.
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 5_000_000)


def test_acquisition_monitor_wakes_on_retention_volume_threshold(
    mock_plugin, mock_database
):
    fc, _ = _make_fc(mock_plugin, mock_database)
    episode = {
        **_episode(started_at=int(time.time()), baseline=1),
        "baseline_base_fee_msat": 0,
    }
    fc._sync_acquisition_monitor_episodes(
        {CHANNEL_ID: episode}, reset_pending=True
    )

    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 25_000_000)
    assert fc.should_wake_acquisition_cycle(CHANNEL_ID, 25_000_000)


def test_acquisition_monitor_retention_quote_and_phase_reset(
    mock_plugin, mock_database
):
    fc, _ = _make_fc(mock_plugin, mock_database)
    acquisition = {
        **_episode(started_at=int(time.time())),
        "baseline_fee_ppm": 100,
        "baseline_base_fee_msat": 500,
    }
    fc._sync_acquisition_monitor_episodes(
        {CHANNEL_ID: acquisition}, reset_pending=True
    )
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 10_000_000)

    retention = {
        **acquisition,
        "phase": "retention",
        "retention_fee_ppm": 0,
        "retention_base_fee_msat": 499,
    }
    fc._sync_acquisition_monitor_episodes({CHANNEL_ID: retention})
    # Phase change discards the free-acquisition pending counters.  A 1-msat
    # base discount needs 5,000 forwards to reach the 5-sat loss step.
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 1_000_000)


def test_acquisition_monitor_absent_and_malformed_evidence_is_neutral(
    mock_plugin, mock_database
):
    fc, _ = _make_fc(mock_plugin, mock_database)
    malformed = {
        **_episode(started_at=int(time.time())),
        "baseline_fee_ppm": "bad",
    }
    fc._sync_acquisition_monitor_episodes(
        {CHANNEL_ID: malformed}, reset_pending=True
    )

    assert not fc.should_wake_acquisition_cycle("999x1x0", 100_000_000)
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, "bad")
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 100_000_000)


def test_acquisition_monitor_restart_read_failure_does_not_crash_or_storm(
    mock_plugin, mock_database
):
    fc, _ = _make_fc(mock_plugin, mock_database)
    fc._acquisition_monitor_initialized = False
    mock_database.get_active_acquisition_experiments.side_effect = RuntimeError(
        "database unavailable"
    )

    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 100_000_000)
    assert not fc.should_wake_acquisition_cycle(CHANNEL_ID, 100_000_000)
    assert mock_database.get_active_acquisition_experiments.call_count == 1


def test_database_enforces_four_distinct_active_markets_and_persists_exact_restore(
    tmp_path,
):
    db_path = str(tmp_path / "acquisition.db")
    db = Database(db_path, MagicMock())
    db.initialize()
    assert db.get_acquisition_forward_evidence_since("1x1x0", 1_000) == {
        "volume_sats": 0,
        "forward_count": 0,
        "min_out_msat": None,
    }
    db.record_forward("9x1x0", "1x1x0", 5_001_000, 5_000_000, 1, 1_100, 1_101)
    db.record_forward("9x1x0", "1x1x0", 30_001_000, 30_000_000, 1, 1_200, 1_201)
    assert db.get_acquisition_forward_evidence_since("1x1x0", 1_000) == {
        "volume_sats": 35_000,
        "forward_count": 2,
        "min_out_msat": 5_000_000,
    }
    first = db.start_acquisition_experiment(
        channel_id="1x1x0",
        peer_id="02a",
        baseline_fee_ppm=37,
        target_fee_ppm=0,
        baseline_base_fee_msat=23,
        target_base_fee_msat=0,
        starting_outbound_ratio=0.91,
        competitor_floor_ppm=1,
        started_at=1_000,
    )
    assert first is not None
    second = db.start_acquisition_experiment(
        channel_id="2x1x0",
        peer_id="02b",
        baseline_fee_ppm=44,
        target_fee_ppm=0,
        starting_outbound_ratio=0.92,
        competitor_floor_ppm=1,
        started_at=1_001,
    )
    assert second is not None
    # A duplicate peer cannot consume another bounded slot.
    assert db.start_acquisition_experiment(
        channel_id="3x1x0",
        peer_id="02b",
        baseline_fee_ppm=45,
        target_fee_ppm=0,
        starting_outbound_ratio=0.93,
        competitor_floor_ppm=1,
        started_at=1_002,
    ) is None
    third = db.start_acquisition_experiment(
        channel_id="4x1x0",
        peer_id="02c",
        baseline_fee_ppm=46,
        target_fee_ppm=0,
        starting_outbound_ratio=0.94,
        competitor_floor_ppm=1,
        started_at=1_003,
    )
    assert third is not None
    fourth = db.start_acquisition_experiment(
        channel_id="5x1x0",
        peer_id="02d",
        baseline_fee_ppm=47,
        target_fee_ppm=0,
        starting_outbound_ratio=0.95,
        competitor_floor_ppm=1,
        started_at=1_004,
    )
    assert fourth is not None
    # A fifth distinct market cannot exceed the node-wide four-episode cap.
    assert db.start_acquisition_experiment(
        channel_id="6x1x0",
        peer_id="02e",
        baseline_fee_ppm=48,
        target_fee_ppm=0,
        starting_outbound_ratio=0.96,
        competitor_floor_ppm=1,
        started_at=1_005,
    ) is None

    # A new Database instance sees all active rows: restart does not orphan
    # any sub-economic quote.
    restarted = Database(db_path, MagicMock())
    restarted.initialize()
    assert {
        row["id"] for row in restarted.get_active_acquisition_experiments()
    } == {first["id"], second["id"], third["id"], fourth["id"]}
    retained = restarted.transition_acquisition_to_retention(
        first["id"],
        retention_fee_ppm=2,
        retention_base_fee_msat=4,
        phase_start_volume_sats=50_000,
        phase_start_forward_count=10,
        transitioned_at=1_500,
    )
    assert retained["phase"] == "retention"
    assert retained["target_fee_ppm"] == 0
    assert retained["retention_fee_ppm"] == 2
    assert retained["retention_base_fee_msat"] == 4
    assert retained["phase_start_volume_sats"] == 50_000
    assert retained["phase_start_forward_count"] == 10
    assert restarted.complete_acquisition_experiment(
        first["id"],
        exit_reason="duration_cap",
        observed_volume_sats=123_456,
        opportunity_cost_sats=4.567,
        ending_outbound_ratio=0.79,
        restored_fee_ppm=37,
        restored_base_fee_msat=23,
        completed_at=2_000,
    )
    completed = restarted.get_acquisition_experiment(first["id"])
    assert completed["state"] == "completed"
    assert completed["restored_fee_ppm"] == 37
    assert completed["restored_base_fee_msat"] == 23
    fifth = restarted.start_acquisition_experiment(
        channel_id="6x1x0",
        peer_id="02e",
        baseline_fee_ppm=48,
        target_fee_ppm=0,
        starting_outbound_ratio=0.96,
        competitor_floor_ppm=1,
        started_at=2_001,
    )
    assert fifth is not None
    assert {
        row["id"] for row in restarted.get_active_acquisition_experiments()
    } == {second["id"], third["id"], fourth["id"], fifth["id"]}
    assert restarted.channel_acquisition_on_cooldown(
        "1x1x0", now=2_001, cooldown_seconds=100
    )
    assert not restarted.channel_acquisition_on_cooldown(
        "1x1x0", now=2_101, cooldown_seconds=100
    )


def test_database_migrates_pre_base_fee_acquisition_schema(tmp_path):
    db_path = str(tmp_path / "legacy-acquisition.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE acquisition_experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            peer_id TEXT NOT NULL,
            state TEXT NOT NULL,
            started_at INTEGER NOT NULL,
            baseline_fee_ppm INTEGER NOT NULL,
            target_fee_ppm INTEGER NOT NULL,
            starting_outbound_ratio REAL NOT NULL,
            competitor_floor_ppm INTEGER NOT NULL,
            completed_at INTEGER,
            exit_reason TEXT,
            observed_volume_sats INTEGER NOT NULL DEFAULT 0,
            opportunity_cost_sats REAL NOT NULL DEFAULT 0.0,
            ending_outbound_ratio REAL,
            restored_fee_ppm INTEGER,
            phase TEXT NOT NULL DEFAULT 'acquisition',
            phase_started_at INTEGER NOT NULL DEFAULT 0,
            phase_start_volume_sats INTEGER NOT NULL DEFAULT 0,
            retention_fee_ppm INTEGER
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX idx_acquisition_one_active
        ON acquisition_experiments(state) WHERE state = 'active'
    """)
    conn.commit()
    conn.close()

    db = Database(db_path, MagicMock())
    db.initialize()
    columns = {
        row["name"]
        for row in db._get_connection().execute(
            "PRAGMA table_info(acquisition_experiments)"
        )
    }
    assert {
        "baseline_base_fee_msat",
        "target_base_fee_msat",
        "phase_start_forward_count",
        "retention_base_fee_msat",
        "restored_base_fee_msat",
    } <= columns
    first = db.start_acquisition_experiment(
        channel_id="1x1x0",
        peer_id="02a",
        baseline_fee_ppm=20,
        target_fee_ppm=0,
        starting_outbound_ratio=0.9,
        competitor_floor_ppm=1,
    )
    second = db.start_acquisition_experiment(
        channel_id="2x1x0",
        peer_id="02b",
        baseline_fee_ppm=20,
        target_fee_ppm=0,
        starting_outbound_ratio=0.9,
        competitor_floor_ppm=1,
    )
    assert first is not None and second is not None


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
    _set_acquisition_evidence(
        mock_database,
        volume_sats=fc.ACQUISITION_RETENTION_MIN_VOLUME_SATS,
        forward_count=10,
        min_out_msat=5_000_000,
    )
    mock_database.transition_acquisition_to_retention.return_value = {
        **episode,
        "phase": "retention",
        "phase_started_at": now,
        "phase_start_volume_sats": fc.ACQUISITION_RETENTION_MIN_VOLUME_SATS,
        "phase_start_forward_count": 10,
        "retention_fee_ppm": 0,
        "retention_base_fee_msat": 1,
    }
    fc.set_channel_fee = MagicMock(
        return_value={"success": True, "fee_ppm": 0, "base_fee_msat": 1}
    )
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
    assert result.new_fee_ppm == 0
    assert result.reason_code == FeeReasonCode.ACQUISITION_RETENTION.value
    assert result.algorithm_values["acquisition_phase"] == "retention"
    assert result.algorithm_values["acquisition_phase_volume_sats"] == 0
    assert result.algorithm_values["acquisition_retention_fee_ppm"] == 0
    assert result.algorithm_values["acquisition_retention_base_fee_msat"] == 1
    transition = mock_database.transition_acquisition_to_retention.call_args.kwargs
    assert transition["phase_start_volume_sats"] == 50_000
    assert transition["phase_start_forward_count"] == 10
    assert transition["retention_base_fee_msat"] == 1
    mock_database.complete_acquisition_experiment.assert_not_called()


def test_retention_requires_a_positive_strict_undercut(
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
    _set_acquisition_evidence(
        mock_database,
        volume_sats=50_000,
        forward_count=10,
        min_out_msat=1_000_000,
    )
    # A 1-ppm competitor charges only 1 msat at this minimum size, so no
    # positive integer-msat quote can be strictly cheaper.
    fc.set_channel_fee = MagicMock(
        return_value={"success": True, "fee_ppm": 100, "base_fee_msat": 0}
    )
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
    mock_database.transition_acquisition_to_retention.assert_not_called()
    complete = mock_database.complete_acquisition_experiment.call_args.kwargs
    assert complete["exit_reason"] == "retention_positive_undercut_unavailable"


def test_positive_retention_base_is_bounded_and_malformed_evidence_is_neutral():
    from modules.fee_controller import FeeController

    assert FeeController._positive_retention_base_fee_msat(5_000_000, 2) == 1
    assert FeeController._positive_retention_base_fee_msat(5_000_000, 1) == 1
    assert FeeController._positive_retention_base_fee_msat(2_000_000, 1) == 1
    assert FeeController._positive_retention_base_fee_msat(2_000_000_000, 10) == 1
    assert FeeController._positive_retention_base_fee_msat(1_000_000, 1) is None
    assert FeeController._positive_retention_base_fee_msat(1_500_000, 1) is None
    for malformed in (None, True, 0, -1, "bad"):
        assert FeeController._positive_retention_base_fee_msat(malformed, 1) is None
        assert FeeController._positive_retention_base_fee_msat(5_000_000, malformed) is None
    assert FeeController._positive_retention_base_fee_msat(5_000_000.5, 1) is None
    assert FeeController._positive_retention_base_fee_msat(5_000_000, 1.5) is None


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
    episode = {
        **_episode(started_at=now - 10),
        "baseline_base_fee_msat": 23,
    }
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=2)
    fc._calculate_floor = MagicMock(return_value=10)
    _set_acquisition_evidence(
        mock_database,
        volume_sats=50_000,
        forward_count=10,
        min_out_msat=5_000_000,
    )
    mock_database.transition_acquisition_to_retention.return_value = {
        **episode,
        "phase": "retention",
        "phase_started_at": now,
        "phase_start_volume_sats": 50_000,
        "phase_start_forward_count": 10,
        "retention_fee_ppm": 0,
        "retention_base_fee_msat": 1,
    }
    fc.set_channel_fee = MagicMock()
    info = _channel_info(0)
    info["fee_base_msat"] = 1
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
        "baseline_base_fee_msat": 10,
        "phase": "retention",
        "phase_started_at": now - fc.ACQUISITION_RETENTION_DURATION_SECONDS,
        "phase_start_volume_sats": 50_000,
        "phase_start_forward_count": 10,
        "retention_fee_ppm": 0,
        "retention_base_fee_msat": 4,
    }
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    fc._calculate_floor = MagicMock(return_value=10)
    _set_acquisition_evidence(
        mock_database,
        volume_sats=70_000,
        forward_count=14,
        min_out_msat=5_000_000,
    )
    fc.set_channel_fee = MagicMock(
        return_value={"success": True, "fee_ppm": 100, "base_fee_msat": 10}
    )
    info = _channel_info(0)
    info["fee_base_msat"] = 4
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
    assert complete["opportunity_cost_sats"] == 7.124
    assert complete["restored_fee_ppm"] == 100
    assert complete["restored_base_fee_msat"] == 10


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
    _set_acquisition_evidence(
        mock_database,
        volume_sats=50_000,
        forward_count=10,
        min_out_msat=5_000_000,
    )
    mock_database.transition_acquisition_to_retention = None
    fc.set_channel_fee = MagicMock(
        return_value={"success": True, "fee_ppm": 100, "base_fee_msat": 0}
    )
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
        "phase_start_forward_count": 10,
        "retention_fee_ppm": "bad-fee",
        "retention_base_fee_msat": "bad-base",
    }
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._calculate_floor = MagicMock(return_value=10)
    _set_acquisition_evidence(
        mock_database,
        volume_sats=50_000,
        forward_count=10,
        min_out_msat=5_000_000,
    )
    fc.set_channel_fee = MagicMock(
        return_value={"success": True, "fee_ppm": 100, "base_fee_msat": 0}
    )
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


def test_prepare_is_default_off_then_selects_four_distinct_best_markets(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(mock_plugin, mock_database)
    now = int(time.time())
    states = [
        {"channel_id": "1x1x0", "peer_id": "02a", "state": "source"},
        {"channel_id": "2x1x0", "peer_id": "02b", "state": "source"},
        {"channel_id": "3x1x0", "peer_id": "02c", "state": "source"},
        {"channel_id": "4x1x0", "peer_id": "02d", "state": "source"},
        {"channel_id": "5x1x0", "peer_id": "02e", "state": "source"},
    ]
    channels = {
        channel_id: {
            "capacity": 1_000_000,
            "spendable_msat": spendable_msat,
            "fee_proportional_millionths": fee_ppm,
        }
        for channel_id, spendable_msat, fee_ppm in (
            ("1x1x0", 910_000_000, 50),
            ("2x1x0", 950_000_000, 75),
            ("3x1x0", 940_000_000, 70),
            ("4x1x0", 930_000_000, 65),
            ("5x1x0", 920_000_000, 60),
        )
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
    mock_database.start_acquisition_experiment.side_effect = lambda **kwargs: {
        "id": int(kwargs["channel_id"].split("x", 1)[0]),
        "channel_id": kwargs["channel_id"],
        "peer_id": kwargs["peer_id"],
    }
    active = fc._prepare_acquisition_experiments(states, channels, cfg, now)
    assert set(active) == {"2x1x0", "3x1x0", "4x1x0", "5x1x0"}
    assert mock_database.start_acquisition_experiment.call_count == 4
    assert [
        call.kwargs["baseline_fee_ppm"]
        for call in mock_database.start_acquisition_experiment.call_args_list
    ] == [75, 70, 65, 60]


def test_prepare_adds_second_market_but_not_second_channel_to_active_peer(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm_saturated=0,
        acquisition_experiment_enabled=True,
    )
    now = int(time.time())
    existing = {"id": 1, "channel_id": "1x1x0", "peer_id": "02a"}
    mock_database.get_active_acquisition_experiments.return_value = [existing]
    mock_database.channel_acquisition_on_cooldown.return_value = False
    mock_database.get_channel_probe.return_value = None
    mock_database.get_forward_count_since.return_value = 0
    mock_database.start_acquisition_experiment.return_value = {
        "id": 2,
        "channel_id": "3x1x0",
        "peer_id": "02b",
    }
    fc._effective_min_fee_ppm = MagicMock(return_value=0)
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    states = [
        {"channel_id": "1x1x0", "peer_id": "02a", "state": "sink"},
        {"channel_id": "2x1x0", "peer_id": "02a", "state": "sink"},
        {"channel_id": "3x1x0", "peer_id": "02b", "state": "sink"},
    ]
    channels = {
        channel_id: {
            "capacity": 1_000_000,
            "spendable_msat": 900_000_000,
            "fee_proportional_millionths": 50,
        }
        for channel_id in ("1x1x0", "2x1x0", "3x1x0")
    }

    active = fc._prepare_acquisition_experiments(states, channels, cfg, now)

    assert set(active) == {"1x1x0", "3x1x0"}
    assert mock_database.start_acquisition_experiment.call_count == 1
    assert (
        mock_database.start_acquisition_experiment.call_args.kwargs["peer_id"]
        == "02b"
    )


def test_prepare_preserves_active_episode_when_new_entry_gate_closes(
    mock_plugin, mock_database
):
    fc, cfg = _make_fc(
        mock_plugin,
        mock_database,
        min_fee_ppm_saturated=1,
        acquisition_experiment_enabled=True,
    )
    existing = {"id": 1, "channel_id": "1x1x0", "peer_id": "02a"}
    mock_database.get_active_acquisition_experiments.return_value = [existing]

    active = fc._prepare_acquisition_experiments([], {}, cfg, int(time.time()))

    assert active == {"1x1x0": existing}
    mock_database.start_acquisition_experiment.assert_not_called()


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
    episode = {
        **_episode(started_at=now - 10),
        "baseline_base_fee_msat": 23,
    }
    fc._acquisition_cycle_experiments = {CHANNEL_ID: episode}
    fc._get_neighbor_fee_percentile = MagicMock(return_value=1)
    fc._calculate_floor = MagicMock(return_value=10)
    fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": 0})
    _set_acquisition_evidence(
        mock_database, volume_sats=0, forward_count=0
    )

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
    fc.set_channel_fee = MagicMock(
        return_value={"success": True, "fee_ppm": 100, "base_fee_msat": 23}
    )
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
    assert complete["restored_base_fee_msat"] == 23
    assert fc.set_channel_fee.call_args.kwargs["base_fee_msat_override"] == 23


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
    _set_acquisition_evidence(
        mock_database, volume_sats=0, forward_count=0
    )
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
    _set_acquisition_evidence(
        mock_database,
        volume_sats=fc.ACQUISITION_VOLUME_CAP_SATS,
        forward_count=0,
    )
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


def test_acquisition_plugin_option_is_dynamic_default_on():
    from tests.plugin_test_utils import load_plugin_module

    mod = load_plugin_module()
    option = mod.plugin.options["revenue-ops-acquisition-experiment-enabled"]
    assert option["default"] is True
    assert option["opt_type"] == "bool"
    assert option["dynamic"] is True
    assert Config().acquisition_experiment_enabled is True


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
