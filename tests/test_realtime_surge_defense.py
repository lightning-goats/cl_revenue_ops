import pytest

from modules.realtime_surge_defense import RealtimeSurgeDefense, SurgeSample


class FakeClock:
    def __init__(self, start: float = 1_000.0):
        self._now = start

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_manager(
    clock: FakeClock,
    *,
    capacity_msat: int = 1_000_000_000,
    baseline_fee_ppm: int = 100,
    cooldown_seconds: int = 120,
    min_interval_seconds: int = 15,
    apply_fee_callback=None,
):
    applied = []

    def apply_fee(channel_id: str, fee_ppm: int) -> bool:
        applied.append((channel_id, fee_ppm))
        return True

    if apply_fee_callback is None:
        apply_fee_callback = apply_fee

    manager = RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=cooldown_seconds,
        surge_setchannel_min_interval_seconds=min_interval_seconds,
        channel_capacity_msat=lambda channel_id: capacity_msat,
        current_fee_ppm=lambda channel_id: baseline_fee_ppm,
        apply_fee_callback=apply_fee_callback,
        time_fn=clock.time,
    )
    return manager, applied


def _sample(
    clock: FakeClock,
    *,
    amount_msat: int,
    incoming_peer_id: str,
    incoming_channel_id: str = "1x1x1",
    outgoing_channel_id: str = "2x2x2",
) -> SurgeSample:
    return SurgeSample(
        ts=clock.time(),
        amount_msat=amount_msat,
        incoming_peer_id=incoming_peer_id,
        incoming_channel_id=incoming_channel_id,
        outgoing_channel_id=outgoing_channel_id,
    )


def test_burst_trigger_fires_when_moved_pct_and_peer_concentration_cross_threshold():
    clock = FakeClock()
    manager, applied = _make_manager(clock)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()

    assert len(applied) == 1
    assert applied[0][0] == "2x2x2"
    assert applied[0][1] > 100
    assert applied[0][1] <= 500

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert status["enabled"] is True
    assert status["active_channel_count"] == 1
    assert channel["active"] is True
    assert channel["baseline_fee_ppm"] == 100
    assert channel["active_fee_ppm"] == applied[0][1]
    assert channel["moved_pct"] == pytest.approx(0.12)
    assert channel["htlc_count"] == 4
    assert channel["top_incoming_peer_volume_share"] == pytest.approx(1.0)
    assert channel["top_incoming_peer_htlc_share"] == pytest.approx(1.0)


def test_status_reports_active_overlay_details():
    clock = FakeClock()
    manager, applied = _make_manager(clock)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert status["enabled"] is True
    assert status["active_channel_count"] == 1
    assert channel["active"] is True
    assert channel["baseline_fee_ppm"] == 100
    assert channel["active_fee_ppm"] == applied[0][1]
    assert channel["cooldown_remaining_sec"] == pytest.approx(119.0)
    assert channel["last_trigger_reason"].startswith("moved_pct=")
    assert channel["last_apply_result"] == "applied"


def test_status_reports_trigger_counts_for_recent_windows():
    clock = FakeClock()
    manager, _ = _make_manager(clock)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()
    clock.advance(180)
    manager.process_pending_actions()
    clock.advance(3_601)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()

    status = manager.get_status()

    assert status["trigger_count_1h"] == 1
    assert status["trigger_count_24h"] == 2


def test_does_not_trigger_on_normal_mixed_flow():
    clock = FakeClock()
    manager, applied = _make_manager(clock)

    for peer_id in ("peer-a", "peer-b", "peer-c", "peer-d"):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id=peer_id)
        )
        clock.advance(1)

    assert applied == []

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert status["active_channel_count"] == 0
    assert channel["active"] is False
    assert channel["moved_pct"] == pytest.approx(0.12)
    assert channel["top_incoming_peer_volume_share"] == pytest.approx(0.25)
    assert channel["top_incoming_peer_htlc_share"] == pytest.approx(0.25)


def test_trigger_is_debounced_by_min_setchannel_interval():
    clock = FakeClock()
    manager, applied = _make_manager(clock, min_interval_seconds=15)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()

    clock.advance(5)

    for _ in range(2):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    assert len(applied) == 1

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert channel["active"] is True
    assert channel["last_apply_result"] == "debounced"


def test_cooldown_extends_while_burst_continues():
    clock = FakeClock()
    manager, applied = _make_manager(clock, cooldown_seconds=120, min_interval_seconds=300)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()
    first_cooldown_until = manager.get_status()["channels"]["2x2x2"]["cooldown_until"]

    clock.advance(30)
    manager.ingest_sample(
        _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
    )

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert len(applied) == 1
    assert channel["active"] is True
    assert channel["cooldown_until"] > first_cooldown_until


def test_trigger_captures_exact_baseline_and_applies_surge_fee():
    clock = FakeClock()
    current_fee = {"value": 100}
    applied = []

    def apply_fee(channel_id: str, fee_ppm: int) -> bool:
        applied.append((channel_id, fee_ppm))
        return True

    manager = RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=lambda channel_id: 1_000_000_000,
        current_fee_ppm=lambda channel_id: current_fee["value"],
        apply_fee_callback=apply_fee,
        time_fn=clock.time,
    )

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    assert applied == []

    current_fee["value"] = 250
    manager.process_pending_actions()

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert applied == [("2x2x2", 340)]
    assert channel["active"] is True
    assert channel["baseline_fee_ppm"] == 100
    assert channel["active_fee_ppm"] == 340


def test_calm_window_reverts_to_exact_baseline_after_cooldown():
    clock = FakeClock()
    current_fee = {"value": 125}
    applied = []

    def apply_fee(channel_id: str, fee_ppm: int) -> bool:
        applied.append((channel_id, fee_ppm))
        return True

    manager = RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=lambda channel_id: 1_000_000_000,
        current_fee_ppm=lambda channel_id: current_fee["value"],
        apply_fee_callback=apply_fee,
        time_fn=clock.time,
    )

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()
    clock.advance(180)
    current_fee["value"] = 500
    manager.process_pending_actions()

    status = manager.get_status()

    assert applied == [("2x2x2", 425), ("2x2x2", 125)]
    assert status["active_channel_count"] == 0
    assert "2x2x2" not in status["channels"]


def test_failed_apply_does_not_mark_overlay_active():
    clock = FakeClock()
    applied = []

    def apply_fee(channel_id: str, fee_ppm: int) -> bool:
        applied.append((channel_id, fee_ppm))
        return False

    manager = RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=lambda channel_id: 1_000_000_000,
        current_fee_ppm=lambda channel_id: 100,
        apply_fee_callback=apply_fee,
        time_fn=clock.time,
    )

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert applied == [("2x2x2", 340)]
    assert status["active_channel_count"] == 0
    assert channel["active"] is False
    assert channel["baseline_fee_ppm"] is None
    assert channel["active_fee_ppm"] is None
    assert channel["last_apply_result"] == "apply_failed"


def test_stale_queued_apply_is_dropped_after_window_goes_calm():
    clock = FakeClock()
    manager, applied = _make_manager(clock)

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    clock.advance(180)
    manager.process_pending_actions()

    status = manager.get_status()

    assert applied == []
    assert status["active_channel_count"] == 0
    assert "2x2x2" not in status["channels"]


def test_apply_callback_exception_fails_safe_without_phantom_pending_state():
    clock = FakeClock()

    def apply_fee(channel_id: str, fee_ppm: int) -> bool:
        raise RuntimeError("rpc failed")

    manager = RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=lambda channel_id: 1_000_000_000,
        current_fee_ppm=lambda channel_id: 100,
        apply_fee_callback=apply_fee,
        time_fn=clock.time,
    )

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()

    status = manager.get_status()
    channel = status["channels"]["2x2x2"]

    assert status["active_channel_count"] == 0
    assert channel["active"] is False
    assert channel["baseline_fee_ppm"] is None
    assert channel["active_fee_ppm"] is None
    assert channel["last_apply_result"] == "apply_failed"


def test_failed_revert_keeps_overlay_active_for_retry():
    clock = FakeClock()
    applied = []
    outcomes = iter([True, False, True])

    def apply_fee(channel_id: str, fee_ppm: int) -> bool:
        applied.append((channel_id, fee_ppm))
        return next(outcomes)

    manager = RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=lambda channel_id: 1_000_000_000,
        current_fee_ppm=lambda channel_id: 100,
        apply_fee_callback=apply_fee,
        time_fn=clock.time,
    )

    for _ in range(4):
        manager.ingest_sample(
            _sample(clock, amount_msat=30_000_000, incoming_peer_id="peer-a")
        )
        clock.advance(1)

    manager.process_pending_actions()
    clock.advance(180)
    manager.process_pending_actions()

    failed_revert_status = manager.get_status()
    failed_revert_channel = failed_revert_status["channels"]["2x2x2"]

    assert applied == [("2x2x2", 340), ("2x2x2", 100)]
    assert failed_revert_status["active_channel_count"] == 1
    assert failed_revert_channel["active"] is True
    assert failed_revert_channel["active_fee_ppm"] == 340
    assert failed_revert_channel["baseline_fee_ppm"] == 100
    assert failed_revert_channel["last_apply_result"] == "revert_failed"

    manager.process_pending_actions()
    still_failed_status = manager.get_status()

    assert applied == [("2x2x2", 340), ("2x2x2", 100)]
    assert still_failed_status["active_channel_count"] == 1

    clock.advance(15)
    manager.process_pending_actions()
    final_status = manager.get_status()

    assert applied == [("2x2x2", 340), ("2x2x2", 100), ("2x2x2", 100)]
    assert final_status["active_channel_count"] == 0
