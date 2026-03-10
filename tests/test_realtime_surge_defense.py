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
):
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
        surge_cooldown_seconds=cooldown_seconds,
        surge_setchannel_min_interval_seconds=min_interval_seconds,
        channel_capacity_msat=lambda channel_id: capacity_msat,
        current_fee_ppm=lambda channel_id: baseline_fee_ppm,
        apply_fee=apply_fee,
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
