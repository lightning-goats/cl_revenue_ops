"""Actual plugin ingestion adapters with real SQLite; no live CLN actions."""

from copy import deepcopy
from dataclasses import replace
import sqlite3
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.fee_authority import FeeAuthorityGate
from modules.forward_identity import ForwardIdentityError, ForwardSource
from tests.plugin_test_utils import load_plugin_module


SOURCE = ForwardSource("02" + "a" * 64, "regtest", "verified-test-wallet")
PEER = "03" + "b" * 64


def event(**changes):
    return dict(
        status="settled", in_channel="672x1x0", out_channel="690x1x0",
        in_htlc_id=23, created_index=86, updated_index=86,
        in_msat=50_018_300, out_msat=50_000_000, fee_msat=18_300,
        received_time="1788657630.1243427", resolved_time="1788657630.2436247",
    ) | changes


def pair():
    return [event(), event(in_htlc_id=24, created_index=87, updated_index=87,
                           received_time="1788657630.6604204", resolved_time="1788657630.7836533")]


@pytest.fixture
def mod(tmp_path):
    module = load_plugin_module()
    module.database = Database(str(tmp_path / "adapters.db"), module.plugin)
    module.database.initialize()
    module.database.initialize_native_forward_ingestion(SOURCE)
    module._resolve_scid_to_peer = MagicMock(return_value=PEER)
    module.safe_plugin = MagicMock()
    module.data_service = MagicMock()
    module.fee_controller = MagicMock()
    module.fee_controller.should_wake_acquisition_cycle.return_value = True
    module.fee_controller.should_wake_yield_inventory_cycle.return_value = False
    module._request_fee_adjustment_wake = MagicMock()
    module.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 10_000)
    yield module
    # Only fake read RPCs are permitted in hydration. Notification tests use
    # a fake peer resolver, so no node/graph data is needed for admission tests.
    assert module.plugin.rpc.mock_calls == []
    assert all(call[0] == "rpc.listforwards" for call in module.safe_plugin.mock_calls)
    assert all(call[0] == "get_forwards" for call in module.data_service.mock_calls)
    module.database.close()


def totals(mod):
    return tuple(mod.database._get_connection().execute(
        "SELECT count(*),coalesce(sum(fee_msat),0) FROM forwards"
    ).fetchone())


def deliver(mod, payload):
    mod.on_forward_event(payload, mod.plugin)


def hydrate(mod, rows, *, fallback=False, start_time=1788657629):
    if fallback:
        mod.safe_plugin.rpc.listforwards.side_effect = RuntimeError("offline paging")
        mod.data_service.get_forwards.return_value = {"forwards": rows}
    else:
        mod.safe_plugin.rpc.listforwards.return_value = {"forwards": rows}
    return mod._hydrate_settled_forward_rows(start_time)


@pytest.mark.parametrize("peer_known", [True, False])
def test_real_notification_keeps_distinct_native_ids_times_and_wakes_once(mod, peer_known):
    mod._resolve_scid_to_peer.return_value = PEER if peer_known else None
    events = pair()
    original = deepcopy(events)
    for payload in events + events:
        deliver(mod, payload)
    assert totals(mod) == (2, 36_600)
    assert mod._request_fee_adjustment_wake.call_count == 2
    assert mod.fee_controller.should_wake_acquisition_cycle.call_count == 2
    assert mod.fee_controller.should_wake_yield_inventory_cycle.call_count == 4
    c = mod.database._get_connection()
    assert [r[0] for r in c.execute("SELECT received_time_ns FROM forwards ORDER BY id")] == [
        1788657630124342700, 1788657630660420400,
    ]
    assert c.execute("SELECT coalesce(sum(success_count),0) FROM peer_reputation").fetchone()[0] == (
        2 if peer_known else 0
    )
    assert events == original


@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize("hydration_first", [True, False])
def test_actual_startup_adapter_and_notification_share_receipts(mod, fallback, hydration_first):
    history = pair()
    notifications = [dict(row, created_index=None, updated_index=None) for row in history]
    if hydration_first:
        assert hydrate(mod, history, fallback=fallback) == (2, 2)
    for payload in notifications:
        deliver(mod, payload)
    assert hydrate(mod, history, fallback=fallback) == (0, 2)
    assert totals(mod) == (2, 36_600)
    c = mod.database._get_connection()
    assert c.execute("SELECT count(*) FROM forward_receipts_v1").fetchone()[0] == 2
    assert mod._request_fee_adjustment_wake.call_count == (0 if hydration_first else 2)


@pytest.mark.parametrize("payload", [
    None, [], {}, event(in_htlc_id=None), event(in_htlc_id=True),
    event(received_time=None), event(resolved_time=None), event(in_msat="broken"),
    event(in_msat="broken", in_msatoshi=50_018_300), event(out_channel=[]),
    event(fee_msat=-1), event(received_time="NaN"),
])
def test_malformed_or_unknown_notification_never_creates_reputation_or_wake(mod, payload):
    deliver(mod, payload)
    assert totals(mod) == (0, 0)
    c = mod.database._get_connection()
    assert c.execute("SELECT count(*) FROM forward_receipts_v1").fetchone()[0] == 0
    assert c.execute("SELECT count(*) FROM peer_reputation").fetchone()[0] == 0
    mod._request_fee_adjustment_wake.assert_not_called()
    mod.fee_controller.should_wake_acquisition_cycle.assert_not_called()


@pytest.mark.parametrize("fallback", [True, False])
def test_hydration_filters_bad_times_without_losing_valid_native_rows(mod, fallback):
    rows = [None, [], event(received_time=None), event(received_time="NaN"),
            event(in_msat="unknown"), *pair()]
    assert hydrate(mod, rows, fallback=fallback) == (2, 3)
    assert totals(mod) == (2, 36_600)
    mod._request_fee_adjustment_wake.assert_not_called()


@pytest.mark.parametrize("fallback", [True, False])
def test_fractional_received_time_in_first_second_of_window_is_not_discarded(mod, fallback):
    assert hydrate(mod, pair(), fallback=fallback, start_time=1788657630) == (2, 2)
    assert totals(mod) == (2, 36_600)


def test_new_database_instance_requires_source_reverification_before_any_ingestion(mod):
    old = mod.database
    deliver(mod, event())
    path = old.db_path
    old.close()
    mod.database = Database(path, mod.plugin)
    mod.database.initialize()  # Schema validation is not wallet verification.
    mod._request_fee_adjustment_wake.reset_mock()
    with pytest.raises(ForwardIdentityError, match="continuity"):
        hydrate(mod, [pair()[1]])
    mod.safe_plugin.rpc.listforwards.assert_not_called()
    deliver(mod, pair()[1])
    assert totals(mod) == (1, 18_300)
    mod._request_fee_adjustment_wake.assert_not_called()
    # Explicit test-fixture caller re-admits the same verified wallet, without
    # creating a new generation or resetting the existing receipts.
    mod.database.initialize_native_forward_ingestion(SOURCE)
    assert hydrate(mod, pair()) == (1, 2)
    assert totals(mod) == (2, 36_600)


def test_failed_readmission_revokes_process_admission(mod):
    assert mod.database.get_native_forward_source() == SOURCE
    with pytest.raises(ForwardIdentityError, match="continuity"):
        mod.database.initialize_native_forward_ingestion(replace(SOURCE, generation="other"))
    with pytest.raises(ForwardIdentityError, match="continuity"):
        mod.database.get_native_forward_source()
    deliver(mod, event())
    assert totals(mod) == (0, 0)
    mod._request_fee_adjustment_wake.assert_not_called()


@pytest.mark.parametrize("table", ["forwards", "peer_reputation"])
def test_notification_write_failure_never_advances_wake_and_can_retry(mod, table):
    c = mod.database._get_connection()
    c.execute(f"CREATE TRIGGER injected_failure BEFORE INSERT ON {table} "
              "BEGIN SELECT RAISE(ABORT,'injected'); END")
    deliver(mod, event())
    assert totals(mod) == (0, 0)
    assert c.execute("SELECT count(*) FROM forward_receipts_v1").fetchone()[0] == 0
    mod._request_fee_adjustment_wake.assert_not_called()
    c.execute("DROP TRIGGER injected_failure")
    deliver(mod, event())
    assert totals(mod) == (1, 18_300)
    mod._request_fee_adjustment_wake.assert_called_once()


def test_hydration_database_failure_propagates_instead_of_reporting_duplicates(mod):
    c = mod.database._get_connection()
    c.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON forwards "
              "BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        hydrate(mod, pair())
    assert c.execute("SELECT count(*) FROM forward_receipts_v1").fetchone()[0] == 0
    mod._request_fee_adjustment_wake.assert_not_called()


def test_disabled_fee_authority_keeps_native_accounting_but_does_not_wake(mod):
    mod.fee_authority_gate.set_enabled(False, reason="test")
    deliver(mod, event())
    assert totals(mod) == (1, 18_300)
    mod._request_fee_adjustment_wake.assert_not_called()
    mod.fee_controller.should_wake_acquisition_cycle.assert_not_called()


def test_pruned_replay_through_actual_adapters_does_not_wake_or_recredit(mod, monkeypatch):
    deliver(mod, event())
    monkeypatch.setattr("modules.database.time.time", lambda: 1788657630 + 10 * 86400)
    mod.database.cleanup_old_data(8)
    mod._request_fee_adjustment_wake.reset_mock()
    deliver(mod, event())
    assert hydrate(mod, [event()]) == (0, 1)
    assert totals(mod) == (0, 0)
    c = mod.database._get_connection()
    assert c.execute("SELECT sum(total_fee_msat) FROM daily_forwarding_stats").fetchone()[0] == 18_300
    mod._request_fee_adjustment_wake.assert_not_called()


def test_source_getter_is_read_only_and_legacy_database_is_not_auto_activated(tmp_path):
    db = Database(str(tmp_path / "legacy.db"), MagicMock())
    try:
        db.initialize()
        c = db._get_connection()
        before = list(c.iterdump())
        c.execute("PRAGMA query_only=ON")
        assert db.get_native_forward_source() is None
        assert list(c.iterdump()) == before
        assert db.plugin.rpc.mock_calls == []
    finally:
        db.close()


def test_actual_init_does_not_turn_native_admission_failure_into_stale_decision_loops(
    tmp_path, monkeypatch
):
    from tests.test_operator_surface import _run_init_with_stubbed_dependencies

    path = str(tmp_path / "readmission.db")
    db = Database(path, MagicMock())
    db.initialize()
    db.initialize_native_forward_ingestion(SOURCE)
    db.close()
    reopened = Database(path, MagicMock())
    reopened.initialize()
    module = load_plugin_module()

    def unavailable_admission():
        # Pin actual init ordering as well as the real Database refusal.
        module._test_fake_db.initialize.assert_called_once()
        return reopened.get_native_forward_source()

    monkeypatch.setattr(module, "_native_forward_ingestion_source", unavailable_admission)
    try:
        with pytest.raises(ForwardIdentityError, match="continuity"):
            _run_init_with_stubbed_dependencies(module, monkeypatch)
        assert module._test_threads == []
        assert module.fee_controller is None
        assert module.rebalancer is None
        module._test_fake_db.bulk_insert_forwards.assert_not_called()
        assert all(call[0] in {"getinfo", "listconfigs", "plugin", "listplugins"}
                   for call in module._test_fake_rpc.mock_calls)
    finally:
        reopened.close()
