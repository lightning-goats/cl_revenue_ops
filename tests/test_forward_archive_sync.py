"""Read-only Core Lightning synchronization for canonical forward evidence."""

import sqlite3
from unittest.mock import MagicMock, call

import pytest

from modules.forward_archive import ForwardArchiveStore
from modules.forward_archive_sync import (
    ForwardArchiveSyncError,
    ForwardArchiveSynchronizer,
)


def _log(*_args, **_kwargs):
    return None


@pytest.fixture
def store(tmp_path):
    connection = sqlite3.connect(
        tmp_path / "forward-archive.db",
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    archive = ForwardArchiveStore(lambda: connection, _log)
    archive.initialize_schema(connection)
    archive._test_connection = connection
    return archive


def _record(
    *,
    created_index,
    updated_index=None,
    status="offered",
    received_time="1700000000",
):
    record = {
        "created_index": created_index,
        "status": status,
        "received_time": received_time,
    }
    if updated_index is not None:
        record["updated_index"] = updated_index
    if status == "settled":
        record.update({
            "in_channel": "1x1x1",
            "out_channel": "2x2x2",
            "in_msat": 2000,
            "out_msat": 1900,
            "fee_msat": 100,
            "resolved_time": "1700000001",
        })
    return record


def test_sync_probes_and_pages_cursor_families_independently(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 12},
    ]
    rpc.listforwards.side_effect = [
        {"forwards": [
            _record(created_index=1, updated_index=11, status="settled"),
            _record(created_index=2, updated_index=12, status="settled"),
        ]},
        {"forwards": [
            _record(created_index=1, updated_index=11, status="settled"),
            _record(created_index=2, updated_index=12, status="settled"),
        ]},
    ]

    result = ForwardArchiveSynchronizer(rpc, store, _log).sync_once(
        now_ns=1_700_006_401_000_000_000
    )

    assert result.created_live_max == 2
    assert result.updated_live_max == 12
    assert result.created_pages == 1
    assert result.updated_pages == 1
    assert store.get_sync_state("created")["next_index"] == 3
    assert store.get_sync_state("updated")["next_index"] == 13
    assert rpc.wait.call_args_list == [
        call(subsystem="forwards", indexname="created", nextvalue=0),
        call(subsystem="forwards", indexname="updated", nextvalue=0),
    ]
    assert rpc.listforwards.call_args_list == [
        call(index="created", start=1, limit=500),
        call(index="updated", start=1, limit=500),
    ]


def test_sync_rejects_stored_cursor_ahead_of_its_own_live_max(store):
    store.apply_page(
        "updated",
        [_record(created_index=1, updated_index=8, status="settled")],
        observed_at_ns=10,
        live_max_index=8,
    )
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 20},
        {"subsystem": "forwards", "updated": 4},
    ]

    with pytest.raises(
        ForwardArchiveSyncError,
        match="updated cursor 9 exceeds live maximum 4",
    ):
        ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=11)

    rpc.listforwards.assert_not_called()


def test_malformed_first_page_preserves_both_last_successful_cursors(store):
    store.apply_page(
        "created", [_record(created_index=4)], 10, live_max_index=4
    )
    store.apply_page(
        "updated",
        [_record(created_index=4, updated_index=2, status="settled")],
        10,
        live_max_index=2,
    )
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 6},
        {"subsystem": "forwards", "updated": 4},
    ]
    rpc.listforwards.return_value = {"forwards": [0]}

    with pytest.raises(ForwardArchiveSyncError, match="expected object"):
        ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=11)

    assert store.get_sync_state("created")["next_index"] == 5
    assert store.get_sync_state("updated")["next_index"] == 3


@pytest.mark.parametrize(
    "payload, error",
    [
        ({}, "malformed payload"),
        ({"subsystem": "forwards", "created": True}, "invalid index"),
        ({"subsystem": "forwards", "created": -1}, "invalid index"),
        ({"subsystem": "invoices", "created": 1}, "malformed payload"),
    ],
)
def test_live_max_payload_fails_closed(store, payload, error):
    rpc = MagicMock()
    rpc.wait.return_value = payload

    with pytest.raises(ForwardArchiveSyncError, match=error):
        ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)

    rpc.listforwards.assert_not_called()


def test_unknown_archive_schema_version_disables_sync_before_rpc(store):
    store.apply_page(
        "created", [_record(created_index=1)], 10, live_max_index=1
    )
    store._test_connection.execute(
        "UPDATE forward_archive_sync_state_v1 SET schema_version = 2"
    )
    rpc = MagicMock()

    with pytest.raises(
        ForwardArchiveSyncError,
        match="unsupported archive schema version 2",
    ):
        ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=11)

    rpc.wait.assert_not_called()
    rpc.listforwards.assert_not_called()


def test_page_limit_returns_checkpointed_backlog_without_sync_error(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
    ]
    rpc.listforwards.return_value = {
        "forwards": [_record(created_index=1)]
    }
    sync = ForwardArchiveSynchronizer(rpc, store, _log)
    sync.PAGE_LIMIT = 1
    sync.MAX_PAGES_PER_FAMILY = 1

    result = sync.sync_once(now_ns=10)

    assert result.caught_up is False
    assert result.backlog_family == "created"
    assert result.created_pages == 1
    assert result.updated_pages == 0
    assert result.touched_dates == (1699920000,)
    assert store.get_sync_state("created")["next_index"] == 2
    assert store.get_sync_state("created")["last_error"] is None
    assert rpc.listforwards.call_count == 1


def test_next_cycle_resumes_checkpoint_and_reaches_catch_up(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
    ]
    rpc.listforwards.side_effect = [
        {"forwards": [_record(created_index=1)]},
        {"forwards": [_record(created_index=2)]},
    ]
    sync = ForwardArchiveSynchronizer(rpc, store, _log)
    sync.PAGE_LIMIT = 1
    sync.MAX_PAGES_PER_FAMILY = 1

    first = sync.sync_once(now_ns=10)
    second = sync.sync_once(now_ns=11)

    assert first.caught_up is False
    assert second.caught_up is True
    assert second.backlog_family is None
    assert rpc.listforwards.call_args_list == [
        call(index="created", start=1, limit=1),
        call(index="created", start=2, limit=1),
    ]
    assert store.get_sync_state("created")["next_index"] == 3


def test_caught_up_empty_source_calls_no_listforwards(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 0},
        {"subsystem": "forwards", "updated": 0},
    ]

    result = ForwardArchiveSynchronizer(rpc, store, _log).sync_once(
        now_ns=1_700_006_401_000_000_000
    )

    assert result.created_pages == 0
    assert result.updated_pages == 0
    rpc.listforwards.assert_not_called()
    assert store.get_sync_state("created")["last_success_at"] == result.observed_at_ns
    assert store.get_sync_state("updated")["last_success_at"] == result.observed_at_ns


def test_updated_backlog_preserves_dates_from_all_committed_records(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 1},
        {"subsystem": "forwards", "updated": 2},
    ]
    rpc.listforwards.side_effect = [
        {"forwards": [_record(created_index=1)]},
        {"forwards": [
            _record(
                created_index=2,
                updated_index=1,
                received_time="1700086400",
            )
        ]},
    ]
    sync = ForwardArchiveSynchronizer(rpc, store, _log)
    sync.PAGE_LIMIT = 1
    sync.MAX_PAGES_PER_FAMILY = 1

    result = sync.sync_once(now_ns=10)

    assert result.caught_up is False
    assert result.backlog_family == "updated"
    assert result.created_pages == 1
    assert result.updated_pages == 1
    assert result.touched_dates == (1699920000, 1700006400)
    assert store.get_sync_state("updated")["next_index"] == 2
    assert store.get_sync_state("updated")["last_error"] is None




def test_sync_ignores_records_newer_than_probed_snapshot(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
    ]
    rpc.listforwards.return_value = {
        "forwards": [
            _record(created_index=1),
            _record(created_index=2),
            _record(created_index=3),
        ]
    }

    result = ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)

    assert result.created_pages == 1
    assert store.get_sync_state("created")["next_index"] == 3
    row = store._test_connection.execute("SELECT COUNT(*) FROM forward_archive_v1").fetchone()
    assert row[0] == 2


def test_caught_up_cycle_recovers_missing_closed_day_coverage(store):
    day = 1699920000
    observed = (day + 2 * 86400) * 1_000_000_000
    record = _record(
        created_index=1,
        updated_index=1,
        status="settled",
        received_time=str(day + 3600),
    )
    store.apply_page("created", [record], observed, live_max_index=1)
    store.apply_page("updated", [record], observed, live_max_index=1)
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 1},
        {"subsystem": "forwards", "updated": 1},
    ]

    result = ForwardArchiveSynchronizer(rpc, store, _log).sync_once(
        now_ns=observed
    )

    assert result.caught_up is True
    assert day in result.touched_dates
    assert store.history(day, day + 86400, None, 100)["complete"] is True


def test_sync_rpc_allowlist_is_wait_and_listforwards_only(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 0},
        {"subsystem": "forwards", "updated": 0},
    ]

    ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)

    assert {method[0] for method in rpc.method_calls} <= {
        "wait",
        "listforwards",
    }
