"""Regression tests for clboss_manager P0 fixes."""

from unittest.mock import MagicMock, patch, call
from pyln.client import RpcError

import pytest

from modules.clboss_manager import ClbossManager, ClbossTags


def _make_manager(clboss_enabled=True, dry_run=False):
    plugin = MagicMock()
    plugin.log = MagicMock()
    plugin.rpc.call = MagicMock(return_value={})
    config = MagicMock()
    config.clboss_enabled = clboss_enabled
    config.dry_run = dry_run
    mgr = ClbossManager(plugin, config)
    mgr._clboss_available = True  # skip availability check
    return mgr


class TestRemanageUsesClbossManage:
    """P0-2: remanage() must use clboss-manage, not clboss-unmanage with empty string."""

    def test_remanage_all_tags_uses_clboss_manage(self):
        mgr = _make_manager()
        peer_id = "02" + "ab" * 32

        result = mgr.remanage(peer_id, tag=None)

        # Should call clboss-manage for each tag in ALL, NOT clboss-unmanage
        calls = mgr.plugin.rpc.call.call_args_list
        for c in calls:
            assert c[0][0] == "clboss-manage", \
                f"Expected clboss-manage but got {c[0][0]}"
        # Should have called for all 4 tags
        assert len(calls) == len(ClbossTags.ALL)
        assert result["success"] is True

    def test_remanage_single_tag_uses_clboss_manage(self):
        mgr = _make_manager()
        peer_id = "02" + "ab" * 32

        result = mgr.remanage(peer_id, tag="lnfee")

        calls = mgr.plugin.rpc.call.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == "clboss-manage"
        assert calls[0][0][1] == [peer_id, "lnfee"]

    def test_remanage_list_tag_uses_clboss_manage(self):
        mgr = _make_manager()
        peer_id = "02" + "ab" * 32

        result = mgr.remanage(peer_id, tag=["lnfee", "balance"])

        calls = mgr.plugin.rpc.call.call_args_list
        assert len(calls) == 2
        assert all(c[0][0] == "clboss-manage" for c in calls)


class TestRemanageDBCleanup:
    """P0-1: remanage() must clean up DB unmanage records."""

    def test_remanage_cleans_up_db(self):
        mgr = _make_manager()
        peer_id = "02" + "cd" * 32
        database = MagicMock()

        result = mgr.remanage(peer_id, tag="lnfee", database=database)

        assert result["success"] is True
        database.remove_unmanage.assert_called_once_with(peer_id, "lnfee")

    def test_remanage_all_cleans_up_db_all(self):
        mgr = _make_manager()
        peer_id = "02" + "cd" * 32
        database = MagicMock()

        result = mgr.remanage(peer_id, tag=None, database=database)

        assert result["success"] is True
        # tag=None should call remove_unmanage with no tag (remove all)
        database.remove_unmanage.assert_called_once_with(peer_id)

    def test_remanage_no_db_still_works(self):
        mgr = _make_manager()
        peer_id = "02" + "cd" * 32

        # No database passed - should not crash
        result = mgr.remanage(peer_id, tag="lnfee")
        assert result["success"] is True

    def test_remanage_db_failure_non_fatal(self):
        mgr = _make_manager()
        peer_id = "02" + "cd" * 32
        database = MagicMock()
        database.remove_unmanage.side_effect = Exception("DB error")

        result = mgr.remanage(peer_id, tag="lnfee", database=database)
        # RPC succeeded, DB cleanup failed but non-fatal
        assert result["success"] is True


class TestReconcileUnmanaged:
    """P0-1: Startup reconciliation must re-manage orphaned peers."""

    def test_reconcile_remanages_orphaned_peers(self):
        mgr = _make_manager()
        database = MagicMock()
        database.get_all_unmanaged.return_value = [
            {"peer_id": "02" + "aa" * 32, "tag": "lnfee"},
            {"peer_id": "02" + "bb" * 32, "tag": "lnfee,balance"},
        ]

        result = mgr.reconcile_unmanaged(database)

        assert result["orphaned_count"] == 2
        assert result["remanaged"] == 2
        # Should have called clboss-manage for each
        manage_calls = [c for c in mgr.plugin.rpc.call.call_args_list if c[0][0] == "clboss-manage"]
        assert len(manage_calls) >= 2  # at least one per peer
        # Should have cleaned up DB records
        assert database.remove_unmanage.call_count == 2

    def test_reconcile_no_orphans(self):
        mgr = _make_manager()
        database = MagicMock()
        database.get_all_unmanaged.return_value = []

        result = mgr.reconcile_unmanaged(database)
        assert result["orphaned_count"] == 0
        assert result["remanaged"] == 0

    def test_reconcile_skipped_when_clboss_unavailable(self):
        mgr = _make_manager(clboss_enabled=False)
        database = MagicMock()

        result = mgr.reconcile_unmanaged(database)
        assert result["skipped"] is True
