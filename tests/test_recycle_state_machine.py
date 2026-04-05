"""Tests for recycling state machine."""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


class TestRecycleOpsTable:

    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = Database(self.tmpfile.name, MagicMock())
        self.db.initialize()

    def test_create_recycle_op(self):
        op_id = self.db.create_recycle_op(
            close_scid="100x1x0", close_peer_id="peer_a",
            open_peer_id="peer_b", open_amount_sats=2_000_000,
            recycle_ev_sats=8000, funding_source="close",
        )
        assert op_id > 0

    def test_get_pending_recycle(self):
        self.db.create_recycle_op(
            close_scid="100x1x0", close_peer_id="a", open_peer_id="b",
            open_amount_sats=2_000_000, recycle_ev_sats=8000,
        )
        pending = self.db.get_pending_recycle_op()
        assert pending is not None
        assert pending["status"] == "pending_close"

    def test_update_recycle_status(self):
        op_id = self.db.create_recycle_op(
            close_scid="100x1x0", close_peer_id="a", open_peer_id="b",
            open_amount_sats=2_000_000, recycle_ev_sats=8000,
        )
        self.db.update_recycle_op(op_id, status="pending_open", close_action_id=42)
        op = self.db.get_pending_recycle_op()
        assert op["status"] == "pending_open"
        assert op["close_action_id"] == 42

    def test_complete_recycle(self):
        op_id = self.db.create_recycle_op(
            close_scid="100x1x0", close_peer_id="a", open_peer_id="b",
            open_amount_sats=2_000_000, recycle_ev_sats=8000,
        )
        self.db.update_recycle_op(op_id, status="completed", open_action_id=43)
        pending = self.db.get_pending_recycle_op()
        assert pending is None

    def test_increment_cycles_waited(self):
        op_id = self.db.create_recycle_op(
            close_scid="100x1x0", close_peer_id="a", open_peer_id="b",
            open_amount_sats=2_000_000, recycle_ev_sats=8000,
        )
        self.db.increment_recycle_cycles_waited(op_id)
        self.db.increment_recycle_cycles_waited(op_id)
        self.db.increment_recycle_cycles_waited(op_id)
        self.db.increment_recycle_cycles_waited(op_id)
        op = self.db.get_pending_recycle_op()
        assert op["cycles_waited"] == 4
