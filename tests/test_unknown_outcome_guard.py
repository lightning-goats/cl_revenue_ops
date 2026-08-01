"""Task 26 (task-23 unknown-outcome siblings): broadcast-capable RPCs must not
record a DEFINITE failure — nor release a possibly-COMMITTED reservation —
when the exception leaves the outcome unknown.

CLN's ``fundchannel`` broadcasts the funding tx as part of the call, and
``close`` can fall back to a unilateral close after an RPC-level timeout, so
a transport/timeout exception can arrive AFTER real sats moved. The old
handlers wrote ``status="failed"`` and released the reservation in both
cases, dropping a committed on-chain cost off the unified budget rail — the
exact thing ``_settle_capex_reservation``'s own docstring forbids.

Contract pinned here:
- structured/definite rejections (``RuntimeError("boom")``-shaped) keep the
  old behavior: release + failed (tests in test_p4_018 already pin that);
- timeout/transport exceptions probe ``listpeerchannels`` once:
  evidence of the open/close on-chain -> settle as committed;
  positive evidence of absence            -> release as before;
  probe unavailable                        -> HOLD the reservation and
  record ``status="unknown"`` — never a definite answer that was not
  measured.
"""

import os
import sys
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_p4_018_planner_unified_budget import (  # noqa: E402
    _make_db, _make_planner, _active_total,
)

PEER = "02" + "ab" * 32


def _planner(tmp_path, budget=1000):
    db = _make_db(tmp_path, name="unknown.db")
    planner, cfg = _make_planner(db, budget=budget)
    planner._estimate_open_cost = MagicMock(return_value=400)
    return db, planner, cfg


def _peer_channels_probe(planner, channels):
    probe = MagicMock(return_value={"channels": channels})
    planner.plugin.rpc.listpeerchannels = probe
    planner.data_service = None
    return probe


def _last_action_status(db):
    conn = db._get_connection()
    row = conn.execute(
        "SELECT status FROM planner_actions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


class TestExecuteOpenUnknownOutcome:
    def test_timeout_with_opening_channel_evidence_settles_as_committed(self, tmp_path):
        db, planner, cfg = _planner(tmp_path)
        planner._rpc_fundchannel = MagicMock(
            side_effect=TimeoutError("rpc timed out"))
        _peer_channels_probe(planner, [{
            "peer_id": PEER, "state": "CHANNELD_AWAITING_LOCKIN",
            "channel_id": "c" * 64,
        }])
        res = planner._execute_open(PEER, 1_000_000, cfg, reason="test")
        assert res.get("status") == "completed", res
        conn = db._get_connection()
        events = conn.execute(
            "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events "
            "WHERE category='channel_open'").fetchone()[0]
        assert int(events) == 400, "committed cost must land on the rail"
        assert _active_total(db) == 0

    def test_timeout_with_proven_absence_releases_as_before(self, tmp_path):
        db, planner, cfg = _planner(tmp_path)
        planner._rpc_fundchannel = MagicMock(
            side_effect=TimeoutError("rpc timed out"))
        _peer_channels_probe(planner, [])
        res = planner._execute_open(PEER, 1_000_000, cfg, reason="test")
        assert res.get("status") == "failed", res
        assert _active_total(db) == 0

    def test_timeout_with_failing_probe_holds_reservation_and_says_unknown(self, tmp_path):
        db, planner, cfg = _planner(tmp_path)
        planner._rpc_fundchannel = MagicMock(
            side_effect=TimeoutError("rpc timed out"))
        planner.data_service = None
        planner.plugin.rpc.listpeerchannels = MagicMock(
            side_effect=RuntimeError("node unreachable"))
        res = planner._execute_open(PEER, 1_000_000, cfg, reason="test")
        assert res.get("status") == "unknown", (
            f"an unmeasured outcome must never be reported definite: {res}")
        assert _active_total(db) == 400, (
            "a possibly-committed reservation must be HELD, never released")
        assert _last_action_status(db) == "unknown"

    def test_preexisting_normal_channel_is_not_evidence_of_our_open(self, tmp_path):
        """A CHANNELD_NORMAL channel predates this call — only an
        in-progress opening state proves OUR broadcast."""
        db, planner, cfg = _planner(tmp_path)
        planner._rpc_fundchannel = MagicMock(
            side_effect=TimeoutError("rpc timed out"))
        _peer_channels_probe(planner, [{
            "peer_id": PEER, "state": "CHANNELD_NORMAL",
            "channel_id": "d" * 64,
        }])
        res = planner._execute_open(PEER, 1_000_000, cfg, reason="test")
        assert res.get("status") == "failed", res
        assert _active_total(db) == 0

    def test_min_size_retry_never_runs_on_unknown_outcome(self, tmp_path):
        """The retry branch releases and recurses into a SECOND open. If the
        first outcome is unknown, that risks a double-open — the retry must
        be gated on a definite pre-broadcast rejection."""
        db, planner, cfg = _planner(tmp_path)
        calls = []

        def fund(*a, **k):
            calls.append(a)
            raise TimeoutError("they sent error channel_open failed: min chan size of 0.02 BTC; connection timed out")

        planner._rpc_fundchannel = fund
        planner._check_reserve = MagicMock(return_value=(True, ""))
        planner.data_service = None
        planner.plugin.rpc.listpeerchannels = MagicMock(
            side_effect=RuntimeError("node unreachable"))
        res = planner._execute_open(PEER, 1_000_000, cfg, reason="test")
        assert len(calls) == 1, "unknown outcome must not trigger a second open"
        assert res.get("status") == "unknown"

    def test_min_size_retry_gated_even_when_probe_proves_absence(self, tmp_path):
        """Absence at probe time does not preclude a LATE broadcast from the
        first, timed-out call — lightningd may still complete it after we
        probed. The retry stays gated on a definite rejection even on the
        proven-absence fall-through."""
        db, planner, cfg = _planner(tmp_path)
        calls = []

        def fund(*a, **k):
            calls.append(a)
            raise TimeoutError("they sent error channel_open failed: min chan size of 0.02 BTC; connection timed out")

        planner._rpc_fundchannel = fund
        planner._check_reserve = MagicMock(return_value=(True, ""))
        _peer_channels_probe(planner, [])
        # Advance the clock per call so a recursive retry would mint a
        # DISTINCT reservation id — without this, a same-second id collision
        # masks the recursion instead of the gate stopping it.
        import itertools
        from unittest.mock import patch
        import modules.capacity_planner as cp_mod
        with patch.object(cp_mod.time, "time",
                          side_effect=itertools.count(1_700_000_000, 60).__next__):
            res = planner._execute_open(PEER, 1_000_000, cfg, reason="test")
        assert len(calls) == 1, (
            "a probe-proven absence still must not re-open: the first tx "
            "can broadcast after the probe")
        assert res.get("status") == "failed"


class TestExecuteCloseUnknownOutcome:
    SCID = "100x1x0"

    def _close_planner(self, tmp_path):
        db, planner, cfg = _planner(tmp_path)
        planner._close_fee_plan = MagicMock(return_value={
            "ok": True, "estimated_cost_sats": 300, "reserve_sats": 300,
            "source": "test",
        })
        planner.rebalancer = None
        return db, planner, cfg

    def test_timeout_with_closing_state_evidence_settles_reserved_cap(self, tmp_path):
        db, planner, cfg = self._close_planner(tmp_path)
        planner._rpc_close = MagicMock(side_effect=TimeoutError("rpc timed out"))
        _peer_channels_probe(planner, [{
            "peer_id": PEER, "state": "CLOSINGD_SIGEXCHANGE",
            "short_channel_id": self.SCID,
        }])
        res = planner._execute_close(self.SCID, PEER, cfg, reason="test")
        assert res.get("status") == "completed", res
        conn = db._get_connection()
        events = conn.execute(
            "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events "
            "WHERE category='channel_close'").fetchone()[0]
        assert int(events) == 300
        assert _active_total(db) == 0

    def test_timeout_with_channel_still_normal_releases(self, tmp_path):
        db, planner, cfg = self._close_planner(tmp_path)
        planner._rpc_close = MagicMock(side_effect=TimeoutError("rpc timed out"))
        _peer_channels_probe(planner, [{
            "peer_id": PEER, "state": "CHANNELD_NORMAL",
            "short_channel_id": self.SCID,
        }])
        res = planner._execute_close(self.SCID, PEER, cfg, reason="test")
        assert res.get("status") == "failed", res
        assert _active_total(db) == 0

    def test_timeout_with_failing_probe_holds_and_says_unknown(self, tmp_path):
        db, planner, cfg = self._close_planner(tmp_path)
        planner._rpc_close = MagicMock(side_effect=TimeoutError("rpc timed out"))
        planner.data_service = None
        planner.plugin.rpc.listpeerchannels = MagicMock(
            side_effect=RuntimeError("node unreachable"))
        res = planner._execute_close(self.SCID, PEER, cfg, reason="test")
        assert res.get("status") == "unknown", res
        assert _active_total(db) == 300
        assert _last_action_status(db) == "unknown"

    def test_channel_gone_from_listing_is_close_evidence(self, tmp_path):
        """A channel absent from listpeerchannels after a close attempt has
        left the channel set — that is evidence the close progressed, not
        that it never happened."""
        db, planner, cfg = self._close_planner(tmp_path)
        planner._rpc_close = MagicMock(side_effect=TimeoutError("rpc timed out"))
        _peer_channels_probe(planner, [])
        res = planner._execute_close(self.SCID, PEER, cfg, reason="test")
        assert res.get("status") == "completed", res
        assert _active_total(db) == 0

    def test_definite_rejection_still_releases(self, tmp_path):
        db, planner, cfg = self._close_planner(tmp_path)
        planner._rpc_close = MagicMock(side_effect=RuntimeError("unknown channel"))
        probe = _peer_channels_probe(planner, [])
        res = planner._execute_close(self.SCID, PEER, cfg, reason="test")
        assert res.get("status") == "failed", res
        assert _active_total(db) == 0
        probe.assert_not_called()


# ---------------------------------------------------------------------------
# Site 3: a boltzcli timeout on swap CREATE. boltzd may already have created
# the swap server-side; releasing the reservation resolved "unknown" as
# "not created" and left NO local record at all.
# ---------------------------------------------------------------------------
from tests.test_p4_014_boltz_unified_budget import (  # noqa: E402
    _make_db as _make_boltz_db, _make_manager,
)
from modules.boltz_manager import BoltzCliError  # noqa: E402


def _boltz_active_total(db):
    conn = db._get_connection()
    return int(conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active' AND category='boltz'").fetchone()[0] or 0)


class TestBoltzCreateTimeout:
    def test_create_timeout_holds_reservation(self, tmp_path):
        db = _make_boltz_db(tmp_path, name="unknown_boltz.db")
        mgr = _make_manager(db)
        mgr._run_json = MagicMock(side_effect=BoltzCliError(
            "boltzcli timed out after 120s: createreverseswap ..."))
        try:
            mgr.loop_out(amount_sats=250_000)
        except BoltzCliError:
            pass
        assert _boltz_active_total(db) == 400, (
            "a create timeout leaves the swap outcome UNKNOWN — the "
            "reservation must be HELD, not released")

    def test_create_definite_failure_still_releases(self, tmp_path):
        db = _make_boltz_db(tmp_path, name="unknown_boltz2.db")
        mgr = _make_manager(db)
        mgr._run_json = MagicMock(side_effect=BoltzCliError(
            "boltzcli exited 1: insufficient wallet balance"))
        try:
            mgr.loop_out(amount_sats=250_000)
        except BoltzCliError:
            pass
        assert _boltz_active_total(db) == 0, (
            "a structured rejection is definite — release as before")
