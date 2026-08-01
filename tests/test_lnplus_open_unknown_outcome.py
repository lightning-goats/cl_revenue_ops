"""Audit 2026-08-01 (high): LN+ fundchannel unknown-outcome reservation hold.

fundchannel is broadcast-capable: a transport death/timeout may fire AFTER the
funding tx broadcast. `_execute_swap_open` used to release the channel_open
budget reservation on ANY exception, so when the channel then materialized the
next watcher pass matched it by capacity with first_fund=False and the
committed on-chain cost vanished from the unified budget rail (repo rule from
51491da: never record a definite failure after a broadcast-capable RPC dies
unresolved).

Contract under test:
  * unknown-outcome exception -> reservation HELD (stays active), marker
    persisted on the row, no deadline-miss breaker trip;
  * next pass, channel materialized -> the capacity-match path settles the
    held reservation (spend event recorded);
  * next pass, successful probe with no channel -> held reservation released,
    then the ordinary retry flow proceeds;
  * next pass, probe failed -> nothing is resolved: reservation stays held
    and no new fundchannel is attempted;
  * definite rejection (CLN answered with an error) -> release, as before.
"""

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402
from modules.config import Config  # noqa: E402
from modules.lnplus_swaps import SwapLifecycle, _DEFAULT_OPEN_COST_SATS  # noqa: E402

PEER = "02" + "aa" * 32


def _make_db():
    path = os.path.join(tempfile.mkdtemp(prefix="lnplus_unknown_"), "test.db")
    db = Database(path, MagicMock())
    db.initialize()
    return db


def _make_lifecycle(capacity_sats=2_000_000, deadline_offset=40 * 3600):
    plugin, rpc = MagicMock(), MagicMock()
    db = _make_db()
    client = MagicMock()
    policy = MagicMock()
    policy.get_policy.return_value.has_tag.return_value = False
    lc = SwapLifecycle(plugin, rpc, db, Config(), client, policy,
                       ignore_peer_fn=MagicMock())
    db.lnplus_record_swap("s1", "applied", capacity_sats, 3,
                          outbound_peer=PEER)
    db.lnplus_update_swap("s1", deadline_at=int(time.time()) + deadline_offset)
    return lc, db, rpc, client


def _reservations(db):
    conn = db._get_connection()
    cur = conn.execute(
        "SELECT reservation_id, reserved_sats, status FROM spend_reservations")
    return [dict(zip([d[0] for d in cur.description], r))
            for r in cur.fetchall()]


def _spend_events_total(db):
    conn = db._get_connection()
    return int(conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events "
        "WHERE category='channel_open'").fetchone()[0])


def _run_open(lc, db):
    row = db.lnplus_get_swap("s1")
    return lc._execute_swap_open(row)


class TestUnknownOutcomeHold:
    def _timeout_attempt(self, lc, db, rpc):
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.side_effect = TimeoutError("RPC call timed out")
        return _run_open(lc, db)

    def test_unknown_outcome_holds_reservation(self):
        lc, db, rpc, _ = _make_lifecycle()
        assert self._timeout_attempt(lc, db, rpc) is False
        res = _reservations(db)
        assert len(res) == 1
        assert res[0]["status"] == "active", (
            "an unknown-outcome fundchannel must HOLD its budget reservation")
        assert res[0]["reserved_sats"] == _DEFAULT_OPEN_COST_SATS

    def test_unknown_outcome_persists_marker_on_row(self):
        lc, db, rpc, _ = _make_lifecycle()
        self._timeout_attempt(lc, db, rpc)
        row = db.lnplus_get_swap("s1")
        assert "held_reservation=" in str(row.get("outcome") or ""), (
            "the held reservation must be recoverable from the row")

    def test_unknown_outcome_does_not_trip_deadline_breaker(self):
        # Past-deadline + unknown outcome: the channel may have funded, so
        # the deadline-miss breaker must not fire on this pass.
        lc, db, rpc, _ = _make_lifecycle(deadline_offset=-3600)
        self._timeout_attempt(lc, db, rpc)
        assert lc.breaker_tripped() is None

    def test_definite_rejection_still_releases(self):
        lc, db, rpc, _ = _make_lifecycle()
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.side_effect = Exception(
            "Unknown peer, and no address known")
        assert _run_open(lc, db) is False
        res = _reservations(db)
        assert len(res) == 1
        assert res[0]["status"] == "released"


class TestNextPassResolution:
    def _hold(self, lc, db, rpc):
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.side_effect = TimeoutError("RPC call timed out")
        assert _run_open(lc, db) is False
        rpc.fundchannel.reset_mock(side_effect=True)
        held = [r for r in _reservations(db) if r["status"] == "active"]
        assert len(held) == 1
        return held[0]

    def test_channel_materialized_settles_held_reservation(self):
        lc, db, rpc, client = _make_lifecycle()
        held = self._hold(lc, db, rpc)
        # Next pass: the funding tx made it — capacity-matched channel exists.
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": PEER, "state": "CHANNELD_AWAITING_LOCKIN",
             "total_msat": 2_000_000_000, "funding_txid": "ff" * 32}]}
        assert _run_open(lc, db) is True
        rpc.fundchannel.assert_not_called()
        rows = {r["reservation_id"]: r for r in _reservations(db)}
        assert rows[held["reservation_id"]]["status"] == "spent", (
            "the committed on-chain cost must be settled onto the budget rail")
        assert _spend_events_total(db) == _DEFAULT_OPEN_COST_SATS
        assert db.lnplus_get_swap("s1")["status"] == "opened"

    def test_never_funded_releases_then_retries(self):
        lc, db, rpc, _ = _make_lifecycle()
        held = self._hold(lc, db, rpc)
        # Next pass: a successful listing shows no channel — the earlier
        # attempt definitively never funded. Release, then retry normally.
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ee" * 32}
        assert _run_open(lc, db) is True
        rpc.fundchannel.assert_called_once()
        rows = {r["reservation_id"]: r for r in _reservations(db)}
        assert rows[held["reservation_id"]]["status"] == "released"
        # The fresh attempt settled its own (new) reservation.
        spent = [r for r in rows.values() if r["status"] == "spent"]
        assert len(spent) == 1
        assert spent[0]["reservation_id"] != held["reservation_id"]

    def test_failed_probe_keeps_hold_and_skips_attempt(self):
        lc, db, rpc, _ = _make_lifecycle()
        held = self._hold(lc, db, rpc)
        # Next pass: listpeerchannels itself fails — nothing is resolved.
        rpc.listpeerchannels.side_effect = ConnectionError("socket closed")
        assert _run_open(lc, db) is False
        rpc.fundchannel.assert_not_called()
        rows = {r["reservation_id"]: r for r in _reservations(db)}
        assert rows[held["reservation_id"]]["status"] == "active", (
            "a failed probe resolves nothing — the reservation stays held")
