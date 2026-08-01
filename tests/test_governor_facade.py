"""Phase 1: governor facade — delegates to existing budget checks, adds
no new authority. Includes the oversubscription concurrency proof against
the real Database.reserve_spend."""
import os
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.econ_intents import Explanation, make_intent
from modules.econ_ledger import EconLedger
from modules.econ_types import Micro, Msat, SignedMsat, UnixTime
from modules.governor_facade import GovernorFacade

NOW = 1_752_400_000


def _intent(target="123x456x0", max_cost_msat=3_000, bucket="rebalance"):
    return make_intent(
        intent_type="REBALANCE",
        snapshot_id="snap-000001",
        created_at=UnixTime(NOW),
        expires_at=UnixTime(NOW + 600),
        target=target,
        amount_msat=Msat(500_000_000),
        expected_benefit_msat=SignedMsat(10_000),
        max_cost_msat=Msat(max_cost_msat),
        capital_committed_msat=Msat(500_000_000),
        confidence_micro=Micro(800_000),
        reason_codes=(),
        explanation=Explanation("rebalance", (("pair", target),)),
        preconditions=(),
        priority=50,
        budget_bucket=bucket,
        origin_policy="rebalance_policy",
        reversible=False,
    )


def _facade(reserve=None, release=None, paused=False, ledger=None):
    return GovernorFacade(
        reserve_spend=reserve or MagicMock(return_value=True),
        release_spend=release or MagicMock(return_value=True),
        is_paused=lambda: paused,
        ledger=ledger,
    )


def test_paused_rejects_without_reserving():
    reserve = MagicMock(return_value=True)
    decision = _facade(reserve=reserve, paused=True).authorize(_intent(), NOW)
    assert not decision.authorized
    assert decision.reason_code == "PAUSED"
    assert decision.token is None
    reserve.assert_not_called()


def test_stale_intent_rejected():
    reserve = MagicMock(return_value=True)
    decision = _facade(reserve=reserve).authorize(_intent(), NOW + 600)
    assert not decision.authorized
    assert decision.reason_code == "INTENT_STALE"
    reserve.assert_not_called()


def test_budget_exhausted_when_delegate_refuses():
    decision = _facade(reserve=MagicMock(return_value=False)).authorize(
        _intent(), NOW)
    assert not decision.authorized
    assert decision.reason_code == "BUDGET_EXHAUSTED"


def test_success_reserves_worst_case_with_ceil_conversion():
    reserve = MagicMock(return_value=True)
    decision = _facade(reserve=reserve).authorize(
        _intent(max_cost_msat=1_500), NOW)
    assert decision.authorized and decision.token is not None
    kwargs = reserve.call_args.kwargs
    assert kwargs["amount_sats"] == 2  # 1500 msat -> ceil 2 sats
    assert kwargs["category"] == "rebalance"
    assert decision.token.reserved_msat == 2_000


def test_zero_cost_intent_authorized_without_reservation():
    reserve = MagicMock(return_value=True)
    decision = _facade(reserve=reserve).authorize(
        _intent(max_cost_msat=0), NOW)
    assert decision.authorized
    assert decision.token.reserved_msat == 0
    reserve.assert_not_called()


def test_ledger_records_authorization(tmp_path):
    ledger = EconLedger(str(tmp_path / "ledger.db"))
    facade = _facade(ledger=ledger)
    decision = facade.authorize(_intent(), NOW)
    assert decision.authorized
    facade.release(decision.token, NOW + 10)
    types = [e["event_type"] for e in ledger.events()]
    assert types == ["intent_authorized", "budget_reserved",
                     "reservation_released"]
    state = ledger.replay()
    assert state.reserved_msat == {}  # released


class _FailingLedger:
    """Ledger stub whose append always raises (lock timeout / disk full).

    Only `append` is exercised by the facade paths under test."""

    def __init__(self):
        self.attempts = 0

    def append(self, **kwargs):
        self.attempts += 1
        import sqlite3
        raise sqlite3.OperationalError("database is locked")


def test_ledger_append_failure_does_not_block_authorization():
    """A ledger journaling failure AFTER the reservation committed must
    neither refuse the intent nor strand the reservation: the
    spend_reservations DB is the authority, the ledger a shadow journal
    (the hourly reconciler heals ledger-missing-reservation divergence)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        db = Database(path, MagicMock())
        db.initialize()
        ledger = _FailingLedger()

        def reserve(**kwargs):
            return db.reserve_spend(
                reservation_id=kwargs["reservation_id"],
                amount_sats=kwargs["amount_sats"],
                category=kwargs["category"],
                effective_budget_sats=100,
            )

        facade = GovernorFacade(
            reserve_spend=reserve,
            release_spend=lambda reservation_id: db.release_spend_reservation(
                reservation_id),
            is_paused=lambda: False,
        )
        facade._ledger = ledger

        decision = facade.authorize(_intent(), NOW)
        assert decision.authorized
        assert decision.token is not None
        assert decision.reason_code == ""
        assert ledger.attempts >= 1  # append was attempted, and raised

        # The reservation is live and owned by the caller via the token —
        # release must succeed even though its ledger append also raises.
        assert facade.release(decision.token, NOW + 10)
        row = db._get_connection().execute(
            "SELECT status FROM spend_reservations WHERE reservation_id = ?",
            (decision.token.reservation_id,)).fetchone()
        assert row is not None and row["status"] != "active"
    finally:
        os.unlink(path)


def test_ledger_append_failure_does_not_block_release():
    release = MagicMock(return_value=True)
    facade = _facade(release=release, ledger=None)
    decision = facade.authorize(_intent(), NOW)
    facade._ledger = _FailingLedger()
    assert facade.release(decision.token, NOW + 10)
    release.assert_called_once_with(decision.token.reservation_id)


def test_release_delegates():
    release = MagicMock(return_value=True)
    facade = _facade(release=release)
    decision = facade.authorize(_intent(), NOW)
    facade.release(decision.token, NOW + 10)
    release.assert_called_once_with(decision.token.reservation_id)


def test_concurrent_authorizations_cannot_oversubscribe():
    """8 threads race for a 10-sat budget with 3-sat worst-case costs:
    at most 3 may win. Delegation inherits Database.reserve_spend's
    BEGIN IMMEDIATE serialization — the facade adds no new authority and
    loses none."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        db = Database(path, MagicMock())
        db.initialize()  # schema lives in initialize(), not __init__
        cap_sats = 10

        def reserve(**kwargs):
            return db.reserve_spend(
                reservation_id=kwargs["reservation_id"],
                amount_sats=kwargs["amount_sats"],
                category=kwargs["category"],
                effective_budget_sats=cap_sats,
            )

        facade = GovernorFacade(
            reserve_spend=reserve,
            release_spend=lambda reservation_id: db.release_spend_reservation(
                reservation_id),
            is_paused=lambda: False,
        )

        granted = []
        lock = threading.Lock()

        def worker(i):
            env = _intent(target=f"{100 + i}x1x0")
            decision = facade.authorize(env, NOW)
            if decision.authorized:
                with lock:
                    granted.append(decision.token)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_reserved_sats = sum(t.reserved_msat for t in granted) // 1000
        assert total_reserved_sats <= cap_sats
        assert len(granted) == 3  # 3 x 3 sats = 9 <= 10; a 4th would be 12
    finally:
        os.unlink(path)
