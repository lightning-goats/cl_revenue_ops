"""Phase 2J: reservations unification (4→1, stage 1).

reserve_budget is now a compatibility wrapper over the generic spend
ledger. These tests prove (ok, remaining) parity against the retained
legacy implementation, weekly-cap preservation, mixed-path concurrency
safety, dual-path transition handling, and journal coverage."""
import os
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.database import Database, _reserve_budget_atomic

NOW = 1_752_400_000
SINCE = NOW - 24 * 3600
WEEK_SINCE = NOW - 7 * 86400


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path, MagicMock())
    database.initialize()
    yield database
    os.unlink(path)


def _legacy_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path, MagicMock())
    database.initialize()
    return database, path


def _legacy_reserve(database, **kw):
    return _reserve_budget_atomic(
        database._get_connection(),
        kw["reservation_id"], kw["amount_sats"], kw.get("channel_id", "c"),
        kw["budget_limit"], kw.get("since_timestamp", SINCE),
        kw.get("weekly_budget_limit"), kw.get("weekly_since_timestamp"),
    )


class TestParity:
    """Same scenario on mirrored DBs: legacy atomic vs unified wrapper
    must return identical (ok, remaining)."""

    def _both(self, scenario_setup, **reserve_kw):
        legacy, lp = _legacy_db()
        unified, up = _legacy_db()
        try:
            scenario_setup(legacy)
            scenario_setup(unified)
            legacy_result = _legacy_reserve(legacy, **reserve_kw)
            unified_result = unified.reserve_budget(
                reservation_id=reserve_kw["reservation_id"],
                amount_sats=reserve_kw["amount_sats"],
                channel_id=reserve_kw.get("channel_id", "c"),
                budget_limit=reserve_kw["budget_limit"],
                since_timestamp=reserve_kw.get("since_timestamp", SINCE),
                weekly_budget_limit=reserve_kw.get("weekly_budget_limit"),
                weekly_since_timestamp=reserve_kw.get(
                    "weekly_since_timestamp"),
            )
            assert unified_result == legacy_result
            return unified_result
        finally:
            os.unlink(lp)
            os.unlink(up)

    def test_fresh_reserve(self):
        ok, remaining = self._both(
            lambda d: None, reservation_id="r1", amount_sats=100,
            budget_limit=1000)
        assert ok is True and remaining == 900

    def test_exact_fit(self):
        ok, remaining = self._both(
            lambda d: None, reservation_id="r1", amount_sats=1000,
            budget_limit=1000)
        assert ok is True and remaining == 0

    def test_over_limit_refusal_reports_remaining(self):
        def setup(d):
            d.reserve_spend(reservation_id="g1", amount_sats=300,
                            category="planner")
        ok, remaining = self._both(
            setup, reservation_id="r1", amount_sats=800,
            budget_limit=1000)
        assert ok is False and remaining == 700

    def test_mixed_state_counts_everything(self):
        def setup(d):
            d.reserve_spend(reservation_id="g1", amount_sats=200,
                            category="boltz")
            d.record_spend_event(event_id="e1", category="channel_open",
                                 amount_sats=100)
        ok, remaining = self._both(
            setup, reservation_id="r1", amount_sats=100,
            budget_limit=1000)
        assert ok is True and remaining == 600  # 1000-200-100-100

    def test_weekly_binding_refusal(self):
        def setup(d):
            d.record_spend_event(event_id="e1", category="rebalance",
                                 amount_sats=400)
        ok, remaining = self._both(
            setup, reservation_id="r1", amount_sats=200,
            budget_limit=1000, weekly_budget_limit=500,
            weekly_since_timestamp=WEEK_SINCE)
        assert ok is False and remaining == 100  # weekly 500-400

    def test_weekly_limited_success_remaining(self):
        ok, remaining = self._both(
            lambda d: None, reservation_id="r1", amount_sats=100,
            budget_limit=1000, weekly_budget_limit=300,
            weekly_since_timestamp=WEEK_SINCE)
        assert ok is True and remaining == 200  # min(900, 200)


class TestUnifiedBehavior:
    def test_reservation_lands_in_generic_ledger(self, db):
        ok, _ = db.reserve_budget(
            reservation_id="reb-1", amount_sats=50, channel_id="111x222x0",
            budget_limit=1000, since_timestamp=SINCE)
        assert ok
        states = db.get_spend_reservation_states(["reb-1"])
        assert states["reb-1"]["status"] == "active"

    def test_release_and_settle_dual_path_unified_rows(self, db):
        db.reserve_budget(reservation_id="reb-1", amount_sats=50,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        assert db.mark_budget_spent("reb-1", actual_spent=30)
        db.reserve_budget(reservation_id="reb-2", amount_sats=50,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        assert db.release_budget_reservation("reb-2")
        states = db.get_spend_reservation_states(["reb-1", "reb-2"])
        assert states["reb-1"]["status"] == "spent"
        assert states["reb-2"]["status"] == "released"

    def test_transition_legacy_rows_still_serviced(self, db):
        # A pre-unification in-flight reservation (legacy table).
        _legacy_reserve(db, reservation_id="old-1", amount_sats=40,
                        budget_limit=1000)
        _legacy_reserve(db, reservation_id="old-2", amount_sats=40,
                        budget_limit=1000)
        assert db.release_budget_reservation("old-1")
        assert db.mark_budget_spent("old-2", actual_spent=25)
        # And they can no longer be released twice.
        assert not db.release_budget_reservation("old-1")

    def test_settle_records_no_spend_event(self, db):
        """Actual rebalance costs live in rebalance_costs — the unified
        settle must not add a spend_event (double count)."""
        db.reserve_budget(reservation_id="reb-1", amount_sats=50,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        db.mark_budget_spent("reb-1", actual_spent=30)
        summary = db.get_spend_ledger_summary(window_hours=24)
        assert summary["spent_24h_sats"] == 0

    def test_journal_covers_rebalance_reservations(self, db):
        journal = MagicMock()
        db.spend_journal = journal
        db.reserve_budget(reservation_id="reb-1", amount_sats=50,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        journal.note_spend_reserved.assert_called_once_with(
            "reb-1", 50, "rebalance")

    def test_get_budget_status_sees_unified_holds(self, db):
        db.reserve_budget(reservation_id="reb-1", amount_sats=70,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        status = db.get_budget_status(SINCE)
        assert status["reserved"] == 70

    def test_daily_rebalance_spend_sees_unified_holds(self, db):
        db.reserve_budget(reservation_id="reb-1", amount_sats=60,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        spend = db.get_daily_rebalance_spend(window_hours=24)
        assert spend["total_reserved_sats"] == 60

    def test_stale_cleanup_covers_unified_rows(self, db):
        db.reserve_budget(reservation_id="reb-old", amount_sats=30,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        # Age the row past the 4h default.
        conn = db._get_connection()
        conn.execute(
            "UPDATE spend_reservations SET reserved_at = ? "
            "WHERE reservation_id = 'reb-old'",
            (NOW - 20_000,))
        conn.commit()
        cleaned = db.cleanup_stale_reservations(max_age_seconds=14_400)
        assert cleaned == 1
        states = db.get_spend_reservation_states(["reb-old"])
        assert states["reb-old"]["status"] == "released"

    def test_stale_cleanup_skips_pending_settlement(self, db):
        """P4-015 in-flight HTLC guard must protect unified rows too."""
        rebalance_id = db.record_rebalance(
            from_channel="100x1x0", to_channel="200x1x0",
            amount_sats=1000, max_fee_sats=3, expected_profit_sats=1,
            status="pending_settlement")
        db.reserve_budget(reservation_id=str(rebalance_id), amount_sats=3,
                          channel_id="200x1x0", budget_limit=1000,
                          since_timestamp=SINCE)
        conn = db._get_connection()
        conn.execute(
            "UPDATE spend_reservations SET reserved_at = ? "
            "WHERE reservation_id = ?", (NOW - 20_000, str(rebalance_id)))
        conn.commit()
        assert db.cleanup_stale_reservations(max_age_seconds=14_400) == 0
        states = db.get_spend_reservation_states([str(rebalance_id)])
        assert states[str(rebalance_id)]["status"] == "active"

    def test_unified_breakdown_counts_each_hold_once(self, db):
        """_total_cost_budget_status: unified rebalance holds appear in
        the rebalance bucket and are excluded from the ledger bucket."""
        from tests.plugin_test_utils import load_plugin_module
        mod = load_plugin_module()
        mod.database = db
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            daily_budget_sats=1000, weekly_budget_sats=35_000,
            total_cost_budget_window_hours=24,
            growth_budget_enabled=False, paused=False)
        mod.config = cfg
        db.reserve_budget(reservation_id="reb-1", amount_sats=70,
                          channel_id="c", budget_limit=1000,
                          since_timestamp=SINCE)
        db.reserve_spend(reservation_id="gen-1", amount_sats=20,
                         category="channel_open")
        status = mod._total_cost_budget_status(force_fresh=True)
        by_cat = status["reserved_by_category"]
        assert by_cat["rebalance"] == 70
        assert by_cat["ledger"] == 20  # excludes the rebalance hold
        assert sum(by_cat.values()) == 90  # each hold exactly once

    def test_mixed_path_concurrency_cannot_oversubscribe(self, db):
        cap = 10
        granted = []
        lock = threading.Lock()

        def worker(i):
            if i % 2 == 0:
                ok = db.reserve_spend(
                    reservation_id=f"g-{i}", amount_sats=3,
                    category="planner", effective_budget_sats=cap)
                amount = 3 if ok else 0
            else:
                ok, _ = db.reserve_budget(
                    reservation_id=f"r-{i}", amount_sats=3,
                    channel_id="c", budget_limit=cap,
                    since_timestamp=SINCE)
                amount = 3 if ok else 0
            if amount:
                with lock:
                    granted.append(amount)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(granted) <= cap
        assert len(granted) == 3  # 3x3=9 <= 10; a 4th would be 12
