"""Closure-resolution sweep: post-close HTLC-sweep fees are accumulated.

record_channel_closure fires once at close detection, but unilateral closes
keep paying sweep fees for hours or days. _reconcile_closure_resolutions
re-queries the bookkeeper for every unresolved closure row, adds new fees
via update_closure_resolution, and marks rows complete when the bookkeeper
account balance reaches zero (or after the 90-day timeout).
"""

import os
import sys
import time

from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database
from tests.plugin_test_utils import load_plugin_module


PEER = "a" * 66
CHAN = "111x222x0"
ACCOUNT = "f" * 64


def _make_db(tmp_path):
    db_path = os.path.join(tmp_path, "closure.db")
    db = Database(db_path, MagicMock())
    db.initialize()
    return db


def _row(db, chan=CHAN):
    return db._get_connection().execute(
        "SELECT * FROM channel_closure_costs WHERE channel_id = ?", (chan,)
    ).fetchone()


def _make_mod(tmp_path, *, bkpr_totals, balances_msat, closed_channels=None):
    """Load the plugin module with a real Database and mocked bkpr RPCs."""
    mod = load_plugin_module()
    db = _make_db(tmp_path)
    mod.database = db
    mod.plugin.log = MagicMock()
    mod.data_service = None

    rpc = MagicMock()

    def rpc_call(method, params=None):
        if method == "bkpr-listbalances":
            return {
                "accounts": [
                    {"account": acct, "balances": [{"balance_msat": bal}]}
                    for acct, bal in balances_msat.items()
                ]
            }
        if method == "bkpr-inspect":
            account = (params or {}).get("account")
            totals = bkpr_totals.get(account)
            if totals is None:
                return {"txs": []}
            closure, sweep = totals
            txs = []
            if closure:
                txs.append({
                    "txid": "close0",
                    "fees_paid_msat": closure * 1000,
                    "outputs": [{"output_tag": "channel_close"}],
                })
            if sweep:
                txs.append({
                    "txid": "sweep0",
                    "fees_paid_msat": sweep * 1000,
                    "outputs": [{"output_tag": "htlc_timeout"}],
                })
            return {"txs": txs}
        if method == "listclosedchannels":
            return {"closedchannels": closed_channels or []}
        raise AssertionError(f"unexpected rpc {method}")

    rpc.call.side_effect = rpc_call
    from types import SimpleNamespace
    mod.safe_plugin = SimpleNamespace(rpc=rpc)
    return mod, db


def test_sweep_accumulates_new_fees_and_completes_on_zero_balance(tmp_path):
    mod, db = _make_mod(
        tmp_path,
        # bookkeeper now reports 1000 close + 700 sweep for the account
        bkpr_totals={ACCOUNT: (1000, 700)},
        balances_msat={ACCOUNT: 0},
    )
    # Recorded at close time with only the close fee known
    db.record_channel_closure(
        channel_id=CHAN, peer_id=PEER, close_type="local_unilateral",
        closure_fee_sats=1000, bkpr_account=ACCOUNT,
    )

    summary = mod._reconcile_closure_resolutions()

    assert summary["checked"] == 1
    assert summary["updated"] == 1
    assert summary["added_fee_sats"] == 700
    assert summary["completed"] == 1
    row = _row(db)
    assert row["htlc_sweep_fee_sats"] == 700
    assert row["total_closure_cost_sats"] == 1700
    assert row["resolution_complete"] == 1


def test_sweep_leaves_row_open_while_balance_remains(tmp_path):
    mod, db = _make_mod(
        tmp_path,
        bkpr_totals={ACCOUNT: (1000, 0)},
        balances_msat={ACCOUNT: 250_000_000},  # outputs still unswept
    )
    db.record_channel_closure(
        channel_id=CHAN, peer_id=PEER, close_type="local_unilateral",
        closure_fee_sats=1000, bkpr_account=ACCOUNT,
    )

    summary = mod._reconcile_closure_resolutions()

    assert summary["completed"] == 0
    row = _row(db)
    assert row["resolution_complete"] == 0
    # No new fees -> no update
    assert summary["updated"] == 0
    assert row["total_closure_cost_sats"] == 1000


def test_sweep_never_subtracts_when_bkpr_reports_less(tmp_path):
    mod, db = _make_mod(
        tmp_path,
        bkpr_totals={ACCOUNT: (400, 0)},  # less than stored (bkpr hiccup)
        balances_msat={ACCOUNT: 1},
    )
    db.record_channel_closure(
        channel_id=CHAN, peer_id=PEER, close_type="mutual",
        closure_fee_sats=1000, bkpr_account=ACCOUNT,
    )

    summary = mod._reconcile_closure_resolutions()

    assert summary["updated"] == 0
    assert _row(db)["total_closure_cost_sats"] == 1000


def test_legacy_row_resolves_account_via_listclosedchannels(tmp_path):
    mod, db = _make_mod(
        tmp_path,
        bkpr_totals={ACCOUNT: (1000, 300)},
        balances_msat={ACCOUNT: 0},
        closed_channels=[
            {"short_channel_id": CHAN, "channel_id": ACCOUNT},
        ],
    )
    # Pre-upgrade row: no bkpr_account stored
    db.record_channel_closure(
        channel_id=CHAN, peer_id=PEER, close_type="remote_unilateral",
        closure_fee_sats=1000,
    )
    assert _row(db)["bkpr_account"] is None

    summary = mod._reconcile_closure_resolutions()

    assert summary["updated"] == 1
    assert summary["added_fee_sats"] == 300
    row = _row(db)
    assert row["bkpr_account"] == ACCOUNT  # backfilled for the next pass
    assert row["resolution_complete"] == 1


def test_unresolvable_row_ages_out_after_timeout(tmp_path):
    mod, db = _make_mod(
        tmp_path,
        bkpr_totals={},
        balances_msat={},
        closed_channels=[],  # account unresolvable
    )
    db.record_channel_closure(
        channel_id=CHAN, peer_id=PEER, close_type="unknown",
        closure_fee_sats=500,
    )
    # Age the row past the timeout
    db._get_connection().execute(
        "UPDATE channel_closure_costs SET closed_at = ? WHERE channel_id = ?",
        (int(time.time()) - 91 * 86400, CHAN),
    )

    summary = mod._reconcile_closure_resolutions()

    assert summary["completed"] == 1
    assert _row(db)["resolution_complete"] == 1


def test_resolved_rows_are_not_revisited(tmp_path):
    mod, db = _make_mod(
        tmp_path,
        bkpr_totals={ACCOUNT: (1000, 700)},
        balances_msat={ACCOUNT: 0},
    )
    db.record_channel_closure(
        channel_id=CHAN, peer_id=PEER, close_type="local_unilateral",
        closure_fee_sats=1000, bkpr_account=ACCOUNT,
    )
    first = mod._reconcile_closure_resolutions()
    assert first["completed"] == 1

    second = mod._reconcile_closure_resolutions()
    assert second["checked"] == 0
