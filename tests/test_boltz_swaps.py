import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.boltz_swaps import BoltzSwapManager


class _TestDB:
    def __init__(self):
        # Match production DB behavior (autocommit mode) to avoid implicit
        # long-lived transactions affecting reservation semantics.
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.conn.row_factory = sqlite3.Row

    def _get_connection(self):
        return self.conn


def _make_manager(config=None):
    db = _TestDB()
    plugin = MagicMock()
    plugin.rpc = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "1" * 64}
    if config is None:
        config = {
            "swap_daily_budget_sats": 50000,
            "swap_max_fee_ppm": 5000,
            "swap_min_amount_sats": 100000,
            "swap_max_amount_sats": 10000000,
            "swap_currency": "btc",
        }
    manager = BoltzSwapManager(db, plugin, config=config)
    return manager, plugin


def _mock_boltzcli(manager, responses):
    """
    Mock _run_boltzcli to return sequential responses.
    responses: list of dicts to return in order.
    """
    call_count = {"n": 0}
    orig = responses

    def side_effect(*args, **kwargs):
        idx = min(call_count["n"], len(orig) - 1)
        call_count["n"] += 1
        return orig[idx]

    manager._run_boltzcli = MagicMock(side_effect=side_effect)


# -----------------------------------------------------------------------
# Quote tests
# -----------------------------------------------------------------------

def test_quote_reverse():
    manager, _ = _make_manager()
    quote_resp = {
        "serviceFee": 250,
        "networkFee": 150,
    }
    manager._run_boltzcli = MagicMock(return_value=quote_resp)

    result = manager.quote(500000, swap_type="reverse")

    assert result["swap_type"] == "reverse"
    assert result["amount_sats"] == 500000
    assert result["boltz_fee_sats"] == 250
    assert result["network_fee_sats"] == 150
    assert result["total_fee_sats"] == 400
    assert result["received_sats"] == 499600
    assert result["fee_ppm"] == 800  # 400/500000 * 1M
    assert "budget" in result


def test_quote_submarine():
    manager, _ = _make_manager()
    quote_resp = {
        "serviceFee": 300,
        "networkFee": 200,
    }
    manager._run_boltzcli = MagicMock(return_value=quote_resp)

    result = manager.quote(500000, swap_type="submarine")

    assert result["swap_type"] == "submarine"
    assert result["total_fee_sats"] == 500
    assert result["fee_ppm"] == 1000


# -----------------------------------------------------------------------
# Loop-out (reverse swap) tests
# -----------------------------------------------------------------------

def test_loop_out_creates_swap_and_records():
    manager, _ = _make_manager()

    # Mock quote then createreverseswap
    call_results = [
        # quote response
        {"serviceFee": 200, "networkFee": 100},
        # createreverseswap response
        {"id": "rev-swap-1", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    result = manager.loop_out(500000)

    assert result["status"] == "created"
    assert result["swap_id"] == "rev-swap-1"
    assert result["total_fee_sats"] == 300
    assert result["fee_ppm"] == 600

    # Check DB record
    row = manager.db._get_connection().execute(
        "SELECT * FROM boltz_swaps WHERE id = ?", ("rev-swap-1",)
    ).fetchone()
    assert row is not None
    assert row["swap_type"] == "reverse"
    assert row["amount_sats"] == 500000
    assert row["total_fee_sats"] == 300
    assert row["status"] == "created"


def test_loop_out_with_address():
    manager, _ = _make_manager()
    call_results = [
        {"serviceFee": 100, "networkFee": 50},
        {"id": "rev-swap-addr", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    result = manager.loop_out(200000, address="bc1qtest")

    assert result["swap_id"] == "rev-swap-addr"
    # Verify the address was passed to boltzcli
    call_args = manager._run_boltzcli.call_args_list[1]
    assert "bc1qtest" in call_args[0]


def test_loop_out_budget_block_daily():
    manager, _ = _make_manager(config={
        "swap_daily_budget_sats": 100,  # Very low budget
        "swap_max_fee_ppm": 5000,
        "swap_min_amount_sats": 100000,
        "swap_max_amount_sats": 10000000,
        "swap_currency": "btc",
    })
    # First complete a swap to consume budget
    conn = manager.db._get_connection()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES (?, ?, ?, 'reverse', 500000, 499500, 300, 200, 500, 1000, 'completed')
    """, ("prior-swap", manager._now_ts(), manager._now_ts()))

    # Quote for new swap
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 200, "networkFee": 100})

    result = manager.loop_out(500000)

    assert "error" in result
    assert "budget" in result["error"].lower() or "budget" in str(result.get("budget", ""))


def test_loop_out_budget_block_fee_ppm():
    manager, _ = _make_manager(config={
        "swap_daily_budget_sats": 50000,
        "swap_max_fee_ppm": 100,  # Very low max fee rate
        "swap_min_amount_sats": 100000,
        "swap_max_amount_sats": 10000000,
        "swap_currency": "btc",
    })
    # Quote returns fee that exceeds max ppm
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 5000, "networkFee": 1000})

    result = manager.loop_out(500000)

    assert "error" in result
    assert "ppm" in result["error"].lower()


def test_loop_out_amount_below_min():
    manager, _ = _make_manager()
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 10, "networkFee": 5})

    result = manager.loop_out(1000)  # Below default min of 100000

    assert "error" in result
    assert "minimum" in result["error"].lower() or "below" in result["error"].lower()


# -----------------------------------------------------------------------
# Loop-in (submarine swap) tests
# -----------------------------------------------------------------------

def test_loop_in_creates_swap():
    manager, _ = _make_manager()
    call_results = [
        {"serviceFee": 150, "networkFee": 100},
        {"id": "sub-swap-1", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    result = manager.loop_in(500000)

    assert result["status"] == "created"
    assert result["swap_id"] == "sub-swap-1"

    row = manager.db._get_connection().execute(
        "SELECT * FROM boltz_swaps WHERE id = ?", ("sub-swap-1",)
    ).fetchone()
    assert row is not None
    assert row["swap_type"] == "submarine"


def test_loop_in_rejects_channel_and_peer_together():
    manager, _ = _make_manager()
    result = manager.loop_in(100000, channel_id="1x1x1", peer_id="02" + "a" * 64)
    assert "error" in result


def test_loop_in_with_channel_hint():
    manager, _ = _make_manager()
    call_results = [
        {"serviceFee": 100, "networkFee": 50},
        {"id": "sub-swap-ch", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    result = manager.loop_in(500000, channel_id="123x1x0")

    assert result["swap_id"] == "sub-swap-ch"

    row = manager.db._get_connection().execute(
        "SELECT target_channel_id FROM boltz_swaps WHERE id = ?", ("sub-swap-ch",)
    ).fetchone()
    assert row["target_channel_id"] == "123x1x0"


# -----------------------------------------------------------------------
# Status / history tests
# -----------------------------------------------------------------------

def test_status_returns_local_and_remote():
    manager, _ = _make_manager()

    # Insert a local record
    conn = manager.db._get_connection()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('swap-status-1', ?, ?, 'reverse', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (manager._now_ts(), manager._now_ts()))

    # swapinfo doesn't support --json, so we mock _run_boltzcli_raw (fix #5)
    manager._run_boltzcli_raw = MagicMock(return_value="Id: swap-status-1\nStatus: invoice.settled")

    result = manager.status("swap-status-1")

    assert result["local"] is not None
    assert result["local"]["status"] == "completed"  # invoice.settled -> completed
    assert result["boltzd"]["status"] == "invoice.settled"


def test_history_returns_swaps_and_totals():
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    for i in range(3):
        conn.execute("""
            INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
                amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
                total_fee_sats, fee_ppm, status)
            VALUES (?, ?, ?, 'reverse', 100000, 99500, 300, 200, 500, 5000, 'completed')
        """, (f"hist-{i}", now - i * 60, now - i * 60))

    # Mock stats call
    manager._run_boltzcli = MagicMock(return_value={"totalSwaps": 3})

    result = manager.history(limit=10)

    assert len(result["swaps"]) == 3
    assert result["totals"]["completed"] == 3
    assert result["totals"]["total_fees_sats"] == 1500
    assert "budget" in result


# -----------------------------------------------------------------------
# Budget tests
# -----------------------------------------------------------------------

def test_budget_status():
    manager, _ = _make_manager(config={
        "swap_daily_budget_sats": 50000,
        "swap_max_fee_ppm": 5000,
        "swap_min_amount_sats": 100000,
        "swap_max_amount_sats": 10000000,
        "swap_currency": "btc",
    })

    budget = manager.get_budget_status()

    assert budget["daily_budget_sats"] == 50000
    assert budget["daily_spent_sats"] == 0
    assert budget["daily_remaining_sats"] == 50000


def test_daily_budget_uses_completion_time_not_creation_time():
    manager, _ = _make_manager(config={
        "swap_daily_budget_sats": 50000,
        "swap_max_fee_ppm": 5000,
        "swap_min_amount_sats": 100000,
        "swap_max_amount_sats": 10000000,
        "swap_currency": "btc",
    })
    conn = manager.db._get_connection()
    now = manager._now_ts()
    # Created long ago, but completed recently -> should count in daily spend.
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('old-created-new-complete', ?, ?, 'reverse', 500000, 499500, 300, 200, 500, 1000, 'completed')
    """, (now - 172800, now - 10))

    budget = manager.get_budget_status()
    assert budget["daily_completed_sats"] == 500
    assert budget["daily_spent_sats"] == 500


def test_budget_reservation_blocks_second_swap_before_completion():
    manager, _ = _make_manager(config={
        "swap_daily_budget_sats": 500,
        "swap_max_fee_ppm": 5000,
        "swap_min_amount_sats": 100000,
        "swap_max_amount_sats": 10000000,
        "swap_currency": "btc",
    })

    call_count = {"n": 0}

    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"serviceFee": 200, "networkFee": 100}
        return {"id": f"swap-{call_count['n']}", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    first = manager.loop_out(500000)
    second = manager.loop_out(500000)

    assert first["status"] == "created"
    assert "error" in second
    assert "budget" in second["error"].lower()

    conn = manager.db._get_connection()
    row = conn.execute(
        "SELECT state FROM swap_budget_reservations WHERE swap_id = ?",
        (first["swap_id"],),
    ).fetchone()
    assert row is not None
    assert row["state"] == "active"


def test_completed_swap_finalizes_budget_reservation():
    manager, _ = _make_manager()
    _mock_boltzcli(manager, [
        {"serviceFee": 200, "networkFee": 100},
        {"id": "reserve-finalize-1", "status": "created"},
    ])
    created = manager.loop_out(500000)
    assert created["status"] == "created"

    manager._run_boltzcli_raw = MagicMock(
        return_value="Id: reserve-finalize-1\nStatus: invoice.settled"
    )
    manager.status("reserve-finalize-1")

    conn = manager.db._get_connection()
    row = conn.execute(
        "SELECT state FROM swap_budget_reservations WHERE swap_id = 'reserve-finalize-1'"
    ).fetchone()
    assert row is not None
    assert row["state"] == "finalized"

    budget = manager.get_budget_status()
    assert budget["daily_reserved_sats"] == 0


def test_failed_swap_releases_budget_reservation():
    manager, _ = _make_manager()
    _mock_boltzcli(manager, [
        {"serviceFee": 200, "networkFee": 100},
        {"id": "reserve-release-1", "status": "created"},
    ])
    created = manager.loop_out(500000)
    assert created["status"] == "created"

    manager._run_boltzcli_raw = MagicMock(
        return_value="Id: reserve-release-1\nStatus: invoice.expired"
    )
    manager.status("reserve-release-1")

    conn = manager.db._get_connection()
    row = conn.execute(
        "SELECT state FROM swap_budget_reservations WHERE swap_id = 'reserve-release-1'"
    ).fetchone()
    assert row is not None
    assert row["state"] == "released"


# -----------------------------------------------------------------------
# Swap monitoring tests
# -----------------------------------------------------------------------

def test_check_pending_swaps_completes():
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('pending-1', ?, ?, 'reverse', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    # invoice.settled is the actual terminal success for reverse swaps (fix #3)
    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [{"id": "pending-1", "status": "invoice.settled"}],
        "swaps": [],
    })

    result = manager.check_pending_swaps()

    assert result["checked"] == 1
    assert result["completed"] == 1

    # Verify status updated
    row = conn.execute(
        "SELECT status FROM boltz_swaps WHERE id = 'pending-1'"
    ).fetchone()
    assert row["status"] == "completed"

    # Verify cost recorded
    cost_row = conn.execute(
        "SELECT * FROM swap_costs WHERE swap_id = 'pending-1'"
    ).fetchone()
    assert cost_row is not None
    assert cost_row["cost_sats"] == 500


def test_check_pending_swaps_completes_transaction_claimed():
    """transaction.claimed is the terminal success for submarine swaps."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('pending-sub', ?, ?, 'submarine', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [],
        "swaps": [{"id": "pending-sub", "status": "transaction.claimed"}],
    })

    result = manager.check_pending_swaps()
    assert result["completed"] == 1

    row = conn.execute("SELECT status FROM boltz_swaps WHERE id = 'pending-sub'").fetchone()
    assert row["status"] == "completed"


def test_check_pending_swaps_fails():
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('pending-fail', ?, ?, 'submarine', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [],
        "swaps": [{"id": "pending-fail", "status": "swap.error"}],
    })

    result = manager.check_pending_swaps()

    assert result["failed"] == 1

    row = conn.execute(
        "SELECT status, error FROM boltz_swaps WHERE id = 'pending-fail'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "swap.error"


def test_check_pending_swaps_no_pending():
    manager, _ = _make_manager()
    result = manager.check_pending_swaps()
    assert result["checked"] == 0
    assert result["updated"] == 0


def test_check_pending_swaps_includes_chain_swaps():
    """Fix #7: chainSwaps from boltzd response should be checked."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('chain-swap-1', ?, ?, 'reverse', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [],
        "swaps": [],
        "chainSwaps": [{"id": "chain-swap-1", "status": "transaction.claimed"}],
    })

    result = manager.check_pending_swaps()
    assert result["completed"] == 1


def test_check_pending_swaps_auto_refund_submarine():
    """Fix #9: Failed submarine swap with locked funds triggers auto-refund."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('sub-refund', ?, ?, 'submarine', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    # listswaps uses _run_boltzcli (with --json)
    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [],
        "swaps": [{"id": "sub-refund", "status": "invoice.failedToPay"}],
    })
    # refundswap uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(return_value='{"id": "sub-refund", "status": "refunded"}')

    result = manager.check_pending_swaps()

    assert result["failed"] == 1
    assert result["refund_attempts"] == 1

    # Verify refundswap was called via _run_boltzcli_raw
    assert manager._run_boltzcli_raw.call_count == 1
    assert manager._run_boltzcli_raw.call_args[0] == ("refundswap", "sub-refund", "wallet")


def test_check_pending_swaps_no_refund_for_reverse():
    """Auto-refund should NOT trigger for reverse swaps (only submarine)."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('rev-expired', ?, ?, 'reverse', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [{"id": "rev-expired", "status": "swap.expired"}],
        "swaps": [],
    })

    result = manager.check_pending_swaps()

    assert result["failed"] == 1
    assert result["refund_attempts"] == 0  # No refund for reverse swaps


def test_check_pending_swaps_retries_refund_for_failed_submarine():
    """Failed refundable swaps remain monitored until refund succeeds."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status, error)
        VALUES ('sub-refund-retry', ?, ?, 'submarine', 500000, 0, 300, 200, 500, 1000, 'failed', 'invoice.failedToPay')
    """, (now, now))

    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [],
        "swaps": [{"id": "sub-refund-retry", "status": "invoice.failedToPay"}],
    })

    call_count = {"n": 0}

    def mock_raw(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("timelock not expired")
        return '{"status": "refunded"}'

    manager._run_boltzcli_raw = MagicMock(side_effect=mock_raw)

    first = manager.check_pending_swaps()
    second = manager.check_pending_swaps()

    assert first["refund_attempts"] == 1
    assert second["refund_attempts"] == 1
    assert second["updated"] == 1
    assert manager._run_boltzcli_raw.call_count == 2

    row = conn.execute(
        "SELECT status FROM boltz_swaps WHERE id = 'sub-refund-retry'"
    ).fetchone()
    assert row["status"] == "refunded"


def test_check_pending_swaps_skips_non_refundable_failed_rows():
    """Only refundable failed swaps are re-polled for retry refunds."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status, error)
        VALUES ('failed-no-retry', ?, ?, 'submarine', 500000, 0, 300, 200, 500, 1000, 'failed', 'transaction.refunded')
    """, (now, now))

    manager._run_boltzcli = MagicMock(return_value={"reverseSwaps": [], "swaps": []})

    result = manager.check_pending_swaps()

    assert result["checked"] == 0
    assert result["updated"] == 0
    manager._run_boltzcli.assert_not_called()


# -----------------------------------------------------------------------
# Swap cost recording tests
# -----------------------------------------------------------------------

def test_swap_cost_recorded_in_table():
    manager, _ = _make_manager()
    manager._record_swap_cost(
        swap_id="cost-test-1",
        cost_sats=500,
        amount_sats=500000,
        swap_type="reverse",
        channel_id="123x1x0",
        peer_id="02" + "ab" * 32,
    )

    conn = manager.db._get_connection()
    row = conn.execute(
        "SELECT * FROM swap_costs WHERE swap_id = 'cost-test-1'"
    ).fetchone()
    assert row is not None
    assert row["cost_sats"] == 500
    assert row["amount_sats"] == 500000
    assert row["swap_type"] == "reverse"
    assert row["channel_id"] == "123x1x0"


# -----------------------------------------------------------------------
# boltzcli error handling tests
# -----------------------------------------------------------------------

def test_boltzcli_not_found():
    manager, _ = _make_manager()

    # Don't mock - let it fail naturally (boltzcli not in PATH)
    # Instead, mock subprocess.run to raise FileNotFoundError
    with patch("modules.boltz_swaps.subprocess.run", side_effect=FileNotFoundError):
        try:
            manager._run_boltzcli("getinfo")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not found" in str(e)


def test_boltzcli_nonzero_exit():
    manager, _ = _make_manager()
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "connection refused"
    mock_result.stdout = ""

    with patch("modules.boltz_swaps.subprocess.run", return_value=mock_result):
        try:
            manager._run_boltzcli("getinfo")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "connection refused" in str(e)


def test_boltzcli_invalid_json():
    manager, _ = _make_manager()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    mock_result.stdout = "not json"

    with patch("modules.boltz_swaps.subprocess.run", return_value=mock_result):
        try:
            manager._run_boltzcli("getinfo")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "non-JSON" in str(e)


# -----------------------------------------------------------------------
# Table migration test
# -----------------------------------------------------------------------

def test_old_table_migrated():
    """Verify the old boltz_swaps table is renamed to boltz_swaps_v1."""
    db = _TestDB()
    conn = db._get_connection()

    # Create old-style table
    conn.execute("""
        CREATE TABLE boltz_swaps (
            id TEXT PRIMARY KEY,
            preimage TEXT,
            claim_privkey TEXT
        )
    """)
    conn.execute("INSERT INTO boltz_swaps VALUES ('old-swap', 'preimage', 'key')")

    plugin = MagicMock()
    plugin.rpc = MagicMock()

    manager = BoltzSwapManager(db, plugin, config={"swap_daily_budget_sats": 50000, "swap_currency": "btc"})

    # Old table should be renamed
    old = conn.execute(
        "SELECT * FROM boltz_swaps_v1 WHERE id = 'old-swap'"
    ).fetchone()
    assert old is not None

    # New table should exist with new schema
    new_cols = {row[1] for row in conn.execute("PRAGMA table_info(boltz_swaps)").fetchall()}
    assert "swap_type" in new_cols
    assert "total_fee_sats" in new_cols
    # Old crypto columns should not be in new table
    assert "preimage" not in new_cols
    assert "claim_privkey" not in new_cols


def test_legacy_rows_backfilled_and_monitored():
    """Legacy actionable swaps are backfilled into new table and still monitored."""
    db = _TestDB()
    conn = db._get_connection()
    now = int(1_700_000_000)

    conn.execute("""
        CREATE TABLE boltz_swaps (
            id TEXT PRIMARY KEY,
            created_at INTEGER,
            updated_at INTEGER,
            swap_type TEXT,
            target_channel_id TEXT,
            target_peer_id TEXT,
            invoice_amount_sats INTEGER,
            onchain_amount_sats INTEGER,
            boltz_fee_sats INTEGER,
            miner_fee_lockup_sats INTEGER,
            miner_fee_claim_sats INTEGER,
            total_cost_sats INTEGER,
            cost_ppm INTEGER,
            status TEXT,
            error TEXT
        )
    """)
    conn.execute("""
        INSERT INTO boltz_swaps
        VALUES ('legacy-live-1', ?, ?, 'loop_out', '123x1x0', NULL,
                500000, 499500, 300, 100, 100, 500, 1000, 'created', NULL)
    """, (now - 100, now - 100))

    plugin = MagicMock()
    plugin.rpc = MagicMock()

    manager = BoltzSwapManager(db, plugin, config={"swap_daily_budget_sats": 50000, "swap_currency": "btc"})

    # Verify backfill to current table happened
    new_row = conn.execute(
        "SELECT * FROM boltz_swaps WHERE id = 'legacy-live-1'"
    ).fetchone()
    assert new_row is not None
    assert new_row["swap_type"] == "reverse"
    assert new_row["status"] == "created"

    # Monitor should still progress legacy-backed swap state
    manager._run_boltzcli = MagicMock(return_value={
        "reverseSwaps": [{"id": "legacy-live-1", "status": "invoice.settled"}],
        "swaps": [],
    })
    result = manager.check_pending_swaps()
    assert result["completed"] == 1


def test_audit_log_recorded():
    manager, _ = _make_manager()
    manager._record_audit_event(
        "test_event", "test message",
        swap_id="audit-test",
        details={"key": "value"},
    )

    conn = manager.db._get_connection()
    row = conn.execute(
        "SELECT * FROM boltz_audit_log WHERE swap_id = 'audit-test'"
    ).fetchone()
    assert row is not None
    assert row["event_type"] == "test_event"
    assert row["message"] == "test message"
    parsed = json.loads(row["details_json"])
    assert parsed["key"] == "value"


# -----------------------------------------------------------------------
# Fix #2: Status detection tests
# -----------------------------------------------------------------------

def test_is_completed_status_invoice_settled():
    """invoice.settled is the terminal success for reverse swaps."""
    manager, _ = _make_manager()
    assert manager._is_completed_status("invoice.settled") is True


def test_is_completed_status_transaction_claimed():
    """transaction.claimed is the terminal success for submarine swaps."""
    manager, _ = _make_manager()
    assert manager._is_completed_status("transaction.claimed") is True


def test_is_completed_status_invoice_paid_not_final():
    """Fix #3: invoice.paid is NOT a final success — swap can still fail."""
    manager, _ = _make_manager()
    assert manager._is_completed_status("invoice.paid") is False


def test_is_failed_status_transaction_refunded():
    """Fix #2: transaction.refunded should be detected as failed."""
    manager, _ = _make_manager()
    assert manager._is_failed_status("transaction.refunded") is True


def test_is_failed_status_invoice_expired():
    """Fix #2: invoice.expired should be detected as failed."""
    manager, _ = _make_manager()
    assert manager._is_failed_status("invoice.expired") is True


def test_is_failed_status_transaction_lockup_failed():
    """Fix #2: transaction.lockupFailed should be detected as failed."""
    manager, _ = _make_manager()
    assert manager._is_failed_status("transaction.lockupFailed") is True


def test_is_failed_status_invoice_failed_to_pay():
    """invoice.failedToPay should be detected as failed."""
    manager, _ = _make_manager()
    assert manager._is_failed_status("invoice.failedToPay") is True


def test_is_refundable_status():
    """Fix #9: Only certain failure states indicate locked funds needing refund."""
    manager, _ = _make_manager()
    assert manager._is_refundable_status("invoice.failedToPay") is True
    assert manager._is_refundable_status("swap.expired") is True
    assert manager._is_refundable_status("transaction.lockupFailed") is True
    # These are NOT refundable (no locked funds or already refunded)
    assert manager._is_refundable_status("transaction.refunded") is False
    assert manager._is_refundable_status("transaction.failed") is False
    assert manager._is_refundable_status("invoice.expired") is False


# -----------------------------------------------------------------------
# Fix #1: Recovery command tests
# -----------------------------------------------------------------------

def test_refund_swap():
    """Fix #1: refund_swap should call boltzcli refundswap."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status, error)
        VALUES ('refund-1', ?, ?, 'submarine', 500000, 0, 300, 200, 500, 1000, 'failed', 'invoice.failedToPay')
    """, (now, now))

    # refund_swap uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(return_value='{"id": "refund-1", "status": "refunded"}')

    result = manager.refund_swap("refund-1")

    assert result["status"] == "refund_initiated"
    assert manager._run_boltzcli_raw.call_count == 1
    assert manager._run_boltzcli_raw.call_args[0] == ("refundswap", "refund-1", "wallet")

    # Verify local record updated
    row = conn.execute("SELECT status FROM boltz_swaps WHERE id = 'refund-1'").fetchone()
    assert row["status"] == "refunded"


def test_refund_swap_with_address():
    """refund_swap should accept a BTC address destination."""
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(return_value='{"status": "refunded"}')

    result = manager.refund_swap("refund-addr", "bc1qtest")

    assert manager._run_boltzcli_raw.call_args[0] == ("refundswap", "refund-addr", "bc1qtest")
    assert result["status"] == "refund_initiated"


def test_refund_swap_error():
    """refund_swap should return error when boltzcli fails."""
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(side_effect=RuntimeError("timelock not expired"))

    result = manager.refund_swap("refund-err")

    assert "error" in result
    assert "timelock" in result["error"]


def test_claim_swaps():
    """Fix #1: claim_swaps should call boltzcli claimswaps."""
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(return_value='{"status": "claimed"}')

    result = manager.claim_swaps(["claim-1", "claim-2"])

    assert result["status"] == "claim_initiated"
    assert manager._run_boltzcli_raw.call_args[0] == ("claimswaps", "wallet", "claim-1", "claim-2")


def test_claim_swaps_string_is_single_id():
    """String input is normalized to a single swap ID, not split into characters."""
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(return_value='{"status": "claimed"}')

    result = manager.claim_swaps("claim-1")

    assert result["status"] == "claim_initiated"
    assert manager._run_boltzcli_raw.call_args[0] == ("claimswaps", "wallet", "claim-1")


def test_claim_swaps_invalid_type():
    manager, _ = _make_manager()
    result = manager.claim_swaps(12345)
    assert "error" in result


def test_claim_swaps_with_address():
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(return_value='{"status": "claimed"}')

    result = manager.claim_swaps(["claim-1"], destination="bc1qtest")

    assert manager._run_boltzcli_raw.call_args[0] == ("claimswaps", "bc1qtest", "claim-1")


def test_claim_swaps_empty_ids():
    manager, _ = _make_manager()
    result = manager.claim_swaps([])
    assert "error" in result


# -----------------------------------------------------------------------
# Fix #4: Fee extraction from creation response
# -----------------------------------------------------------------------

def test_extract_fees_from_creation_response():
    """Fix #4: Fees from creation response should override quote."""
    manager, _ = _make_manager()

    result = {"serviceFee": 300, "onchainFee": 180}
    quote = {"boltz_fee_sats": 250, "network_fee_sats": 150}

    fees = manager._extract_fees_from_response(result, 500000, quote)

    # Should use creation response values, not quote
    assert fees["boltz_fee_sats"] == 300
    assert fees["network_fee_sats"] == 180
    assert fees["total_fee_sats"] == 480
    assert fees["fee_ppm"] == 960


def test_extract_fees_fallback_to_quote():
    """Fix #4: Falls back to quote when creation response has no fee fields."""
    manager, _ = _make_manager()

    result = {"id": "swap-1", "status": "created"}
    quote = {"boltz_fee_sats": 250, "network_fee_sats": 150}

    fees = manager._extract_fees_from_response(result, 500000, quote)

    assert fees["boltz_fee_sats"] == 250
    assert fees["network_fee_sats"] == 150
    assert fees["total_fee_sats"] == 400


# -----------------------------------------------------------------------
# Fix #5: swapinfo raw parsing
# -----------------------------------------------------------------------

def test_parse_swapinfo_raw():
    """Fix #5: Parse raw swapinfo output into structured dict."""
    manager, _ = _make_manager()

    raw = """Id: abc123
Status: invoice.settled
Amount: 500000
Service Fee: 250
Onchain Fee: 150"""

    parsed = manager._parse_swapinfo_raw(raw)

    assert parsed["id"] == "abc123"
    assert parsed["status"] == "invoice.settled"
    assert parsed["amount"] == "500000"


def test_swapinfo_syncs_status_from_raw():
    """Fix #5: get_swap_info should parse raw swapinfo and sync status."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('raw-sync', ?, ?, 'submarine', 500000, 499500, 300, 200, 500, 1000, 'created')
    """, (now, now))

    manager._run_boltzcli_raw = MagicMock(return_value="Id: raw-sync\nStatus: transaction.claimed")

    result = manager.get_swap_info("raw-sync")

    assert result["local"]["status"] == "completed"
    # Cost should be recorded
    cost_row = conn.execute("SELECT * FROM swap_costs WHERE swap_id = 'raw-sync'").fetchone()
    assert cost_row is not None


def test_swapinfo_failed_triggers_refund_for_submarine():
    """Fix #9: get_swap_info should auto-refund failed submarine swaps."""
    manager, _ = _make_manager()

    conn = manager.db._get_connection()
    now = manager._now_ts()
    conn.execute("""
        INSERT INTO boltz_swaps (id, created_at, updated_at, swap_type,
            amount_sats, received_sats, boltz_fee_sats, network_fee_sats,
            total_fee_sats, fee_ppm, status)
        VALUES ('sub-fail-info', ?, ?, 'submarine', 500000, 0, 300, 200, 500, 1000, 'created')
    """, (now, now))

    # swapinfo uses _run_boltzcli_raw (already correct — no --json)
    # refundswap also uses _run_boltzcli_raw via _run_boltzcli_auto
    # Both go through _run_boltzcli_raw, so mock it with sequential responses
    call_count = {"n": 0}
    def mock_raw(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # swapinfo response
            return "Id: sub-fail-info\nStatus: invoice.failedToPay"
        else:
            # refundswap response
            return '{"status": "refunded"}'
    manager._run_boltzcli_raw = MagicMock(side_effect=mock_raw)

    result = manager.get_swap_info("sub-fail-info")

    assert result["local"]["status"] == "failed"
    # refundswap should have been called via _run_boltzcli_raw
    assert manager._run_boltzcli_raw.call_count == 2
    refund_call = manager._run_boltzcli_raw.call_args_list[1]
    assert refund_call[0] == ("refundswap", "sub-fail-info", "wallet")


# -----------------------------------------------------------------------
# Fix #6: --refund wallet on createswap
# -----------------------------------------------------------------------

def test_createswap_includes_refund_wallet():
    """Fix #6: createswap should include --refund wallet flag."""
    manager, _ = _make_manager()

    call_results = [
        {"serviceFee": 150, "networkFee": 100},
        {"id": "sub-refund-flag", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    manager.loop_in(500000)

    # Second call is createswap
    create_call = manager._run_boltzcli.call_args_list[1]
    args = create_call[0]
    assert args[0] == "createswap"
    assert "--refund" in args
    assert "wallet" in args


# -----------------------------------------------------------------------
# Fix #8: --chan-id on createreverseswap
# -----------------------------------------------------------------------

def test_createreverseswap_passes_chan_id():
    """Fix #8: createreverseswap should pass --chan-id when channel_id provided."""
    manager, _ = _make_manager()

    call_results = [
        {"serviceFee": 200, "networkFee": 100},
        {"id": "rev-chan", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    manager.loop_out(500000, channel_id="123x1x0")

    create_call = manager._run_boltzcli.call_args_list[1]
    args = create_call[0]
    assert args[0] == "createreverseswap"
    assert "--chan-id" in args
    assert "123x1x0" in args


def test_createreverseswap_no_chan_id_without_channel():
    """--chan-id should NOT be passed when channel_id is None."""
    manager, _ = _make_manager()

    call_results = [
        {"serviceFee": 200, "networkFee": 100},
        {"id": "rev-no-chan", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    manager.loop_out(500000)

    create_call = manager._run_boltzcli.call_args_list[1]
    args = create_call[0]
    assert "--chan-id" not in args


# -----------------------------------------------------------------------
# Idempotent cost recording
# -----------------------------------------------------------------------

def test_maybe_record_completion_cost_idempotent():
    """Cost should only be recorded once per swap, even if called multiple times."""
    manager, _ = _make_manager()

    local = {
        "id": "idem-1",
        "total_fee_sats": 500,
        "amount_sats": 500000,
        "swap_type": "reverse",
        "target_channel_id": None,
        "target_peer_id": None,
    }

    manager._maybe_record_completion_cost(local)
    manager._maybe_record_completion_cost(local)

    conn = manager.db._get_connection()
    rows = conn.execute("SELECT * FROM swap_costs WHERE swap_id = 'idem-1'").fetchall()
    assert len(rows) == 1


# -----------------------------------------------------------------------
# LBTC (Liquid) integration tests
# -----------------------------------------------------------------------

def _make_lbtc_manager(config=None):
    """Create a manager with LBTC as default currency."""
    if config is None:
        config = {
            "swap_daily_budget_sats": 50000,
            "swap_max_fee_ppm": 5000,
            "swap_min_amount_sats": 100000,
            "swap_max_amount_sats": 10000000,
            "swap_currency": "lbtc",
        }
    return _make_manager(config=config)


def test_get_currency_default_lbtc():
    """_get_currency returns config value when valid."""
    manager, _ = _make_lbtc_manager()
    assert manager._get_currency() == "lbtc"


def test_get_currency_default_btc():
    manager, _ = _make_manager()
    assert manager._get_currency() == "btc"


def test_get_currency_override():
    """Override parameter takes precedence over config."""
    manager, _ = _make_lbtc_manager()
    assert manager._get_currency("btc") == "btc"

    manager2, _ = _make_manager()
    assert manager2._get_currency("lbtc") == "lbtc"


def test_get_currency_invalid_falls_back_to_lbtc():
    """Invalid currency string falls back to lbtc."""
    manager, _ = _make_manager(config={"swap_currency": "invalid"})
    assert manager._get_currency() == "lbtc"


def test_ensure_lbtc_wallet_finds_existing():
    """_ensure_lbtc_wallet returns existing LBTC wallet name."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [
            {"name": "btc-wallet", "currency": "BTC"},
            {"name": "my-liquid", "currency": "LBTC"},
        ]
    })

    name = manager._ensure_lbtc_wallet()

    assert name == "my-liquid"
    assert manager._wallet_names.get("LBTC") == "my-liquid"


def test_ensure_lbtc_wallet_creates_new():
    """_ensure_lbtc_wallet creates a wallet when none exists."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "btc-wallet", "currency": "BTC"}]
    })
    manager._run_boltzcli_raw = MagicMock(return_value="Wallet created")

    name = manager._ensure_lbtc_wallet()

    assert name == "liquid"
    manager._run_boltzcli_raw.assert_called_once_with("wallet", "create", "liquid", "LBTC")


def test_ensure_lbtc_wallet_caches():
    """_ensure_lbtc_wallet caches the result."""
    manager, _ = _make_lbtc_manager()
    manager._wallet_names["LBTC"] = "cached-wallet"

    name = manager._ensure_lbtc_wallet()

    assert name == "cached-wallet"


def test_get_lbtc_balance():
    """_get_lbtc_balance extracts confirmed balance."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [
            {"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 500000, "unconfirmed": 10000}},
        ]
    })

    balance = manager._get_lbtc_balance("liquid")
    assert balance == 500000


def test_get_lbtc_balance_missing_wallet():
    """_get_lbtc_balance returns 0 if wallet not found."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={"wallets": []})

    balance = manager._get_lbtc_balance("liquid")
    assert balance == 0


def test_get_wallet_balances():
    """get_wallet_balances returns wallet list from boltzd."""
    manager, _ = _make_lbtc_manager()
    expected = {"wallets": [{"name": "liquid", "currency": "LBTC"}]}
    manager._run_boltzcli = MagicMock(return_value=expected)

    result = manager.get_wallet_balances()
    assert result == expected


def test_quote_reverse_lbtc():
    """Reverse quote with LBTC includes --to LBTC flag."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 125, "networkFee": 47})

    result = manager.quote_reverse(500000)

    call_args = manager._run_boltzcli.call_args[0]
    assert "--to" in call_args
    assert "LBTC" in call_args
    assert "reverse" in call_args
    assert result["currency"] == "lbtc"
    assert result["total_fee_sats"] == 172


def test_quote_reverse_btc_no_to_flag():
    """Reverse quote with BTC should NOT include --to flag."""
    manager, _ = _make_manager()
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 250, "networkFee": 530})

    result = manager.quote_reverse(500000)

    call_args = manager._run_boltzcli.call_args[0]
    assert "--to" not in call_args
    assert result["currency"] == "btc"


def test_quote_submarine_lbtc():
    """Submarine quote with LBTC includes --from LBTC flag."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 100, "networkFee": 19})

    result = manager.quote_submarine(500000)

    call_args = manager._run_boltzcli.call_args[0]
    assert "--from" in call_args
    assert "LBTC" in call_args
    assert "submarine" in call_args
    assert result["currency"] == "lbtc"


def test_quote_both_currencies():
    """currency='both' returns both BTC and LBTC quotes."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"serviceFee": 250, "networkFee": 530}  # BTC
        else:
            return {"serviceFee": 125, "networkFee": 47}   # LBTC

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.quote(500000, swap_type="reverse", currency="both")

    assert "btc" in result
    assert "lbtc" in result
    assert result["btc"]["currency"] == "btc"
    assert result["lbtc"]["currency"] == "lbtc"


def test_loop_out_lbtc_to_wallet():
    """LBTC reverse swap routes to LBTC wallet when no address given."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            # wallet list (for _ensure_lbtc_wallet) and quote
            if args[0] == "wallet":
                return {"wallets": [{"name": "liquid", "currency": "LBTC"}]}
            return {"serviceFee": 125, "networkFee": 47}
        return {"id": "rev-lbtc-1", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.loop_out(500000)

    assert result["status"] == "created"
    assert result["currency"] == "lbtc"

    # Find the createreverseswap call
    for call in manager._run_boltzcli.call_args_list:
        args = call[0]
        if args[0] == "createreverseswap":
            assert args[1] == "lbtc"
            assert "--to-wallet" in args
            assert "liquid" in args
            break
    else:
        assert False, "createreverseswap call not found"


def test_loop_out_lbtc_with_address_no_wallet():
    """LBTC reverse swap with explicit address should NOT add --to-wallet."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"serviceFee": 125, "networkFee": 47}
        return {"id": "rev-lbtc-addr", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.loop_out(500000, address="lq1someaddress")

    assert result["status"] == "created"
    for call in manager._run_boltzcli.call_args_list:
        args = call[0]
        if args[0] == "createreverseswap":
            assert "lq1someaddress" in args
            assert "--to-wallet" not in args
            break


def test_loop_in_lbtc_from_wallet():
    """LBTC submarine swap uses --from-wallet with LBTC wallet."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "wallet":
            return {"wallets": [{"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 1000000}}]}
        if args[0] == "quote":
            return {"serviceFee": 100, "networkFee": 19}
        return {"id": "sub-lbtc-1", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.loop_in(500000)

    assert result["status"] == "created"
    assert result["currency"] == "lbtc"

    for call in manager._run_boltzcli.call_args_list:
        args = call[0]
        if args[0] == "createswap":
            assert "--from-wallet" in args
            assert "liquid" in args
            assert "lbtc" in args
            break
    else:
        assert False, "createswap call not found"


def test_loop_in_lbtc_auto_fallback_to_btc():
    """LBTC submarine swap falls back to BTC when LBTC wallet balance insufficient."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "wallet":
            # LBTC wallet with very low balance
            return {"wallets": [{"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 100}}]}
        if args[0] == "quote":
            return {"serviceFee": 150, "networkFee": 302}
        return {"id": "sub-fallback", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.loop_in(500000)

    assert result["status"] == "created"
    assert result["currency"] == "btc"
    assert result.get("fallback") == "lbtc_insufficient_balance"

    # createswap should use btc, not lbtc
    for call in manager._run_boltzcli.call_args_list:
        args = call[0]
        if args[0] == "createswap":
            assert "btc" in args
            assert "--from-wallet" not in args
            break


def test_currency_recorded_in_db():
    """Currency is recorded in both boltz_swaps and swap_costs tables."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "wallet":
            return {"wallets": [{"name": "liquid", "currency": "LBTC"}]}
        if args[0] == "quote":
            return {"serviceFee": 125, "networkFee": 47}
        return {"id": "cur-track-1", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)
    manager.loop_out(500000)

    conn = manager.db._get_connection()
    row = conn.execute("SELECT currency FROM boltz_swaps WHERE id = 'cur-track-1'").fetchone()
    assert row is not None
    assert row["currency"] == "lbtc"


def test_currency_column_migration():
    """Currency column is added to existing tables via migration."""
    manager, _ = _make_manager()
    conn = manager.db._get_connection()

    # Verify currency column exists in boltz_swaps
    cols = {row[1] for row in conn.execute("PRAGMA table_info(boltz_swaps)").fetchall()}
    assert "currency" in cols

    # Verify currency column exists in swap_costs
    cols = {row[1] for row in conn.execute("PRAGMA table_info(swap_costs)").fetchall()}
    assert "currency" in cols


def test_loop_out_btc_override_on_lbtc_config():
    """currency='btc' override on LBTC-configured manager uses BTC."""
    manager, _ = _make_lbtc_manager()

    call_results = [
        {"serviceFee": 250, "networkFee": 530},
        {"id": "rev-btc-override", "status": "created"},
    ]
    _mock_boltzcli(manager, call_results)

    result = manager.loop_out(500000, currency="btc")

    assert result["currency"] == "btc"
    # createreverseswap should use btc
    create_call = manager._run_boltzcli.call_args_list[1]
    args = create_call[0]
    assert args[0] == "createreverseswap"
    assert args[1] == "btc"


# -----------------------------------------------------------------------
# Audit fix: C3 - Wallet error handling in swap creation
# -----------------------------------------------------------------------

def test_loop_out_lbtc_wallet_error_returns_error():
    """C3: If _ensure_lbtc_wallet fails during reverse swap, return error."""
    manager, _ = _make_lbtc_manager()

    # Quote succeeds, but wallet list will fail
    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"serviceFee": 125, "networkFee": 47}
        if args[0] == "wallet":
            raise RuntimeError("boltzd connection refused")
        return {"id": "should-not-reach", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)
    manager._run_boltzcli_raw = MagicMock(side_effect=RuntimeError("boltzd connection refused"))

    result = manager.loop_out(500000)

    assert "error" in result
    assert "LBTC wallet unavailable" in result["error"]


def test_loop_in_lbtc_wallet_error_falls_back_to_btc():
    """C3: If wallet operations fail during submarine swap, fallback to BTC."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "wallet":
            raise RuntimeError("boltzd connection refused")
        if args[0] == "quote":
            return {"serviceFee": 150, "networkFee": 302}
        return {"id": "sub-fallback-err", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)
    manager._run_boltzcli_raw = MagicMock(side_effect=RuntimeError("boltzd connection refused"))

    result = manager.loop_in(500000)

    assert result["status"] == "created"
    assert result["currency"] == "btc"
    assert result.get("fallback") == "lbtc_insufficient_balance"


# -----------------------------------------------------------------------
# Audit fix: H1 - Fee margin in LBTC balance check
# -----------------------------------------------------------------------

def test_loop_in_lbtc_fee_margin_triggers_fallback():
    """H1: Balance must cover amount + 2% fee margin, not just amount."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "wallet":
            # Balance is exactly the swap amount but not enough with 2% margin
            return {"wallets": [{"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 500000}}]}
        if args[0] == "quote":
            return {"serviceFee": 150, "networkFee": 302}
        return {"id": "sub-margin", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.loop_in(500000)

    # 500000 * 1.02 = 510000 > 500000 balance, so should fall back
    assert result["currency"] == "btc"
    assert result.get("fallback") == "lbtc_insufficient_balance"


def test_loop_in_lbtc_sufficient_with_margin():
    """Balance covers amount + 2% fee margin — no fallback."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "wallet":
            # 520000 > 510000 (500000 * 1.02)
            return {"wallets": [{"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 520000}}]}
        if args[0] == "quote":
            return {"serviceFee": 100, "networkFee": 19}
        return {"id": "sub-lbtc-ok", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.loop_in(500000)

    assert result["status"] == "created"
    assert result["currency"] == "lbtc"


# -----------------------------------------------------------------------
# Chain swap (LBTC <-> BTC) tests
# -----------------------------------------------------------------------

def test_create_chain_swap_lbtc_to_btc():
    """Chain swap from LBTC to BTC."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"boltzFee": 200, "networkFee": 100}
        if args[0] == "wallet":
            return {"wallets": [
                {"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 1000000}},
                {"name": "CLN", "currency": "BTC"},
            ]}
        return {"id": "chain-1", "status": "created", "serviceFee": 200, "networkFee": 100}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.create_chain_swap(500000, from_currency="lbtc", to_currency="btc")

    assert result["status"] == "created"
    assert result["swap_id"] == "chain-1"
    assert result["from_currency"] == "lbtc"
    assert result["to_currency"] == "btc"

    # Verify createchainswap was called with correct args
    for call in manager._run_boltzcli.call_args_list:
        a = call[0]
        if a[0] == "createchainswap":
            assert "--from-wallet" in a
            assert "--to-wallet" in a
            break
    else:
        assert False, "createchainswap call not found"

    # Verify DB record
    row = manager.db._get_connection().execute(
        "SELECT * FROM boltz_swaps WHERE id = 'chain-1'"
    ).fetchone()
    assert row is not None
    assert row["swap_type"] == "chain"
    assert row["currency"] == "lbtc->btc"


def test_create_chain_swap_same_currency_rejected():
    """Chain swap from BTC to BTC should be rejected."""
    manager, _ = _make_manager()
    result = manager.create_chain_swap(500000, from_currency="btc", to_currency="btc")
    assert "error" in result
    assert "itself" in result["error"]


def test_create_chain_swap_insufficient_lbtc():
    """Chain swap fails when LBTC balance < amount + 2% margin."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"boltzFee": 200, "networkFee": 100}
        if args[0] == "wallet":
            return {"wallets": [{"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 100}}]}
        return {}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.create_chain_swap(500000, from_currency="lbtc", to_currency="btc")

    assert "error" in result
    assert "balance" in result["error"].lower()


def test_create_chain_swap_with_to_address():
    """Chain swap with explicit destination address."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"boltzFee": 200, "networkFee": 100}
        if args[0] == "wallet":
            return {"wallets": [{"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 1000000}}]}
        return {"id": "chain-addr", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.create_chain_swap(500000, to_address="bc1qtest")

    assert result["status"] == "created"
    for call in manager._run_boltzcli.call_args_list:
        a = call[0]
        if a[0] == "createchainswap":
            assert "--to-address" in a
            assert "bc1qtest" in a
            break


# -----------------------------------------------------------------------
# Wallet send/receive tests
# -----------------------------------------------------------------------

def test_wallet_receive_lbtc():
    """wallet_receive returns deposit address for LBTC wallet."""
    manager, _ = _make_lbtc_manager()

    # _ensure_lbtc_wallet uses _run_boltzcli("wallet", "list")
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "liquid", "currency": "LBTC"}]
    })
    # wallet receive uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(return_value='{"address": "lq1testaddr123"}')

    result = manager.wallet_receive(currency="lbtc")

    assert result["currency"] == "lbtc"
    assert result["wallet"] == "liquid"
    assert result["result"]["address"] == "lq1testaddr123"


def test_wallet_receive_wallet_error():
    """wallet_receive returns error when LBTC wallet fails."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(side_effect=RuntimeError("boltzd down"))
    manager._run_boltzcli_raw = MagicMock(side_effect=RuntimeError("boltzd down"))

    result = manager.wallet_receive()
    assert "error" in result


def test_wallet_send_lbtc():
    """wallet_send sends from LBTC wallet."""
    manager, _ = _make_lbtc_manager()

    # _ensure_lbtc_wallet uses _run_boltzcli("wallet", "list")
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "liquid", "currency": "LBTC"}]
    })
    # wallet send uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(return_value='{"txid": "abc123"}')

    result = manager.wallet_send("lq1dest", 100000)

    assert result["status"] == "sent"
    assert result["amount_sats"] == 100000
    assert result["currency"] == "lbtc"


def test_wallet_send_with_sweep():
    """wallet_send with sweep flag."""
    manager, _ = _make_lbtc_manager()

    # _ensure_lbtc_wallet uses _run_boltzcli("wallet", "list")
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "liquid", "currency": "LBTC"}]
    })
    # wallet send uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(return_value='{"txid": "sweep123"}')

    result = manager.wallet_send("lq1dest", 0, sweep=True)

    assert result["status"] == "sent"
    # Verify --sweep was in the raw command
    raw_call = manager._run_boltzcli_raw.call_args[0]
    assert "--sweep" in raw_call


def test_wallet_send_error():
    """wallet_send returns error on boltzcli failure."""
    manager, _ = _make_lbtc_manager()

    # _ensure_lbtc_wallet uses _run_boltzcli("wallet", "list")
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "liquid", "currency": "LBTC"}]
    })
    # wallet send uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(side_effect=RuntimeError("insufficient funds"))

    result = manager.wallet_send("lq1dest", 100000)

    assert "error" in result
    assert "insufficient funds" in result["error"]


def test_wallet_send_rejects_option_like_destination():
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(return_value='{"txid": "abc"}')

    result = manager.wallet_send("--sweep", 100000, currency="btc")

    assert "error" in result
    assert "cannot start with '-'" in result["error"]
    manager._run_boltzcli_raw.assert_not_called()


def test_loop_out_rejects_option_like_address():
    manager, _ = _make_manager()
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 200, "networkFee": 100})

    result = manager.loop_out(500000, address="--to-wallet")

    assert "error" in result
    assert "cannot start with '-'" in result["error"]
    manager._run_boltzcli.assert_not_called()


def test_chain_swap_rejects_option_like_to_address():
    manager, _ = _make_manager()
    manager._run_boltzcli = MagicMock(return_value={"serviceFee": 200, "networkFee": 100})

    result = manager.create_chain_swap(500000, from_currency="btc", to_currency="lbtc", to_address="--to-wallet")

    assert "error" in result
    assert "cannot start with '-'" in result["error"]
    manager._run_boltzcli.assert_not_called()


def test_refund_rejects_option_like_destination():
    manager, _ = _make_manager()
    manager._run_boltzcli_raw = MagicMock(return_value='{"status": "refunded"}')

    result = manager.refund_swap("refund-1", destination="--wallet")

    assert "error" in result
    assert "cannot start with '-'" in result["error"]
    manager._run_boltzcli_raw.assert_not_called()


# -----------------------------------------------------------------------
# Audit round 3: Dynamic wallet name discovery (C1)
# -----------------------------------------------------------------------

def test_get_wallet_name_btc_discovers_cln():
    """C1: BTC wallet name is discovered dynamically, not hardcoded 'default'."""
    manager, _ = _make_manager()
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "CLN", "currency": "BTC"}]
    })

    name = manager._get_wallet_name("btc")

    assert name == "CLN"
    assert manager._wallet_names.get("BTC") == "CLN"


def test_get_wallet_name_btc_missing_raises():
    """C1: Missing BTC wallet raises RuntimeError (no auto-create for BTC)."""
    manager, _ = _make_manager()
    manager._run_boltzcli = MagicMock(return_value={"wallets": []})

    try:
        manager._get_wallet_name("btc")
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "BTC" in str(e)


def test_get_wallet_name_lbtc_auto_creates():
    """LBTC wallet is auto-created if not found (unchanged behavior)."""
    manager, _ = _make_lbtc_manager()
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "CLN", "currency": "BTC"}]
    })
    manager._run_boltzcli_raw = MagicMock(return_value="Wallet created")

    name = manager._get_wallet_name("lbtc")

    assert name == "liquid"
    manager._run_boltzcli_raw.assert_called_once_with("wallet", "create", "liquid", "LBTC")


def test_get_wallet_name_caches():
    """Wallet names are cached after first discovery."""
    manager, _ = _make_manager()
    manager._wallet_names["BTC"] = "cached-btc"

    name = manager._get_wallet_name("btc")

    assert name == "cached-btc"


def test_wallet_receive_btc_uses_discovered_name():
    """C1: wallet_receive for BTC uses dynamically discovered wallet name."""
    manager, _ = _make_manager()

    # _get_wallet_name uses _run_boltzcli("wallet", "list")
    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "CLN", "currency": "BTC"}]
    })
    # wallet receive uses _run_boltzcli_auto -> _run_boltzcli_raw
    manager._run_boltzcli_raw = MagicMock(return_value='{"address": "bc1qtestaddr"}')

    result = manager.wallet_receive(currency="btc")

    assert result["currency"] == "btc"
    assert result["wallet"] == "CLN"
    # Verify the actual wallet name "CLN" was passed to receive
    assert manager._run_boltzcli_raw.call_args[0] == ("wallet", "receive", "CLN")


def test_wallet_send_btc_uses_discovered_name():
    """C1: wallet_send for BTC uses dynamically discovered wallet name."""
    manager, _ = _make_manager()

    manager._run_boltzcli = MagicMock(return_value={
        "wallets": [{"name": "CLN", "currency": "BTC"}]
    })
    manager._run_boltzcli_raw = MagicMock(return_value='{"txid": "btctx123"}')

    result = manager.wallet_send("bc1qdest", 50000, currency="btc")

    assert result["status"] == "sent"
    assert result["currency"] == "btc"
    # Verify "CLN" not "default" was used
    raw_call = manager._run_boltzcli_raw.call_args[0]
    assert "CLN" in raw_call


def test_chain_swap_btc_uses_discovered_wallet_name():
    """C1: chain swap uses discovered BTC wallet name, not 'default'."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"boltzFee": 200, "networkFee": 100}
        if args[0] == "wallet":
            return {"wallets": [
                {"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 1000000}},
                {"name": "CLN", "currency": "BTC"},
            ]}
        return {"id": "chain-btc-name", "status": "created", "serviceFee": 200, "networkFee": 100}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.create_chain_swap(500000, from_currency="lbtc", to_currency="btc")

    assert result["status"] == "created"
    # Verify --to-wallet uses "CLN" not "default"
    for call in manager._run_boltzcli.call_args_list:
        a = call[0]
        if a[0] == "createchainswap":
            assert "--to-wallet" in a
            assert "CLN" in a
            assert "default" not in a
            break
    else:
        assert False, "createchainswap call not found"


# -----------------------------------------------------------------------
# Audit round 3: --to-wallet / --to-address mutual exclusion (M1)
# -----------------------------------------------------------------------

def test_chain_swap_to_address_excludes_to_wallet():
    """M1: When to_address is given, --to-wallet should NOT be present."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"boltzFee": 200, "networkFee": 100}
        if args[0] == "wallet":
            return {"wallets": [
                {"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 1000000}},
                {"name": "CLN", "currency": "BTC"},
            ]}
        return {"id": "chain-addr-only", "status": "created"}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.create_chain_swap(500000, to_address="bc1qexternal")

    assert result["status"] == "created"
    for call in manager._run_boltzcli.call_args_list:
        a = call[0]
        if a[0] == "createchainswap":
            assert "--to-address" in a
            assert "bc1qexternal" in a
            # Must NOT have --to-wallet when --to-address is given
            assert "--to-wallet" not in a
            break
    else:
        assert False, "createchainswap call not found"


def test_chain_swap_no_address_uses_to_wallet():
    """M1: When no to_address, --to-wallet is used."""
    manager, _ = _make_lbtc_manager()

    call_count = {"n": 0}
    def mock_boltzcli(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "quote":
            return {"boltzFee": 200, "networkFee": 100}
        if args[0] == "wallet":
            return {"wallets": [
                {"name": "liquid", "currency": "LBTC", "balance": {"confirmed": 1000000}},
                {"name": "CLN", "currency": "BTC"},
            ]}
        return {"id": "chain-wallet-only", "status": "created", "serviceFee": 200, "networkFee": 100}

    manager._run_boltzcli = MagicMock(side_effect=mock_boltzcli)

    result = manager.create_chain_swap(500000, from_currency="lbtc", to_currency="btc")

    assert result["status"] == "created"
    for call in manager._run_boltzcli.call_args_list:
        a = call[0]
        if a[0] == "createchainswap":
            assert "--to-wallet" in a
            assert "--to-address" not in a
            break
    else:
        assert False, "createchainswap call not found"
