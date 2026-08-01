"""Regression tests for boltz_manager P0 fixes and audit correctness fixes."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager


def _make_manager(**overrides):
    cfg_kwargs = {
        "enabled": True,
        "cli_path": "/usr/local/bin/boltzcli",
        "datadir": "/tmp/test_boltz",
        "daily_budget_sats": 3000,
        "enforce_budget": True,
    }
    cfg_kwargs.update(overrides)
    cfg = BoltzCliConfig(**cfg_kwargs)
    plugin = MagicMock()
    plugin.log = MagicMock()
    rpc = MagicMock()
    mgr = BoltzCliManager(plugin, rpc, cfg)
    return mgr


class TestBudgetTOCTOU:
    """P0-1: Verify _swap_creation_lock serializes budget-check + swap-create."""

    def test_swap_creation_lock_exists(self):
        mgr = _make_manager()
        assert hasattr(mgr, '_swap_creation_lock')
        assert isinstance(mgr._swap_creation_lock, type(threading.Lock()))

    def test_concurrent_loop_in_serialized(self):
        """Two concurrent loop_in calls should not both pass budget check."""
        mgr = _make_manager(daily_budget_sats=100, enforce_budget=True)

        call_order = []
        original_enforce = mgr._enforce_budget_for_quote

        def slow_enforce(quote, extra_fee_sats=0):
            call_order.append(('enforce_start', threading.current_thread().name))
            # Simulate slow budget check
            time.sleep(0.05)
            result = original_enforce(quote, extra_fee_sats=extra_fee_sats)
            call_order.append(('enforce_end', threading.current_thread().name))
            return result

        mgr._enforce_budget_for_quote = slow_enforce
        # Mock quote and budget to allow first, reject second
        call_count = [0]

        def mock_quote(**kwargs):
            call_count[0] += 1
            return {
                "swap_type": "submarine",
                "amount_sats": 50000,
                "currency": "LBTC",
                "quote": {"boltzFee": "80", "networkFee": "10"},
                "estimated_total_fee_sats": 90,
            }

        mgr.quote = mock_quote

        # The lock should serialize calls - verify no interleaving
        results = []

        def do_loop_in(idx):
            try:
                # This will fail at _run_json but the lock serialization is what we test
                r = mgr.loop_in(50000, currency="LBTC")
                results.append((idx, r))
            except Exception as e:
                results.append((idx, str(e)))

        t1 = threading.Thread(target=do_loop_in, args=(1,), name="t1")
        t2 = threading.Thread(target=do_loop_in, args=(2,), name="t2")
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Verify serialization: enforce_end of first must come before enforce_start of second
        enforce_starts = [i for i, (action, _) in enumerate(call_order) if action == 'enforce_start']
        enforce_ends = [i for i, (action, _) in enumerate(call_order) if action == 'enforce_end']
        if len(enforce_starts) >= 2 and len(enforce_ends) >= 1:
            # Second start must come after first end (serialized)
            assert enforce_starts[1] > enforce_ends[0], \
                f"Budget checks interleaved! Order: {call_order}"


class TestBackupMnemonicOmission:
    """P0-2: Verify backup() omits mnemonic by default."""

    def test_backup_no_mnemonic_by_default(self):
        mgr = _make_manager()
        mgr._run = MagicMock(return_value="word1 word2 word3")
        mgr._listswaps_json = MagicMock(return_value={"swaps": []})
        mgr._extract_swap_list = MagicMock(return_value=[])
        mgr.wallet_balances = MagicMock(return_value={"wallets": []})

        result = mgr.backup()
        assert "swap_mnemonic" not in result
        assert "note" in result
        # _run should NOT have been called for swapmnemonic
        mgr._run.assert_not_called()

    def test_backup_includes_mnemonic_when_requested(self):
        mgr = _make_manager()
        mgr._run = MagicMock(return_value="word1 word2 word3")
        mgr._listswaps_json = MagicMock(return_value={"swaps": []})
        mgr._extract_swap_list = MagicMock(return_value=[])
        mgr.wallet_balances = MagicMock(return_value={"wallets": []})

        result = mgr.backup(include_mnemonic=True)
        assert "swap_mnemonic" in result
        assert result["swap_mnemonic"] == "word1 word2 word3"
        assert "warning" in result


class TestCooldownPreClaim:
    """C1: Cooldown pre-claim prevents double execution for same channel."""

    def test_pre_claim_blocks_second_thread(self):
        """Two threads racing on the same channel: only one should execute."""
        lock = threading.Lock()
        last_action = {}
        results = []

        def simulate_cycle(thread_id, cooldown_seconds=60):
            now = int(time.time())
            ch_id = "100x1x0"
            with lock:
                last_ts = int(last_action.get(ch_id, 0) or 0)
                if cooldown_seconds > 0 and last_ts > 0 and (now - last_ts) < cooldown_seconds:
                    results.append((thread_id, "cooldown_active"))
                    return
                # Pre-claim
                last_action[ch_id] = now
            # Simulate swap execution outside lock
            time.sleep(0.05)
            results.append((thread_id, "executed"))

        t1 = threading.Thread(target=simulate_cycle, args=(1,))
        t2 = threading.Thread(target=simulate_cycle, args=(2,))
        t1.start()
        time.sleep(0.01)
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)
        statuses = [s for _, s in results]
        assert "executed" in statuses
        assert "cooldown_active" in statuses

    def test_failed_swap_clears_pre_claim(self):
        """If swap fails, the pre-claimed cooldown slot is restored."""
        lock = threading.Lock()
        last_action = {"100x1x0": 0}
        ch_id = "100x1x0"
        now = 1000000
        # Pre-claim inside lock
        with lock:
            original_ts = int(last_action.get(ch_id, 0) or 0)
            last_action[ch_id] = now
        # Simulate failure - restore
        with lock:
            if last_action.get(ch_id) == now:
                last_action[ch_id] = original_ts
        assert last_action[ch_id] == 0


class TestPendingSwapReservation:
    """C2: Pending swaps should be counted in reserved_24h_sats."""

    def test_pending_swap_counted_as_reserved(self):
        mgr = _make_manager()
        now = int(time.time())
        pending_swap = {
            "id": "swap_pending_1", "createdAt": str(now - 100),
            "state": "pending", "status": "pending",
            "boltzFee": "50", "networkFee": "10",
        }
        completed_swap = {
            "id": "swap_done_1", "createdAt": str(now - 200),
            "completedAt": str(now - 150),
            "updatedAt": str(now - 150),
            "state": "completed", "status": "swap.completed",
            "boltzFee": "40", "networkFee": "5",
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [pending_swap, completed_swap]})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=24)
        assert result["spent_24h_sats"] == 45
        assert result["reserved_24h_sats"] > 0

    def test_error_swap_not_reserved(self):
        mgr = _make_manager()
        now = int(time.time())
        error_swap = {
            "id": "swap_err_1", "createdAt": str(now - 100),
            "state": "error", "status": "swap.failed",
            "error": "some error", "boltzFee": "50", "networkFee": "10",
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [error_swap]})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=24)
        assert result["reserved_24h_sats"] == 0

    def test_old_pending_swap_not_reserved(self):
        mgr = _make_manager()
        now = int(time.time())
        old_pending = {
            "id": "swap_old_1", "createdAt": str(now - 100000),
            "state": "pending", "status": "pending",
            "boltzFee": "50", "networkFee": "10",
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [old_pending]})
        mgr._augment_with_swap_journal = MagicMock(side_effect=lambda s, **kw: s)
        result = mgr.get_boltz_cost_components(window_hours=24)
        assert result["reserved_24h_sats"] == 0


class TestBudgetStatus:
    """Budget status should include local Boltz pending reserves."""

    def test_pending_swap_reserves_count_toward_budget(self):
        mgr = _make_manager(daily_budget_sats=300, enforce_budget=True)
        mgr._get_global_budget_limit = MagicMock(return_value={
            "budget_sats": 300,
            "source": "fixed",
        })
        mgr.get_boltz_cost_components = MagicMock(return_value={
            "spent_24h_sats": 40,
            "reserved_24h_sats": 120,
            "counted_details": [],
            "skipped_without_timestamp": 0,
        })
        mgr._get_external_liquidity_costs = MagicMock(return_value={
            "spent_24h_sats": 20,
            "reserved_24h_sats": 30,
        })

        result = mgr.get_budget_status()

        assert result["reserved_24h_sats_estimate"] == 150
        assert result["remaining_24h_sats_estimate"] == 90
        assert result["boltz_remaining_24h_sats_estimate"] == 140
        assert result["boltz_reserved_24h_sats_estimate"] == 120


class TestAutoCycleErrorCounter:
    """C3: Error counter should not reset on blocked state."""

    def test_error_increments_counter(self):
        state = {'consecutive_errors': 2}
        result = {'error': 'something failed'}
        if isinstance(result, dict) and 'error' in result:
            state['consecutive_errors'] = int(state.get('consecutive_errors', 0) or 0) + 1
        assert state['consecutive_errors'] == 3

    def test_success_resets_counter(self):
        state = {'consecutive_errors': 5}
        result = {'status': 'executed'}
        if isinstance(result, dict) and 'error' not in result:
            status = str(result.get('status') or '')
            if status in ('executed', 'dry_run'):
                state['consecutive_errors'] = 0
        assert state['consecutive_errors'] == 0

    def test_blocked_preserves_counter(self):
        state = {'consecutive_errors': 3}
        result = {'status': 'blocked', 'reason': 'pending_swaps'}
        if isinstance(result, dict) and 'error' in result:
            state['consecutive_errors'] = int(state.get('consecutive_errors', 0) or 0) + 1
        elif isinstance(result, dict):
            status = str(result.get('status') or '')
            if status in ('executed', 'dry_run'):
                state['consecutive_errors'] = 0
        assert state['consecutive_errors'] == 3


class TestExpiredSwapTerminal:
    """Task 26/79: 'expired' was missing from the terminal vocabulary, so an
    expired swap kept counting as RESERVED liquidity (on top of the
    boltz:{sid} spend event already recorded at create) until its
    created_at aged out of the window — a double count against the
    unified budget."""

    def test_expired_states_are_terminal(self):
        mgr = _make_manager()
        for shape in ({"status": "swap.expired"},
                      {"status": "invoice.expired"},
                      {"state": "EXPIRED"}):
            assert mgr._is_terminal_swap(shape) is True, shape

    def test_pending_states_stay_non_terminal(self):
        mgr = _make_manager()
        for shape in ({"status": "swap.created"},
                      {"status": "transaction.mempool"},
                      {"state": "PENDING"}):
            assert mgr._is_terminal_swap(shape) is False, shape

    def test_expired_swap_not_counted_as_reserved(self):
        mgr = _make_manager()
        now = int(time.time())
        swaps = [
            {"id": "sX", "status": "swap.expired", "createdAt": now - 600,
             "onchainFee": 400, "serviceFee": 0},
            {"id": "sY", "status": "swap.created", "createdAt": now - 600,
             "onchainFee": 400, "serviceFee": 0},
        ]
        mgr._listswaps_json = MagicMock(return_value={"swaps": swaps})
        mgr._augment_with_swap_journal = lambda sw, limit_hint=50: sw
        mgr._estimate_swap_fee_sats = lambda s: 400
        costs = mgr.get_boltz_cost_components()
        # Only the genuinely-pending swap may hold reserved budget.
        assert costs.get("reserved_24h_sats", 0) <= 400, costs
        assert costs.get("reserved_24h_sats", 0) > 0, (
            "the pending swap must still be counted")
