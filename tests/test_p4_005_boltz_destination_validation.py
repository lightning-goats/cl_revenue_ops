"""P4-005: _validate_onchain_address (P1-006) must gate the operator-supplied
destination on ALL swap paths that send on-chain proceeds, not just withdraw:
loop_out (address), chainswap (to_address), refund/claim (destination). A
malformed / wrong-network address is rejected before any subprocess call.
"""

from unittest.mock import MagicMock

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliError, BoltzCliManager


BTC_BECH32 = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
LBTC_EX1 = "ex1qw508d6qejxtdg4y5r3zarvary0c5xw7kxw5fx4"
BAD = "--sweep"
GARBAGE = "notanaddress"


def _make_manager(**overrides):
    cfg_kwargs = {
        "enabled": True,
        "cli_path": "/usr/local/bin/boltzcli",
        "datadir": "/tmp/test_boltz",
        "daily_budget_sats": 100_000,
        "enforce_budget": True,
    }
    cfg_kwargs.update(overrides)
    cfg = BoltzCliConfig(**cfg_kwargs)
    plugin = MagicMock()
    plugin.log = MagicMock()
    mgr = BoltzCliManager(plugin, MagicMock(), cfg)
    mgr._resolve_wallet_name = lambda cur, explicit_name=None: "CLN" if cur == "BTC" else "LOOP-LBTC"
    return mgr


def _capture_run(mgr, return_value="{}"):
    calls = []
    mgr._run = lambda args, timeout=None: (calls.append(list(args)) or return_value)
    return calls


def _capture_run_json(mgr):
    calls = []
    mgr._run_json = lambda args, timeout=None: (
        calls.append(list(args)) or {"swaps": [{"id": "s1"}]}
    )
    return calls


# ---------------------------------------------------------------------------
# loop_out
# ---------------------------------------------------------------------------
class TestLoopOut:
    def _prep(self, mgr, calls):
        mgr.quote = lambda **kw: {"quote": {"boltzFee": "80"}}
        mgr._enforce_budget_for_quote = lambda q: {"allowed": True, "estimated_fee_sats": 80, "budget": {}}
        mgr.check_tactical_budget = lambda **kw: {"allowed": True}
        mgr.check_channel_capex_budget = lambda **kw: {"allowed": True}
        mgr._detect_reverse_chanids_support = lambda: True
        mgr._record_swap_result = lambda *a, **k: None

    def test_valid_address_passes(self):
        mgr = _make_manager()
        calls = _capture_run_json(mgr)
        self._prep(mgr, calls)
        mgr.loop_out(amount_sats=50_000, address=BTC_BECH32, currency="BTC")
        assert any(c and c[0] == "createreverseswap" for c in calls)

    @pytest.mark.parametrize("bad", [BAD, GARBAGE, LBTC_EX1])  # LBTC addr wrong network for BTC
    def test_bad_address_rejected(self, bad):
        mgr = _make_manager()
        calls = _capture_run_json(mgr)
        self._prep(mgr, calls)
        with pytest.raises(BoltzCliError):
            mgr.loop_out(amount_sats=50_000, address=bad, currency="BTC")
        assert calls == []  # never reached boltzcli


# ---------------------------------------------------------------------------
# chainswap
# ---------------------------------------------------------------------------
class TestChainswap:
    def _prep(self, mgr):
        mgr._enforce_budget_for_quote = lambda q: {"allowed": True, "estimated_fee_sats": 80, "budget": {}}
        mgr._record_swap_result = lambda *a, **k: None

    def test_valid_to_address_passes(self):
        mgr = _make_manager()
        calls = _capture_run_json(mgr)
        self._prep(mgr)
        # to_currency default is BTC
        mgr.chainswap(amount_sats=50_000, from_currency="LBTC", to_currency="BTC",
                      to_address=BTC_BECH32)
        assert any(c and c[0] == "createchainswap" for c in calls)

    @pytest.mark.parametrize("bad", [BAD, GARBAGE, LBTC_EX1])
    def test_bad_to_address_rejected(self, bad):
        mgr = _make_manager()
        calls = _capture_run_json(mgr)
        self._prep(mgr)
        with pytest.raises(BoltzCliError):
            mgr.chainswap(amount_sats=50_000, from_currency="LBTC", to_currency="BTC",
                          to_address=bad)
        assert calls == []


# ---------------------------------------------------------------------------
# refund
# ---------------------------------------------------------------------------
class TestRefund:
    def test_wallet_keyword_skips_validation(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.refund("swap-1")  # default destination "wallet"
        assert len(calls) == 1

    def test_valid_address_passes(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.refund("swap-1", BTC_BECH32)
        assert len(calls) == 1

    @pytest.mark.parametrize("bad", [BAD, GARBAGE])
    def test_bad_destination_rejected(self, bad):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        with pytest.raises(BoltzCliError):
            mgr.refund("swap-1", bad)
        assert calls == []


# ---------------------------------------------------------------------------
# claim
# ---------------------------------------------------------------------------
class TestClaim:
    def test_wallet_keyword_skips_validation(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.claim(["swap-1"])  # default destination "wallet"
        assert len(calls) == 1

    def test_valid_address_passes(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.claim(["swap-1"], LBTC_EX1)  # Liquid address is a valid on-chain dest
        assert len(calls) == 1

    @pytest.mark.parametrize("bad", [BAD, GARBAGE])
    def test_bad_destination_rejected(self, bad):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        with pytest.raises(BoltzCliError):
            mgr.claim(["swap-1"], bad)
        assert calls == []
