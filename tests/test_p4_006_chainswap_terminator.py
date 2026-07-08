"""P4-006: chainswap is the one createswap path that was missing the P1-015
`--` end-of-options terminator before its positional args. Add it so the
positional (amount, and any future free-form positional) is passed as data,
never reparsed by boltzcli as a flag — matching the sibling commands.
"""

from unittest.mock import MagicMock

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager


BTC_BECH32 = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"


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
    mgr._enforce_budget_for_quote = lambda q, extra_fee_sats=0: {"allowed": True, "estimated_fee_sats": 80, "budget": {}}
    mgr._record_swap_result = lambda *a, **k: None
    return mgr


def _capture_run_json(mgr):
    calls = []
    mgr._run_json = lambda args, timeout=None: (calls.append(list(args)) or {"swaps": [{"id": "s1"}]})
    return calls


def _create_call(calls):
    return next(c for c in calls if c and c[0] == "createchainswap")


def test_p4_006_amount_positional_after_terminator_to_wallet():
    mgr = _make_manager()
    calls = _capture_run_json(mgr)
    mgr.chainswap(amount_sats=50_000, from_currency="LBTC", to_currency="BTC")
    args = _create_call(calls)
    assert "--" in args, f"missing -- terminator in {args}"
    term = args.index("--")
    # amount is the positional; it must sit after the terminator.
    assert "50000" in args[term + 1:], f"amount not after -- in {args}"
    # flags stay before the terminator.
    assert "--from-wallet" in args[:term]
    assert "--to-wallet" in args[:term]


def test_p4_006_amount_positional_after_terminator_to_address():
    mgr = _make_manager()
    calls = _capture_run_json(mgr)
    mgr.chainswap(amount_sats=25_000, from_currency="LBTC", to_currency="BTC",
                  to_address=BTC_BECH32)
    args = _create_call(calls)
    assert "--" in args
    term = args.index("--")
    assert "25000" in args[term + 1:]
    # --to-address is a flag+value pair and stays before the terminator.
    assert "--to-address" in args[:term]
    assert BTC_BECH32 in args[:term]
