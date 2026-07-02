"""P1-015: a `--` end-of-options terminator precedes free-form positional
args passed to boltzcli, so a value beginning with `-` is treated as data,
not reparsed as a flag. No shell is involved (list argv); this is
argument-injection hardening only.
"""

from unittest.mock import MagicMock

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
    return BoltzCliManager(plugin, rpc, cfg)


def _capture_run(mgr, return_value="{}"):
    calls = []

    def fake_run(args, timeout=None):
        calls.append(list(args))
        return return_value

    mgr._run = fake_run
    return calls


def _assert_dash_after_terminator(args, malicious):
    assert "--" in args, f"missing -- terminator in {args}"
    term = args.index("--")
    assert malicious in args[term + 1:], f"{malicious!r} not after -- in {args}"


class TestRefundTerminator:
    def test_swap_id_and_dest_after_terminator(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.refund("--sweep", "wallet")
        _assert_dash_after_terminator(calls[0], "--sweep")
        assert calls[0][0] == "refundswap"

    def test_dash_dest_after_terminator(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.refund("abc123", "--to-wallet")
        _assert_dash_after_terminator(calls[0], "--to-wallet")


class TestClaimTerminator:
    def test_dest_and_ids_after_terminator(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.claim(["--sweep"], "wallet")
        _assert_dash_after_terminator(calls[0], "--sweep")
        assert calls[0][0] == "claimswaps"


class TestSwapStatusTerminator:
    def test_swap_id_after_terminator(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        # swap_status also calls _run_json(listswaps) and json.loads(raw);
        # return valid JSON so downstream parsing does not raise.
        mgr._run = lambda args, timeout=None: calls.append(list(args)) or "{}"
        mgr._run_json = lambda args, timeout=None: {}
        mgr.swap_status("--sweep")
        swapinfo_call = next(c for c in calls if c and c[0] == "swapinfo")
        _assert_dash_after_terminator(swapinfo_call, "--sweep")


class TestLoopOutAddressTerminator:
    def test_address_after_terminator(self):
        mgr = _make_manager()
        captured = []

        def fake_run_json(args, timeout=None):
            captured.append(list(args))
            return {"swaps": [{"id": "swap-1"}]}

        mgr._run_json = fake_run_json
        mgr.quote = lambda **kw: {"quote": {"boltzFee": "80"}, "estimated_total_fee_sats": 80}
        mgr._enforce_budget_for_quote = lambda q: {"allowed": True, "estimated_fee_sats": 80, "budget": {}}
        mgr._detect_reverse_chanids_support = lambda: True
        mgr._record_swap_result = lambda *a, **k: None
        # Malicious address beginning with '-' must be passed as data.
        mgr.loop_out(amount_sats=50_000, address="--sweep", currency="BTC")
        create_call = next(c for c in captured if c and c[0] == "createreverseswap")
        _assert_dash_after_terminator(create_call, "--sweep")


class TestWithdrawTerminator:
    def test_valid_destination_after_terminator_flags_before(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        # Valid BTC bech32 address; currency BTC.
        addr = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
        mgr._resolve_wallet_name = lambda cur, explicit_name=None: "CLN"
        mgr.withdraw(amount_sats=1000, destination=addr, currency="BTC",
                     sat_per_vbyte=5, sweep=False)
        args = calls[0]
        assert "--" in args
        term = args.index("--")
        # Flag placed before the terminator, positional destination after it.
        assert "--sat-per-vbyte" in args[:term]
        assert addr in args[term + 1:]
