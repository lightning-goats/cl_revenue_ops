"""DD3 / P1-006 (remainder): a configurable hard cap bounds a single Boltz
on-chain withdraw so a typo or automation bug cannot sweep the wallet in one
call. Sweep mode bypasses the amount cap and requires an explicit confirmation.
"""

from unittest.mock import MagicMock

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliError, BoltzCliManager

BTC_BECH32 = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"


def _make_manager(**overrides):
    cfg_kwargs = {"enabled": True, "datadir": "/tmp/test_boltz"}
    cfg_kwargs.update(overrides)
    cfg = BoltzCliConfig(**cfg_kwargs)
    mgr = BoltzCliManager(MagicMock(), MagicMock(), cfg)
    mgr._resolve_wallet_name = lambda cur, explicit_name=None: "CLN"
    return mgr


def _capture_run(mgr):
    calls = []
    mgr._run = lambda args, timeout=None: (calls.append(list(args)) or "{}")
    return calls


def test_over_cap_withdraw_rejected_before_subprocess():
    mgr = _make_manager(max_withdraw_sats=1_000_000)
    calls = _capture_run(mgr)
    with pytest.raises(BoltzCliError, match="exceeds max_withdraw_sats"):
        mgr.withdraw(amount_sats=2_000_000, destination=BTC_BECH32, currency="BTC")
    assert calls == []  # never reached boltzcli


def test_under_cap_withdraw_passes():
    mgr = _make_manager(max_withdraw_sats=1_000_000)
    calls = _capture_run(mgr)
    mgr.withdraw(amount_sats=500_000, destination=BTC_BECH32, currency="BTC")
    assert len(calls) == 1
    assert "500000" in calls[0]


def test_cap_is_configurable():
    mgr = _make_manager(max_withdraw_sats=3_000_000)
    calls = _capture_run(mgr)
    # 2M is under the raised cap now.
    mgr.withdraw(amount_sats=2_000_000, destination=BTC_BECH32, currency="BTC")
    assert len(calls) == 1


def test_cap_zero_disables():
    mgr = _make_manager(max_withdraw_sats=0)
    calls = _capture_run(mgr)
    mgr.withdraw(amount_sats=999_000_000, destination=BTC_BECH32, currency="BTC")
    assert len(calls) == 1


def test_sweep_requires_explicit_confirmation():
    mgr = _make_manager(max_withdraw_sats=1_000_000)
    calls = _capture_run(mgr)
    with pytest.raises(BoltzCliError, match="confirm_sweep"):
        mgr.withdraw(amount_sats=None, destination=BTC_BECH32, currency="BTC", sweep=True)
    assert calls == []


def test_sweep_with_confirmation_proceeds():
    mgr = _make_manager(max_withdraw_sats=1_000_000)
    calls = _capture_run(mgr)
    mgr.withdraw(amount_sats=None, destination=BTC_BECH32, currency="BTC",
                 sweep=True, confirm_sweep=True)
    assert len(calls) == 1
    assert "--sweep" in calls[0]
