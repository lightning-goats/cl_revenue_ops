"""P1-006 (address-validation part): withdraw validates the destination as a
structurally valid on-chain address for the wallet currency before the
subprocess call. A malformed/empty/wrong-network destination is rejected
cleanly and never reaches boltzcli.

(The amount cap and budget gate are behavioral and intentionally out of
scope here.)
"""

from unittest.mock import MagicMock

import pytest

from modules.boltz_manager import (
    BoltzCliConfig,
    BoltzCliError,
    BoltzCliManager,
    _validate_onchain_address,
)


# Valid vectors (checksums verified with the reference bech32 implementation).
BTC_BECH32 = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
BTC_TESTNET_BECH32 = "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"
BTC_P2PKH = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
BTC_P2SH = "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"
LBTC_EX1 = "ex1qw508d6qejxtdg4y5r3zarvary0c5xw7kxw5fx4"


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
    mgr._resolve_wallet_name = lambda cur, explicit_name=None: "CLN" if cur == "BTC" else "LOOP-LBTC"
    return mgr


def _capture_run(mgr):
    calls = []
    mgr._run = lambda args, timeout=None: (calls.append(list(args)) or "{}")
    return calls


# ---------------------------------------------------------------------------
# Pure validator
# ---------------------------------------------------------------------------

class TestValidator:
    @pytest.mark.parametrize("addr", [BTC_BECH32, BTC_TESTNET_BECH32, BTC_P2PKH, BTC_P2SH])
    def test_valid_btc(self, addr):
        assert _validate_onchain_address(addr, "BTC") is True

    def test_valid_lbtc_segwit(self):
        assert _validate_onchain_address(LBTC_EX1, "LBTC") is True

    @pytest.mark.parametrize("addr", ["", "   ", "--sweep", "notanaddress",
                                      "bc1qinvalidchecksum000000000000000000",
                                      "0BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"])
    def test_malformed_rejected(self, addr):
        assert _validate_onchain_address(addr, "BTC") is False

    def test_wrong_network_btc_bech32_for_lbtc(self):
        # A bitcoin bech32 address must not validate as a Liquid address.
        assert _validate_onchain_address(BTC_BECH32, "LBTC") is False

    def test_wrong_network_lbtc_for_btc(self):
        assert _validate_onchain_address(LBTC_EX1, "BTC") is False

    def test_btc_bech32_uppercase_valid(self):
        assert _validate_onchain_address(BTC_BECH32.upper(), "BTC") is True

    def test_mixed_case_bech32_rejected(self):
        mixed = BTC_BECH32[:6].upper() + BTC_BECH32[6:]
        assert _validate_onchain_address(mixed, "BTC") is False


# ---------------------------------------------------------------------------
# withdraw() integration
# ---------------------------------------------------------------------------

class TestWithdrawValidation:
    def test_valid_btc_address_reaches_boltzcli(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.withdraw(amount_sats=1000, destination=BTC_BECH32, currency="BTC")
        assert len(calls) == 1
        assert BTC_BECH32 in calls[0]

    def test_valid_lbtc_address_reaches_boltzcli(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        mgr.withdraw(amount_sats=1000, destination=LBTC_EX1, currency="LBTC")
        assert len(calls) == 1
        assert LBTC_EX1 in calls[0]

    def test_malformed_destination_rejected(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        with pytest.raises(BoltzCliError):
            mgr.withdraw(amount_sats=1000, destination="--sweep", currency="LBTC")
        assert calls == []  # never reached boltzcli

    def test_empty_destination_rejected(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        with pytest.raises(BoltzCliError):
            mgr.withdraw(amount_sats=1000, destination="   ", currency="LBTC")
        assert calls == []

    def test_wrong_network_destination_rejected(self):
        mgr = _make_manager()
        calls = _capture_run(mgr)
        # BTC address passed to an LBTC withdraw.
        with pytest.raises(BoltzCliError):
            mgr.withdraw(amount_sats=1000, destination=BTC_BECH32, currency="LBTC")
        assert calls == []
