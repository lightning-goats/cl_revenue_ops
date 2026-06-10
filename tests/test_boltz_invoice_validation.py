"""External-pay reverse swaps must validate the Boltz-supplied invoice amount.

The invoice comes from boltzd / the Boltz API. Paying it blind means a
compromised or buggy upstream can hand back an invoice for an arbitrary
amount and the plugin pays the principal in full. The payer must reject any
invoice whose decoded amount does not match the requested swap amount.
"""

from unittest.mock import MagicMock

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliError, BoltzCliManager


PAYEE = "02" + "c" * 64


def _make_manager():
    cfg = BoltzCliConfig(
        enabled=True,
        cli_path="/usr/local/bin/boltzcli",
        datadir="/tmp/test_boltz",
        daily_budget_sats=3000,
        enforce_budget=True,
    )
    plugin = MagicMock()
    plugin.log = MagicMock()
    rpc = MagicMock()
    mgr = BoltzCliManager(plugin, rpc, cfg)
    mgr.data_service = None
    return mgr


def _rpc_responses(mgr, decode_response, pay_response=None):
    def fake_call(method, params=None):
        if method == "decode":
            return decode_response
        if method == "pay":
            return pay_response if pay_response is not None else {"status": "complete"}
        if method == "listpeerchannels":
            return {"channels": []}
        return {}

    mgr.rpc.call.side_effect = fake_call
    return mgr


def _pay(mgr, expected_amount_sats=100_000):
    return mgr._pay_invoice_via_first_hop(
        "lnbc1invoice",
        preferred_peer_id=PAYEE,
        preferred_channel_id=None,
        retry_for=1,
        expected_amount_sats=expected_amount_sats,
    )


def _pay_methods(mgr):
    return [call.args[0] for call in mgr.rpc.call.call_args_list]


def test_rejects_invoice_exceeding_expected_amount():
    mgr = _make_manager()
    _rpc_responses(mgr, {"amount_msat": 200_000_000, "payee": PAYEE})

    with pytest.raises(BoltzCliError, match="amount"):
        _pay(mgr, expected_amount_sats=100_000)

    assert "pay" not in _pay_methods(mgr)


def test_rejects_amountless_invoice_when_amount_expected():
    mgr = _make_manager()
    _rpc_responses(mgr, {"payee": PAYEE})

    with pytest.raises(BoltzCliError, match="amount"):
        _pay(mgr, expected_amount_sats=100_000)

    assert "pay" not in _pay_methods(mgr)


def test_accepts_invoice_matching_expected_amount():
    mgr = _make_manager()
    _rpc_responses(mgr, {"amount_msat": 100_000_000, "payee": PAYEE})

    result = _pay(mgr, expected_amount_sats=100_000)

    assert result.get("status") == "submitted"
    assert "pay" in _pay_methods(mgr)


def test_accepts_msat_string_amount_format():
    mgr = _make_manager()
    _rpc_responses(mgr, {"amount_msat": "100000000msat", "payee": PAYEE})

    result = _pay(mgr, expected_amount_sats=100_000)

    assert result.get("status") == "submitted"


def test_validation_skipped_when_no_expected_amount_given():
    # Backward compatibility: callers that don't know the amount keep the
    # old behavior.
    mgr = _make_manager()
    _rpc_responses(mgr, {"amount_msat": 200_000_000, "payee": PAYEE})

    result = mgr._pay_invoice_via_first_hop(
        "lnbc1invoice",
        preferred_peer_id=PAYEE,
        preferred_channel_id=None,
        retry_for=1,
    )

    assert result.get("status") == "submitted"


def test_loop_out_external_pay_threads_expected_amount():
    mgr = _make_manager()
    mgr._detect_reverse_chanids_support = MagicMock(return_value=False)
    mgr.quote = MagicMock(return_value={"quote": {"estimated_total_fee_sats": 50}})
    mgr._enforce_budget_for_quote = MagicMock(
        return_value={"allowed": True, "estimated_fee_sats": 50, "budget": {}}
    )
    mgr.check_tactical_budget = MagicMock(return_value={"allowed": True})
    mgr._resolve_first_hop_target = MagicMock(return_value=(PAYEE, "100x1x0", []))
    mgr._resolve_wallet_name = MagicMock(return_value="default")
    mgr._run_json = MagicMock(return_value={"id": "swap-1", "invoice": "lnbc1invoice"})
    mgr._record_swap_result = MagicMock()
    mgr._primary_swap_entry = MagicMock(return_value={"id": "swap-1"})
    mgr.swap_status = MagicMock(return_value={})
    mgr._pay_invoice_via_first_hop = MagicMock(return_value={"status": "submitted"})

    mgr.loop_out(amount_sats=250_000, channel_id="100x1x0")

    kwargs = mgr._pay_invoice_via_first_hop.call_args.kwargs
    assert kwargs.get("expected_amount_sats") == 250_000
