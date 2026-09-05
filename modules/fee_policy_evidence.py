"""Observational fee execution context. No RPC, clock, model or authority."""

from typing import Any, Optional


def _amount(value: Any) -> Optional[int]:
    if isinstance(value, str):
        value = value.removesuffix("msat")
        if value.isascii() and value.isdecimal():
            # Bound parsing effort and avoid Python's long-integer string limit.
            value = int(value) if len(value) <= 19 else None
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def capture_fee_request(channel_info: Any, rpc_params: Any) -> dict:
    """Copy primitive pre-RPC observations; never synthesize absent values."""
    info = channel_info if isinstance(channel_info, dict) else {}
    params = rpc_params if isinstance(rpc_params, dict) else {}
    return {
        "schema_version": 1,
        "prior_policy": {
            "fee_ppm": _amount(info.get("fee_proportional_millionths")),
            "base_fee_msat": _amount(info.get("fee_base_msat")),
        },
        "requested_policy": {
            "fee_ppm": _amount(params.get("feeppm")),
            "base_fee_msat": _amount(params.get("feebase")),
            "htlcmin_msat": _amount(params.get("htlcmin")),
            "htlcmax_msat": _amount(params.get("htlcmax")),
        },
        "prior_context": {
            "capacity_sats": _amount(info.get("capacity")),
            "to_us_msat": _amount(info.get("to_us_msat")),
            "spendable_msat": _amount(info.get("spendable_msat")),
            "receivable_msat": _amount(info.get("receivable_msat")),
        },
    }


def complete_fee_execution(request: dict, rpc_result: Any,
                           channel_id: str, acknowledged_at: int) -> dict:
    """Record acknowledgement, not proof of gossip propagation/exposure.

    Only a uniquely matching response can supply reported policy. A generic
    success response does not turn the requested amounts into readback facts.
    The existing fee-change row ID supplies the local action-record identity.
    """
    candidates = rpc_result.get("channels", []) if isinstance(rpc_result, dict) else []
    matches = []
    if isinstance(candidates, list):
        for row in candidates:
            if not isinstance(row, dict):
                continue
            ids = [row.get("short_channel_id"), row.get("channel_id")]
            if any(isinstance(value, str) and value.replace(":", "x") == channel_id
                   for value in ids):
                matches.append(row)
    reported = None
    if len(matches) == 1:
        row = matches[0]
        reported = {
            "fee_ppm": _amount(row.get("fee_proportional_millionths")),
            "base_fee_msat": _amount(row.get("fee_base_msat")),
            "htlcmin_msat": _amount(row.get("minimum_htlc_out_msat")),
            "htlcmax_msat": _amount(row.get("maximum_htlc_out_msat")),
        }
    return {
        **request,
        "reported_policy": reported,
        "rpc_acknowledged": True,
        "acknowledged_at": _amount(acknowledged_at),
        "time_resolution_seconds": 1,
        "attribution_status": "pending",
    }
