"""Rebalance executor v2 — single execution model, no fleet/network split.

Executes a priced circular route using:
  invoice → sendpay → waitsendpay → cleanup

Retries on route failures with exclude lists. Does not switch between
separate fleet and network execution models.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of a v2 rebalance execution attempt."""

    success: bool = False
    attempts: int = 0
    fee_sats: int = 0
    fee_msat: int = 0
    amount_sats: int = 0
    error: str = ""
    excluded_channels: List[str] = field(default_factory=list)
    payment_pending: bool = False


# Timeout and retry constants
SENDPAY_TIMEOUT = 60
INVOICE_EXPIRY = 300
# The executor does not re-route on failure — the orchestrator handles
# retry-with-exclude at a higher level. One attempt per route.
MAX_ATTEMPTS = 1


class RebalanceExecutor:
    """Execute rebalance routes using standard CLN RPCs.

    No fleet/network distinction — one code path for all routes.
    """

    def __init__(self, plugin, database=None):
        self.plugin = plugin
        self.database = database
        self._our_id: Optional[str] = None

    def _log(self, msg: str, level: str = "info") -> None:
        if self.plugin:
            self.plugin.log(f"[ExecutorV2] {msg}", level=level)

    def _get_our_id(self) -> Optional[str]:
        if self._our_id:
            return self._our_id
        try:
            info = self.plugin.rpc.getinfo()
            self._our_id = info["id"]
        except Exception:
            pass
        return self._our_id

    def execute(
        self,
        route: List[Dict[str, Any]],
        amount_sats: int,
        source_channel_id: str,
        dest_channel_id: str,
        max_fee_sats: int,
    ) -> ExecutionResult:
        """Execute a circular rebalance using invoice + sendpay + waitsendpay.

        Args:
            route: List of hop dicts from getroute (already priced).
            amount_sats: Amount to rebalance.
            source_channel_id: Channel to drain (first hop).
            dest_channel_id: Channel to fill (last hop back to us).
            max_fee_sats: Maximum fee budget in sats.

        Returns:
            ExecutionResult with success/failure details.
        """
        our_id = self._get_our_id()
        if not our_id:
            return ExecutionResult(error="no_node_id")

        amount_msat = amount_sats * 1000
        result = ExecutionResult(amount_sats=amount_sats)

        # Create self-paying invoice
        label = f"rebal-v2-{secrets.token_hex(8)}"
        payment_hash = None
        try:
            inv = self.plugin.rpc.invoice(
                amount_msat=amount_msat,
                label=label,
                description="rebalance-v2",
                expiry=INVOICE_EXPIRY,
            )
            payment_hash = inv["payment_hash"]
            bolt11 = inv.get("bolt11", "")
            payment_secret = inv.get("payment_secret", "")
        except Exception as e:
            result.error = f"invoice_error: {e}"
            self._log(result.error, level="warn")
            return result

        try:
            return self._execute_with_retries(
                route=route,
                amount_msat=amount_msat,
                payment_hash=payment_hash,
                bolt11=bolt11,
                payment_secret=payment_secret,
                label=label,
                max_fee_sats=max_fee_sats,
                result=result,
            )
        finally:
            self._cleanup(label, payment_hash, result)

    def _execute_with_retries(
        self,
        route: List[Dict[str, Any]],
        amount_msat: int,
        payment_hash: str,
        bolt11: str,
        payment_secret: str,
        label: str,
        max_fee_sats: int,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Execute a single sendpay attempt on the given route.

        The executor does not re-route — on failure it records the erring
        channel in excluded_channels and returns. The orchestrator can
        re-price with excludes at a higher level.
        """
        result.attempts = 1

        # Validate route fee before sending
        if route:
            route_fee_msat = route[0].get("amount_msat", amount_msat) - amount_msat
            route_fee_sats = (route_fee_msat + 999) // 1000
            if route_fee_sats > max_fee_sats:
                result.error = (
                    f"route_over_budget: {route_fee_sats} sats exceeds "
                    f"budget {max_fee_sats} sats"
                )
                self._log(result.error, level="info")
                return result

        # Send
        try:
            self.plugin.rpc.sendpay(
                route=route,
                payment_hash=payment_hash,
                amount_msat=amount_msat,
                bolt11=bolt11,
                payment_secret=payment_secret,
            )
        except Exception as e:
            result.error = f"sendpay_error: {e}"
            self._log(f"sendpay failed: {e}", level="warn")
            return result

        # Wait
        try:
            wait_result = self.plugin.rpc.waitsendpay(
                payment_hash=payment_hash,
                timeout=SENDPAY_TIMEOUT,
            )
            fee_msat = wait_result.get("amount_sent_msat", amount_msat) - amount_msat
            result.success = True
            result.fee_msat = fee_msat
            result.fee_sats = (fee_msat + 999) // 1000
            self._log(
                f"Success: {result.amount_sats} sats, fee {result.fee_sats} sats"
            )
            return result

        except Exception as e:
            # Full-fidelity diagnostic log BEFORE structured extraction, so we
            # never lose information when the exception shape is unexpected.
            # This replaces the old 'Failed:  erring_channel=None' log that
            # masked any non-conforming failure mode (timeouts, non-RpcError
            # exceptions, missing error.data fields). See Phase B Task 3
            # investigation on nexus-01 2026-04-10: the blank-failcode log
            # turned out to hide a completely opaque failure, and adding
            # this diagnostic is the first step to figuring out why.
            self._log_executor_failure(e)

            error_dict = getattr(e, "error", {})
            error_code = error_dict.get("code") if isinstance(error_dict, dict) else None
            if error_code == 200:
                result.error = "payment_pending_timeout"
                result.payment_pending = True
                self._log(
                    "waitsendpay timed out with payment still pending",
                    level="info",
                )
                return result

            error_data = self._extract_error_data(e)
            erring_channel = error_data.get("erring_channel")
            erring_direction = error_data.get("erring_direction")
            failcode = error_data.get("failcodename", "")

            # Combine channel + direction into CLN's canonical exclude
            # format (``scid/dir``). Both getroute and askrene's update-channel
            # reject bare SCIDs:
            #   "exclude: should be short_channel_id_dir or node_id: invalid token"
            # (observed live on nexus-01 2026-04-10 18:54Z during the first
            # engine retry attempt). When direction is missing from the
            # error data, fall back to the bare SCID — a future call that
            # needs directional precision can expand both dirs.
            excluded_entry: Optional[str] = None
            if erring_channel:
                if erring_direction is not None:
                    excluded_entry = f"{erring_channel}/{int(erring_direction)}"
                else:
                    excluded_entry = str(erring_channel)

            self._log(
                f"Failed: {failcode} erring_channel={excluded_entry}",
                level="info",
            )

            if excluded_entry:
                result.excluded_channels = [excluded_entry]

            if failcode in (
                "WIRE_PERMANENT_CHANNEL_FAILURE",
                "WIRE_UNKNOWN_NEXT_PEER",
                "WIRE_CHANNEL_DISABLED",
            ):
                result.error = f"permanent_failure: {failcode}"
            else:
                result.error = f"retriable_failure: {failcode}"

        return result

    def _log_executor_failure(self, exc: Exception) -> None:
        """Emit a full diagnostic dump of a sendpay/waitsendpay failure.

        Captures exception type, repr, whether it has an .error attribute,
        the shape of that .error (dict vs str), the top-level keys, and the
        keys of .error.data if present. This produces one machine-grep'able
        line per failure so operators can diagnose cases the structured
        extractor misses without re-running with strace.
        """
        exc_type = type(exc).__name__
        exc_repr = repr(exc)
        err_attr = getattr(exc, "error", None)

        if err_attr is None:
            self._log(
                f"EXECUTOR_FAIL_DIAG type={exc_type} has_error_attr=no "
                f"repr={exc_repr}",
                level="warn",
            )
            return

        if isinstance(err_attr, dict):
            top_keys = sorted(err_attr.keys())
            data = err_attr.get("data")
            data_keys = (
                sorted(data.keys()) if isinstance(data, dict) else None
            )
            message = err_attr.get("message", "")
            code = err_attr.get("code")
            self._log(
                f"EXECUTOR_FAIL_DIAG type={exc_type} "
                f"err_code={code} err_message={message!r} "
                f"err_keys={top_keys} data_keys={data_keys} "
                f"err_dict={err_attr!r}",
                level="warn",
            )
            return

        self._log(
            f"EXECUTOR_FAIL_DIAG type={exc_type} err_attr_type={type(err_attr).__name__} "
            f"err_attr={err_attr!r} repr={exc_repr}",
            level="warn",
        )

    def _extract_error_data(self, error: Exception) -> Dict[str, Any]:
        """Extract structured error data from RPC exception.

        Returns error.error['data'] as a dict if the exception is an
        RpcError with a nested 'data' dict. Returns {} for any other
        shape — the full diagnostic is logged separately by
        _log_executor_failure so nothing is silently discarded.
        """
        if hasattr(error, "error"):
            err = error.error
            if isinstance(err, dict):
                data = err.get("data")
                if isinstance(data, dict):
                    return data
        return {}

    def _cleanup(
        self,
        label: str,
        payment_hash: Optional[str],
        result: ExecutionResult,
    ) -> None:
        """Clean up invoice and failed payment records."""
        try:
            if result.payment_pending:
                return
            if payment_hash and not result.success:
                try:
                    self.plugin.rpc.delpay(
                        payment_hash=payment_hash,
                        status="failed",
                    )
                except Exception:
                    pass
            try:
                self.plugin.rpc.delinvoice(
                    label=label,
                    status="unpaid" if not result.success else "paid",
                )
            except Exception:
                pass
        except Exception:
            pass
