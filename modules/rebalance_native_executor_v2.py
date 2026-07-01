"""Route-aware native rebalance executor for v2 engine.

This executor consumes the sendpay-ready route produced by router v3 and
executes that exact route via Core Lightning RPCs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .rebalance_execution import ExecutionResult
from .utils import base_to_sats_ceil, parse_msat


class NativeRouteExecutor:
    """Execute an already-priced circular route with CLN sendpay."""

    INVOICE_EXPIRY_SEC = 300
    SENDPAY_TIMEOUT_SEC = 60

    def __init__(self, plugin, database=None, observation_store=None, our_id=None):
        self.plugin = plugin
        self.database = database
        self.observation_store = observation_store
        # Callers that already know the node id (the engine caches it) can
        # inject it to skip the per-instantiation getinfo RPC issued by
        # is_available()/_get_our_id(). Default None preserves the lazy
        # getinfo lookup.
        self._our_id: Optional[str] = str(our_id) if our_id else None

    def _log(self, msg: str, level: str = "info") -> None:
        if self.plugin:
            self.plugin.log(f"[NativeRebalanceExecutor] {msg}", level=level)

    def _rpc_call(self, method: str, params: Any = None, *, timeout: Optional[int] = None) -> Any:
        if params is None:
            params = {}
        if timeout is not None:
            # The ThreadSafeRpcProxy consumes a ``timeout=`` kwarg to extend
            # its deadline past config.rpc_timeout_seconds. Plain pyln rpc
            # objects don't accept it, so fall back on TypeError (raised at
            # call time, before any RPC is issued).
            try:
                return self.plugin.rpc.call(method, params, timeout=timeout)
            except TypeError:
                return self.plugin.rpc.call(method, params)
        return self.plugin.rpc.call(method, params)

    def is_available(self) -> bool:
        try:
            self._get_our_id()
            return bool(self._our_id)
        except Exception as exc:
            self._log(f"native executor unavailable: {exc}", level="debug")
            return False

    def _get_our_id(self) -> str:
        if self._our_id:
            return self._our_id
        info = self._rpc_call("getinfo", {})
        if not isinstance(info, dict) or not info.get("id"):
            raise ValueError("getinfo returned no node id")
        self._our_id = str(info["id"])
        return self._our_id

    @staticmethod
    def stable_failure_reason(error: Optional[str]) -> str:
        normalized = str(error or "").strip().lower()
        if not normalized:
            return "local_execution_failed"
        if normalized.startswith("native_route_invalid:"):
            return "local_policy_block"
        if normalized.startswith("native_route_over_budget:"):
            return "route_segment_exhausted"
        if "temporary_channel_failure" in normalized or "fee_insufficient" in normalized:
            return "shared_conflict_changed"
        if "timeout" in normalized or "deadline" in normalized:
            return "executor_timeout"
        return "local_execution_failed"

    @staticmethod
    def _error_details(exc: Exception) -> Tuple[str, Dict[str, Any]]:
        details: Dict[str, Any] = {}
        err = getattr(exc, "error", None)
        if isinstance(err, dict):
            for key in ("code", "message"):
                if key in err:
                    details[key] = err.get(key)
            data = err.get("data")
            if isinstance(data, dict):
                for key in (
                    "erring_channel",
                    "erring_direction",
                    "erring_node",
                    "erring_index",
                    "failcode",
                    "failcodename",
                    "failcode_name",
                ):
                    if key in data:
                        details[key] = data.get(key)
            msg = err.get("message")
            if msg:
                return str(msg), details
            return str(err), details
        return str(exc), details

    @staticmethod
    def _route_summary(route: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for index, hop in enumerate(route or []):
            if not isinstance(hop, dict):
                continue
            summary.append(
                {
                    "index": index,
                    "channel": str(hop.get("channel", "") or ""),
                    "direction": hop.get("direction"),
                    "id": str(hop.get("id", "") or ""),
                    "amount_msat": hop.get("amount_msat"),
                    "delay": hop.get("delay"),
                }
            )
        return summary

    @staticmethod
    def _failure_class(error_text: str) -> str:
        normalized = str(error_text or "").lower()
        if "temporary_channel_failure" in normalized:
            return "liquidity"
        if "fee_insufficient" in normalized or "incorrect_cltv_expiry" in normalized:
            return "fee"
        if "timeout" in normalized or "deadline" in normalized:
            return "timeout"
        return "unknown"

    @staticmethod
    def _compact_detail_suffix(details: Dict[str, Any]) -> str:
        keys = (
            "failcodename",
            "failcode_name",
            "failcode",
            "erring_channel",
            "erring_direction",
            "erring_index",
            "erring_node",
        )
        parts = [
            f"{key}={details[key]}"
            for key in keys
            if details.get(key) not in (None, "")
        ]
        return f" ({', '.join(parts)})" if parts else ""

    @staticmethod
    def _exclude_from_failure(
        route: List[Dict[str, Any]],
        details: Dict[str, Any],
        error_text: str,
    ) -> List[str]:
        channel = str(details.get("erring_channel") or "").strip()
        direction = details.get("erring_direction")
        if channel and direction in (0, 1, "0", "1"):
            return [f"{channel}/{int(direction)}"]
        if channel:
            return [channel]

        failure_class = NativeRouteExecutor._failure_class(error_text)
        if failure_class not in {"liquidity", "fee"}:
            return []

        # If CLN does not name the erring channel, avoid replaying the exact
        # failed external middle path on the immediate retry. The first and
        # final hops are pinned local channels, so only middle hops are useful
        # exclusions for alternate route discovery.
        exclusions: List[str] = []
        seen: set[str] = set()
        for hop in (route or [])[1:-1]:
            if not isinstance(hop, dict):
                continue
            scid = str(hop.get("channel") or "").strip()
            direction = hop.get("direction")
            if not scid:
                continue
            entry = (
                f"{scid}/{int(direction)}"
                if direction in (0, 1, "0", "1")
                else scid
            )
            if entry not in seen:
                seen.add(entry)
                exclusions.append(entry)
        return exclusions

    # Confidence model for failure observations (audit finding 4):
    # - CLN names channel AND direction: full attributed confidence.
    # - CLN names only the channel: both directions at half confidence
    #   (previously dropped silently — the recording loop skipped entries
    #   without a '/dir' suffix).
    # - No attribution at all (all-middle-hops fallback): the blame is
    #   split across the suspect set, 0.85 / n_middle_hops with a 0.2
    #   floor, so mostly-innocent segments are never gossiped fleet-wide
    #   at full confidence.
    ATTRIBUTED_CONFIDENCE = 0.85
    INFERRED_CONFIDENCE_FLOOR = 0.2

    def _record_failure_observations(
        self,
        *,
        result: ExecutionResult,
        route: List[Dict[str, Any]],
        amount_sats: int,
        observation_store=None,
        observation_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        store = observation_store or self.observation_store
        if store is None or result.success:
            return
        failure_data = dict(result.failure_data or {})
        failure_class = str(failure_data.get("failure_class") or "unknown")

        attributed_channel = str(failure_data.get("erring_channel") or "").strip()
        attributed_direction = failure_data.get("erring_direction")

        # Build (scid, direction, confidence) targets per attribution quality.
        targets: List[Tuple[str, int, float]] = []
        if attributed_channel and attributed_direction in (0, 1, "0", "1"):
            targets.append(
                (attributed_channel, int(attributed_direction), self.ATTRIBUTED_CONFIDENCE)
            )
        elif attributed_channel:
            # Attributed channel, unknown direction: record both directions
            # at half confidence instead of dropping the observation.
            for direction in (0, 1):
                targets.append(
                    (attributed_channel, direction, self.ATTRIBUTED_CONFIDENCE / 2.0)
                )
        else:
            # Inferred attribution: excluded_channels is the all-middle-hops
            # fallback. Split the evidence across the suspects.
            directed: List[Tuple[str, int]] = []
            for entry in result.excluded_channels:
                scid, sep, direction_text = str(entry).partition("/")
                if not scid or not sep:
                    continue
                try:
                    directed.append((scid, int(direction_text)))
                except (TypeError, ValueError):
                    continue
            if not directed:
                return
            confidence = max(
                self.INFERRED_CONFIDENCE_FLOOR,
                self.ATTRIBUTED_CONFIDENCE / len(directed),
            )
            targets = [(scid, direction, confidence) for scid, direction in directed]

        context = observation_context or {}
        for scid, direction, confidence in targets:
            try:
                store.record_failure(
                    short_channel_id=scid,
                    direction=direction,
                    amount_sats=int(amount_sats),
                    failure_class=failure_class,
                    confidence=confidence,
                    source_channel_id=str(context.get("source_channel_id", "") or ""),
                    dest_channel_id=str(context.get("dest_channel_id", "") or ""),
                    route_policy=str(context.get("route_policy", "") or ""),
                    router_kind=str(context.get("router_kind", "") or ""),
                    correlation_id=str(context.get("correlation_id", "") or ""),
                )
            except Exception as exc:
                self._log(f"failed to record native segment observation: {exc}", level="debug")

    def _validate_route(
        self,
        route: List[Dict[str, Any]],
        *,
        amount_sats: int,
        source_channel_id: str,
        dest_channel_id: str,
        max_fee_sats: int,
    ) -> tuple[bool, str, int]:
        if amount_sats <= 0:
            return False, "invalid_amount", 0
        if not isinstance(route, list) or not route:
            return False, "missing_route", 0

        first = route[0]
        final = route[-1]
        for index, hop in enumerate(route):
            if not isinstance(hop, dict):
                return False, f"hop_{index}_not_object", 0
            for field in ("id", "channel", "amount_msat", "delay"):
                if field not in hop:
                    return False, f"hop_{index}_missing_{field}", 0

        if str(first.get("channel", "")) != source_channel_id:
            return False, "first_hop_not_source_channel", 0
        if str(final.get("channel", "")) != dest_channel_id:
            return False, "final_hop_not_dest_channel", 0

        our_id = self._get_our_id()
        if str(final.get("id", "")) != our_id:
            return False, "final_hop_not_our_node", 0

        expected_final_msat = amount_sats * 1000
        final_msat = parse_msat(final.get("amount_msat"))
        if final_msat != expected_final_msat:
            return False, "final_amount_mismatch", 0

        previous_msat: Optional[int] = None
        for index, hop in enumerate(route):
            amount_msat = parse_msat(hop.get("amount_msat"))
            if amount_msat <= 0:
                return False, f"hop_{index}_invalid_amount", 0
            if previous_msat is not None and amount_msat > previous_msat:
                return False, "increasing_route_amount", 0
            previous_msat = amount_msat

        first_msat = parse_msat(first.get("amount_msat"))
        fee_msat = max(0, first_msat - expected_final_msat)
        max_fee_msat = max(0, int(max_fee_sats or 0)) * 1000
        if fee_msat > max_fee_msat:
            return (
                False,
                f"route_over_budget: {base_to_sats_ceil(fee_msat)} > {max_fee_sats}",
                fee_msat,
            )
        return True, "", fee_msat

    @staticmethod
    def _is_proxy_timeout(exc: Exception) -> bool:
        """Detect the main plugin's RPCTimeoutError without importing it."""
        for klass in type(exc).__mro__:
            if klass.__name__ == "RPCTimeoutError":
                return True
        return "rpc timeout" in str(exc).lower()

    def _is_payment_unresolved(
        self,
        exc: Exception,
        error_text: str,
        details: Dict[str, Any],
        payment_attempted: bool,
    ) -> bool:
        """True when the payment may still settle despite the exception.

        CLN's waitsendpay code 200 means "timed out, payment still pending".
        A proxy deadline on sendpay/waitsendpay leaves the HTLC state unknown.
        Treating these as terminal failures would delete the invoice, release
        the budget, and allow an immediate repeat payment while the first one
        can still settle.
        """
        if not payment_attempted:
            return False
        if details.get("code") == 200:
            return True
        if self._is_proxy_timeout(exc):
            return True
        return "waitsendpay_status=pending" in error_text

    def _fail_malformed_invoice_response(
        self, result: ExecutionResult, label: str, reason: str
    ) -> ExecutionResult:
        """Terminal failure for a mangled invoice RPC response (NX-4).

        CLN may have created the invoice server-side even though the proxy
        response was malformed, so attempt the same best-effort cleanup as
        the thrown-exception path (no payment_hash: sendpay never ran, so
        there is no payment to delete). Also stamp failure_class so this
        path's failure_data has the same shape as every other failure.
        """
        result.error = f"native_invoice_error: {reason}"
        if not isinstance(result.failure_data, dict):
            result.failure_data = {}
        result.failure_data["failure_class"] = self._failure_class(result.error)
        self._cleanup_failed_payment("", label)
        return result

    def _cleanup_failed_payment(self, payment_hash: str, label: str) -> None:
        if payment_hash:
            try:
                self._rpc_call("delpay", {"payment_hash": payment_hash, "status": "failed"})
            except Exception:
                pass
        if label:
            try:
                self._rpc_call("delinvoice", {"label": label, "status": "unpaid"})
            except Exception:
                pass

    def execute(
        self,
        route: List[Dict[str, Any]],
        amount_sats: int,
        source_channel_id: str,
        dest_channel_id: str,
        max_fee_sats: int,
        observation_store=None,
        observation_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        result = ExecutionResult(
            amount_sats=amount_sats,
            route_type="native",
            hops=len(route or []),
            parts=1,
            attempts=1,
            failure_data={"route_summary": self._route_summary(route or [])},
        )
        ok, reason, planned_fee_msat = self._validate_route(
            route,
            amount_sats=amount_sats,
            source_channel_id=source_channel_id,
            dest_channel_id=dest_channel_id,
            max_fee_sats=max_fee_sats,
        )
        if not ok:
            prefix = "native_route_over_budget" if reason.startswith("route_over_budget") else "native_route_invalid"
            result.error = f"{prefix}: {reason}"
            result.fee_msat = planned_fee_msat
            result.fee_sats = base_to_sats_ceil(planned_fee_msat) if planned_fee_msat else 0
            return result

        label = f"rebal-native-{int(time.time() * 1000)}-{dest_channel_id}"
        payment_hash = ""
        payment_attempted = False
        try:
            invoice = self._rpc_call(
                "invoice",
                {
                    "amount_msat": amount_sats * 1000,
                    "label": label,
                    "description": f"cl_revenue_ops rebalance {source_channel_id}->{dest_channel_id}",
                    "expiry": self.INVOICE_EXPIRY_SEC,
                },
            )
            if not isinstance(invoice, dict):
                return self._fail_malformed_invoice_response(
                    result, label, "unexpected response"
                )
            payment_hash = str(invoice.get("payment_hash") or "")
            if not payment_hash:
                return self._fail_malformed_invoice_response(
                    result, label, "missing payment_hash"
                )

            sendpay_params: Dict[str, Any] = {
                "route": route,
                "payment_hash": payment_hash,
                "amount_msat": amount_sats * 1000,
            }
            if invoice.get("bolt11"):
                sendpay_params["bolt11"] = invoice["bolt11"]
            if invoice.get("payment_secret"):
                sendpay_params["payment_secret"] = invoice["payment_secret"]

            payment_attempted = True
            self._rpc_call("sendpay", sendpay_params)
            paid = self._rpc_call(
                "waitsendpay",
                {
                    "payment_hash": payment_hash,
                    "timeout": self.SENDPAY_TIMEOUT_SEC,
                },
                timeout=self.SENDPAY_TIMEOUT_SEC,
            )
            if isinstance(paid, dict) and paid.get("status") not in (None, "complete"):
                raise RuntimeError(f"waitsendpay_status={paid.get('status')}")

            actual_sent_msat = parse_msat(
                paid.get("amount_sent_msat") if isinstance(paid, dict) else 0
            )
            actual_fee_msat = max(
                0,
                (actual_sent_msat or parse_msat(route[0].get("amount_msat")))
                - amount_sats * 1000,
            )
            result.success = True
            result.fee_msat = actual_fee_msat
            result.fee_sats = base_to_sats_ceil(actual_fee_msat)
            if amount_sats > 0:
                result.fee_ppm = (actual_fee_msat * 1_000_000) // (amount_sats * 1000)
            self._log(
                f"Native route success: {amount_sats} sats to {dest_channel_id}, "
                f"fee {result.fee_sats} sats",
                level="debug",
            )
            return result
        except Exception as exc:
            error_text, details = self._error_details(exc)
            if self._is_payment_unresolved(exc, error_text, details, payment_attempted):
                # The HTLC may still settle. Do NOT delete the invoice or the
                # payment, do NOT emit route exclusions, and surface the
                # planned fee so the engine can keep the budget held until
                # the reconciliation sweep resolves the payment.
                result.payment_pending = True
                result.error = f"payment_pending_timeout: {error_text}"
                result.fee_msat = planned_fee_msat
                result.fee_sats = base_to_sats_ceil(planned_fee_msat)
                result.failure_data = {
                    "failure_class": "pending",
                    "payment_hash": payment_hash,
                    "invoice_label": label,
                    "route_summary": self._route_summary(route or []),
                }
                self._log(
                    f"payment unresolved after timeout (hash={payment_hash}); "
                    "holding for settlement reconciliation",
                    level="warn",
                )
                return result
            failure_data = dict(details)
            failure_data["failure_class"] = self._failure_class(error_text)
            failure_data["route_summary"] = self._route_summary(route or [])
            result.failure_data = failure_data
            result.excluded_channels = self._exclude_from_failure(
                route or [],
                failure_data,
                error_text,
            )
            result.error = (
                f"native_sendpay_error: {error_text}"
                f"{self._compact_detail_suffix(failure_data)}"
            )
            self._cleanup_failed_payment(payment_hash, label)
            self._record_failure_observations(
                result=result,
                route=route or [],
                amount_sats=amount_sats,
                observation_store=observation_store,
                observation_context=observation_context,
            )
            self._log(result.error, level="debug")
            return result
