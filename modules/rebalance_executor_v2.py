"""Rebalance executor v2 — sling-backed execution.

Executes a selected rebalance intent by:
  sling-once -> poll sling-stats -> optional sling-stop on timeout

The v2 engine still prices pairs locally, but the executor no longer
constructs or executes circular routes itself.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of a rebalance execution attempt."""

    success: bool = False
    attempts: int = 0
    fee_sats: int = 0
    fee_msat: int = 0
    amount_sats: int = 0
    fee_ppm: int = 0
    hops: int = 0
    route_type: str = "sling"
    parts: int = 1
    error: str = ""
    excluded_channels: List[str] = field(default_factory=list)
    payment_pending: bool = False


SLING_POLL_INTERVAL_SEC = 0.5
SLING_STATUS_TIMEOUT_S = 30.0
SLING_STOP_WAIT_SEC = 5.0

RUNNING_STATES = {"Starting", "Rebalancing", "Stopping", "NotStarted"}
SLING_FAILURE_STATES = {
    "Error",
    "NoRoutes",
    "NoCheapRoute",
    "NoCandidates",
    "PeerBad",
    "Stopped",
}


class RebalanceExecutor:
    """Execute rebalance intents through the sling plugin only."""

    @staticmethod
    def stable_failure_reason(error: Optional[str]) -> str:
        """Map executor-local errors to stable coordination reasons."""
        normalized = str(error or "").strip().lower()
        if not normalized:
            return "local_execution_failed"
        if normalized == "route_over_budget" or normalized.startswith("route_over_budget:"):
            return "route_segment_exhausted"
        if normalized == "sling_unavailable":
            return "local_execution_failed"
        if normalized.startswith("sling_preflight_error:") or normalized.startswith("sling_error:"):
            return "local_execution_failed"
        if normalized.startswith("sling_timeout"):
            return "executor_timeout"
        if normalized.startswith("retriable_failure:"):
            if any(
                token in normalized
                for token in ("temporary_channel_failure", "fee_insufficient")
            ):
                return "shared_conflict_changed"
            if "incorrect_cltv_expiry" in normalized:
                return "shared_conflict_changed"
            return "local_execution_failed"
        if normalized == "payment_pending_timeout":
            return "executor_timeout"
        return "local_execution_failed"

    def __init__(self, plugin, database=None, observation_store=None):
        self.plugin = plugin
        self.database = database
        self.observation_store = observation_store
        self.observe_timeout_sec = SLING_STATUS_TIMEOUT_S
        self._timeout_from_config_loaded = False
        self.poll_interval_sec = SLING_POLL_INTERVAL_SEC
        self.stop_wait_sec = SLING_STOP_WAIT_SEC

    def _log(self, msg: str, level: str = "info") -> None:
        if self.plugin:
            self.plugin.log(f"[ExecutorV2] {msg}", level=level)

    def is_available(self) -> bool:
        """Return True iff the sling RPC surface is currently loaded."""
        try:
            self.plugin.rpc.call("sling-stats", {"json": True})
            return True
        except Exception as exc:
            self._log(f"sling unavailable: {exc}", level="debug")
            return False

    @staticmethod
    def _maxppm_for_budget(amount_sats: int, max_fee_sats: int) -> int:
        if amount_sats <= 0:
            return 0
        return max(0, (max_fee_sats * 1_000_000) // amount_sats)

    def _load_observe_timeout_sec(self) -> float:
        """Use sling-timeoutpay when CLN exposes it, otherwise keep a safe fallback."""
        try:
            configs = self.plugin.rpc.call("listconfigs", {})
        except Exception as exc:
            self._log(f"listconfigs failed for sling-timeoutpay: {exc}", level="debug")
            return SLING_STATUS_TIMEOUT_S

        timeout_config = {}
        if isinstance(configs, dict):
            timeout_config = (
                (configs.get("configs") or {}).get("sling-timeoutpay")
                or configs.get("sling-timeoutpay")
                or {}
            )

        raw_value = None
        if isinstance(timeout_config, dict):
            raw_value = timeout_config.get("value_int")
            if raw_value is None:
                raw_value = timeout_config.get("value_str")

        try:
            timeout_s = float(raw_value)
        except (TypeError, ValueError):
            return SLING_STATUS_TIMEOUT_S
        return timeout_s if timeout_s > 0 else SLING_STATUS_TIMEOUT_S

    def _ensure_observe_timeout_sec(self) -> None:
        if self._timeout_from_config_loaded:
            return
        self.observe_timeout_sec = self._load_observe_timeout_sec()
        self._timeout_from_config_loaded = True

    @staticmethod
    def _parse_statuses(row: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(row, dict):
            return []
        statuses = row.get("status") or []
        parsed: List[str] = []
        for item in statuses:
            text = str(item)
            if ":" in text:
                parsed.append(text.split(":", 1)[1])
            elif text:
                parsed.append(text)
        return parsed

    def _sling_row(self, scid: str) -> Optional[Dict[str, Any]]:
        try:
            rows = self.plugin.rpc.call("sling-stats", {"json": True})
        except Exception as exc:
            self._log(f"sling-stats row lookup failed for {scid}: {exc}", level="warn")
            return None
        if not isinstance(rows, list):
            return None
        for row in rows:
            if isinstance(row, dict) and str(row.get("scid", "")) == scid:
                return row
        return None

    def _scid_stats(self, scid: str) -> Dict[str, Any]:
        try:
            result = self.plugin.rpc.call("sling-stats", {"scid": scid, "json": True})
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            self._log(f"sling-stats detail lookup failed for {scid}: {exc}", level="warn")
            return {}

    def _stop_once_job(self, scid: str) -> None:
        try:
            self.plugin.rpc.call("sling-stop", {"scid": scid})
        except Exception as exc:
            self._log(f"sling-stop failed for {scid}: {exc}", level="warn")

    @staticmethod
    def _success_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, int]:
        before_s = before.get("successes_in_time_window") or {}
        after_s = after.get("successes_in_time_window") or {}
        return {
            "amount_sats": max(
                0,
                int(after_s.get("total_amount_sats", 0) or 0)
                - int(before_s.get("total_amount_sats", 0) or 0),
            ),
            "spent_sats": max(
                0,
                int(after_s.get("total_spent_sats", 0) or 0)
                - int(before_s.get("total_spent_sats", 0) or 0),
            ),
            "rebalances": max(
                0,
                int(after_s.get("total_rebalances", 0) or 0)
                - int(before_s.get("total_rebalances", 0) or 0),
            ),
        }

    @staticmethod
    def _failure_reason(before: Dict[str, Any], after: Dict[str, Any]) -> str:
        before_f = before.get("failures_in_time_window") or {}
        after_f = after.get("failures_in_time_window") or {}
        before_counts = {
            str(item.get("failure_reason", "")): int(item.get("failure_count", 0) or 0)
            for item in (before_f.get("top_5_failure_reasons") or [])
            if item.get("failure_reason")
        }
        best_reason = ""
        best_delta = 0
        for item in after_f.get("top_5_failure_reasons") or []:
            reason = str(item.get("failure_reason", "") or "")
            count = int(item.get("failure_count", 0) or 0)
            delta = count - before_counts.get(reason, 0)
            if delta > best_delta:
                best_delta = delta
                best_reason = reason
        return best_reason

    def _wait_for_terminal_state(
        self,
        dest_channel_id: str,
        amount_sats: int,
        baseline_stats: Dict[str, Any],
        result: ExecutionResult,
    ) -> ExecutionResult:
        deadline = time.monotonic() + self.observe_timeout_sec

        while time.monotonic() < deadline:
            row = self._sling_row(dest_channel_id)
            statuses = self._parse_statuses(row)

            if "Balanced" in statuses:
                after_stats = self._scid_stats(dest_channel_id)
                delta = self._success_delta(baseline_stats, after_stats)
                fee_sats = delta["spent_sats"]
                moved_sats = delta["amount_sats"] or amount_sats
                result.success = True
                result.amount_sats = moved_sats
                result.fee_sats = fee_sats
                result.fee_msat = fee_sats * 1000
                if moved_sats > 0:
                    result.fee_ppm = math.ceil(
                        result.fee_msat * 1_000_000 / (moved_sats * 1000)
                    )
                self._log(
                    f"Sling success: {moved_sats} sats to {dest_channel_id}, fee {fee_sats} sats"
                )
                return result

            for status in statuses:
                if status in SLING_FAILURE_STATES:
                    after_stats = self._scid_stats(dest_channel_id)
                    failure_reason = self._failure_reason(baseline_stats, after_stats)
                    result.amount_sats = amount_sats
                    result.error = (
                        f"retriable_failure: {failure_reason}"
                        if failure_reason
                        else f"retriable_failure: {status}"
                    )
                    self._log(f"Sling failed for {dest_channel_id}: {result.error}", level="info")
                    return result

            time.sleep(self.poll_interval_sec)

        self._stop_once_job(dest_channel_id)
        self._scid_stats(dest_channel_id)
        result.amount_sats = amount_sats
        result.error = "sling_timeout"
        self._log(f"Sling failed for {dest_channel_id}: {result.error}", level="info")
        return result

    @staticmethod
    def _is_unknown_command(exc: Exception) -> bool:
        err = getattr(exc, "error", {})
        if isinstance(err, dict):
            code = err.get("code")
            msg = str(err.get("message", "") or "")
            if code == -32601 or "Unknown command" in msg:
                return True
        return "Unknown command" in str(exc)

    @staticmethod
    def _failure_class(error: str) -> str:
        normalized = str(error or "").strip().lower()
        if not normalized:
            return "unknown"
        if "fee_insufficient" in normalized or "nocheaproute" in normalized:
            return "fee"
        if "timeout" in normalized:
            return "timeout"
        if (
            normalized.startswith("retriable_failure:")
            or "temporary_channel_failure" in normalized
            or "incorrect_cltv_expiry" in normalized
            or "noroutes" in normalized
            or "nocandidates" in normalized
        ):
            return "liquidity"
        return "unknown"

    @staticmethod
    def _failure_confidence(error: str) -> float:
        failure_class = RebalanceExecutor._failure_class(error)
        if failure_class == "liquidity":
            return 0.8
        if failure_class == "fee":
            return 0.75
        if failure_class == "timeout":
            return 0.6
        return 0.5

    def _record_failure_observation(
        self,
        *,
        result: ExecutionResult,
        amount_sats: int,
        observation_store=None,
        observation_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        store = observation_store or self.observation_store
        if store is None or not observation_context or result.success:
            return

        error = str(result.error or "").strip()
        normalized = error.lower()
        if not (
            normalized.startswith("retriable_failure:")
            or normalized == "sling_timeout"
        ):
            return

        try:
            store.record_failure(
                short_channel_id=str(
                    observation_context.get("short_channel_id", "")
                ).strip(),
                direction=int(observation_context.get("direction", -1)),
                amount_sats=int(amount_sats),
                failure_class=self._failure_class(error),
                confidence=self._failure_confidence(error),
                source_channel_id=str(
                    observation_context.get("source_channel_id", "")
                ).strip(),
                dest_channel_id=str(
                    observation_context.get("dest_channel_id", "")
                ).strip(),
                route_policy=str(
                    observation_context.get("route_policy", "")
                ).strip(),
                router_kind=str(
                    observation_context.get("router_kind", "")
                ).strip(),
                correlation_id=str(
                    observation_context.get("correlation_id", "")
                ).strip(),
            )
        except Exception as exc:
            self._log(f"failed to record segment observation: {exc}", level="debug")

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
        """Execute a rebalance through sling-once and observe completion."""
        del route
        self._ensure_observe_timeout_sec()

        result = ExecutionResult(
            amount_sats=amount_sats,
            route_type="sling",
            hops=0,
            parts=1,
            attempts=1,
        )
        if amount_sats <= 0:
            result.error = "invalid_amount"
            return result

        baseline_stats = self._scid_stats(dest_channel_id)

        # Iter2: proactively stop any stale sling-once job for this scid.
        # sling rejects new sling-once with "already a job for that scid running"
        # if a prior cycle's job is still flagged active. sling-stop is
        # idempotent (returns stopped_count=0 when nothing to stop) so it's
        # always safe to call.
        try:
            self.plugin.rpc.call("sling-stop", [dest_channel_id])
        except Exception as exc:
            self._log(
                f"sling-stop precheck failed for {dest_channel_id}: {exc}",
                level="debug",
            )

        payload = {
            "scid": dest_channel_id,
            "direction": "pull",
            "amount": amount_sats,
            "onceamount": amount_sats,
            "maxppm": self._maxppm_for_budget(amount_sats, max_fee_sats),
            "candidates": [source_channel_id],
        }

        try:
            response = self.plugin.rpc.call("sling-once", payload)
        except Exception as exc:
            if self._is_unknown_command(exc):
                result.error = "sling_unavailable"
                return result
            result.error = f"sling_preflight_error: {exc}"
            self._log(result.error, level="warn")
            return result

        if not isinstance(response, dict) or response.get("result") != "started":
            result.error = f"sling_preflight_error: unexpected response {response!r}"
            self._log(result.error, level="warn")
            return result

        result = self._wait_for_terminal_state(
            dest_channel_id=dest_channel_id,
            amount_sats=amount_sats,
            baseline_stats=baseline_stats,
            result=result,
        )
        self._record_failure_observation(
            result=result,
            amount_sats=amount_sats,
            observation_store=observation_store,
            observation_context=observation_context,
        )
        return result
