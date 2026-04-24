"""Local route-segment failure observation storage and export."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class SegmentObservationStore:
    """Thread-safe in-memory store for local failure-derived observations."""

    DATASTORE_KEY = ["revenue", "segment-observations"]
    SCHEMA_VERSION = 1
    DEFAULT_TTL_SECONDS = 900
    DEFAULT_MAX_OBSERVATIONS = 200
    BUCKETS = (
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    )

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_observations: int = DEFAULT_MAX_OBSERVATIONS,
    ) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_observations = max(1, int(max_observations))
        self._observations: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next_observation_id = 1

    @classmethod
    def bucket_amount_sats(cls, amount_sats: int) -> int:
        """Return the largest configured amount bucket not exceeding amount_sats."""
        try:
            amount = int(amount_sats)
        except (TypeError, ValueError):
            return 0
        if amount <= 0:
            return 0

        bucket = cls.BUCKETS[0]
        for candidate in cls.BUCKETS:
            if amount < candidate:
                break
            bucket = candidate
        return bucket

    def _valid_observation(self, entry: Dict[str, Any], now: int) -> Optional[Dict[str, Any]]:
        try:
            short_channel_id = str(entry.get("short_channel_id", "")).strip()
            direction = int(entry.get("direction"))
            amount_bucket_sats = int(entry.get("amount_bucket_sats", 0))
            confidence = float(entry.get("confidence", 0.0))
            observed_at = int(entry.get("observed_at", 0))
        except (TypeError, ValueError):
            return None

        if not short_channel_id or direction not in (0, 1) or amount_bucket_sats <= 0:
            return None
        if observed_at <= 0 or (now - observed_at) > self.ttl_seconds:
            return None

        failure_class = str(entry.get("failure_class", "")).strip() or "unknown"
        if failure_class not in {"liquidity", "fee", "timeout", "unknown"}:
            failure_class = "unknown"

        normalized = {
            "observation_id": str(entry.get("observation_id", "")).strip(),
            "short_channel_id": short_channel_id,
            "direction": direction,
            "amount_bucket_sats": amount_bucket_sats,
            "outcome": "failure",
            "failure_class": failure_class,
            "confidence": max(0.0, min(1.0, confidence)),
            "observed_at": observed_at,
            "source_channel_id": str(entry.get("source_channel_id", "")).strip(),
            "dest_channel_id": str(entry.get("dest_channel_id", "")).strip(),
            "route_policy": str(entry.get("route_policy", "")).strip(),
            "router_kind": str(entry.get("router_kind", "")).strip(),
            "correlation_id": str(entry.get("correlation_id", "")).strip(),
        }
        if not normalized["observation_id"]:
            return None
        return normalized

    def record_failure(
        self,
        *,
        short_channel_id: str,
        direction: int,
        amount_sats: int,
        failure_class: str,
        confidence: float,
        observed_at: Optional[int] = None,
        source_channel_id: str = "",
        dest_channel_id: str = "",
        route_policy: str = "",
        router_kind: str = "",
        correlation_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Record one local failure-derived segment observation."""
        observed_ts = int(time.time()) if observed_at is None else int(observed_at)
        bucket = self.bucket_amount_sats(amount_sats)
        if bucket <= 0:
            return None

        with self._lock:
            observation_id = f"obs-{observed_ts}-{self._next_observation_id}"
            self._next_observation_id += 1
            entry = {
                "observation_id": observation_id,
                "short_channel_id": short_channel_id,
                "direction": direction,
                "amount_bucket_sats": bucket,
                "outcome": "failure",
                "failure_class": failure_class,
                "confidence": confidence,
                "observed_at": observed_ts,
                "source_channel_id": source_channel_id,
                "dest_channel_id": dest_channel_id,
                "route_policy": route_policy,
                "router_kind": router_kind,
                "correlation_id": correlation_id,
            }
            self._observations.append(entry)
            self._observations = self._observations[-self.max_observations :]
            return dict(entry)

    def export_snapshot(
        self,
        *,
        observer_member_id: str,
        now: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Export the current validated observation snapshot."""
        now_ts = int(time.time()) if now is None else int(now)
        with self._lock:
            valid = []
            for entry in self._observations:
                normalized = self._valid_observation(entry, now_ts)
                if normalized is not None:
                    valid.append(normalized)
            valid.sort(key=lambda item: item["observed_at"], reverse=True)
            self._observations = valid[-self.max_observations :]

        return {
            "generated_at": now_ts,
            "ttl_seconds": self.ttl_seconds,
            "schema_version": self.SCHEMA_VERSION,
            "observer_member_id": str(observer_member_id or "").strip(),
            "segment_observations": valid,
        }
