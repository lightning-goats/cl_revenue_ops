"""
Hive hints adapter -- sole integration boundary with cl_hive.

Prefers the local CLN datastore key ["hive", "hints"] for a compact
hint snapshot, falls back to the live hive-export-hints RPC when the
datastore payload is missing, stale, or invalid, then validates and
caches the snapshot with TTL.

If hints are missing, stale, invalid, or both transports fail, all
lookups silently return 1.0 (neutral / no effect).
"""

import time

# Hard-coded bias caps -- not configurable by design
MAX_FEE_BIAS = 0.10          # +/-10% max fee effect
MAX_REBALANCE_BIAS = 0.15    # +/-15% max rebalance score effect
MAX_CORRIDOR_UTILIZATION_BIAS = 0.10  # +/-10% max utilization effect

# Per-field contribution weights
FEE_CORRIDOR_WEIGHT = 0.03   # corridor_role: +/-3%
FEE_COMPETITION_WEIGHT = 0.02  # competition_bias: +/-2%
REBAL_PREFERENCE_WEIGHT = 0.05  # rebalance_preference: +/-5%
REBAL_QUALITY_WEIGHT = 0.05    # peer_quality_score: +/-5%
CORRIDOR_OWNER_UTILIZATION_WEIGHT = 0.10
CORRIDOR_SECONDARY_UTILIZATION_WEIGHT = 0.05
SEGMENT_BUCKETS = (
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
)


class HiveHintAdapter:
    """Adapter that reads cl_hive fleet hints and exposes bounded bias lookups."""

    VALID_CORRIDOR_ROLES = {"owner", "secondary", "contested", "none"}
    STALE_FALLBACK_MIN_SECONDS = 6 * 3600
    STALE_FALLBACK_TTL_MULTIPLIER = 24

    def __init__(self, plugin, ttl_override: int = 0):
        self._plugin = plugin
        self._ttl_override = ttl_override
        self._snapshot = None
        self._snapshot_fetched_at = 0
        self._using_stale_fallback = False
        self.data_service = None

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self):
        """Fetch a fresh hint snapshot. Prefers CLN datastore (fast local read),
        falls back to hive-export-hints RPC (cross-plugin, may timeout).

        cl-hive pushes hints to datastore key ["hive", "hints"] each cycle.
        Reading from datastore is a direct lightningd call with no cross-plugin
        round-trip, eliminating the timeout problem.
        """
        raw = None

        # Priority 1: Read from CLN datastore (fast, no cross-plugin RPC)
        try:
            import json as _json
            ds = self.data_service.list_datastore(["hive", "hints"]) if self.data_service else self._plugin.rpc.listdatastore(key=["hive", "hints"])
            entries = ds.get("datastore", [])
            if entries:
                entry = entries[0]
                data_str = entry.get("string", "")
                if data_str:
                    raw = _json.loads(data_str)
                else:
                    data_hex = entry.get("hex", "")
                    if data_hex:
                        raw = _json.loads(bytes.fromhex(data_hex).decode())
        except Exception:
            pass

        candidate = self._validate_and_normalize_snapshot(raw)
        stale_candidate = (
            candidate
            if candidate is not None and not self._snapshot_is_fresh(candidate)
            else None
        )
        if candidate is None or stale_candidate is not None:
            try:
                raw = self._plugin.rpc.call("hive-export-hints")
            except Exception as e:
                self._plugin.log(f"HIVE_HINTS: poll failed: {e}", level='debug')
                if stale_candidate is not None and self._snapshot_is_recent_enough_for_fallback(stale_candidate):
                    self._snapshot = stale_candidate
                    self._snapshot_fetched_at = int(time.time())
                    self._using_stale_fallback = True
                elif self._error_indicates_hive_unavailable(e):
                    self._clear_snapshot()
                return
            candidate = self._validate_and_normalize_snapshot(raw)

        if candidate is None:
            if stale_candidate is not None and self._snapshot_is_recent_enough_for_fallback(stale_candidate):
                self._snapshot = stale_candidate
                self._snapshot_fetched_at = int(time.time())
                self._using_stale_fallback = True
                return
            if self._raw_indicates_hive_unavailable(raw):
                self._clear_snapshot()
                return
            self._clear_snapshot()
            self._plugin.log("HIVE_HINTS: invalid snapshot schema, ignoring", level='debug')
            return

        if not self._snapshot_is_fresh(candidate):
            if stale_candidate is not None and self._snapshot_is_recent_enough_for_fallback(stale_candidate):
                self._snapshot = stale_candidate
                self._snapshot_fetched_at = int(time.time())
                self._using_stale_fallback = True
                return
            self._clear_snapshot()
            self._plugin.log("HIVE_HINTS: stale snapshot, ignoring", level='debug')
            return

        self._snapshot = candidate
        self._snapshot_fetched_at = int(time.time())
        self._using_stale_fallback = False

    def _clear_snapshot(self) -> None:
        self._snapshot = None
        self._snapshot_fetched_at = 0
        self._using_stale_fallback = False

    @staticmethod
    def _error_indicates_hive_unavailable(error) -> bool:
        text = str(error).lower()
        return (
            "unknown command" in text
            or "hive-export-hints" in text and "not found" in text
            or "not a hive member" in text
        )

    @staticmethod
    def _raw_indicates_hive_unavailable(raw) -> bool:
        if not isinstance(raw, dict):
            return False
        text = " ".join(str(raw.get(key) or "") for key in ("error", "stderr", "message")).lower()
        return (
            "unknown command" in text
            or "hive-export-hints" in text and "not found" in text
            or "not a hive member" in text
        )

    @staticmethod
    def _validate_snapshot(raw) -> bool:
        if not isinstance(raw, dict):
            return False
        if not isinstance(raw.get("generated_at"), (int, float)):
            return False
        if not isinstance(raw.get("hints"), dict):
            return False
        return True

    @staticmethod
    def _normalize_hint_map(hints) -> dict:
        if not isinstance(hints, dict):
            return {}
        normalized = {}
        for peer_id, hint in hints.items():
            normalized[peer_id] = dict(hint) if isinstance(hint, dict) else {}
        return normalized

    def _validate_and_normalize_snapshot(self, raw) -> dict | None:
        if not self._validate_snapshot(raw):
            return None
        snapshot = dict(raw)
        snapshot["hints"] = self._normalize_hint_map(raw.get("hints"))
        return snapshot

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def _effective_ttl_for(self, snapshot: dict | None) -> int:
        if self._ttl_override > 0:
            return self._ttl_override
        if snapshot and isinstance(snapshot.get("ttl_seconds"), (int, float)):
            return int(snapshot["ttl_seconds"])
        return 900

    def _effective_ttl(self) -> int:
        return self._effective_ttl_for(self._snapshot)

    def _snapshot_is_fresh(self, snapshot: dict | None) -> bool:
        if snapshot is None:
            return False
        age = int(time.time()) - int(snapshot.get("generated_at", 0))
        return age <= self._effective_ttl_for(snapshot)

    def is_fresh(self) -> bool:
        return self._snapshot_is_fresh(self._snapshot)

    def _snapshot_is_recent_enough_for_fallback(self, snapshot: dict | None) -> bool:
        if snapshot is None:
            return False
        age = int(time.time()) - int(snapshot.get("generated_at", 0))
        max_age = max(
            self.STALE_FALLBACK_MIN_SECONDS,
            self._effective_ttl_for(snapshot) * self.STALE_FALLBACK_TTL_MULTIPLIER,
        )
        return age <= max_age

    def is_usable(self) -> bool:
        return self.is_fresh() or self._using_stale_fallback

    # ------------------------------------------------------------------
    # Peer hint lookup
    # ------------------------------------------------------------------

    def _get_peer_hint(self, peer_id: str) -> dict:
        if not self.is_usable():
            return {}
        hints = self._snapshot.get("hints", {})
        hint = hints.get(peer_id, {})
        return hint if isinstance(hint, dict) else {}

    @staticmethod
    def _normalize_str_list(values) -> list:
        if isinstance(values, str):
            return [values] if values else []
        if not isinstance(values, list):
            return []
        normalized = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_route_segments(values) -> list:
        if not isinstance(values, list):
            return []

        normalized = []
        for value in values:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    normalized.append(text)
                continue
            if not isinstance(value, dict):
                return []
            source = str(value.get("source", "")).strip()
            destination = str(value.get("destination", "")).strip()
            if not source or not destination:
                return []
            normalized.append({"source": source, "destination": destination})
        return normalized

    def _get_section_entries(self, section_name: str, validator) -> list:
        if not self.is_usable():
            return []
        section = self._snapshot.get(section_name, [])
        if not isinstance(section, list):
            return []

        entries = []
        for raw in section:
            entry = validator(raw)
            if entry is not None:
                entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def is_hive_member(self, peer_id: str) -> bool:
        """Return True if peer is a hive fleet member. False if unavailable/stale."""
        hint = self._get_peer_hint(peer_id)
        return bool(hint.get("member", False))

    def get_corridor_role(self, peer_id: str) -> str:
        """Return validated corridor_role, or 'none' if unavailable."""
        hint = self._get_peer_hint(peer_id)
        role = hint.get("corridor_role")
        if role in self.VALID_CORRIDOR_ROLES:
            return role
        return "none"

    def get_corridor_utilization_bias(self, peer_id: str) -> float:
        """Return utilization multiplier from corridor role in [0.9, 1.1]. 1.0 if unavailable."""
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return 1.0

        confidence = hint.get("traffic_confidence")
        if not isinstance(confidence, (int, float)) or confidence <= 0:
            return 1.0
        confidence = min(confidence, 1.0)

        role = self.get_corridor_role(peer_id)
        bias = 0.0
        if role == "owner":
            bias += CORRIDOR_OWNER_UTILIZATION_WEIGHT
        elif role == "secondary":
            bias -= CORRIDOR_SECONDARY_UTILIZATION_WEIGHT

        bias *= confidence
        bias = max(-MAX_CORRIDOR_UTILIZATION_BIAS, min(MAX_CORRIDOR_UTILIZATION_BIAS, bias))
        return 1.0 + bias

    # ------------------------------------------------------------------
    # Fee bias
    # ------------------------------------------------------------------

    def get_fee_bias(self, peer_id: str) -> float:
        """Return multiplicative fee bias in [0.9, 1.1]. 1.0 if unavailable."""
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return 1.0

        confidence = hint.get("traffic_confidence")
        if not isinstance(confidence, (int, float)) or confidence <= 0:
            return 1.0
        confidence = min(confidence, 1.0)

        bias = 0.0

        role = hint.get("corridor_role")
        if role == "owner":
            bias += FEE_CORRIDOR_WEIGHT
        elif role == "secondary":
            bias -= FEE_CORRIDOR_WEIGHT

        comp = hint.get("competition_bias")
        if isinstance(comp, (int, float)):
            comp = max(-1.0, min(1.0, comp))
            bias += comp * FEE_COMPETITION_WEIGHT

        bias *= confidence
        bias = max(-MAX_FEE_BIAS, min(MAX_FEE_BIAS, bias))
        return 1.0 + bias

    # ------------------------------------------------------------------
    # Rebalance bias
    # ------------------------------------------------------------------

    def get_rebalance_bias(self, peer_id: str) -> float:
        """Return multiplicative rebalance score bias in [0.85, 1.15]. 1.0 if unavailable."""
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return 1.0

        confidence = hint.get("traffic_confidence")
        if not isinstance(confidence, (int, float)) or confidence <= 0:
            return 1.0
        confidence = min(confidence, 1.0)

        bias = 0.0

        pref = hint.get("rebalance_preference")
        if pref == "sink":
            bias += REBAL_PREFERENCE_WEIGHT
        elif pref == "source":
            bias -= REBAL_PREFERENCE_WEIGHT

        quality = hint.get("peer_quality_score")
        if isinstance(quality, (int, float)):
            quality = max(0.0, min(1.0, quality))
            bias += (quality - 0.5) * 2.0 * REBAL_QUALITY_WEIGHT

        bias *= confidence
        bias = max(-MAX_REBALANCE_BIAS, min(MAX_REBALANCE_BIAS, bias))
        return 1.0 + bias

    def get_centrality(self, peer_id: str) -> float:
        """Return external centrality for peer (0.0 if unavailable)."""
        hint = self._get_peer_hint(peer_id)
        val = hint.get("external_centrality")
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
        return 0.0

    def get_reputation_score(self, peer_id: str) -> int:
        """Return fleet-aggregated reputation score (50 if unavailable)."""
        hint = self._get_peer_hint(peer_id)
        val = hint.get("reputation_score")
        if isinstance(val, (int, float)):
            return max(0, min(100, int(val)))
        return 50

    def get_traffic_confidence(self, peer_id: str) -> float:
        """Return traffic confidence score in [0.0, 1.0] (0.0 if unavailable)."""
        hint = self._get_peer_hint(peer_id)
        val = hint.get("traffic_confidence")
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
        return 0.0

    def get_peak_hours(self, peer_id: str) -> list:
        """Return peak traffic hours UTC (empty list if unavailable)."""
        hint = self._get_peer_hint(peer_id)
        val = hint.get("peak_hours_utc")
        if isinstance(val, list):
            return [int(h) for h in val if isinstance(h, (int, float)) and 0 <= h <= 23]
        return []

    def get_drain_direction(self, peer_id: str) -> str:
        """Return drain direction for askrene/diagnostic consumers.

        The active fee controller intentionally does not use this directly;
        directional pressure is applied through cl-hive's askrene traffic
        layers to avoid double-counting the same signal.
        """
        hint = self._get_peer_hint(peer_id)
        val = hint.get("drain_direction")
        if val in ("inbound_heavy", "outbound_heavy", "balanced"):
            return val
        return "balanced"

    def get_fee_elasticity(self, peer_id: str) -> float:
        """Return estimated price elasticity (0.0 if unavailable)."""
        hint = self._get_peer_hint(peer_id)
        val = hint.get("fee_elasticity")
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    def get_optimal_fee_estimate(self, peer_id: str) -> int:
        """Return fleet-estimated optimal fee PPM (0 if unavailable)."""
        hint = self._get_peer_hint(peer_id)
        val = hint.get("optimal_fee_estimate_ppm")
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        return 0

    def get_fleet_balance(self, peer_id: str) -> dict:
        """Return fleet member balance data from hints (pushed by cl-hive).

        Returns dict with capacity_sats, available_sats, topology — or empty dict.
        Eliminates the need for a separate hive-fleet-balances RPC.
        """
        hint = self._get_peer_hint(peer_id)
        cap = hint.get("fleet_capacity_sats")
        avail = hint.get("fleet_available_sats")
        if isinstance(cap, (int, float)) and isinstance(avail, (int, float)):
            return {
                "capacity_sats": int(cap),
                "available_sats": int(avail),
                "topology": hint.get("fleet_topology", []),
            }
        return {}

    def get_fleet_topology(self, peer_id: str) -> list:
        """Return validated topology entries advertised by a hive member hint."""
        hint = self._get_peer_hint(peer_id)
        merged = []
        for value in (
            *self._normalize_str_list(hint.get("fleet_hive_topology")),
            *self._normalize_str_list(hint.get("fleet_topology")),
        ):
            if value not in merged:
                merged.append(value)
        return merged

    def get_member_peer_ids(self) -> list:
        """Return hive member peer ids represented in the current hint snapshot."""
        if not self.is_usable():
            return []
        hints = self._snapshot.get("hints", {})
        if not isinstance(hints, dict):
            return []
        return [
            str(peer_id)
            for peer_id, hint in hints.items()
            if peer_id and isinstance(hint, dict) and bool(hint.get("member", False))
        ]

    # ------------------------------------------------------------------
    # Coordination sections
    # ------------------------------------------------------------------

    def _validate_route_segment_lease(self, raw) -> dict | None:
        if not isinstance(raw, dict):
            return None
        lease_id = raw.get("lease_id")
        route_segments = raw.get("route_segments")
        if not isinstance(lease_id, str) or not lease_id.strip():
            return None
        if not isinstance(route_segments, list):
            return None
        normalized_route_segments = self._normalize_route_segments(route_segments)
        if route_segments and not normalized_route_segments:
            return None
        lease = dict(raw)
        lease["lease_id"] = lease_id.strip()
        lease["route_segments"] = normalized_route_segments
        return lease

    def _validate_rebalance_recommendation(self, raw) -> dict | None:
        if not isinstance(raw, dict):
            return None
        recommendation_id = raw.get("recommendation_id")
        if not isinstance(recommendation_id, str) or not recommendation_id.strip():
            return None
        route_segments_missing = "route_segments" not in raw
        route_segments = raw.get("route_segments", [])
        if route_segments is None:
            route_segments_missing = True
            route_segments = []
        if not isinstance(route_segments, list):
            return None
        normalized_route_segments = self._normalize_route_segments(route_segments)
        if route_segments and not normalized_route_segments:
            return None
        if route_segments_missing and not normalized_route_segments:
            has_source_sink = bool(
                str(raw.get("source_scid") or "").strip()
                and str(raw.get("sink_scid") or "").strip()
            )
            has_peer_pair = bool(
                str(raw.get("source_peer_id") or raw.get("from_peer_id") or "").strip()
                and str(
                    raw.get("destination_peer_id")
                    or raw.get("dest_peer_id")
                    or raw.get("to_peer_id")
                    or raw.get("target_peer_id")
                    or ""
                ).strip()
            )
            if not (has_source_sink or has_peer_pair):
                return None
        recommendation = dict(raw)
        recommendation["recommendation_id"] = recommendation_id.strip()
        recommendation["route_segments"] = normalized_route_segments
        return recommendation

    def _validate_rebalance_campaign(self, raw) -> dict | None:
        if not isinstance(raw, dict):
            return None
        campaign_id = str(raw.get("campaign_id", "")).strip()
        status = str(raw.get("status", "")).strip()
        if not campaign_id or not status:
            return None
        campaign = dict(raw)
        campaign["campaign_id"] = campaign_id
        campaign["status"] = status
        return campaign

    def _validate_segment_observation(self, raw) -> dict | None:
        if not isinstance(raw, dict):
            return None
        try:
            observation = {
                "observation_id": str(raw.get("observation_id", "")).strip(),
                "observer_member_id": str(raw.get("observer_member_id", "")).strip(),
                "short_channel_id": str(raw.get("short_channel_id", "")).strip(),
                "direction": int(raw.get("direction")),
                "amount_bucket_sats": int(raw.get("amount_bucket_sats", 0)),
                "outcome": str(raw.get("outcome", "")).strip(),
                "failure_class": str(raw.get("failure_class", "")).strip(),
                "confidence": float(raw.get("confidence", 0.0)),
                "observed_at": int(raw.get("observed_at", 0)),
                "source_channel_id": str(raw.get("source_channel_id", "")).strip(),
                "dest_channel_id": str(raw.get("dest_channel_id", "")).strip(),
                "route_policy": str(raw.get("route_policy", "")).strip(),
                "router_kind": str(raw.get("router_kind", "")).strip(),
                "correlation_id": str(raw.get("correlation_id", "")).strip(),
            }
        except (TypeError, ValueError):
            return None
        if (
            not observation["observation_id"]
            or not observation["short_channel_id"]
            or observation["direction"] not in (0, 1)
            or observation["amount_bucket_sats"] <= 0
            or observation["observed_at"] <= 0
        ):
            return None
        observation["outcome"] = "failure"
        observation["confidence"] = max(0.0, min(1.0, observation["confidence"]))
        return observation

    def _validate_segment_score(self, raw) -> dict | None:
        if not isinstance(raw, dict):
            return None
        try:
            score = {
                "short_channel_id": str(raw.get("short_channel_id", "")).strip(),
                "direction": int(raw.get("direction")),
                "amount_bucket_sats": int(raw.get("amount_bucket_sats", 0)),
                "success_score": float(raw.get("success_score", 0.0)),
                "failure_score": float(raw.get("failure_score", 0.0)),
                "net_utility": float(raw.get("net_utility", 0.0)),
                "confidence": float(raw.get("confidence", 0.0)),
                "observer_count": int(raw.get("observer_count", 0)),
                "last_observed_at": int(raw.get("last_observed_at", 0)),
            }
        except (TypeError, ValueError):
            return None
        if (
            not score["short_channel_id"]
            or score["direction"] not in (0, 1)
            or score["amount_bucket_sats"] <= 0
            or score["last_observed_at"] <= 0
        ):
            return None
        score["success_score"] = max(0.0, min(1.0, score["success_score"]))
        score["failure_score"] = max(0.0, min(1.0, score["failure_score"]))
        score["confidence"] = max(0.0, min(1.0, score["confidence"]))
        score["net_utility"] = max(-1.0, min(1.0, score["net_utility"]))
        return score

    @staticmethod
    def _segment_bucket(amount_sats: int) -> int:
        try:
            amount = int(amount_sats)
        except (TypeError, ValueError):
            return 0
        if amount <= 0:
            return 0
        bucket = SEGMENT_BUCKETS[0]
        for candidate in SEGMENT_BUCKETS:
            if amount < candidate:
                break
            bucket = candidate
        return bucket

    def get_route_segment_leases(self) -> list[dict]:
        """Return validated route-segment leases or [] if unavailable/stale."""
        return self._get_section_entries(
            "route_segment_leases",
            self._validate_route_segment_lease,
        )

    def get_rebalance_recommendations(self) -> list[dict]:
        """Return validated rebalance recommendations or [] if unavailable/stale."""
        return self._get_section_entries(
            "rebalance_recommendations",
            self._validate_rebalance_recommendation,
        )

    def get_rebalance_campaigns(self) -> list[dict]:
        """Return validated rebalance campaigns or [] if unavailable/stale."""
        return self._get_section_entries(
            "rebalance_campaigns",
            self._validate_rebalance_campaign,
        )

    def get_segment_observations(self) -> list[dict]:
        """Return validated segment observations or [] if unavailable/stale."""
        return self._get_section_entries(
            "segment_observations",
            self._validate_segment_observation,
        )

    def get_segment_scores(self) -> list[dict]:
        """Return validated segment scores or [] if unavailable/stale."""
        return self._get_section_entries(
            "segment_scores",
            self._validate_segment_score,
        )

    def get_segment_score(
        self,
        short_channel_id: str,
        direction: int,
        amount_sats: int | None = None,
    ) -> dict:
        """Return the best matching score for a directed channel segment."""
        normalized_scid = str(short_channel_id or "").strip().replace(":", "x")
        if not normalized_scid or direction not in (0, 1):
            return {}

        matches = [
            entry
            for entry in self.get_segment_scores()
            if entry["short_channel_id"].replace(":", "x") == normalized_scid
            and int(entry["direction"]) == int(direction)
        ]
        if not matches:
            return {}

        bucket = self._segment_bucket(amount_sats or 0)
        if bucket > 0:
            exact = [
                entry for entry in matches if entry["amount_bucket_sats"] == bucket
            ]
            if exact:
                matches = exact
            else:
                bounded = [
                    entry for entry in matches if entry["amount_bucket_sats"] <= bucket
                ]
                if bounded:
                    max_bucket = max(entry["amount_bucket_sats"] for entry in bounded)
                    matches = [
                        entry
                        for entry in bounded
                        if entry["amount_bucket_sats"] == max_bucket
                    ]

        matches.sort(
            key=lambda entry: (
                -float(entry.get("confidence", 0.0) or 0.0),
                -int(entry.get("last_observed_at", 0) or 0),
            )
        )
        return dict(matches[0])

    # ------------------------------------------------------------------
    # Channel-open hints
    # ------------------------------------------------------------------

    VALID_OPEN_PREFS = {"open", "neutral", "avoid"}
    VALID_SIZE_BUCKETS = {"small", "medium", "large"}
    VALID_OPEN_REASONS = {
        "underserved_corridor", "improve_coverage", "reduce_overlap",
        "member_connectivity", "none",
    }

    def get_channel_open_hint(self, peer_id: str) -> dict:
        """Return validated channel_open_hint for peer, or {} if unavailable/invalid."""
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return {}
        raw = hint.get("channel_open_hint")
        if not isinstance(raw, dict):
            return {}
        result = {}
        pref = raw.get("open_preference")
        if pref in self.VALID_OPEN_PREFS:
            result["open_preference"] = pref
        conf = raw.get("topology_confidence")
        if isinstance(conf, (int, float)):
            result["topology_confidence"] = max(0.0, min(1.0, float(conf)))
        bucket = raw.get("suggested_size_bucket")
        if bucket in self.VALID_SIZE_BUCKETS:
            result["suggested_size_bucket"] = bucket
        reason = raw.get("reason")
        if reason in self.VALID_OPEN_REASONS:
            result["reason"] = reason
        return result

    def get_open_candidates(self) -> list:
        """Return list of (peer_id, hint_dict) for peers with open_preference='open'."""
        if not self.is_fresh():
            return []
        results = []
        for peer_id, hint in self._snapshot.get("hints", {}).items():
            coh = hint.get("channel_open_hint")
            if not isinstance(coh, dict):
                continue
            if coh.get("open_preference") != "open":
                continue
            validated = self.get_channel_open_hint(peer_id)
            if validated.get("open_preference") == "open":
                results.append((peer_id, validated))
        return results

    # ------------------------------------------------------------------
    # Closure recommendations
    # ------------------------------------------------------------------

    def is_closure_recommended(self, peer_id: str) -> bool:
        """Return True if cl-hive reputation layer recommends closing this peer."""
        hint = self._get_peer_hint(peer_id)
        return bool(hint.get("closure_recommended", False))

    def get_closure_reason(self, peer_id: str) -> str:
        """Return closure reason string, or '' if no recommendation."""
        hint = self._get_peer_hint(peer_id)
        return str(hint.get("closure_reason", ""))

    # ------------------------------------------------------------------
    # Fleet fee prior
    # ------------------------------------------------------------------

    def get_fleet_fee_prior(self, peer_id: str) -> int | None:
        """Return fleet-observed fee median for a peer, or None."""
        hint = self._get_peer_hint(peer_id)
        if not hint:
            return None
        fee = hint.get("fleet_fee_median")
        if isinstance(fee, (int, float)) and fee > 0:
            return int(fee)
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        if self._snapshot is None:
            return {
                "snapshot_fresh": False,
                "snapshot_age_seconds": None,
                "hints_count": 0,
            }
        age = int(time.time()) - int(self._snapshot.get("generated_at", 0))
        hints = self._snapshot.get("hints", {})
        member_hints = [
            hint
            for hint in hints.values()
            if isinstance(hint, dict) and bool(hint.get("member", False))
        ]
        return {
            "snapshot_fresh": self.is_fresh(),
            "snapshot_usable": self.is_usable(),
            "stale_fallback": self._using_stale_fallback,
            "snapshot_age_seconds": age,
            "hints_count": len(hints),
            "member_hints_count": len(member_hints),
            "rebalance_recommendations_count": len(self.get_rebalance_recommendations()),
            "rebalance_campaigns_count": len(self.get_rebalance_campaigns()),
            "route_segment_leases_count": len(self.get_route_segment_leases()),
        }
