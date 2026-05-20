"""
Hive hints adapter -- sole integration boundary with cl_hive.

Prefers the local CLN datastore key ["hive", "hints"] for a compact
hint snapshot, falls back to the live hive-export-hints RPC when the
datastore payload is missing, stale, or invalid, then validates and
caches the snapshot with TTL.

If hints are missing, stale, invalid, or both transports fail, all
lookups silently return 1.0 (neutral / no effect).
"""

from __future__ import annotations

import json
import threading
import time

# Hard-coded bias caps -- not configurable by design
MAX_FEE_BIAS = 0.10          # +/-10% max fee effect
MAX_REBALANCE_BIAS = 0.15    # +/-15% max rebalance score effect
MAX_CORRIDOR_UTILIZATION_BIAS = 0.10  # +/-10% max utilization effect

VALID_M2_SCOPES = {
    "legacy_seed_only",
    "channel_peers",
    "channel_and_fleet_peers",
    "all_hints",
}
DEFAULT_M2_SCOPE = "channel_and_fleet_peers"
VALID_STALE_FALLBACK_POLICIES = {
    "diagnostics_only",
    "bounded_bias",
    "full_legacy_fallback",
}
DEFAULT_STALE_FALLBACK_POLICY = "bounded_bias"
STALE_FALLBACK_BOUNDED_FIELDS = ("fee_bias", "rebalance_bias")
STALE_FALLBACK_NEUTRALIZED_FIELDS = (
    "membership",
    "corridor_utilization_bias",
    "channel_open_hint",
    "closure_recommended",
    "route_segment_leases",
    "rebalance_recommendations",
    "rebalance_campaigns",
    "segment_scores",
    "segment_observations",
)
M2_SENSITIVE_PEER_FIELDS = {
    "fee_bias",
    "rebalance_bias",
    "membership",
    "corridor_role",
    "corridor_utilization_bias",
    "peer_quality_score",
    "traffic_confidence",
    "drain_direction",
    "competition_bias",
    "fee_elasticity",
    "optimal_fee_estimate_ppm",
    "reputation_score",
    "external_centrality",
    "fleet_fee_prior",
    "fleet_balance",
    "fleet_topology",
    "channel_open_hint",
    "closure_recommended",
    "closure_reason",
}
M2_SENSITIVE_SECTIONS = {
    "route_segment_leases",
    "rebalance_recommendations",
    "rebalance_campaigns",
    "segment_scores",
}
PEER_IDENTIFIER_FIELDS = (
    "peer_id",
    "peer_ids",
    "member_id",
    "member_ids",
    "member_peer_ids",
    "source_peer_id",
    "source_member_id",
    "from_peer_id",
    "destination_peer_id",
    "destination_member_id",
    "dest_peer_id",
    "to_peer_id",
    "target_peer_id",
    "owner_member_id",
    "observer_member_id",
    "observer_peer_id",
    "primary_executor_member_id",
    "executor_member_id",
    "executor_member_ids",
    "fallback_executor_member_ids",
    "participant_peer_ids",
    "candidate_peer_ids",
)
NESTED_PEER_IDENTIFIER_SECTIONS = (
    "route_segments",
    "segments",
    "path_segments",
)

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

    def __init__(self, plugin, ttl_override: int = 0, stale_fallback_policy: str = DEFAULT_STALE_FALLBACK_POLICY):
        self._plugin = plugin
        self._ttl_override = ttl_override
        self._stale_fallback_policy = self._normalize_stale_fallback_policy(stale_fallback_policy)
        self._lock = threading.RLock()
        self._snapshot = None
        self._snapshot_fetched_at = 0
        self._using_stale_fallback = False
        self._snapshot_source = "none"
        self.data_service = None

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _read_datastore_raw(self):
        info = {
            "queried": True,
            "source": "datastore",
            "transport": "data_service" if self.data_service else "rpc.listdatastore",
            "available": False,
            "reason": "missing",
            "error": "",
        }
        try:
            ds = (
                self.data_service.list_datastore(["hive", "hints"])
                if self.data_service
                else self._plugin.rpc.listdatastore(key=["hive", "hints"])
            )
            entries = ds.get("datastore", []) if isinstance(ds, dict) else []
            if not isinstance(entries, list) or not entries:
                return None, info
            entry = entries[0] if isinstance(entries[0], dict) else {}
            data_str = entry.get("string", "")
            data_hex = entry.get("hex", "")
            if data_str:
                info["available"] = True
                info["encoding"] = "string"
                info["reason"] = ""
                return json.loads(data_str), info
            if data_hex:
                info["available"] = True
                info["encoding"] = "hex"
                info["reason"] = ""
                return json.loads(bytes.fromhex(data_hex).decode()), info
            info["reason"] = "empty_payload"
            return None, info
        except Exception as e:
            info["available"] = bool(info.get("available", False))
            info["reason"] = "read_error" if not info["available"] else "decode_error"
            info["error"] = str(e)
            return None, info

    def _read_live_export_raw(self):
        info = {
            "queried": True,
            "source": "hive_export_rpc",
            "transport": "hive-export-hints",
            "available": False,
            "reason": "",
            "error": "",
        }
        try:
            raw = self._plugin.rpc.call("hive-export-hints")
            info["available"] = raw is not None
            info["reason"] = "" if raw is not None else "empty_response"
            return raw, info
        except Exception as e:
            info["reason"] = "rpc_error"
            info["error"] = str(e)
            return None, info

    def _store_snapshot(self, snapshot: dict, source: str, *, stale_fallback: bool = False) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._snapshot_fetched_at = int(time.time())
            self._using_stale_fallback = bool(stale_fallback)
            self._snapshot_source = source

    def poll(self):
        """Fetch a fresh hint snapshot. Prefers CLN datastore (fast local read),
        falls back to hive-export-hints RPC (cross-plugin, may timeout).

        cl-hive pushes hints to datastore key ["hive", "hints"] each cycle.
        Reading from datastore is a direct lightningd call with no cross-plugin
        round-trip, eliminating the timeout problem.
        """
        raw = None
        raw_source = "none"

        # Priority 1: Read from CLN datastore (fast, no cross-plugin RPC)
        raw, datastore_info = self._read_datastore_raw()
        if raw is not None:
            raw_source = "datastore"

        candidate = self._validate_and_normalize_snapshot(raw)
        stale_candidate_source = raw_source
        stale_candidate = (
            candidate
            if candidate is not None and not self._snapshot_is_fresh(candidate)
            else None
        )
        if candidate is None or stale_candidate is not None:
            raw, export_info = self._read_live_export_raw()
            if export_info.get("error"):
                self._plugin.log(f"HIVE_HINTS: poll failed: {export_info['error']}", level='debug')
                if stale_candidate is not None and self._snapshot_is_recent_enough_for_fallback(stale_candidate):
                    self._store_snapshot(
                        stale_candidate,
                        f"{stale_candidate_source or 'datastore'}_stale_fallback",
                        stale_fallback=True,
                    )
                elif self._error_indicates_hive_unavailable(export_info.get("error", "")):
                    self._clear_snapshot()
                return
            raw_source = "hive_export_rpc"
            candidate = self._validate_and_normalize_snapshot(raw)

        if candidate is None:
            if stale_candidate is not None and self._snapshot_is_recent_enough_for_fallback(stale_candidate):
                self._store_snapshot(
                    stale_candidate,
                    f"{stale_candidate_source or 'datastore'}_stale_fallback",
                    stale_fallback=True,
                )
                return
            if self._raw_indicates_hive_unavailable(raw):
                self._clear_snapshot()
                return
            self._clear_snapshot()
            self._plugin.log("HIVE_HINTS: invalid snapshot schema, ignoring", level='debug')
            return

        if not self._snapshot_is_fresh(candidate):
            if stale_candidate is not None and self._snapshot_is_recent_enough_for_fallback(stale_candidate):
                self._store_snapshot(
                    stale_candidate,
                    f"{stale_candidate_source or 'datastore'}_stale_fallback",
                    stale_fallback=True,
                )
                return
            self._clear_snapshot()
            self._plugin.log("HIVE_HINTS: stale snapshot, ignoring", level='debug')
            return

        self._store_snapshot(candidate, raw_source or "unknown", stale_fallback=False)

    def _clear_snapshot(self) -> None:
        with self._lock:
            self._snapshot = None
            self._snapshot_fetched_at = 0
            self._using_stale_fallback = False
            self._snapshot_source = "none"

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
        with self._lock:
            return self._effective_ttl_for(self._snapshot)

    def _snapshot_is_fresh(self, snapshot: dict | None) -> bool:
        if snapshot is None:
            return False
        age = int(time.time()) - int(snapshot.get("generated_at", 0))
        return age <= self._effective_ttl_for(snapshot)

    def is_fresh(self) -> bool:
        with self._lock:
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
        with self._lock:
            return self._snapshot_is_fresh(self._snapshot) or self._using_stale_fallback

    # ------------------------------------------------------------------
    # Influence policy
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_stale_fallback_policy(policy: str) -> str:
        normalized = str(policy or DEFAULT_STALE_FALLBACK_POLICY).strip().lower()
        if normalized in VALID_STALE_FALLBACK_POLICIES:
            return normalized
        return DEFAULT_STALE_FALLBACK_POLICY

    @staticmethod
    def _snapshot_has_m2_marker(snapshot: dict | None) -> bool:
        if not isinstance(snapshot, dict):
            return False
        if "m2_scope" in snapshot:
            return True
        producer = str(snapshot.get("producer") or "").lower()
        compat = str(snapshot.get("compat_schema") or "").lower()
        schema = str(snapshot.get("schema_version") or "").lower()
        return "mycelium" in producer or "m2" in compat or "m2" in schema

    def _normalized_m2_scope_for(self, snapshot: dict | None) -> str:
        if not isinstance(snapshot, dict):
            return DEFAULT_M2_SCOPE
        raw = str(snapshot.get("m2_scope") or "").strip()
        if raw in VALID_M2_SCOPES:
            return raw
        if self._snapshot_has_m2_marker(snapshot):
            return DEFAULT_M2_SCOPE
        return "legacy_seed_only"

    @staticmethod
    def _legacy_seed_peer_ids(snapshot: dict | None) -> set:
        if not isinstance(snapshot, dict):
            return set()
        result = set()
        for key in ("legacy_seed_peer_ids", "legacy_peer_ids", "seed_peer_ids"):
            values = snapshot.get(key)
            if isinstance(values, str):
                if values.strip():
                    result.add(values.strip())
            elif isinstance(values, list):
                for value in values:
                    text = str(value or "").strip()
                    if text:
                        result.add(text)
        hints = snapshot.get("hints", {})
        if isinstance(hints, dict):
            for peer_id, hint in hints.items():
                if isinstance(hint, dict) and bool(
                    hint.get("legacy_seed_peer") or hint.get("legacy_seed")
                ):
                    result.add(str(peer_id))
        return result

    def _peer_in_m2_scope(self, peer_id: str, hint: dict, snapshot: dict | None) -> bool:
        if not self._snapshot_has_m2_marker(snapshot):
            return True
        scope = self._normalized_m2_scope_for(snapshot)
        if scope == "all_hints":
            return True
        if scope == "channel_and_fleet_peers":
            return bool(hint.get("direct_channel_peer", False) or hint.get("member", False))
        if scope == "channel_peers":
            return bool(hint.get("direct_channel_peer", False))
        if scope == "legacy_seed_only":
            return str(peer_id or "") in self._legacy_seed_peer_ids(snapshot)
        return False

    @staticmethod
    def _entry_peer_ids(entry: dict) -> list:
        peer_ids = []

        def add_value(value) -> None:
            if value is None:
                return
            if isinstance(value, list):
                for item in value:
                    add_value(item)
                return
            if isinstance(value, dict):
                collect_from_mapping(value)
                return
            text = str(value or "").strip()
            if text and text not in peer_ids:
                peer_ids.append(text)

        def collect_from_mapping(mapping: dict) -> None:
            if not isinstance(mapping, dict):
                return
            for key in PEER_IDENTIFIER_FIELDS:
                add_value(mapping.get(key))
            for key in NESTED_PEER_IDENTIFIER_SECTIONS:
                nested = mapping.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, dict):
                            collect_from_mapping(item)

        collect_from_mapping(entry)
        return peer_ids

    def _entry_in_m2_scope(self, entry: dict, snapshot: dict | None, section_name: str | None = None) -> bool:
        if not self._snapshot_has_m2_marker(snapshot):
            return True
        if self._normalized_m2_scope_for(snapshot) == "all_hints":
            return True
        peer_ids = self._entry_peer_ids(entry)
        if not peer_ids:
            # Segment scores may be route-level aggregate evidence without a
            # peer id. Peer-specific score entries are still scoped when the
            # producer includes peer_id/source_peer_id/etc.
            return section_name == "segment_scores"
        hints = snapshot.get("hints", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(hints, dict):
            return False
        for peer_id in peer_ids:
            hint = hints.get(peer_id, {})
            if not isinstance(hint, dict) or not self._peer_in_m2_scope(peer_id, hint, snapshot):
                return False
        return True

    def _behavior_field_allowed(self, field: str, snapshot: dict | None) -> bool:
        if self._snapshot_is_fresh(snapshot):
            return True
        if not self._using_stale_fallback:
            return False
        policy = self._normalize_stale_fallback_policy(self._stale_fallback_policy)
        if policy == "full_legacy_fallback":
            return True
        if policy == "bounded_bias":
            return str(field or "") in STALE_FALLBACK_BOUNDED_FIELDS
        return False

    def _m2_scope_debug_for(self, snapshot: dict | None) -> dict:
        scope = self._normalized_m2_scope_for(snapshot)
        hints = snapshot.get("hints", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(hints, dict):
            hints = {}
        m2_active = self._snapshot_has_m2_marker(snapshot)
        out_of_scope = []
        neutralized_fields = 0
        if m2_active and scope != "all_hints":
            for peer_id, hint in hints.items():
                hint = hint if isinstance(hint, dict) else {}
                if self._peer_in_m2_scope(str(peer_id), hint, snapshot):
                    continue
                out_of_scope.append(str(peer_id))
                neutralized_fields += sum(
                    1 for key in M2_SENSITIVE_PEER_FIELDS if key in hint
                )
            for section_name in M2_SENSITIVE_SECTIONS:
                section = snapshot.get(section_name, []) if isinstance(snapshot, dict) else []
                if isinstance(section, list):
                    for raw in section:
                        if isinstance(raw, dict) and not self._entry_in_m2_scope(raw, snapshot, section_name):
                            neutralized_fields += 1
        return {
            "m2_scope": scope,
            "m2_scope_enforced_by_consumer": True,
            "m2_scope_lab_only_all_hints": scope == "all_hints",
            "m2_out_of_scope_peer_count": len(out_of_scope),
            "m2_scope_neutralized_field_count": int(neutralized_fields),
        }

    def _stale_fallback_debug_for(self, active: bool | None = None) -> dict:
        policy = self._normalize_stale_fallback_policy(self._stale_fallback_policy)
        active = self._using_stale_fallback if active is None else bool(active)
        if not active:
            allowed = []
            neutralized = []
        elif policy == "full_legacy_fallback":
            allowed = ["all_legacy_behavior"]
            neutralized = []
        elif policy == "bounded_bias":
            allowed = list(STALE_FALLBACK_BOUNDED_FIELDS)
            neutralized = list(STALE_FALLBACK_NEUTRALIZED_FIELDS)
        else:
            allowed = []
            neutralized = list(STALE_FALLBACK_BOUNDED_FIELDS) + list(STALE_FALLBACK_NEUTRALIZED_FIELDS)
        return {
            "stale_fallback_active": active,
            "stale_fallback_policy": policy,
            "stale_fallback_behavior_fields_allowed": allowed,
            "stale_fallback_behavior_fields_neutralized": neutralized,
        }

    # ------------------------------------------------------------------
    # Peer hint lookup
    # ------------------------------------------------------------------

    def _get_peer_hint(self, peer_id: str, behavior_field: str = "generic") -> dict:
        with self._lock:
            snapshot = self._snapshot if isinstance(self._snapshot, dict) else None
            if not self._behavior_field_allowed(behavior_field, snapshot):
                return {}
            hints = snapshot.get("hints", {}) if snapshot else {}
            hint = hints.get(peer_id, {}) if isinstance(hints, dict) else {}
            if not isinstance(hint, dict):
                return {}
            if (
                behavior_field in M2_SENSITIVE_PEER_FIELDS
                and not self._peer_in_m2_scope(peer_id, hint, snapshot)
            ):
                return {}
            return dict(hint)

    def _get_peer_hint_fresh(self, peer_id: str, behavior_field: str = "generic") -> dict:
        with self._lock:
            snapshot = self._snapshot if isinstance(self._snapshot, dict) else None
            if not self._snapshot_is_fresh(snapshot):
                return {}
            hints = snapshot.get("hints", {}) if snapshot else {}
            hint = hints.get(peer_id, {}) if isinstance(hints, dict) else {}
            if not isinstance(hint, dict):
                return {}
            if (
                behavior_field in M2_SENSITIVE_PEER_FIELDS
                and not self._peer_in_m2_scope(peer_id, hint, snapshot)
            ):
                return {}
            return dict(hint)

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
            segment = {"source": source, "destination": destination}
            for key in PEER_IDENTIFIER_FIELDS:
                peer_value = value.get(key)
                if isinstance(peer_value, list):
                    normalized_values = []
                    for item in peer_value:
                        text = str(item or "").strip()
                        if text and text not in normalized_values:
                            normalized_values.append(text)
                    if normalized_values:
                        segment[key] = normalized_values
                else:
                    text = str(peer_value or "").strip()
                    if text:
                        segment[key] = text
            normalized.append(segment)
        return normalized

    def _get_section_entries(self, section_name: str, validator, behavior_field: str | None = None) -> list:
        with self._lock:
            field = behavior_field or section_name
            snapshot = self._snapshot if isinstance(self._snapshot, dict) else None
            if not self._behavior_field_allowed(field, snapshot):
                return []
            section = snapshot.get(section_name, []) if snapshot else []
            if not isinstance(section, list):
                return []

            entries = []
            for raw in section:
                entry = validator(raw)
                if entry is None:
                    continue
                if section_name in M2_SENSITIVE_SECTIONS and not self._entry_in_m2_scope(entry, snapshot, section_name):
                    continue
                entries.append(entry)
            return entries

    def _get_section_entries_fresh(self, section_name: str, validator, behavior_field: str | None = None) -> list:
        with self._lock:
            snapshot = self._snapshot if isinstance(self._snapshot, dict) else None
            if not self._snapshot_is_fresh(snapshot):
                return []
            section = snapshot.get(section_name, []) if snapshot else []
            if not isinstance(section, list):
                return []

            entries = []
            for raw in section:
                entry = validator(raw)
                if entry is None:
                    continue
                if section_name in M2_SENSITIVE_SECTIONS and not self._entry_in_m2_scope(entry, snapshot, section_name):
                    continue
                entries.append(entry)
            return entries

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def is_hive_member(self, peer_id: str) -> bool:
        """Return True if peer is a hive fleet member. False if unavailable/stale."""
        hint = self._get_peer_hint(peer_id, "membership")
        return bool(hint.get("member", False))

    def get_membership_status(self, peer_id: str) -> dict:
        """Return detailed membership/freshness state without changing adapter state."""
        with self._lock:
            snapshot = self._snapshot if isinstance(self._snapshot, dict) else None
            generated_at = snapshot.get("generated_at") if snapshot else None
            generated_at = int(generated_at) if isinstance(generated_at, (int, float)) else None
            age = int(time.time()) - generated_at if generated_at is not None else None
            fresh = self._snapshot_is_fresh(snapshot)
            usable = fresh or self._using_stale_fallback
            hint = self._get_peer_hint(peer_id, "membership") if usable else {}
            known = usable and isinstance(hint, dict) and "member" in hint
            return {
                "peer_id": str(peer_id or ""),
                "known": bool(known),
                "member": bool(hint.get("member", False)) if known else False,
                "fresh": bool(fresh),
                "usable": bool(usable),
                "stale_fallback": bool(self._using_stale_fallback),
                "source": self._snapshot_source,
                "generation": self._snapshot_generation(snapshot),
                "generated_at": generated_at,
                "age_seconds": age,
                "effective_ttl_seconds": self._effective_ttl_for(snapshot),
            }

    def get_corridor_role(self, peer_id: str) -> str:
        """Return validated corridor_role, or 'none' if unavailable."""
        hint = self._get_peer_hint(peer_id, "corridor_role")
        role = hint.get("corridor_role")
        if role in self.VALID_CORRIDOR_ROLES:
            return role
        return "none"

    def get_corridor_utilization_bias(self, peer_id: str) -> float:
        """Return utilization multiplier from corridor role in [0.9, 1.1]. 1.0 if unavailable."""
        hint = self._get_peer_hint(peer_id, "corridor_utilization_bias")
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
        hint = self._get_peer_hint(peer_id, "fee_bias")
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
        hint = self._get_peer_hint(peer_id, "rebalance_bias")
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

    def get_peer_quality_score(self, peer_id: str) -> float:
        """Return peer quality score in [0.0, 1.0], or 0.5 when unavailable."""
        hint = self._get_peer_hint(peer_id, "peer_quality_score")
        val = hint.get("peer_quality_score")
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
        return 0.5

    def get_centrality(self, peer_id: str) -> float:
        """Return external centrality for peer (0.0 if unavailable)."""
        hint = self._get_peer_hint(peer_id, "external_centrality")
        val = hint.get("external_centrality")
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
        return 0.0

    def get_reputation_score(self, peer_id: str) -> int:
        """Return fleet-aggregated reputation score (50 if unavailable)."""
        hint = self._get_peer_hint(peer_id, "reputation_score")
        val = hint.get("reputation_score")
        if isinstance(val, (int, float)):
            return max(0, min(100, int(val)))
        return 50

    def get_traffic_confidence(self, peer_id: str) -> float:
        """Return traffic confidence score in [0.0, 1.0] (0.0 if unavailable)."""
        hint = self._get_peer_hint(peer_id, "traffic_confidence")
        val = hint.get("traffic_confidence")
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
        return 0.0

    def get_peak_hours(self, peer_id: str) -> list:
        """Return peak traffic hours UTC (empty list if unavailable)."""
        hint = self._get_peer_hint(peer_id, "traffic_confidence")
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
        hint = self._get_peer_hint(peer_id, "drain_direction")
        val = hint.get("drain_direction")
        if val in ("inbound_heavy", "outbound_heavy", "balanced"):
            return val
        return "balanced"

    def get_fee_elasticity(self, peer_id: str) -> float:
        """Return estimated price elasticity (0.0 if unavailable)."""
        hint = self._get_peer_hint(peer_id, "fee_elasticity")
        val = hint.get("fee_elasticity")
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    def get_optimal_fee_estimate(self, peer_id: str) -> int:
        """Return fleet-estimated optimal fee PPM (0 if unavailable)."""
        hint = self._get_peer_hint(peer_id, "optimal_fee_estimate_ppm")
        val = hint.get("optimal_fee_estimate_ppm")
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        return 0

    def get_fleet_balance(self, peer_id: str) -> dict:
        """Return fleet member balance data from hints (pushed by cl-hive).

        Returns dict with capacity_sats, available_sats, topology — or empty dict.
        Eliminates the need for a separate hive-fleet-balances RPC.
        """
        hint = self._get_peer_hint(peer_id, "fleet_balance")
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
        hint = self._get_peer_hint(peer_id, "fleet_topology")
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
        with self._lock:
            snapshot = self._snapshot if isinstance(self._snapshot, dict) else None
            if not self._behavior_field_allowed("membership", snapshot):
                return []
            hints = snapshot.get("hints", {}) if snapshot else {}
            if not isinstance(hints, dict):
                return []
            return [
                str(peer_id)
                for peer_id, hint in hints.items()
                if peer_id
                and isinstance(hint, dict)
                and bool(hint.get("member", False))
                and self._peer_in_m2_scope(str(peer_id), hint, snapshot)
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
            for key in ("peer_id", "source_peer_id", "destination_peer_id", "dest_peer_id", "to_peer_id", "target_peer_id"):
                value = str(raw.get(key) or "").strip()
                if value:
                    score[key] = value
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

    def get_route_segment_leases_fresh(self) -> list[dict]:
        """Return route-segment leases only from a fresh snapshot."""
        return self._get_section_entries_fresh(
            "route_segment_leases",
            self._validate_route_segment_lease,
        )

    def get_rebalance_recommendations(self) -> list[dict]:
        """Return validated rebalance recommendations or [] if unavailable/stale."""
        return self._get_section_entries(
            "rebalance_recommendations",
            self._validate_rebalance_recommendation,
        )

    def get_rebalance_recommendations_fresh(self) -> list[dict]:
        """Return rebalance recommendations only from a fresh snapshot."""
        return self._get_section_entries_fresh(
            "rebalance_recommendations",
            self._validate_rebalance_recommendation,
        )

    def get_rebalance_campaigns(self) -> list[dict]:
        """Return validated rebalance campaigns or [] if unavailable/stale."""
        return self._get_section_entries(
            "rebalance_campaigns",
            self._validate_rebalance_campaign,
        )

    def get_rebalance_campaigns_fresh(self) -> list[dict]:
        """Return rebalance campaigns only from a fresh snapshot."""
        return self._get_section_entries_fresh(
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
        "member_connectivity", "organism_high_risk_suppression", "none",
    }

    def get_channel_open_hint(self, peer_id: str) -> dict:
        """Return validated channel_open_hint for peer, or {} if unavailable/invalid."""
        hint = self._get_peer_hint(peer_id, "channel_open_hint")
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
        hint = self._get_peer_hint(peer_id, "closure_recommended")
        return bool(hint.get("closure_recommended", False))

    def is_closure_recommended_fresh(self, peer_id: str) -> bool:
        """Return closure recommendation only from a fresh snapshot."""
        hint = self._get_peer_hint_fresh(peer_id, "closure_recommended")
        return bool(hint.get("closure_recommended", False))

    def get_closure_reason(self, peer_id: str) -> str:
        """Return closure reason string, or '' if no recommendation."""
        hint = self._get_peer_hint(peer_id, "closure_reason")
        return str(hint.get("closure_reason", ""))

    def get_closure_reason_fresh(self, peer_id: str) -> str:
        """Return closure reason only from a fresh snapshot."""
        hint = self._get_peer_hint_fresh(peer_id, "closure_reason")
        return str(hint.get("closure_reason", ""))

    # ------------------------------------------------------------------
    # Fleet fee prior
    # ------------------------------------------------------------------

    def get_fleet_fee_prior(self, peer_id: str) -> int | None:
        """Return fleet-observed fee median for a peer, or None."""
        hint = self._get_peer_hint(peer_id, "fleet_fee_prior")
        if not hint:
            return None
        fee = hint.get("fleet_fee_median")
        if isinstance(fee, (int, float)) and fee > 0:
            return int(fee)
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_generation(raw) -> int | str | None:
        if not isinstance(raw, dict):
            return None
        for key in ("generation", "refresh_generation", "hint_generation"):
            value = raw.get(key)
            if value not in (None, ""):
                return int(value) if isinstance(value, (int, float)) else str(value)
        for section_key in ("runtime", "metadata", "diagnostics"):
            section = raw.get(section_key)
            if not isinstance(section, dict):
                continue
            for key in ("generation", "refresh_generation", "hint_generation"):
                value = section.get(key)
                if value not in (None, ""):
                    return int(value) if isinstance(value, (int, float)) else str(value)
        return None

    def _snapshot_debug_view(self, raw, source: str, transport_info: dict | None = None) -> dict:
        info = dict(transport_info or {})
        available = bool(info.get("available", raw is not None))
        generated_at = raw.get("generated_at") if isinstance(raw, dict) else None
        generated_at = int(generated_at) if isinstance(generated_at, (int, float)) else None
        ttl = self._effective_ttl_for(raw if isinstance(raw, dict) else None)
        age = int(time.time()) - generated_at if generated_at is not None else None
        hints = raw.get("hints") if isinstance(raw, dict) else None
        candidate = self._validate_and_normalize_snapshot(raw)
        valid = candidate is not None
        fresh = bool(valid and age is not None and age <= ttl)
        stale_fallback_usable = bool(valid and self._snapshot_is_recent_enough_for_fallback(candidate))
        if not available:
            reason = str(info.get("reason") or "missing")
        elif not valid:
            reason = str(info.get("reason") or "invalid_schema")
            if reason in {"", "missing"}:
                reason = "invalid_schema"
        elif fresh:
            reason = "fresh"
        else:
            reason = "stale"
        result = {
            "queried": bool(info.get("queried", False)),
            "source": source,
            "transport": info.get("transport"),
            "available": available,
            "valid": valid,
            "fresh": fresh,
            "usable": fresh,
            "stale_fallback_usable": stale_fallback_usable,
            "reason": reason,
            "error": str(info.get("error") or ""),
            "generation": self._snapshot_generation(raw),
            "generated_at": generated_at,
            "age_seconds": age,
            "ttl_seconds": ttl,
            "effective_ttl_seconds": ttl,
            "hints_count": len(hints) if isinstance(hints, dict) else 0,
        }
        result.update(self._m2_scope_debug_for(candidate if candidate is not None else raw))
        result.update(self._stale_fallback_debug_for(active=False))
        return result

    def _cache_debug_view(self) -> dict:
        with self._lock:
            cache = self._snapshot_debug_view(
                self._snapshot,
                self._snapshot_source,
                {
                    "queried": False,
                    "available": self._snapshot is not None,
                    "transport": "adapter_cache",
                    "reason": "missing" if self._snapshot is None else "",
                },
            )
            cache["fresh"] = self._snapshot_is_fresh(self._snapshot)
            cache["usable"] = self._snapshot_is_fresh(self._snapshot) or self._using_stale_fallback
            cache["stale_fallback"] = self._using_stale_fallback
            cache["fetched_at"] = int(self._snapshot_fetched_at or 0) or None
            cache["cache_age_seconds"] = (
                int(time.time()) - int(self._snapshot_fetched_at)
                if self._snapshot_fetched_at
                else None
            )
            cache.update(self._m2_scope_debug_for(self._snapshot))
            cache.update(self._stale_fallback_debug_for())
            return cache

    @staticmethod
    def _fallback_reason_from_probe(prefix: str, probe: dict) -> str:
        reason = str(probe.get("reason") or "")
        if not bool(probe.get("available", False)):
            return f"{prefix}_missing"
        if not bool(probe.get("valid", False)):
            return f"{prefix}_invalid_schema"
        if not bool(probe.get("fresh", False)):
            return f"{prefix}_stale"
        return reason or f"{prefix}_unusable"

    def refresh_status_for_debug(self) -> dict:
        """Refresh stale adapter cache for diagnostics and report each transport separately.

        This only reads cl-hive state and updates the in-memory adapter snapshot,
        matching poll() transport behavior without changing budgets or execution
        decisions.
        """
        cache_before = self._cache_debug_view()
        live_export_default = {
            "queried": False,
            "source": "hive_export_rpc",
            "transport": "hive-export-hints",
            "available": False,
            "valid": False,
            "fresh": False,
            "usable": False,
            "stale_fallback_usable": False,
            "reason": "not_queried",
            "error": "",
            "generation": None,
            "generated_at": None,
            "age_seconds": None,
            "ttl_seconds": None,
            "effective_ttl_seconds": None,
            "hints_count": 0,
        }
        diagnostics = {
            "cache": cache_before,
            "cache_after_refresh": cache_before,
            "refresh_needed": not bool(cache_before.get("fresh", False)),
            "refresh_attempted": False,
            "refresh_result": "cache_fresh" if cache_before.get("fresh") else "not_attempted",
            "live_datastore": {
                "queried": False,
                "source": "datastore",
                "transport": "data_service" if self.data_service else "rpc.listdatastore",
                "available": False,
                "valid": False,
                "fresh": False,
                "usable": False,
                "stale_fallback_usable": False,
                "reason": "not_queried",
                "error": "",
                "generation": None,
                "generated_at": None,
                "age_seconds": None,
                "ttl_seconds": None,
                "effective_ttl_seconds": None,
                "hints_count": 0,
            },
            "live_hive_export": dict(live_export_default),
            "fallback": {
                "needed": False,
                "reason": "",
                "used": False,
                "used_source": "",
                "stale_fallback_used": False,
                "stale_fallback_reason": "",
            },
        }
        if cache_before.get("fresh"):
            return diagnostics

        diagnostics["refresh_attempted"] = True
        raw_datastore, datastore_info = self._read_datastore_raw()
        datastore_probe = self._snapshot_debug_view(raw_datastore, "datastore", datastore_info)
        diagnostics["live_datastore"] = datastore_probe
        datastore_candidate = self._validate_and_normalize_snapshot(raw_datastore)
        if datastore_candidate is not None and datastore_probe["fresh"]:
            self._store_snapshot(datastore_candidate, "datastore", stale_fallback=False)
            diagnostics["refresh_result"] = "refreshed_from_datastore"
            diagnostics["cache_after_refresh"] = self._cache_debug_view()
            return diagnostics

        fallback_reason = self._fallback_reason_from_probe("datastore", datastore_probe)
        diagnostics["fallback"]["needed"] = True
        diagnostics["fallback"]["reason"] = fallback_reason

        raw_export, export_info = self._read_live_export_raw()
        export_probe = self._snapshot_debug_view(raw_export, "hive_export_rpc", export_info)
        diagnostics["live_hive_export"] = export_probe
        export_candidate = self._validate_and_normalize_snapshot(raw_export)
        if export_candidate is not None and export_probe["fresh"]:
            self._store_snapshot(export_candidate, "hive_export_rpc", stale_fallback=False)
            diagnostics["fallback"]["used"] = True
            diagnostics["fallback"]["used_source"] = "hive_export_rpc"
            diagnostics["refresh_result"] = "refreshed_from_hive_export"
            diagnostics["cache_after_refresh"] = self._cache_debug_view()
            return diagnostics

        if datastore_candidate is not None and self._snapshot_is_recent_enough_for_fallback(datastore_candidate):
            self._store_snapshot(datastore_candidate, "datastore_stale_fallback", stale_fallback=True)
            diagnostics["fallback"]["used"] = True
            diagnostics["fallback"]["used_source"] = "datastore_stale_fallback"
            diagnostics["fallback"]["stale_fallback_used"] = True
            diagnostics["fallback"]["stale_fallback_reason"] = self._fallback_reason_from_probe(
                "hive_export",
                export_probe,
            )
            diagnostics["refresh_result"] = "using_stale_datastore_fallback"
            diagnostics["cache_after_refresh"] = self._cache_debug_view()
            return diagnostics

        if self._raw_indicates_hive_unavailable(raw_export) or self._error_indicates_hive_unavailable(
            export_info.get("error", "")
        ):
            self._clear_snapshot()
            diagnostics["refresh_result"] = "hive_unavailable"
        elif export_info.get("error"):
            diagnostics["refresh_result"] = "refresh_failed_cache_retained"
        else:
            self._clear_snapshot()
            diagnostics["refresh_result"] = "invalid_live_hive_export"
        diagnostics["cache_after_refresh"] = self._cache_debug_view()
        return diagnostics

    def _status_from_cache(self) -> dict:
        with self._lock:
            if self._snapshot is None:
                return {
                    "snapshot_fresh": False,
                    "snapshot_usable": False,
                    "stale_fallback": False,
                    "snapshot_age_seconds": None,
                    "snapshot_generated_at": None,
                    "snapshot_fetched_at": None,
                    "adapter_cache_age_seconds": None,
                    "effective_ttl_seconds": self._effective_ttl_for(None),
                    "snapshot_source": self._snapshot_source,
                    "hints_count": 0,
                    **self._m2_scope_debug_for(None),
                    **self._stale_fallback_debug_for(active=False),
                }
            age = int(time.time()) - int(self._snapshot.get("generated_at", 0))
            cache_age = int(time.time()) - int(self._snapshot_fetched_at or 0) if self._snapshot_fetched_at else None
            hints = self._snapshot.get("hints", {})
            member_hints = [
                hint
                for hint in hints.values()
                if isinstance(hint, dict) and bool(hint.get("member", False))
            ]
            status = {
                "snapshot_fresh": self._snapshot_is_fresh(self._snapshot),
                "snapshot_usable": self._snapshot_is_fresh(self._snapshot) or self._using_stale_fallback,
                "stale_fallback": self._using_stale_fallback,
                "snapshot_age_seconds": age,
                "snapshot_generated_at": int(self._snapshot.get("generated_at", 0) or 0),
                "snapshot_fetched_at": int(self._snapshot_fetched_at or 0),
                "adapter_cache_age_seconds": cache_age,
                "effective_ttl_seconds": self._effective_ttl_for(self._snapshot),
                "snapshot_source": self._snapshot_source,
                "hints_count": len(hints),
                "member_hints_count": len(member_hints),
                "rebalance_recommendations_count": len(self._get_section_entries("rebalance_recommendations", self._validate_rebalance_recommendation)),
                "rebalance_campaigns_count": len(self._get_section_entries("rebalance_campaigns", self._validate_rebalance_campaign)),
                "route_segment_leases_count": len(self._get_section_entries("route_segment_leases", self._validate_route_segment_lease)),
            }
            status.update(self._m2_scope_debug_for(self._snapshot))
            status.update(self._stale_fallback_debug_for())
            return status

    @staticmethod
    def _debug_refresh_result_for_status(refresh_result: str) -> str:
        return {
            "refreshed_from_hive_export": "refreshed_from_export",
            "using_stale_datastore_fallback": "stale_fallback",
            "invalid_live_hive_export": "invalid_snapshot",
            "refresh_failed_cache_retained": "no_usable_hints",
        }.get(str(refresh_result or ""), str(refresh_result or ""))

    def _merge_debug_refresh_status(self, status: dict, diagnostics: dict) -> None:
        cache = diagnostics.get("cache", {}) if isinstance(diagnostics, dict) else {}
        datastore = diagnostics.get("live_datastore", {}) if isinstance(diagnostics, dict) else {}
        export = diagnostics.get("live_hive_export", {}) if isinstance(diagnostics, dict) else {}
        fallback = diagnostics.get("fallback", {}) if isinstance(diagnostics, dict) else {}
        if not isinstance(cache, dict):
            cache = {}
        if not isinstance(datastore, dict):
            datastore = {}
        if not isinstance(export, dict):
            export = {}
        if not isinstance(fallback, dict):
            fallback = {}

        fallback_reason = str(fallback.get("reason") or "")
        if fallback.get("stale_fallback_used"):
            fallback_reason = str(fallback.get("stale_fallback_reason") or fallback_reason)

        status.update({
            "cached_snapshot_age_seconds": cache.get("age_seconds"),
            "cached_snapshot_effective_ttl": cache.get("effective_ttl_seconds"),
            "cached_snapshot_usable": bool(cache.get("usable", False)),
            "live_datastore_age_seconds": datastore.get("age_seconds"),
            "live_datastore_ttl_seconds": datastore.get("ttl_seconds"),
            "live_datastore_usable": bool(datastore.get("usable", False)),
            "live_datastore_generation": datastore.get("generation"),
            "live_export_age_seconds": export.get("age_seconds"),
            "live_export_ttl_seconds": export.get("ttl_seconds"),
            "live_export_usable": bool(export.get("usable", False)),
            "live_export_generation": export.get("generation"),
            "refresh_attempted": bool(diagnostics.get("refresh_attempted", False)),
            "refresh_result": self._debug_refresh_result_for_status(
                str(diagnostics.get("refresh_result") or "")
            ),
            "fallback_reason": fallback_reason,
        })

    def get_status(self, live_refresh: bool = True) -> dict:
        diagnostics = None
        if live_refresh:
            try:
                with self._lock:
                    refresh_needed = not self._snapshot_is_fresh(self._snapshot)
                if refresh_needed:
                    diagnostics = self.refresh_status_for_debug()
            except Exception as e:
                diagnostics = {
                    "refresh_attempted": True,
                    "refresh_result": "status_refresh_error",
                    "fallback": {"reason": str(e)},
                }

        status = self._status_from_cache()
        if isinstance(diagnostics, dict):
            self._merge_debug_refresh_status(status, diagnostics)
        return status
