"""
Hive Hints adapter -- sole integration boundary with cl_hive.

Polls a local cl_hive RPC for a compact hint snapshot, validates and
caches it with TTL, and exposes bounded multiplicative bias factors
for the fee controller and rebalancer.

If hints are missing, stale, invalid, or the RPC fails, all lookups
silently return 1.0 (neutral / no effect).
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


class HiveHintAdapter:
    """Adapter that polls cl_hive for fleet hints and exposes bounded bias lookups."""

    VALID_CORRIDOR_ROLES = {"owner", "secondary", "contested", "none"}

    def __init__(self, plugin, ttl_override: int = 0):
        self._plugin = plugin
        self._ttl_override = ttl_override
        self._snapshot = None
        self._snapshot_fetched_at = 0

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
            ds = self._plugin.rpc.listdatastore(key=["hive", "hints"])
            entries = ds.get("datastore", [])
            if entries:
                data_str = entries[0].get("string", "")
                if data_str:
                    raw = _json.loads(data_str)
        except Exception:
            pass

        # Priority 2: Fall back to cross-plugin RPC if datastore empty
        if raw is None:
            try:
                raw = self._plugin.rpc.call("hive-export-hints")
            except Exception as e:
                self._plugin.log(f"HIVE_HINTS: poll failed: {e}", level='debug')
                return

        if not self._validate_snapshot(raw):
            self._plugin.log("HIVE_HINTS: invalid snapshot schema, ignoring", level='debug')
            return

        self._snapshot = raw
        self._snapshot_fetched_at = int(time.time())

    @staticmethod
    def _validate_snapshot(raw) -> bool:
        if not isinstance(raw, dict):
            return False
        if not isinstance(raw.get("generated_at"), (int, float)):
            return False
        if not isinstance(raw.get("hints"), dict):
            return False
        return True

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def _effective_ttl(self) -> int:
        if self._ttl_override > 0:
            return self._ttl_override
        if self._snapshot and isinstance(self._snapshot.get("ttl_seconds"), (int, float)):
            return int(self._snapshot["ttl_seconds"])
        return 900

    def is_fresh(self) -> bool:
        if self._snapshot is None:
            return False
        age = int(time.time()) - int(self._snapshot.get("generated_at", 0))
        return age <= self._effective_ttl()

    # ------------------------------------------------------------------
    # Peer hint lookup
    # ------------------------------------------------------------------

    def _get_peer_hint(self, peer_id: str) -> dict:
        if not self.is_fresh():
            return {}
        hints = self._snapshot.get("hints", {})
        return hints.get(peer_id, {})

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
        """Return drain direction: inbound_heavy|outbound_heavy|balanced."""
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
        return {
            "snapshot_fresh": self.is_fresh(),
            "snapshot_age_seconds": age,
            "hints_count": len(hints),
        }
