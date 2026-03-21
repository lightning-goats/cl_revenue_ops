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

# Per-field contribution weights
FEE_CORRIDOR_WEIGHT = 0.03   # corridor_role: +/-3%
FEE_COMPETITION_WEIGHT = 0.02  # competition_bias: +/-2%
REBAL_PREFERENCE_WEIGHT = 0.05  # rebalance_preference: +/-5%
REBAL_QUALITY_WEIGHT = 0.05    # peer_quality_score: +/-5%


class HiveHintAdapter:
    """Adapter that polls cl_hive for fleet hints and exposes bounded bias lookups."""

    def __init__(self, plugin, ttl_override: int = 0):
        self._plugin = plugin
        self._ttl_override = ttl_override
        self._snapshot = None
        self._snapshot_fetched_at = 0

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self):
        """Fetch a fresh hint snapshot from cl_hive. Fail-open on any error."""
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
