"""LN+ (lightningnetwork.plus) liquidity swap automation.

Three collaborators, wired in cl-revenue-ops.py init:
  LNPlusClient   — stdlib-urllib HTTPS client + signmessage auth
  SwapEvaluator  — pre-application gate chain (spec gates 0-9)
  SwapLifecycle  — obligations watcher / state machine (spec gates 10-14)

Design spec: docs/plans/2026-07-05-lnplus-swap-automation-design.md
Application to a swap is an IRREVERSIBLE COMMITMENT (48h open deadline once
filled). Every gate lives before create_application; everything after only
executes obligations safely.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_URL = "https://lightningnetwork.plus/api/2"
_MAX_RESPONSE_BYTES = 1_000_000
_CHALLENGE_MAX_LEN = 500
_CHALLENGE_FORBIDDEN_PREFIXES = ("lnbc", "lntb", "lnbcrt", "cbor", "psbt")
_PUBKEY_RE = re.compile(r"^0[23][0-9a-fA-F]{64}$")


def _valid_pubkey(value) -> bool:
    return isinstance(value, str) and bool(_PUBKEY_RE.match(value))


def _parse_ts(value) -> Optional[int]:
    """ISO-8601 string or epoch number -> epoch seconds, else None."""
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return None
    return None


class LNPlusError(Exception):
    def __init__(self, message: str, http_status: Optional[int] = None):
        super().__init__(message)
        self.http_status = http_status


class LNPlusClient:
    """Thin HTTPS client for the LN+ v2 API. No business logic."""

    def __init__(self, plugin, rpc, base_url: str = BASE_URL, timeout_seconds: int = 30):
        self._plugin = plugin
        self.rpc = rpc
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    # -- transport -------------------------------------------------------
    def _request(self, path: str, params: Optional[Dict] = None, method: str = "GET") -> Any:
        url = f"{self._base_url}/{path}"
        data = None
        headers = {"Accept": "application/json", "User-Agent": "cl-revenue-ops"}
        if method == "POST":
            data = urllib.parse.urlencode(params or {}).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read(_MAX_RESPONSE_BYTES)
            except Exception:
                pass
            raise LNPlusError(f"LN+ HTTP {e.code} on {path}: {body[:500]!r}",
                              http_status=e.code)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise LNPlusError(f"LN+ unreachable on {path}: {e}")
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LNPlusError(f"LN+ response too large on {path}")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as e:
            raise LNPlusError(f"LN+ invalid JSON on {path}: {e}")

    # -- auth --------------------------------------------------------------
    def _auth_params(self) -> Dict[str, str]:
        challenge = self._request("get_message")
        message = challenge.get("message") if isinstance(challenge, dict) else None
        self._validate_challenge(message)
        signed = self.rpc.signmessage(message)
        signature = signed.get("zbase") if isinstance(signed, dict) else None
        if not signature:
            raise LNPlusError("signmessage returned no zbase signature")
        return {"message": message, "signature": signature}

    @staticmethod
    def _validate_challenge(message) -> None:
        """Gate 15: only sign strings that look like LN+ auth challenges."""
        if not isinstance(message, str) or not message:
            raise LNPlusError("LN+ challenge missing")
        if len(message) > _CHALLENGE_MAX_LEN:
            raise LNPlusError("LN+ challenge suspiciously long — refusing to sign")
        if not message.isprintable():
            raise LNPlusError("LN+ challenge not printable — refusing to sign")
        lowered = message.lower()
        for prefix in _CHALLENGE_FORBIDDEN_PREFIXES:
            if lowered.startswith(prefix):
                raise LNPlusError("LN+ challenge looks like an invoice/PSBT — refusing to sign")

    # -- endpoints ---------------------------------------------------------
    def get_applicable_swaps(self) -> List[Dict]:
        result = self._request("get_applicable_swaps", self._auth_params(), method="POST")
        swaps = result.get("swaps", result) if isinstance(result, dict) else result
        return swaps if isinstance(swaps, list) else []

    def get_swap(self, swap_id) -> Dict:
        return self._request(f"get_swap/id={urllib.parse.quote(str(swap_id), safe='')}")

    def get_my_swaps(self) -> Dict:
        result = self._request("get_my_swaps", self._auth_params(), method="POST")
        if not isinstance(result, dict):
            raise LNPlusError("get_my_swaps: unexpected payload")
        return {k: result.get(k) or [] for k in ("pending", "opening", "completed")}

    def create_application(self, swap_id) -> Dict:
        params = self._auth_params()
        params["id"] = str(swap_id)
        return self._request("create_application", params, method="POST")

    def delete_application(self, swap_id) -> Dict:
        params = self._auth_params()
        params["id"] = str(swap_id)
        return self._request("delete_application", params, method="POST")

    def complete_application(self, swap_id) -> Dict:
        params = self._auth_params()
        params["id"] = str(swap_id)
        return self._request("complete_application", params, method="POST")

    def create_rating(self, swap_id, rating: str) -> Dict:
        if rating not in ("positive", "negative"):
            raise LNPlusError(f"invalid rating {rating!r}")
        params = self._auth_params()
        params["id"] = str(swap_id)
        params["rating"] = rating
        return self._request("create_rating", params, method="POST")


NEG_RATIO_MAX = 0.10
TOR_RELIABILITY = 0.8
RELIABILITY_FLOOR = 0.6
P_UNDERPERFORM = 0.3
BOLTZ_REPLACEMENT_RATE = 0.005
SCORE_FLOOR = 0.5
_IDENTIFIERS = ("A", "B", "C", "D", "E")


class SwapEvaluator:
    """Pre-application gate chain (spec gates 0-9). At most one apply per cycle."""

    def __init__(self, plugin, rpc, database, config, client, planner, lifecycle):
        self._plugin = plugin
        self.rpc = rpc
        self._db = database
        self._config = config
        self._client = client
        self._planner = planner
        self._lifecycle = lifecycle

    def run_cycle(self, cfg, best_regular_ev: float) -> Dict:
        summary = {"applied": False, "recommended": False, "swap_id": None,
                   "swap_ev": 0.0, "best_regular_ev": best_regular_ev,
                   "rejections": []}

        if not getattr(cfg, "lnplus_swaps_enabled", False):
            return summary
        breaker = self._lifecycle.breaker_tripped()
        if breaker:
            self._plugin.log(f"LNPLUS: breaker tripped ({breaker}) — no applications", level="warn")
            return summary
        if self._lifecycle.has_inflight():
            self._plugin.log("LNPLUS: swap in flight — serialization gate holds", level="debug")
            return summary
        if not self._feerate_ok(cfg, summary):
            return summary
        if not self._lifecycle.reconcile_ok():
            self._plugin.log("LNPLUS: reconciliation preflight failed — no applications", level="warn")
            return summary

        try:
            swaps = self._client.get_applicable_swaps()
        except LNPlusError as e:
            self._plugin.log(f"LNPLUS: get_applicable_swaps failed: {e}", level="warn")
            return summary

        qualifying = []
        for swap in swaps:
            reason = self._filter_swap(swap, cfg)
            if reason is None:
                reason = self._check_participants(swap, cfg)
                if reason is not None:
                    self._reject(summary, swap, reason)
                else:
                    qualifying.append(swap)
            else:
                self._reject(summary, swap, reason)

        # EV + apply/recommend implemented in the ranking task.
        return self._select_and_apply(qualifying, cfg, best_regular_ev, summary)

    # -- gates ---------------------------------------------------------
    def _feerate_ok(self, cfg, summary) -> bool:
        try:
            opening = int(self.rpc.feerates("perkw")["perkw"]["opening"])
        except Exception as e:
            self._plugin.log(f"LNPLUS: feerates unavailable ({e}) — skipping cycle", level="warn")
            return False
        if opening > int(cfg.lnplus_apply_feerate_ceiling):
            self._plugin.log(
                f"LNPLUS: opening feerate {opening} perkw above ceiling "
                f"{cfg.lnplus_apply_feerate_ceiling} — no applications", level="info")
            return False
        return True

    def _filter_swap(self, swap, cfg) -> Optional[str]:
        """Gate 3 (fill state) + gate 4 (terms). Returns rejection 'gate:reason' or None."""
        if swap.get("status") != "pending":
            return "fill_state:not pending"
        if int(swap.get("participant_waiting_for_count") or 0) != 1:
            return "fill_state:not the last open slot"
        capacity = int(swap.get("capacity_sats") or 0)
        min_chan = getattr(cfg, "planner_min_channel_sats", 0) or 0
        if capacity < min_chan:
            return f"terms:capacity {capacity} below planner min {min_chan}"
        if int(swap.get("duration_months") or 99) > int(cfg.lnplus_max_duration_months):
            return "terms:duration exceeds cap"
        if int(swap.get("participant_max_count") or 99) > int(cfg.lnplus_max_participants):
            return "terms:too many participants"
        if str(swap.get("platform") or "any").lower() == "lnd":
            return "terms:LND/BOS swap — we are CLN"
        return None

    def _check_participants(self, swap, cfg) -> Optional[str]:
        """Gate 5 (peer quality, one bad peer vetoes) + gate 6 (fleet dedup)."""
        fleet = {pk.strip() for pk in (cfg.lnplus_fleet_pubkeys or "").split(",") if pk.strip()}
        our_id = None
        try:
            our_id = self.rpc.getinfo().get("id")
        except Exception:
            pass
        participants = swap.get("participants") or []
        if not participants:
            return "peer_quality:no visible participants"
        for p in participants:
            pk = p.get("pubkey")
            if not _valid_pubkey(pk):
                return "peer_quality:invalid participant pubkey"
            if pk in fleet or (our_id and pk == our_id):
                return "fleet_dedup:fleet node already in swap"
            pos = int(p.get("positive_ratings_count") or 0)
            neg = int(p.get("negative_ratings_count") or 0)
            if pos < int(cfg.lnplus_min_peer_positive_ratings):
                return f"peer_quality:{pk[:16]} has {pos} positive ratings (< floor)"
            if pos + neg > 0 and neg / (pos + neg) > NEG_RATIO_MAX:
                return f"peer_quality:{pk[:16]} negative ratio too high"
            if not (p.get("address_1") or p.get("address_2")):
                return f"peer_quality:{pk[:16]} publishes no address"
            local = self._db.lnplus_get_peer(pk)
            if local and local.get("defections", 0) > 0:
                return f"peer_quality:{pk[:16]} defected on us before"
            # Spec gate 5: planner reputation scoring must not veto the peer.
            # _score_candidate multiplicatively enriches a base score, so with
            # base 1.0 a result below SCORE_FLOOR means reputation/uptime/profit
            # history cut the peer roughly in half.
            try:
                enriched = float(self._planner._score_candidate(pk, 1.0))
            except Exception:
                enriched = 1.0   # no history is not a veto
            if enriched < SCORE_FLOOR:
                return f"peer_quality:{pk[:16]} planner score {enriched:.2f} below floor"
        return None

    def _infer_assignment(self, swap) -> Dict:
        """LN+ convention: each participant opens to the next letter; last wraps to A.
        We join as the next free identifier. Authoritative assignment is re-read
        from get_my_swaps at open time; this is for pre-apply EV only."""
        participants = {p.get("participant_identifier"): p
                        for p in (swap.get("participants") or [])}
        ours = next(i for i in _IDENTIFIERS if i not in participants)
        count = int(swap.get("participant_max_count") or (len(participants) + 1))
        letters = list(_IDENTIFIERS[:count])
        idx = letters.index(ours)
        outbound_id = letters[(idx + 1) % len(letters)]
        incoming_id = letters[(idx - 1) % len(letters)]
        return {
            "our_identifier": ours,
            "outbound_peer": (participants.get(outbound_id) or {}).get("pubkey"),
            "incoming_peer": (participants.get(incoming_id) or {}).get("pubkey"),
        }

    @staticmethod
    def _reject(summary, swap, gate_reason: str) -> None:
        gate, _, reason = gate_reason.partition(":")
        summary["rejections"].append(
            {"swap_id": swap.get("id"), "gate": gate, "reason": reason})

    # -- placeholder until the ranking task ------------------------------
    def _select_and_apply(self, qualifying, cfg, best_regular_ev, summary) -> Dict:
        return summary
