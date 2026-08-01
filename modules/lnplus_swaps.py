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
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

BASE_URL = "https://lightningnetwork.plus/api/2"
_MAX_RESPONSE_BYTES = 1_000_000
_CHALLENGE_MAX_LEN = 500
_CHALLENGE_FORBIDDEN_PREFIXES = ("lnbc", "lntb", "lnbcrt", "cbor", "psbt")
_PUBKEY_RE = re.compile(r"^0[23][0-9a-fA-F]{64}$")
# I3: sane host:port shape for a connect address sourced from the LN+ API —
# no shell metacharacters, no whitespace; length-capped separately.
_ADDR_RE = re.compile(r"^[A-Za-z0-9.\-_:\[\]]+$")
_ADDR_MAX_LEN = 300


def _valid_pubkey(value) -> bool:
    return isinstance(value, str) and bool(_PUBKEY_RE.match(value))


def _valid_connect_addr(value) -> bool:
    return (isinstance(value, str) and 0 < len(value) < _ADDR_MAX_LEN
            and bool(_ADDR_RE.match(value)))


def _parse_ts(value) -> Optional[int]:
    """ISO-8601 string or epoch number -> epoch seconds, else None."""
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        # B3: a TZ-less ISO string must not be interpreted in the node's
        # local timezone by .timestamp() — LN+ deadlines/ends are UTC.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    return None


class LNPlusError(Exception):
    def __init__(self, message: str, http_status: Optional[int] = None,
                 errors: Optional[Dict] = None):
        super().__init__(message)
        self.http_status = http_status
        # C-4 (2026-07-08 audit): structured validation errors parsed from a
        # 422 body's "errors" key (e.g. {"id": ["already applied"]}), when
        # present. None otherwise — no caller behavior change required.
        self.errors = errors


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
            # C-4 (2026-07-08 audit): best-effort structured error parse —
            # a 422 (or other) body that decodes to a dict with an "errors"
            # key is attached to the raised LNPlusError so callers CAN act
            # on field-level detail (e.g. "already applied"); any parse
            # failure or non-dict/non-"errors" body just leaves errors=None,
            # matching the existing message format exactly either way.
            parsed_errors = None
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and isinstance(parsed.get("errors"), dict):
                    parsed_errors = parsed["errors"]
            except Exception:
                pass
            raise LNPlusError(f"LN+ HTTP {e.code} on {path}: {body[:500]!r}",
                              http_status=e.code, errors=parsed_errors)
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
        swaps = swaps if isinstance(swaps, list) else []
        # Normalize swap ids to strings at the client boundary
        for swap in swaps:
            if isinstance(swap, dict) and "id" in swap:
                swap["id"] = str(swap["id"])
        return swaps

    def get_swap(self, swap_id) -> Dict:
        result = self._request(f"get_swap/id={urllib.parse.quote(str(swap_id), safe='')}")
        result = self._unwrap_list_envelope(result, {})
        # Normalize swap id to string at the client boundary, guard non-dict entries defensively
        if isinstance(result, dict) and "id" in result:
            result["id"] = str(result["id"])
        return result

    def get_my_swaps(self) -> Dict:
        result = self._request("get_my_swaps", self._auth_params(), method="POST")
        result = self._unwrap_list_envelope(
            result, {"pending": [], "opening": [], "completed": []})
        if not isinstance(result, dict):
            raise LNPlusError("get_my_swaps: unexpected payload")
        normalized = {k: result.get(k) or [] for k in ("pending", "opening", "completed")}
        # Normalize swap ids to strings at the client boundary, skip entries without an id
        for bucket in normalized.values():
            for entry in bucket:
                if isinstance(entry, dict) and "id" in entry:
                    entry["id"] = str(entry["id"])
        return normalized

    @staticmethod
    def _unwrap_list_envelope(result: Any, empty_fallback: Any) -> Any:
        """The LN+ docs show get_my_swaps (and get_account_info/get_my_todos)
        wrapping a single response object in a one-element array:
        [ {...} ]. Normalize defensively so the client is the only place
        that has to know this: a list -> its first element if that element
        is a dict (an empty list -> empty_fallback); a dict -> used as-is;
        anything else is returned unchanged for the caller to reject."""
        if isinstance(result, list):
            if not result:
                return empty_fallback
            return result[0] if isinstance(result[0], dict) else result
        return result

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

    def get_notifications(self) -> Dict:
        result = self._request("get_notifications", self._auth_params(), method="POST")
        result = self._unwrap_list_envelope(result, {})
        if not isinstance(result, dict):
            raise LNPlusError("get_notifications: unexpected payload")
        return result

    def mark_read_notifications(self) -> Dict:
        result = self._request("mark_read_notifications", self._auth_params(), method="POST")
        return self._unwrap_list_envelope(result, {}) if isinstance(result, (list, dict)) else {}

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

    def __init__(self, plugin, rpc, database, config, client, planner, lifecycle,
                 policy_manager=None):
        self._plugin = plugin
        self.rpc = rpc
        self._db = database
        self._config = config
        self._client = client
        self._planner = planner
        self._lifecycle = lifecycle
        # Optional: enables the operator-ban veto in gate 5 (revenue-ban).
        self._policy_manager = policy_manager
        # PR 3d: process-constant node id + per-pass frozen peer set.
        self._our_id_cache: Optional[str] = None
        self._pass_peers_with_channels: Optional[set] = None

    def _our_id(self) -> Optional[str]:
        """Our node id — a process constant, cached on first success
        (PR 3d: replaces a per-swap live getinfo read)."""
        if self._our_id_cache:
            return self._our_id_cache
        try:
            our_id = self.rpc.getinfo().get("id")
        except Exception:
            return None
        if our_id:
            self._our_id_cache = str(our_id)
        return self._our_id_cache

    def _capture_peers_with_channels(self) -> Optional[set]:
        """PR 3d: freeze ONE peer-channel observation at pass entry —
        the existing-channel gate consults this set instead of issuing
        a live read per swap. ANY channel row counts (matching the
        legacy per-peer read, including pending/closing states). None on
        failure -> per-swap live fallback (exact pre-adoption path)."""
        try:
            channels = (self.rpc.listpeerchannels() or {}).get(
                "channels", []) or []
            return {str(c.get("peer_id"))
                    for c in channels
                    if isinstance(c, dict) and c.get("peer_id")}
        except Exception:
            return None

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

        # PR 3d: freeze the pass's peer-channel observation before the
        # gate loop — per-swap gates never issue their own live reads.
        self._pass_peers_with_channels = self._capture_peers_with_channels()

        qualifying = []
        for swap in swaps:
            reason = self._filter_swap(swap, cfg)
            if reason is None:
                reason = self._check_participants(swap, cfg)
            if reason is None:
                reason = self._check_assignment_available(swap)
            if reason is None:
                reason = self._check_existing_channel(swap)
            if reason is not None:
                self._reject(summary, swap, reason)
            else:
                qualifying.append(swap)

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
        # I2(a): no upper capacity bound previously existed — an oversized
        # swap (past the planner's own affordability/portfolio-concentration
        # ceiling) would sail through terms and only get caught (or not) by
        # the funds check below, well after evaluation cost was sunk.
        max_chan = getattr(cfg, "planner_max_channel_sats", 0) or 0
        if max_chan and capacity > max_chan:
            return f"terms:capacity {capacity} above planner max {max_chan}"
        if int(swap.get("duration_months") or 99) > int(cfg.lnplus_max_duration_months):
            return "terms:duration exceeds cap"
        if int(swap.get("participant_max_count") or 99) > int(cfg.lnplus_max_participants):
            return "terms:too many participants"
        # D-3 (2026-07-08 Revision 2, operator-directed): dual (2-party)
        # swaps are rejected; 3+ participants required. A missing count
        # already fails the max check above (default 99), so this only
        # bites a genuinely-declared undersized ring.
        min_participants = int(cfg.lnplus_min_participants)
        if int(swap.get("participant_max_count") or 0) < min_participants:
            return f"terms:fewer than {min_participants} participants"
        if str(swap.get("platform") or "any").lower() == "lnd":
            return "terms:LND/BOS swap — we are CLN"
        return None

    def _check_participants(self, swap, cfg) -> Optional[str]:
        """Gate 5 (peer quality, one bad peer vetoes) + own-node check."""
        our_id = self._our_id()  # PR 3d: process-constant, cached
        participants = swap.get("participants") or []
        if not participants:
            return "peer_quality:no visible participants"
        for p in participants:
            # B12: a cancelled/banned participant's slot is effectively free
            # (they are no longer party to the swap) and their historical
            # stats are stale — they must not veto gate 5 on our behalf.
            if p.get("cancelled") or p.get("banned"):
                continue
            pk = p.get("pubkey")
            if not _valid_pubkey(pk):
                return "peer_quality:invalid participant pubkey"
            if our_id and pk == our_id:
                return "own_node:we are already in this swap"
            # Operator ban (revenue-ban): a banned pubkey anywhere in the
            # ring vetoes the swap. Fail closed — an unreadable policy
            # source must not let a banned peer through.
            if self._policy_manager is not None:
                try:
                    if self._policy_manager.is_peer_banned(pk):
                        return f"peer_quality:{pk[:16]} operator-banned"
                except Exception as e:
                    self._plugin.log(
                        f"LNPLUS: ban lookup failed for {pk[:16]} ({e}) — "
                        "rejecting swap (fail closed)", level="warn")
                    return f"peer_quality:{pk[:16]} ban lookup failed (fail closed)"
            if not (p.get("address_1") or p.get("address_2")):
                return f"peer_quality:{pk[:16]} publishes no address"
            pos = int(p.get("positive_ratings_count") or 0)
            neg = int(p.get("negative_ratings_count") or 0)
            if pos < int(cfg.lnplus_min_peer_positive_ratings):
                return f"peer_quality:{pk[:16]} has {pos} positive ratings (< floor)"
            if pos + neg > 0 and neg / (pos + neg) > NEG_RATIO_MAX:
                return f"peer_quality:{pk[:16]} negative ratio too high"
            # D-2 (Revision 2): gold-or-better rank floor. Missing/zero
            # rank is fail-closed (below floor), not a pass.
            rank = int(p.get("lnplus_rank_number") or 0)
            floor = int(cfg.lnplus_min_peer_rank)
            if rank < floor:
                return f"peer_quality:{pk[:16]} rank {rank} below floor {floor}"
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

    def _check_assignment_available(self, swap) -> Optional[str]:
        """B11: a full identifier set (malformed data, or an operator
        raising max-participants above _IDENTIFIERS' length) leaves
        _infer_assignment with no free slot. Reject cleanly here instead of
        letting a bare StopIteration escape through the gate chain."""
        if self._infer_assignment(swap).get("our_identifier") is None:
            return "terms:no free participant slot"
        return None

    def _check_existing_channel(self, swap) -> Optional[str]:
        """I5(a): reject a swap whose INFERRED outbound peer we already have
        a channel to. Without this an idempotency check can later (I5b)
        misread a coincidental pre-existing channel to that peer as "our
        swap channel" and never open the one the swap actually requires;
        pre-empting it here also avoids burning the one-application-per-node
        serialization slot on a swap that cannot add new capacity anyway.
        Fail-open on an RPC hiccup (matches _check_mid_contract_vanish /
        _execute_swap_open's existing-channel lookups elsewhere in this
        module) — real state is re-checked at open time regardless.
        """
        outbound_peer = self._infer_assignment(swap).get("outbound_peer")
        if not outbound_peer:
            return None
        # PR 3d: consult the pass-frozen peer set; live per-peer read
        # only as the fail-open fallback when the capture failed.
        frozen = self._pass_peers_with_channels
        if frozen is not None:
            if str(outbound_peer) not in frozen:
                return None
            return "terms:existing channel to assigned outbound peer"
        try:
            channels = self.rpc.listpeerchannels(outbound_peer).get("channels", []) or []
        except Exception:
            return None
        if not channels:
            return None
        return "terms:existing channel to assigned outbound peer"

    def _infer_assignment(self, swap) -> Dict:
        """LN+ convention: each participant opens to the next letter; last wraps to A.
        We join as the next free identifier. Authoritative assignment is re-read
        from get_my_swaps at open time; this is for pre-apply EV only.

        B12: cancelled/banned participants are excluded from the identifier
        map entirely — their slot is effectively free (not occupied) and
        their pubkey must not be used as an outbound/incoming assignment.
        B11: if no identifier is free (full/malformed participant set),
        return a None-valued assignment instead of letting the underlying
        next() raise a bare StopIteration — callers check our_identifier
        and reject via _check_assignment_available."""
        participants = {p.get("participant_identifier"): p
                        for p in (swap.get("participants") or [])
                        if not (p.get("cancelled") or p.get("banned"))}
        ours = next((i for i in _IDENTIFIERS if i not in participants), None)
        empty = {"our_identifier": None, "outbound_peer": None, "incoming_peer": None}
        if ours is None:
            return empty
        count = int(swap.get("participant_max_count") or (len(participants) + 1))
        letters = list(_IDENTIFIERS[:count])
        if ours not in letters:
            return empty
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

    def _swap_ev(self, swap, cfg, best_regular_ev: float):
        assignment = self._infer_assignment(swap)
        capacity = int(swap.get("capacity_sats") or 0)
        duration = int(swap.get("duration_months") or 0)
        outbound_ev = float(self._planner._calculate_open_ev(
            assignment["outbound_peer"], capacity, cfg) or 0.0)
        # I4: inbound_corridor deliberately reuses _calculate_open_ev, which
        # already nets out open+close on-chain costs internally (see
        # capacity_planner._calculate_open_ev's on_chain_cost term). Using it
        # as a corridor-value proxy here — rather than a bespoke revenue-only
        # estimate — is conservative: it silently charges the inbound side
        # for a hypothetical open/close it will never pay, so inbound_credit
        # is a lower bound on the corridor's true value.
        inbound_corridor = float(self._planner._calculate_open_ev(
            assignment["incoming_peer"], capacity, cfg) or 0.0)
        replacement = capacity * BOLTZ_REPLACEMENT_RATE
        inbound_credit = min(inbound_corridor, replacement)

        # B12: cancelled/banned participants are no longer party to the
        # swap — exclude them from reliability and the Tor discount, same
        # as gate 5 and _infer_assignment already do.
        counterparties = [p for p in (swap.get("participants") or [])
                          if not (p.get("cancelled") or p.get("banned"))]
        min_pos = min((int(p.get("positive_ratings_count") or 0) for p in counterparties),
                      default=0)
        reliability = min(1.0, RELIABILITY_FLOOR + 0.4 * min(1.0, min_pos / 50.0))
        if counterparties and all(self._tor_only(p) for p in counterparties):
            reliability *= TOR_RELIABILITY

        lockup_haircut = P_UNDERPERFORM * max(0.0, best_regular_ev) * (duration / 12.0)
        # I4: _calculate_open_ev (outbound_ev, above) already nets out
        # open_cost + close_cost internally via its on_chain_cost term — see
        # capacity_planner._calculate_open_ev. Regular candidates'
        # _planned_ev never re-subtracts open_cost on top of that (compare
        # capacity_planner.execute_cycle's `ev = self._calculate_open_ev(...)`
        # used as-is), so doing it here double-counted the same on-chain cost
        # twice for swaps only, unfairly penalizing them against regular
        # opens in the unified ranking. open_cost is still used (unmodified)
        # by the capex/funds gates in _select_and_apply, where it correctly
        # represents the swap's own upfront on-chain commitment.
        value = (outbound_ev
                 + inbound_credit * reliability * float(cfg.lnplus_inbound_credit_factor)
                 - lockup_haircut)
        return value, assignment

    @staticmethod
    def _tor_only(participant) -> bool:
        addrs = [a for a in (participant.get("address_1"), participant.get("address_2")) if a]
        return bool(addrs) and all(".onion" in str(a) for a in addrs)

    def _select_and_apply(self, qualifying, cfg, best_regular_ev, summary) -> Dict:
        scored = []
        for swap in qualifying:
            value, assignment = self._swap_ev(swap, cfg, best_regular_ev)
            if value <= 0:
                self._reject(summary, swap, f"economics:non-positive EV ({value:.0f})")
                continue
            scored.append((value, swap, assignment))
        if not scored:
            return summary
        # D-3 (2026-07-08 Revision 2, operator-directed): among equal-EV
        # qualifiers, prefer fewer participants (triangle beats square
        # beats pentagon) — pure tie-break, EV still decides primarily.
        scored.sort(key=lambda t: (-t[0], int(t[1].get("participant_max_count") or 0)))
        value, swap, assignment = scored[0]
        summary["swap_ev"] = value
        summary["swap_id"] = swap.get("id")

        margin = float(cfg.lnplus_swap_preference_margin)
        if best_regular_ev > value * (1.0 + margin):
            self._reject(summary, swap,
                         f"preference_margin:regular open EV {best_regular_ev:.0f} beats "
                         f"swap EV {value:.0f} beyond {margin:.0%} margin")
            summary["swap_id"] = None
            return summary

        open_cost = int(self._planner._estimate_open_cost() or 0)
        capacity = int(swap.get("capacity_sats") or 0)
        # Gate 7: confirmed unreserved on-chain funds must cover capacity + fee buffer.
        # Fail-closed: unparseable listfunds counts as zero.
        try:
            outputs = self.rpc.listfunds().get("outputs", [])
            confirmed_sats = sum(
                int(o.get("amount_msat", 0)) // 1000
                for o in outputs
                if o.get("status") == "confirmed" and not o.get("reserved"))
        except Exception:
            confirmed_sats = 0
        # I2(b): the confirmed-funds check ignored the planner's own on-chain
        # reserve floor (min_wallet_reserve) — a swap could be applied for
        # using funds the operator wants kept as an untouchable buffer, the
        # same reserve the regular-open sizer respects
        # (capacity_planner.py's `available_sats = confirmed - min_reserve`).
        reserve = getattr(cfg, "min_wallet_reserve", 0) or 0
        if confirmed_sats < capacity + open_cost + reserve:
            self._reject(summary, swap,
                         f"economics:confirmed funds {confirmed_sats} below "
                         f"capacity+fees+reserve {capacity + open_cost + reserve}")
            summary["swap_id"] = None
            return summary
        capex = getattr(self._planner, "_capex_engine", None)
        if capex is not None:
            try:
                budget = int(capex.get_fleet_exploration_budget() or 0)
            except Exception:
                budget = 0
            if budget < open_cost:
                self._reject(summary, swap,
                             f"economics:capex budget {budget} below open cost {open_cost}")
                summary["swap_id"] = None
                return summary

        reason = (f"LN+ swap {swap.get('id')}: EV {value:.0f} vs regular {best_regular_ev:.0f}, "
                  f"{capacity} sats for {swap.get('duration_months')}mo")
        action_id = self._db.record_planner_action(
            action_type="swap_apply",
            peer_id=assignment.get("outbound_peer") or "unknown",
            amount_sats=capacity, estimated_cost_sats=open_cost,
            reason=reason,
            metadata={"swap_id": swap.get("id"),
                      "incoming_peer": assignment.get("incoming_peer")})

        if not getattr(cfg, "lnplus_execute_applications", False) or getattr(cfg, "planner_dry_run", False):
            self._db.update_planner_action(action_id, status="recommended")
            self._plugin.log(f"LNPLUS: [RECOMMEND] would apply to swap {swap.get('id')} — {reason}")
            summary["recommended"] = True
            return summary

        # Intent-first: ledger row before the external call.
        self._db.lnplus_record_swap(
            str(swap.get("id")), "applied", capacity,
            int(swap.get("duration_months") or 0),
            outbound_peer=assignment.get("outbound_peer"),
            incoming_peer=assignment.get("incoming_peer"),
            our_identifier=assignment.get("our_identifier"),
            planner_action_id=action_id)
        try:
            self._client.create_application(swap.get("id"))
        except LNPlusError as e:
            self._db.lnplus_update_swap(str(swap.get("id")), status="failed",
                                        outcome=f"apply failed: {e}")
            self._db.update_planner_action(action_id, status="failed")
            self._plugin.log(f"LNPLUS: application to {swap.get('id')} failed: {e}", level="warn")
            return summary
        self._db.update_planner_action(action_id, status="completed")
        self._plugin.log(f"LNPLUS: applied to swap {swap.get('id')} — {reason}")
        summary["applied"] = True
        return summary


_BREAKER_KEY = "_lnplus_breaker"
_BACKFILL_FLAG = "_lnplus_backfill_done"   # config_overrides key, value = epoch str
_OPEN_STATES = ("OPENINGD", "CHANNELD_AWAITING_LOCKIN", "CHANNELD_NORMAL",
                "DUALOPEND_OPEN_INIT", "DUALOPEND_AWAITING_LOCKIN")
# Best-effort fallback used only when the caller does not wire the capacity
# planner's real cost/budget estimators in (estimate_open_cost_fn /
# budget_params_fn on SwapLifecycle.__init__). Still counted on the atomic
# spend rail at settle time either way.
_DEFAULT_OPEN_COST_SATS = 2500
# B9: local 'applied' rows younger than this are exempt from reconcile
# divergence checks — the evaluator may have applied milliseconds after the
# watcher's own get_my_swaps fetch ran.
_RECONCILE_GRACE_SECONDS = 600
# B5(b): local statuses meaning "we have already knowingly walked away from
# this swap" — a pending ghost matching one of these is cleanup, not a
# genuine untracked commitment.
_TERMINAL_PENDING_GHOST_STATUSES = ("failed", "withdrawn", "cancelled_remote")
# Task 26 companion (audit 2026-08-01): outcome-text marker persisted when a
# broadcast-capable fundchannel dies with the outcome UNKNOWN and its budget
# reservation is HELD for the next watcher pass to resolve. Format:
#   "fundchannel outcome unknown; held_reservation=<reservation_id>:<sats>"
_HELD_RESERVATION_MARKER = "held_reservation="


class SwapLifecycle:
    """Obligations watcher / state machine (spec gates 10-14).

    Once an application is filled, the 48h open deadline and the eventual
    protection contract are IRREVERSIBLE COMMITMENTS. Every DB state
    transition here follows crash-safe ordering: write intent before acting
    on it externally, write outcome after the external call returns.
    Ratings and ignore_peer_fn calls are best-effort (try/except + log);
    the DB state machine transitions are not.
    """

    _BACKFILL_FLAG = _BACKFILL_FLAG

    # Phase 2F: optional governor/ledger plumbing (EconShadow), injected
    # at plugin init.
    econ_shadow = None
    # Phase 3E: optional CLN adapter (DataService), injected at plugin
    # init; raw-RPC fallback preserved when absent.
    data_service = None

    def _lnplus_governor_enabled(self) -> bool:
        """Phase 2F: strict flag check (MagicMock/absent snapshots stay
        on the legacy path)."""
        try:
            cfg = self._config.snapshot() \
                if self._config is not None \
                and hasattr(self._config, "snapshot") else self._config
            return getattr(cfg, "econ_governor_lnplus_enabled",
                           False) is True
        except Exception:
            return False

    def _governed_reserve_spend(self, *, reservation_id, amount_sats,
                                metadata, effective_budget_sats,
                                since_timestamp, swap_id, peer_id,
                                capacity_sats):
        """Phase 2F: GovernorFacade.authorize() gates the LN+ swap-open
        reservation. Fulfilling an ACCEPTED swap is a CONTRACTUAL
        OBLIGATION (refactor invariant 6): the legacy path carries no
        pause gate here and neither does the governed one — is_paused is
        pinned False and the intent carries CONTRACT_OBLIGATION. The
        delegate performs the IDENTICAL reserve_spend call; FAILS CLOSED
        on internal error (caller treats False as retry-next-pass, same
        as a legacy budget refusal)."""
        try:
            import time as _time

            from .econ_intents import Explanation, make_intent
            from .econ_types import Micro, Msat, SignedMsat, UnixTime
            from .governor_facade import GovernorFacade

            now = int(_time.time())

            def _lnplus_reserve_delegate(**_facade_kwargs):
                return bool(self._db.reserve_spend(
                    reservation_id=reservation_id,
                    amount_sats=amount_sats,
                    category="channel_open",
                    subcategory="lnplus_swap",
                    metadata=metadata,
                    effective_budget_sats=effective_budget_sats,
                    since_timestamp=since_timestamp,
                ))

            ledger = None
            registry = None
            if self.econ_shadow is not None:
                try:
                    ledger = self.econ_shadow.ledger_for_reconciliation()
                except Exception:
                    ledger = None
                try:
                    registry = self.econ_shadow.arbitration_registry()
                except Exception:
                    registry = None

            facade = GovernorFacade(
                reserve_spend=_lnplus_reserve_delegate,
                release_spend=self._db.release_spend_reservation,
                # Invariant 6: obligation fulfillment is never pause- or
                # authority-gated — the swap was accepted under whatever
                # authority applied at application time.
                is_paused=lambda: False,
                ledger=ledger,
                registry=registry,
            )
            committed = int(capacity_sats or 0)
            env = make_intent(
                intent_type="OPEN_CHANNEL",
                snapshot_id=f"lnplus-swap-{swap_id}",
                created_at=UnixTime(now),
                expires_at=UnixTime(now + 600),
                target=str(peer_id),
                amount_msat=(Msat(committed * 1000) if committed > 0
                             else None),
                expected_benefit_msat=SignedMsat(0),
                max_cost_msat=Msat(int(amount_sats) * 1000),
                capital_committed_msat=Msat(committed * 1000),
                confidence_micro=Micro(0),
                reason_codes=("CONTRACT_OBLIGATION",),
                explanation=Explanation("lnplus_swap_open", (
                    ("swap_id", str(swap_id)),
                    ("peer", str(peer_id)),
                    ("capacity_sats", committed),
                    ("reserve_sats", int(amount_sats)),
                )),
                preconditions=(),
                priority=80,
                budget_bucket="channel_open",
                origin_policy="lnplus_lifecycle_governed",
                reversible=False,
            )
            if ledger is not None:
                try:
                    details = {"target": env.target, "governed": True,
                               "swap_id": str(swap_id)}
                    # PR 3d: canonical-snapshot linkage as EVIDENCE only.
                    # The envelope's swap-scoped snapshot_id is deliberately
                    # untouched — it carries the obligation's cross-attempt
                    # idempotency (the key hashes snapshot_id).
                    try:
                        snap_ref = (self.econ_shadow.snapshot_ref(now)
                                    if getattr(self, "econ_shadow", None)
                                    is not None else None)
                        if snap_ref and snap_ref.get("snapshot_id"):
                            details["canonical_snapshot_id"] = str(
                                snap_ref["snapshot_id"])
                    except Exception:
                        pass
                    ledger.append(
                        event_type="intent_proposed",
                        intent_id=env.intent_id.value,
                        idempotency_key=env.idempotency_key,
                        cycle_id=env.snapshot_id,
                        at=now,
                        details=details,
                    )
                except Exception:
                    pass
            decision = facade.authorize(env, now,
                                        reservation_id=reservation_id)
            if not decision.authorized:
                self._plugin.log(
                    f"LNPLUS: governor blocked swap-open reservation for "
                    f"swap {swap_id}: {decision.reason_code}", level="warn")
            return bool(decision.authorized)
        except Exception as e:
            self._plugin.log(
                f"LNPLUS: governed reservation failed closed for swap "
                f"{swap_id}: {e}", level="warn")
            return False

    def __init__(self, plugin, rpc, database, config, client, policy_manager,
                 ignore_peer_fn=None, estimate_open_cost_fn=None, budget_params_fn=None):
        self._plugin = plugin
        self.rpc = rpc
        self._db = database
        self._config = config
        self._client = client
        self._policy = policy_manager
        self._ignore_peer_fn = ignore_peer_fn
        # Injected from the capacity planner (_estimate_open_cost /
        # _unified_reserve_budget_params) by the caller wiring this class up;
        # when absent, _execute_swap_open falls back to _DEFAULT_OPEN_COST_SATS
        # and a best-effort (unenforced) reservation.
        self._estimate_open_cost_fn = estimate_open_cost_fn
        self._budget_params_fn = budget_params_fn
        self._watcher_lock = threading.Lock()
        # C-7 (2026-07-08 audit): dedicated lock around the backfill-run-once
        # block in _reconcile — the evaluator's reconcile_ok preflight and
        # the watcher can call _reconcile concurrently on separate threads,
        # and without this the unlocked "if not flag" check races and can
        # run backfill_from_lnplus more than once.
        self._backfill_lock = threading.Lock()
        # B13: in-memory only (no DB) — observability of the most recent
        # watcher pass, surfaced via get_status().
        self._last_watcher_pass: Optional[Dict] = None

    # -- breaker -------------------------------------------------------
    def breaker_tripped(self) -> Optional[str]:
        return self._db.get_config_override(_BREAKER_KEY)

    def trip_breaker(self, reason: str) -> None:
        # B10: preserve the FIRST cause. Overwriting the stored reason on a
        # second/later trip destroys the original diagnostic signal the
        # operator needs — later reasons are often downstream noise from the
        # same underlying divergence (e.g. repeated deadline misses).
        existing = self._db.get_config_override(_BREAKER_KEY)
        if existing:
            self._plugin.log(
                f"LNPLUS: CIRCUIT BREAKER TRIP — {reason} (breaker already tripped)",
                level="error")
            return
        self._db.set_config_override(_BREAKER_KEY, f"{int(time.time())}: {reason}")
        self._plugin.log(f"LNPLUS: CIRCUIT BREAKER TRIPPED — {reason}", level="error")

    def clear_breaker(self) -> None:
        self._db.delete_config_override(_BREAKER_KEY)
        self._plugin.log("LNPLUS: circuit breaker cleared by operator")

    def has_inflight(self) -> bool:
        return bool(self._db.lnplus_inflight_swaps())

    # -- reconciliation --------------------------------------------------
    def reconcile_ok(self) -> bool:
        """Live get_my_swaps vs local in-flight rows. Divergence trips the
        breaker and returns False. Also returns False (without tripping)
        if LN+ itself is unreachable — new applications should not proceed
        blind, though obligation phases in run_watcher_once never call
        this and are never blocked by it."""
        try:
            my = self._client.get_my_swaps()
        except LNPlusError as e:
            self._plugin.log(f"LNPLUS: reconcile fetch failed: {e}", level="warn")
            return False
        return self._reconcile(my)

    def _reconcile(self, my: Dict) -> bool:
        # Choke point: adopt pre-existing LN+ state (manual applications/
        # opens/contracts predating this automation) BEFORE the first
        # divergence check ever runs, on both the watcher pass and the
        # evaluator's reconcile_ok preflight. Never trip the breaker
        # against state we have not adopted yet — if backfill itself
        # raises, retry next pass instead of running divergence checks.
        # C-7 (2026-07-08 audit): the evaluator preflight (reconcile_ok) and
        # the watcher can both reach this choke point on separate threads.
        # A cheap unlocked pre-check avoids taking the lock on every pass
        # once backfill has already run; the flag is re-checked INSIDE the
        # lock (double-checked locking) so only one thread ever actually
        # runs backfill_from_lnplus. Blocking acquire is fine — backfill is
        # fast and runs exactly once.
        if not self._db.get_config_override(_BACKFILL_FLAG):
            with self._backfill_lock:
                if not self._db.get_config_override(_BACKFILL_FLAG):
                    try:
                        self.backfill_from_lnplus(my)
                    except Exception as e:
                        self._plugin.log(
                            f"LNPLUS: backfill_from_lnplus failed: {e} — retrying "
                            "next pass", level="error")
                        return False
                    self._db.set_config_override(_BACKFILL_FLAG, str(int(time.time())))

        my = my or {}
        pending_ids = {s.get("id") for s in (my.get("pending") or [])}
        opening_ids = {s.get("id") for s in (my.get("opening") or [])}
        completed_ids = {s.get("id") for s in (my.get("completed") or [])}
        local_inflight = self._db.lnplus_inflight_swaps()
        local_ids = {row["swap_id"] for row in local_inflight}

        ok = True
        now = int(time.time())
        for row in local_inflight:
            sid = row["swap_id"]
            status = row["status"]
            if status == "applied":
                # B9: the evaluator may have applied milliseconds after the
                # watcher's own get_my_swaps fetch — a swap that legitimately
                # exists on LN+ can still be invisible in `my` for one pass.
                # Skip divergence evaluation entirely inside the grace window.
                applied_at = row.get("applied_at") or 0
                if now - applied_at < _RECONCILE_GRACE_SECONDS:
                    continue
                if sid in pending_ids or sid in opening_ids:
                    continue
                if sid in completed_ids:
                    # The swap filled and completed while our row was still
                    # 'applied' (e.g. operator drove it manually on the LN+
                    # site while the watcher was down). This is a LIVE
                    # contract, not a remote cancellation — terminally
                    # marking it cancelled_remote would skip activation and
                    # leave the contract channel unprotected (no no_close
                    # tag). Trip the breaker for operator review instead.
                    self.trip_breaker(
                        f"local swap {sid} still 'applied' but LN+ lists it "
                        "completed — resolve manually (backfill skips swaps "
                        "that already have a local row)")
                    ok = False
                    continue
                # B4: an 'applied' row absent from pending/opening/completed
                # on a successful fetch is a REMOTE cancellation (creator
                # cancelled/deleted the swap), not our defection. Tripping
                # the breaker here every pass forever was a permanent
                # breaker trip-loop while the row also held the
                # serialization slot and its capacity reservation.
                # cancelled_remote is terminal (not in
                # _LNPLUS_INFLIGHT_STATUSES) so both free automatically.
                self._db.lnplus_update_swap(
                    sid, status="cancelled_remote",
                    outcome="swap disappeared from LN+ (creator cancelled/deleted)")
                self._plugin.log(
                    f"LNPLUS: applied swap {sid} vanished from LN+ (not in "
                    "pending/opening/completed) — treating as remote "
                    "cancellation, not tripping the breaker", level="warn")
                continue
            elif status in ("opening", "opened"):
                # Funds may already be committed on-chain for these — this
                # IS a divergence needing an operator, so still trip.
                compatible = sid in opening_ids or sid in completed_ids
            else:
                compatible = True
            if not compatible:
                self.trip_breaker(
                    f"local swap {sid} (status {status}) missing/divergent on LN+")
                ok = False

        for sid in opening_ids:
            if sid and sid not in local_ids:
                self.trip_breaker(
                    f"LN+ shows opening swap {sid} with no local record — "
                    "if this swap was entered manually, run "
                    "revenue-lnplus-backfill to adopt it")
                ok = False

        # I1 fix: gate 0 was one-eyed — it only checked the opening list for
        # LN+-side entries with no local row. A pending application LN+ knows
        # about but we have no in-flight row for (applied/opening/opened) is
        # an untracked live commitment (a "ghost") that the pending-timeout
        # phase will never see (that phase only handles rows WE already know
        # about), so it must trip the breaker here instead of going unnoticed
        # forever.
        for sid in pending_ids:
            if not sid or sid in local_ids:
                continue
            # B5(b): a pending entry LN+ still lists that matches a local row
            # we have already knowingly walked away from (terminal:
            # failed/withdrawn/cancelled_remote) is not a "ghost" — it is a
            # stale application we should clean up on LN+'s side. Only a
            # pending id with NO local row at all is a true untracked ghost.
            local_row = self._db.lnplus_get_swap(sid)
            if local_row and local_row.get("status") in _TERMINAL_PENDING_GHOST_STATUSES:
                try:
                    self._client.delete_application(sid)
                except LNPlusError as e:
                    if self._application_already_gone(e):
                        self._plugin.log(
                            f"LNPLUS: stale pending application {sid} is "
                            "already gone on LN+ — resolved", level="info")
                    else:
                        self._plugin.log(
                            f"LNPLUS: delete_application({sid}) failed for a "
                            f"stale pending application (local status "
                            f"{local_row.get('status')!r}): {e}", level="warn")
                except Exception as e:
                    self._plugin.log(
                        f"LNPLUS: delete_application({sid}) failed for a "
                        f"stale pending application (local status "
                        f"{local_row.get('status')!r}): {e}", level="warn")
                continue
            self.trip_breaker(
                f"LN+ shows pending swap {sid} with no local record — "
                "if this swap was entered manually, run "
                "revenue-lnplus-backfill to adopt it")
            ok = False
        return ok

    # -- backfill of pre-existing (manual) LN+ swaps -----------------------
    def backfill_from_lnplus(self, my: Optional[Dict] = None) -> Dict:
        """Adopt LN+ account state accumulated before this automation
        existed (manual applications/opens/running-or-ended contracts made
        on the LN+ website) into the local ledger.

        Common rule across every category: skip unconditionally if a local
        row for that swap_id already exists — lnplus_record_swap is INSERT
        OR REPLACE and would otherwise clobber automation-owned state or
        resurrect a terminal row. This makes the whole method idempotent
        and safe to call any number of times (the flag in _reconcile only
        prevents redundant AUTO-runs; the revenue-lnplus-backfill RPC path
        relies purely on this per-swap skip).
        """
        if my is None:
            my = self._client.get_my_swaps()
        my = my or {}
        now = int(time.time())
        imported = {"pending": 0, "opening": 0, "active": 0, "ended": 0}
        skipped: List[str] = []
        warnings: List[str] = []

        for entry in (my.get("pending") or []):
            self._backfill_pending(entry, now, imported, skipped, warnings)
        for entry in (my.get("opening") or []):
            self._backfill_opening(entry, now, imported, skipped, warnings)
        for entry in (my.get("completed") or []):
            self._backfill_completed(entry, now, imported, skipped, warnings)

        return {"imported": imported, "skipped": skipped, "warnings": warnings}

    def _backfill_pending(self, entry: Dict, now: int, imported: Dict,
                          skipped: List[str], warnings: List[str]) -> None:
        """Rule 1: applied row, no peer/identifier assignment yet (phase 3
        receives the authoritative assignment when it fills — running
        _infer_assignment on stale data here would be wrong)."""
        sid = entry.get("id")
        if not sid:
            return
        sid = str(sid)
        if self._db.lnplus_get_swap(sid):
            skipped.append(sid)
            return
        capacity, duration = self._resolve_capacity_duration(entry, sid, warnings)
        self._db.lnplus_record_swap(
            sid, "applied", capacity, duration,
            metadata={"imported": True, "imported_at": now})
        self._plugin.log(
            f"LNPLUS: backfill — imported pending swap {sid} "
            f"({capacity} sats, {duration}mo); pending-timeout clock "
            "starts now", level="info")
        imported["pending"] += 1

    def _backfill_opening(self, entry: Dict, now: int, imported: Dict,
                          skipped: List[str], warnings: List[str]) -> None:
        """Rule 2: opening row. outbound_peer only if the LN+-supplied
        pubkey validates (else NULL + warning, phase 3 retries); deadline
        from the entry or a conservative now+48h fallback. Phase 3/3b
        executes the open in the same watcher pass, and its existing
        capacity-match idempotency already handles an operator having
        funded the channel manually — nothing to duplicate here."""
        sid = entry.get("id")
        if not sid:
            return
        sid = str(sid)
        if self._db.lnplus_get_swap(sid):
            skipped.append(sid)
            return
        capacity, duration = self._resolve_capacity_duration(entry, sid, warnings)
        peer = entry.get("outgoing_peer_pubkey")
        outbound_peer = peer if _valid_pubkey(peer) else None
        if peer and not outbound_peer:
            warnings.append(
                f"swap {sid}: invalid outgoing_peer_pubkey on import — "
                "outbound_peer left NULL")
        deadline_ts = _parse_ts(entry.get("deadline"))
        if not deadline_ts:
            deadline_ts = now + 48 * 3600
            warnings.append(
                f"swap {sid}: no parseable deadline on import — using 48h "
                "fallback")
        self._db.lnplus_record_swap(
            sid, "opening", capacity, duration,
            outbound_peer=outbound_peer,
            metadata={"imported": True, "imported_at": now})
        self._db.lnplus_update_swap(sid, deadline_at=deadline_ts)
        self._plugin.log(
            f"LNPLUS: backfill — imported opening swap {sid} "
            f"({capacity} sats)", level="info")
        imported["opening"] += 1

    def _backfill_completed(self, entry: Dict, now: int, imported: Dict,
                            skipped: List[str], warnings: List[str]) -> None:
        sid = entry.get("id")
        if not sid:
            return
        sid = str(sid)
        if self._db.lnplus_get_swap(sid):
            skipped.append(sid)
            return
        ends_ts = _parse_ts(entry.get("ends"))
        incoming = entry.get("incoming_peer_pubkey")
        incoming_peer = incoming if _valid_pubkey(incoming) else None
        if incoming and not incoming_peer:
            warnings.append(
                f"swap {sid}: invalid incoming_peer_pubkey on import — "
                "incoming_peer left NULL")

        if ends_ts and ends_ts > now:
            self._backfill_running_contract(
                sid, entry, ends_ts, incoming_peer, now, imported, warnings)
        else:
            if not ends_ts:
                warnings.append(
                    f"swap {sid}: completed entry has no parseable 'ends' "
                    "— importing as ended anyway")
            self._backfill_ended_contract(
                sid, entry, ends_ts, incoming_peer, now, imported, warnings)

    def _backfill_running_contract(self, sid: str, entry: Dict, ends_ts: int,
                                   incoming_peer: Optional[str], now: int,
                                   imported: Dict, warnings: List[str]) -> None:
        """Rule 3: a still-running manual contract is NOT protected by
        no_close until this import runs — the planner could otherwise
        force-close a live contract (defection). outbound_peer is derived
        from the authoritative, complete participant list on get_swap(sid)
        (exact, unlike the pre-apply letter inference)."""
        outbound_peer, detail = self._derive_outbound_for_import(sid, warnings)
        capacity = entry.get("capacity_sats")
        if capacity is None and isinstance(detail, dict):
            capacity = detail.get("capacity_sats")
        if capacity is None:
            warnings.append(
                f"swap {sid}: capacity_sats unknown on import (defaulted "
                "to 0)")
            capacity = 0
        duration = int(entry.get("duration_months") or 0)
        self._db.lnplus_record_swap(
            sid, "opened", int(capacity), duration,
            outbound_peer=outbound_peer, incoming_peer=incoming_peer,
            metadata={"imported": True, "imported_at": now})
        self._db.lnplus_update_swap(sid, ends_at=ends_ts)
        self._plugin.log(
            f"LNPLUS: backfill — imported running contract {sid} "
            f"(outbound_peer={'set' if outbound_peer else 'NULL'}); phase "
            "4 this pass will activate no_close protection", level="info")
        imported["active"] += 1

    def _derive_outbound_for_import(self, sid: str, warnings: List[str]):
        """Fetch get_swap(sid), locate our own participant by pubkey
        (self.rpc.getinfo()["id"]), and derive outbound as the next
        participant_identifier cyclically — the participant list on a
        completed entry is final, so this is exact (not an inference).
        Never raises: any failure is logged at ERROR naming the swap and
        telling the operator to protect the channel manually. Returns
        (outbound_peer_or_None, detail_or_None)."""
        try:
            detail = self._client.get_swap(sid)
        except Exception as e:
            self._plugin.log(
                f"LNPLUS: backfill — get_swap({sid}) failed while "
                f"importing a running contract: {e}; importing with "
                "outbound_peer NULL — operator must protect this channel "
                "manually", level="error")
            return None, None
        participants = detail.get("participants") if isinstance(detail, dict) else None
        if not participants:
            self._plugin.log(
                f"LNPLUS: backfill — get_swap({sid}) returned no usable "
                "participants while importing a running contract; "
                "importing with outbound_peer NULL — operator must "
                "protect this channel manually", level="error")
            return None, (detail if isinstance(detail, dict) else None)
        try:
            our_id = self.rpc.getinfo().get("id")
        except Exception:
            our_id = None
        by_ident = {}
        our_ident = None
        for p in participants:
            ident = p.get("participant_identifier")
            pk = p.get("pubkey")
            if ident is None:
                continue
            by_ident[ident] = pk
            if our_id and pk == our_id:
                our_ident = ident
        if our_ident is None or not by_ident:
            self._plugin.log(
                f"LNPLUS: backfill — could not locate our own pubkey "
                f"among swap {sid}'s participants; importing with "
                "outbound_peer NULL — operator must protect this channel "
                "manually", level="error")
            return None, detail
        letters = sorted(by_ident.keys())
        idx = letters.index(our_ident)
        outbound_pk = by_ident.get(letters[(idx + 1) % len(letters)])
        if not _valid_pubkey(outbound_pk):
            self._plugin.log(
                f"LNPLUS: backfill — derived outbound pubkey for swap "
                f"{sid} is invalid; importing with outbound_peer NULL — "
                "operator must protect this channel manually", level="error")
            return None, detail
        return outbound_pk, detail

    def _backfill_ended_contract(self, sid: str, entry: Dict, ends_ts: Optional[int],
                                 incoming_peer: Optional[str], now: int,
                                 imported: Dict, warnings: List[str]) -> None:
        """Rule 4: a contract that already ended before this automation
        existed. Never rate it — the still-open heuristic used by
        _finalize is only valid to check AT contract end, and a peer who
        legitimately closed after expiry must not be defamed. Bumping peer
        history (no rating, no defection) is a best-effort seed only."""
        capacity = entry.get("capacity_sats")
        if capacity is None:
            warnings.append(
                f"swap {sid}: capacity_sats unknown on import (defaulted "
                "to 0)")
            capacity = 0
        duration = int(entry.get("duration_months") or 0)
        self._db.lnplus_record_swap(
            sid, "ended", int(capacity), duration,
            incoming_peer=incoming_peer,
            metadata={"imported": True, "imported_at": now})
        fields = {"outcome": "imported_pre_automation"}
        if ends_ts:
            fields["ends_at"] = ends_ts
        self._db.lnplus_update_swap(sid, **fields)
        self._plugin.log(
            f"LNPLUS: backfill — imported ended contract {sid} (no rating "
            "filed — still-open heuristic is only valid at contract end)",
            level="info")
        if incoming_peer:
            try:
                self._db.lnplus_bump_peer(incoming_peer)
            except Exception as e:
                self._plugin.log(
                    f"LNPLUS: backfill — lnplus_bump_peer failed for "
                    f"{incoming_peer}: {e}", level="warn")
        imported["ended"] += 1

    def _resolve_capacity_duration(self, entry: Dict, sid: str, warnings: List[str]):
        """Rule 1/2: capacity_sats/duration_months from the entry, falling
        back to get_swap(sid) detail if either is absent; if still
        unknown, default to 0 and warn."""
        capacity = entry.get("capacity_sats")
        duration = entry.get("duration_months")
        if capacity is None or duration is None:
            try:
                detail = self._client.get_swap(sid)
            except Exception:
                detail = None
            if isinstance(detail, dict):
                if capacity is None:
                    capacity = detail.get("capacity_sats")
                if duration is None:
                    duration = detail.get("duration_months")
        if capacity is None:
            warnings.append(
                f"swap {sid}: capacity_sats unknown on import (defaulted "
                "to 0)")
            capacity = 0
        if duration is None:
            warnings.append(
                f"swap {sid}: duration_months unknown on import (defaulted "
                "to 0)")
            duration = 0
        return int(capacity), int(duration)

    # -- watcher -----------------------------------------------------------
    def run_watcher_once(self) -> Dict:
        if not self._watcher_lock.acquire(blocking=False):
            return {"skipped": "watcher already running"}
        try:
            result = self._run_watcher_once_locked()
            self._record_watcher_pass(result)
            return result
        finally:
            self._watcher_lock.release()

    def _record_watcher_pass(self, summary: Dict) -> None:
        """B13: in-memory only (no DB) — recorded after every pass (success
        or outage-skip alike) for get_status() observability."""
        counts = {k: (len(v) if isinstance(v, list) else v)
                  for k, v in (summary or {}).items()}
        self._poll_notifications()
        self._last_watcher_pass = {"ts": int(time.time()), "summary": counts}

    def _run_watcher_once_locked(self) -> Dict:
        summary = {"opened": [], "activated": [], "finalized": [],
                   "withdrawn": [], "errors": []}

        # Phase 1: fetch live state. LN+ outage must not stall a funded
        # deadline and must NOT trip the breaker — fall through only to
        # phase 3b, which drives obligations off the local ledger.
        try:
            my = self._client.get_my_swaps()
        except LNPlusError as e:
            self._plugin.log(f"LNPLUS: get_my_swaps unreachable: {e}", level="warn")
            summary["skipped"] = "lnplus unreachable"
            # B1: outage path — LN+ is unreachable so we cannot know which
            # swaps it still recognizes. Drive obligations off the local
            # ledger unconditionally (live_ids=None disables the filter) —
            # a funded deadline must not wait on LN+ being back up.
            self._phase_3b(summary, set(), live_ids=None)
            self._prune_terminal_rows()
            self._retry_pending_ratings()
            return summary

        # Phase 2: reconcile (breaker does not block the phases below).
        try:
            self._reconcile(my)
        except Exception as e:
            self._plugin.log(f"LNPLUS: reconcile error: {e}", level="error")

        # Phase 3: entries LN+ shows as filled/opening.
        processed_opening_ids = set()
        for entry in (my.get("opening") or []):
            sid = entry.get("id")
            if not sid:
                continue
            row = self._db.lnplus_get_swap(sid)
            if not row:
                continue
            row_status = row.get("status")
            if row_status == "opened":
                # Already funded and locally marked opened on a prior pass —
                # LN+ may still list it under "opening" for a cycle or two.
                # Do not downgrade status back to "opening" or re-run
                # complete_application; phase 4 handles activation once LN+
                # reports it completed.
                continue
            if row_status not in ("applied", "opening"):
                # C2 fix: a row we abandoned locally (e.g. status "failed" via
                # revenue-lnplus-abandon) must stay terminal even if LN+ still
                # lists the swap under "opening" for a stale cycle or two.
                # Flipping it back to "opening" here would resurrect it and
                # trigger a real fundchannel/complete_application against an
                # application we deliberately walked away from — no write, no
                # open attempt, for any terminal status (failed/withdrawn/
                # ended/active).
                self._plugin.log(
                    f"LNPLUS: swap {sid} is terminal locally (status "
                    f"{row_status!r}) but LN+ still lists it under 'opening' — "
                    "ignoring, not resurrecting", level="warn")
                continue
            try:
                # I3: LN+ is an untrusted API — never write an unvalidated
                # pubkey into the row (it later feeds connect/fundchannel).
                # An invalid value means we skip this row for the pass
                # entirely: no write, no open attempt.
                peer = entry.get("outgoing_peer_pubkey")
                if peer is not None and not _valid_pubkey(peer):
                    self._plugin.log(
                        f"LNPLUS: swap {sid} — LN+ returned an invalid "
                        f"outgoing_peer_pubkey ({peer!r}); refusing to write "
                        "or open this pass", level="error")
                    continue
                fields = {"status": "opening"}
                if peer:
                    if row.get("outbound_peer") and row.get("outbound_peer") != peer:
                        # LN+'s live assignment is authoritative — our
                        # pre-apply letter inference (_infer_assignment) can
                        # be wrong — but the change must be visible, not
                        # silently swapped underneath the operator.
                        self._plugin.log(
                            f"LNPLUS: swap {sid} — LN+-assigned outbound "
                            f"peer {peer[:16]}... differs from our "
                            f"apply-time inference "
                            f"{str(row.get('outbound_peer'))[:16]}...; "
                            "LN+'s value is authoritative", level="warn")
                    fields["outbound_peer"] = peer
                deadline_ts = _parse_ts(entry.get("deadline"))
                if deadline_ts:
                    fields["deadline_at"] = deadline_ts
                elif not row.get("deadline_at"):
                    # Gate 10 backstop: LN+ gave us no parseable deadline and we
                    # have no local one yet. The missed-deadline breaker must
                    # never be silently disabled, so stamp a conservative local
                    # 48h estimate (the LN+ contractual open window) rather than
                    # leaving deadline_at null forever.
                    fallback_deadline = int(time.time()) + 48 * 3600
                    fields["deadline_at"] = fallback_deadline
                    self._plugin.log(
                        f"LNPLUS: swap {sid} — LN+ supplied no parseable "
                        f"deadline; local 48h estimate ({fallback_deadline}) "
                        "in force", level="warn")
                self._db.lnplus_update_swap(sid, **fields)
                row = self._db.lnplus_get_swap(sid)
                opened_now = self._execute_swap_open(row, entry)
                # Touched this pass either way, so 3b does not redo it — but
                # only a row that actually reached 'opened' counts as opened.
                processed_opening_ids.add(sid)
                if opened_now:
                    summary["opened"].append(sid)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error opening swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

        # Phase 3b: local opening rows not touched above this pass — the
        # funding attempt must not wait on LN+ round-trips.
        # B1: on the success path, only fund rows LN+ still recognizes
        # (opening ∪ completed). A local 'opening' row absent from both
        # (creator cancelled / we were banned / swap deleted) is a dead
        # swap — funding it would commit an on-chain channel for nothing.
        opening_ids = {e.get("id") for e in (my.get("opening") or []) if e.get("id")}
        completed_ids = {e.get("id") for e in (my.get("completed") or []) if e.get("id")}
        self._phase_3b(summary, processed_opening_ids, live_ids=opening_ids | completed_ids)

        # Phase 4: swaps LN+ reports completed -> activate protection.
        completed_by_id = {e.get("id"): e for e in (my.get("completed") or []) if e.get("id")}
        for row in self._db.lnplus_get_swaps_by_status(["opened"]):
            sid = row["swap_id"]
            entry = completed_by_id.get(sid)
            if not entry:
                continue
            try:
                self._activate(row, entry)
                summary["activated"].append(sid)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error activating swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

        # Phase 5: active contracts — watch for mid-contract defection and
        # finalize once past ends_at.
        now = int(time.time())
        for row in self._db.lnplus_get_swaps_by_status(["active"]):
            sid = row["swap_id"]
            try:
                ends_at = row.get("ends_at")
                if ends_at and now >= ends_at:
                    self._finalize(row)
                    summary["finalized"].append(sid)
                else:
                    # Self-heal: contracts activated before the incoming-side
                    # protection existed (incoming_tag_added never evaluated)
                    # get the counterparty channel tagged now. Idempotent —
                    # stamps 0/1 on first evaluation.
                    if row.get("incoming_tag_added") is None and row.get("incoming_peer"):
                        self._protect_peer_no_close(sid, row["incoming_peer"],
                                                    "incoming_tag_added")
                    # Lazy-eval audit F3a: retry the OUTBOUND side too — a
                    # transient add_tag failure at activation left the
                    # contract channel unprotected for the whole term.
                    if row.get("tag_added") is None and row.get("outbound_peer"):
                        self._protect_peer_no_close(sid, row["outbound_peer"],
                                                    "tag_added")
                    self._check_mid_contract_vanish(row)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error processing active swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

        # Phase 6: applications that timed out still pending.
        try:
            summary["withdrawn"].extend(self._handle_pending_timeouts(my))
        except Exception as e:
            self._plugin.log(f"LNPLUS: pending timeout phase error: {e}", level="error")

        # Phase 7: cheap terminal-row pruning (C-6, 2026-07-08 audit).
        self._prune_terminal_rows()
        self._retry_pending_ratings()

        return summary

    def _prune_terminal_rows(self) -> None:
        """C-6: keep the ledger from growing unbounded — terminal rows
        (ended/failed/withdrawn/cancelled_remote) old enough to be pure
        history are deleted once per watcher pass. Best-effort: a DB
        hiccup here must never fail the pass."""
        try:
            pruned = self._db.lnplus_prune_terminal()
            if pruned:
                self._plugin.log(f"LNPLUS: pruned {pruned} terminal swap row(s)", level="debug")
        except Exception as e:
            self._plugin.log(f"LNPLUS: terminal-row pruning failed: {e}", level="warn")

    def _phase_3b(self, summary: Dict, processed_opening_ids, live_ids=None) -> None:
        """B1: live_ids is the set of swap ids LN+ still recognizes (opening
        ∪ completed) on a SUCCESSFUL get_my_swaps fetch. When not None, a
        local 'opening' row absent from it is dead (creator cancelled / we
        were banned / swap deleted) and must be skipped, not funded.
        live_ids=None means the fetch itself failed (outage) — drive
        obligations off the local ledger unconditionally, as before."""
        for row in self._db.lnplus_get_swaps_by_status(["opening"]):
            sid = row["swap_id"]
            if sid in processed_opening_ids:
                continue
            if not row.get("outbound_peer") or not row.get("deadline_at"):
                continue
            if live_ids is not None and sid not in live_ids:
                self._plugin.log(
                    f"LNPLUS: swap {sid} is 'opening' locally but LN+ no "
                    "longer recognizes it (missing from opening/completed) "
                    "— not funding a channel for a dead swap", level="warn")
                continue
            try:
                if self._execute_swap_open(row):
                    summary["opened"].append(sid)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error retrying open for swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

    # -- gate 10-11: channel-open execution -------------------------------
    def _execute_swap_open(self, row: Dict, entry: Optional[Dict] = None) -> bool:
        """Return True only when the row actually reached 'opened'. Callers use
        this to decide whether the pass may report the swap as opened — a
        completion step that failed must never be summarised as progress."""
        sid = row["swap_id"]
        peer = row.get("outbound_peer")
        deadline = row.get("deadline_at")
        now = int(time.time())

        if row.get("channel_funding_txid"):
            # Already funded on a prior pass — only complete_application is left.
            return self._complete_and_mark_opened(sid)

        probe_ok = True
        try:
            channels = self.rpc.listpeerchannels().get("channels", []) or []
        except Exception as e:
            self._plugin.log(f"LNPLUS: listpeerchannels failed for swap {sid}: {e}", level="error")
            channels = []
            probe_ok = False
        capacity_sats = int(row.get("capacity_sats") or 0)
        # I5(b): a channel to this peer existing is not, by itself, proof
        # it is OUR swap channel — the idempotency skip used to claim ANY
        # open channel to `peer`, including an unrelated pre-existing one
        # (e.g. from before the swap or from a regular planner open), and
        # never fund the swap's own channel at all. Only trust a match by
        # capacity (this row's committed swap terms) or because we already
        # recorded our own funding txid for it (checked above, kept here for
        # defense-in-depth against any future reordering of this method).
        # B7: under experimental-dual-fund a peer contribution inflates
        # total_msat past our committed capacity, so a crash-retry would
        # never match on total_msat and would fund a SECOND channel. Our
        # own contribution (to_us_msat) on a freshly-opened channel still
        # equals the capacity we promised, so accept that as an equally
        # valid match — accepting a small staleness window where to_us_msat
        # could theoretically drift post-open (e.g. from routed HTLCs) is
        # preferable to a duplicate on-chain open.
        existing = next(
            (ch for ch in channels
             if ch.get("peer_id") == peer and ch.get("state") in _OPEN_STATES
             and (int(ch.get("total_msat", 0) or 0) // 1000 == capacity_sats
                  or int(ch.get("to_us_msat", 0) or 0) // 1000 == capacity_sats
                  or bool(row.get("channel_funding_txid")))),
            None)

        # Task 26 companion: a reservation HELD by a prior unknown-outcome
        # fundchannel attempt (see the exception handler below) is resolved
        # here, by measurement, exactly like capacity_planner's post-timeout
        # probe: a capacity-matched channel proves the broadcast (settle);
        # a successful listing with no such channel proves absence (release,
        # then retry normally); a failed probe resolves NOTHING.
        held = self._parse_held_reservation(row.get("outcome"))

        first_fund = False
        if existing is not None:
            # Idempotent-skip fundchannel; record txid if visible.
            txid = existing.get("funding_txid")
            if txid:
                self._db.lnplus_update_swap(sid, channel_funding_txid=txid, opened_at=now)
            if held:
                held_rid, held_cost = held
                self._plugin.log(
                    f"LNPLUS: swap {sid} — channel materialized after an "
                    f"unknown-outcome fundchannel; settling held reservation "
                    f"{held_rid} ({held_cost} sats)", level="info")
                self._settle_swap_open_reservation(held_rid, held_cost, sid)
                self._db.lnplus_update_swap(
                    sid, outcome="fundchannel unknown outcome resolved: "
                                 "channel found; held reservation settled")
        else:
            if held:
                held_rid, held_cost = held
                if not probe_ok:
                    # No evidence either way — keep holding, retry next pass.
                    self._plugin.log(
                        f"LNPLUS: swap {sid} — fundchannel outcome still "
                        f"unknown and the node cannot be probed; reservation "
                        f"{held_rid} stays HELD ({held_cost} sats); no new "
                        f"attempt this pass", level="error")
                    return False
                # A successful listing with no capacity-matched channel a
                # full pass later: the earlier attempt never funded.
                self._plugin.log(
                    f"LNPLUS: swap {sid} — fundchannel unknown outcome "
                    f"resolved: never funded; releasing held reservation "
                    f"{held_rid}", level="info")
                self._release_swap_open_reservation(held_rid, sid)
                self._db.lnplus_update_swap(
                    sid, outcome="fundchannel unknown outcome resolved: "
                                 "never funded; held reservation released")
            hours_left = ((deadline - now) / 3600.0) if deadline else -1.0
            feerate = "slow" if hours_left > 24 else ("normal" if hours_left > 12 else "urgent")

            addr = None
            if entry:
                addr = entry.get("outgoing_peer_clearnet_address") or entry.get("outgoing_peer_tor_address")
            # I3: API-derived strings never reach an RPC call unvalidated
            # (gate 16). A malformed/oversized address falls back to a bare
            # pubkey connect rather than being interpolated into `target`.
            if addr is not None and not _valid_connect_addr(addr):
                self._plugin.log(
                    f"LNPLUS: swap {sid} — LN+ returned a malformed connect "
                    f"address ({addr!r}); falling back to bare-pubkey connect",
                    level="warn")
                addr = None
            target = f"{peer}@{addr}" if addr else peer
            try:
                # Phase 3E: prefer the CLN adapter (cache-coherent);
                # raw-RPC fallback when not wired.
                if self.data_service is not None:
                    self.data_service.connect_peer(target)
                else:
                    self.rpc.connect(target)
            except Exception as e:
                self._plugin.log(f"LNPLUS: connect to {peer} failed for swap {sid}: {e}", level="warn")
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return False

            # Write intent before the irreversible external call.
            self._db.lnplus_update_swap(sid, status="opening", outcome="fundchannel attempt")

            # Atomic spend reservation immediately before fundchannel — pairs
            # this money-committing call with the repo's atomic rail exactly
            # like capacity_planner._execute_open/_execute_close do (BEGIN
            # IMMEDIATE, cross-category budget). reserve_spend refuses to
            # resurrect a terminal (spent/released) reservation_id, so the id
            # is unique per attempt: a released reservation from a prior
            # failed attempt can never block a retry.
            estimated_cost = (self._estimate_open_cost_fn() if self._estimate_open_cost_fn
                               else _DEFAULT_OPEN_COST_SATS)
            eff_budget, budget_since = (self._budget_params_fn() if self._budget_params_fn
                                         else (None, None))
            # Task 26 companion: the uuid suffix makes the id unique per
            # ATTEMPT even within one second — a held reservation released
            # by the resolution path above must never collide with (and so
            # block) the fresh retry reservation made in the same pass.
            reservation_id = f"lnplus-open-{sid}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
            if self._lnplus_governor_enabled():
                # Phase 2F: governor-gated reservation (same reserve_spend
                # underneath; fail-closed; no pause gate — invariant 6).
                reservation_active = self._governed_reserve_spend(
                    reservation_id=reservation_id,
                    amount_sats=estimated_cost,
                    metadata={"swap_id": sid, "peer_id": peer},
                    effective_budget_sats=eff_budget,
                    since_timestamp=budget_since,
                    swap_id=sid,
                    peer_id=peer,
                    capacity_sats=int(row["capacity_sats"]),
                )
            else:
                try:
                    reservation_active = bool(self._db.reserve_spend(
                        reservation_id=reservation_id,
                        amount_sats=estimated_cost,
                        category="channel_open",
                        subcategory="lnplus_swap",
                        metadata={"swap_id": sid, "peer_id": peer},
                        effective_budget_sats=eff_budget,
                        since_timestamp=budget_since,
                    ))
                except Exception as e:
                    self._plugin.log(
                        f"LNPLUS: budget reservation failed for swap {sid} — "
                        f"retrying next pass ({e})", level="warn")
                    self._maybe_trip_deadline_miss(row, sid, deadline, now)
                    return False
            if not reservation_active:
                self._plugin.log(
                    f"LNPLUS: budget reservation failed for swap {sid} — "
                    "retrying next pass", level="warn")
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return False

            try:
                # Phase 3E: prefer the CLN adapter (invalidates funds/
                # channel caches); raw-RPC fallback when not wired.
                if self.data_service is not None:
                    fund_params = {"id": peer,
                                   "amount": int(row["capacity_sats"])}
                    if feerate is not None:
                        fund_params["feerate"] = feerate
                    result = self.data_service.fund_channel(**fund_params)
                else:
                    result = self.rpc.fundchannel(
                        peer, int(row["capacity_sats"]), feerate=feerate)
            except Exception as e:
                # Task 26 companion (audit 2026-08-01): fundchannel is
                # broadcast-capable — a transport death/timeout may fire
                # AFTER the funding tx broadcast. Releasing the reservation
                # then would drop a committed on-chain cost off the unified
                # budget rail when the next pass matches the channel by
                # capacity (first_fund=False, settle never runs). HOLD the
                # reservation, persist a marker, and let the next watcher
                # pass resolve it by measurement (see `held` above). The
                # deadline-miss breaker is also skipped: the channel may
                # exist, so this is not (yet) a missed obligation.
                from .capacity_planner import _outcome_unknown_exception
                if _outcome_unknown_exception(e):
                    self._db.lnplus_update_swap(
                        sid, outcome=f"fundchannel outcome unknown; "
                                     f"{_HELD_RESERVATION_MARKER}"
                                     f"{reservation_id}:{estimated_cost}")
                    self._plugin.log(
                        f"LNPLUS: UNKNOWN OUTCOME: fundchannel to {peer} for "
                        f"swap {sid} raised after a possible broadcast ({e}). "
                        f"HOLDING reservation {reservation_id} "
                        f"({estimated_cost} sats) — a committed on-chain fee "
                        f"must never be released; the next watcher pass "
                        f"resolves it via listpeerchannels.", level="error")
                    return False
                self._plugin.log(f"LNPLUS: fundchannel to {peer} failed for swap {sid}: {e}", level="error")
                self._release_swap_open_reservation(reservation_id, sid)
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return False
            txid = result.get("txid") if isinstance(result, dict) else None
            if not txid:
                self._plugin.log(f"LNPLUS: fundchannel for swap {sid} returned no txid", level="error")
                self._release_swap_open_reservation(reservation_id, sid)
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return False
            # Outcome, after the call.
            self._db.lnplus_update_swap(sid, channel_funding_txid=txid, opened_at=now)
            first_fund = True
            # Settle loud/bounded-retry — mirrors capacity_planner's
            # _settle_capex_reservation: the on-chain tx already committed the
            # fee, so a settle write failure must never silently release the
            # reservation and drop the spend off the rail.
            self._settle_swap_open_reservation(reservation_id, estimated_cost, sid)

        marked_opened = self._complete_and_mark_opened(sid)
        if first_fund:
            self._record_swap_open_planner_action(
                row, status="completed",
                reason=f"LN+ swap {sid}: channel opened to {peer}")
        return marked_opened

    @staticmethod
    def _application_already_gone(e: LNPlusError) -> bool:
        """True when LN+ rejects delete_application because the application
        no longer EXISTS on their side (task 26 / task-23 finding 12).

        Like _already_past_opening, that is a success signal wearing a 422:
        the delete's goal (no application remains) is already achieved, and
        retrying can never succeed. Treating it as retryable left the row at
        'applied' forever — non-terminal, permanently past the timeout
        cutoff, re-selected every pass, and holding the serialization slot.

        Deliberately narrow: only the GONE class terminalizes. A 422 saying
        the application is "not pending" means the swap ADVANCED remotely —
        the next listing resolves that via promote/reconcile, and writing a
        terminal state here could bury a swap whose channel is about to
        fund. Transport failures and 5xx stay retryable.
        """
        if e.http_status != 422 or not isinstance(e.errors, dict):
            return False
        gone = ("no application", "not found", "does not exist")
        return any(needle in str(message).lower()
                   for value in e.errors.values()
                   for message in (value if isinstance(value, list) else [value])
                   for needle in gone)

    @staticmethod
    def _already_past_opening(e: LNPlusError) -> bool:
        """True when LN+ rejects complete_application because the application
        has ALREADY left the opening state.

        That is a success signal wearing a 422: the transition we are asking
        for has happened. It is the normal shape whenever a swap is completed
        outside this plugin (an operator finishing it on the LN+ site), and
        retrying can never succeed — so treating it as retryable pins the row
        at 'opening' forever, which starves phase 4 activation, skips the
        no_close protection, and jams the in-flight serialization gate.
        """
        if e.http_status != 422 or not isinstance(e.errors, dict):
            return False
        return any("not in opening state" in str(message).lower()
                   for value in e.errors.values()
                   for message in (value if isinstance(value, list) else [value]))

    def _complete_and_mark_opened(self, sid: str) -> bool:
        """Return True once the row is marked opened; False while it is still
        stuck at 'opening' so callers never report progress that didn't happen."""
        try:
            self._client.complete_application(sid)
        except LNPlusError as e:
            if not self._already_past_opening(e):
                self._plugin.log(f"LNPLUS: complete_application failed for {sid} (will retry): {e}",
                                  level="warn")
                return False
            self._plugin.log(
                f"LNPLUS: complete_application for {sid} reports it already left "
                "'opening' — treating as already complete and marking opened",
                level="info")
        self._db.lnplus_update_swap(sid, status="opened")
        return True

    @staticmethod
    def _parse_held_reservation(outcome) -> Optional[Tuple[str, int]]:
        """Extract (reservation_id, estimated_cost_sats) from an outcome text
        written by the unknown-outcome fundchannel path; None when absent or
        malformed (a malformed marker must never crash the watcher — the
        reservation then simply stays active until the operator reconciles)."""
        text = str(outcome or "")
        if _HELD_RESERVATION_MARKER not in text:
            return None
        try:
            payload = text.split(_HELD_RESERVATION_MARKER, 1)[1].strip()
            reservation_id, cost_text = payload.rsplit(":", 1)
            reservation_id = reservation_id.strip()
            cost = int(cost_text)
            if not reservation_id or cost <= 0:
                return None
            return reservation_id, cost
        except (ValueError, IndexError):
            return None

    def _release_swap_open_reservation(self, reservation_id: str, sid: str) -> None:
        """Best-effort release on a failed/aborted fundchannel attempt. Never a
        committed spend, so a release failure is logged and swallowed — the
        hourly retry makes a fresh (unique) reservation_id next pass."""
        try:
            self._db.release_spend_reservation(reservation_id)
        except Exception as e:
            self._plugin.log(
                f"LNPLUS: release_spend_reservation failed for swap {sid} "
                f"({reservation_id}): {e}", level="warn")

    def _settle_swap_open_reservation(self, reservation_id: str, estimated_cost: int, sid: str) -> None:
        """Settle a committed swap-open reservation as a spend event.

        Mirrors capacity_planner._settle_capex_reservation's loud/bounded-retry
        semantics: the on-chain tx already committed the fee, so the settle
        must NOT silently drop the spend-event write. mark_spend_reservation_spent
        is loud/idempotent — on a spend_events write failure it rolls the UPDATE
        back (or returns False) and the reservation stays 'active', keeping the
        committed fee counted on the unified rail. We retry a bounded number of
        times and, on persistent failure, log LOUDLY and leave the reservation
        active — never release a committed spend.
        """
        for attempt in range(3):
            try:
                if self._db.mark_spend_reservation_spent(
                    reservation_id=reservation_id,
                    actual_spent_sats=estimated_cost,
                    source="lnplus_swaps",
                    record_event=True,
                ):
                    return
            except Exception as e:
                self._plugin.log(
                    f"LNPLUS: swap-open settle write failed for swap {sid} "
                    f"(attempt {attempt + 1}/3): {e}", level="warn")
                continue
            self._plugin.log(
                f"LNPLUS: swap-open settle write returned failure for swap {sid} "
                f"(attempt {attempt + 1}/3); retrying", level="warn")
        # Persistent failure: the committed fee stays counted via the still-active
        # reservation. Surface it loudly so the operator can reconcile rather
        # than silently losing the write.
        self._plugin.log(
            f"LNPLUS: swap-open settle write PERSISTENTLY FAILED for swap {sid}: "
            f"committed fee kept as an active reservation ({estimated_cost} sats) "
            "so it stays counted against the unified budget; investigate spend_events.",
            level="error")

    def _maybe_trip_deadline_miss(self, row: Dict, sid: str, deadline, now: int) -> None:
        if not deadline or now <= deadline:
            return
        current = self._db.lnplus_get_swap(sid) or row
        if current.get("channel_funding_txid"):
            return
        self.trip_breaker(f"missed 48h deadline for swap {sid}")
        self._record_swap_open_planner_action(
            current, status="failed", reason=f"LN+ swap {sid}: missed open deadline")

    def _record_swap_open_planner_action(self, row: Dict, status: str, reason: str) -> None:
        try:
            action_id = self._db.record_planner_action(
                action_type="swap_open",
                peer_id=row.get("outbound_peer") or "unknown",
                amount_sats=row.get("capacity_sats"),
                estimated_cost_sats=0,
                reason=reason)
            self._db.update_planner_action(action_id, status=status)
        except Exception as e:
            self._plugin.log(f"LNPLUS: failed to record swap_open planner action: {e}", level="warn")

    # -- gate 12-13: activation --------------------------------------------
    def _activate(self, row: Dict, entry: Dict) -> None:
        sid = row["swap_id"]
        ends_ts = _parse_ts(entry.get("ends"))
        incoming = entry.get("incoming_peer_pubkey") or row.get("incoming_peer")

        # Intent: persist the authoritative contract terms first.
        pre_fields = {}
        if ends_ts:
            pre_fields["ends_at"] = ends_ts
        if incoming:
            pre_fields["incoming_peer"] = incoming
        if pre_fields:
            self._db.lnplus_update_swap(sid, **pre_fields)

        outbound_peer = row.get("outbound_peer")
        if outbound_peer:
            self._protect_peer_no_close(sid, outbound_peer, "tag_added")
        # The LN+ agreement binds BOTH sides of our participation: the
        # channel the counterparty opened TO us is part of the same contract,
        # and it is exactly the shape the planner's loser classifier likes to
        # close (all-inbound balance, no outbound history). Closing it
        # mid-contract is a defection just like closing our outbound side.
        incoming_peer = row.get("incoming_peer") or (self._db.lnplus_get_swap(sid) or {}).get("incoming_peer")
        if incoming_peer:
            self._protect_peer_no_close(sid, incoming_peer, "incoming_tag_added")

        # Outcome: contract is now protected.
        self._db.lnplus_update_swap(sid, status="active")

        # C-5 (2026-07-08 audit): best-effort planner-action breadcrumb —
        # observability parity with swap_apply/swap_open for the moment a
        # contract goes live.
        try:
            current = self._db.lnplus_get_swap(sid) or row
            action_id = self._db.record_planner_action(
                action_type="swap_complete",
                peer_id=outbound_peer or "unknown",
                amount_sats=current.get("capacity_sats"),
                reason=f"LN+ swap {sid} contract active until {current.get('ends_at')}")
            self._db.update_planner_action(action_id, status="completed")
        except Exception as e:
            self._plugin.log(f"LNPLUS: failed to record swap_complete planner action: {e}",
                              level="warn")

    def _check_mid_contract_vanish(self, row: Dict) -> None:
        peer = row.get("outbound_peer")
        if not peer:
            return
        try:
            channels = self.rpc.listpeerchannels().get("channels", [])
        except Exception:
            return
        if not isinstance(channels, list):
            return
        if not any(ch.get("peer_id") == peer for ch in channels):
            self._plugin.log(
                f"LNPLUS: swap channel to {peer} closed mid-contract — operator review needed",
                level="error")

    # -- gate 14: finalize / rate / release ---------------------------------
    def _finalize(self, row: Dict) -> None:
        sid = row["swap_id"]
        outbound_peer = row.get("outbound_peer")
        incoming_peer = row.get("incoming_peer")

        if not incoming_peer:
            # C-8 (2026-07-08 audit): NULL/empty incoming_peer means we have
            # no counterparty channel to check — the still-open heuristic is
            # meaningless without one. Never default a peer we cannot even
            # identify to a negative rating; release protection (subject to
            # the same shared-peer guard as the normal path) and close the
            # row out unjudged.
            self._plugin.log(
                f"LNPLUS: finalize for swap {sid} — cannot judge counterparty "
                "(incoming_peer is NULL/empty); ending unjudged, no rating "
                "filed", level="warn")
            self._release_no_close_if_ours(sid, row, outbound_peer)
            self._release_no_close_if_ours(sid, row, row.get("incoming_peer"),
                                           flag_column="incoming_tag_added")
            self._db.lnplus_update_swap(sid, status="ended", outcome="ended_unjudged")
            return

        positive = self._incoming_channel_open(incoming_peer)
        if positive is None:
            # B2: listpeerchannels FAILED (unknown state) — this is not the
            # same as the RPC answering "no channel" (genuine defection).
            # Filing a permanent negative rating / ignore / defection bump
            # for an innocent peer on a transient RPC hiccup is a much worse
            # outcome than retrying next hourly pass, so make no state
            # change at all and leave the row 'active'.
            self._plugin.log(
                f"LNPLUS: finalize for swap {sid} deferred — listpeerchannels "
                "failed (unknown channel state), not defaulting to a "
                "negative rating; will retry next pass", level="warn")
            return
        rating = "positive" if positive else "negative"

        rating_filed = self._try_create_rating(sid, rating)

        self._release_no_close_if_ours(sid, row, outbound_peer)
        self._release_no_close_if_ours(sid, row, incoming_peer,
                                       flag_column="incoming_tag_added")

        if incoming_peer:
            try:
                self._db.lnplus_bump_peer(incoming_peer, defection=(not positive), rating=rating)
            except Exception as e:
                self._plugin.log(f"LNPLUS: lnplus_bump_peer failed for {incoming_peer}: {e}",
                                  level="warn")
        if outbound_peer:
            try:
                self._db.lnplus_bump_peer(outbound_peer)
            except Exception as e:
                self._plugin.log(f"LNPLUS: lnplus_bump_peer failed for {outbound_peer}: {e}",
                                  level="warn")

        if not positive and incoming_peer and self._ignore_peer_fn:
            try:
                self._ignore_peer_fn(incoming_peer, "LN+ swap defection")
            except Exception as e:
                self._plugin.log(f"LNPLUS: ignore_peer_fn failed for {incoming_peer}: {e}",
                                  level="warn")

        # Outcome, after best-effort ratings/tags/ignore. A transiently
        # failed create_rating parks the row in ended_rating_pending so the
        # watcher retries it (bounded: 7 days past ends_at) instead of the
        # swap silently ending unrated on LN+.
        final_status = "ended" if rating_filed else "ended_rating_pending"
        self._db.lnplus_update_swap(sid, status=final_status, outcome=rating)

    _RATING_RETRY_WINDOW_SECONDS = 7 * 86400

    def _try_create_rating(self, sid: str, rating: str) -> bool:
        """Attempt create_rating once. True = filed (or LN+ says it already
        was); False = transient failure worth retrying next pass."""
        try:
            self._client.create_rating(sid, rating)
            return True
        except LNPlusError as e:
            errors = getattr(e, "errors", None)
            blob = str(errors or e).lower()
            if "already" in blob or "once" in blob:
                # One rating per swap: an already-rated answer is success.
                self._plugin.log(
                    f"LNPLUS: create_rating for swap {sid}: LN+ reports "
                    "already rated — treating as filed", level="info")
                return True
            self._plugin.log(f"LNPLUS: create_rating failed for swap {sid}: {e}",
                              level="warn")
            return False
        except Exception as e:
            self._plugin.log(f"LNPLUS: create_rating failed for swap {sid}: {e}",
                              level="warn")
            return False

    def _retry_pending_ratings(self) -> None:
        """Watcher phase: retry ratings for rows parked in
        ended_rating_pending. Bounded: give up (-> ended) once we are more
        than _RATING_RETRY_WINDOW_SECONDS past the contract end."""
        now = int(time.time())
        for row in self._db.lnplus_get_swaps_by_status(["ended_rating_pending"]):
            sid = row["swap_id"]
            rating = row.get("outcome") or "positive"
            try:
                deadline = (row.get("ends_at") or now) + self._RATING_RETRY_WINDOW_SECONDS
                if self._try_create_rating(sid, rating):
                    self._db.lnplus_update_swap(sid, status="ended")
                elif now > deadline:
                    self._plugin.log(
                        f"LNPLUS: giving up on rating swap {sid} after 7 days "
                        "of retries — ending unrated", level="warn")
                    self._db.lnplus_update_swap(sid, status="ended",
                                                outcome=f"{rating}_rating_unfiled")
            except Exception as e:
                self._plugin.log(f"LNPLUS: rating retry error for swap {sid}: {e}",
                                  level="warn")

    def _protect_peer_no_close(self, sid: str, peer: str, flag_column: str) -> None:
        """Apply no_close protection to a contract peer, recording in
        flag_column ('tag_added' for outbound, 'incoming_tag_added' for the
        counterparty's channel to us) whether WE added the tag. C-3: a
        pre-existing tag (operator-set or another contract's) is recorded as
        0 so release never clobbers it."""
        already_tagged = False
        lookup_failed = False
        try:
            policy = self._policy.get_policy(peer)
            already_tagged = bool(policy and policy.has_tag("no_close"))
        except Exception as e:
            lookup_failed = True
            self._plugin.log(
                f"LNPLUS: get_policy failed for {peer} while protecting "
                f"swap {sid}: {e}", level="warn")
        if lookup_failed:
            # Lazy-eval audit F3b: under uncertainty, protect but NEVER claim
            # ownership — a pre-existing operator tag stamped as ours would be
            # stripped at release. Worst case now: a stale tag needing manual
            # cleanup (the safe direction).
            try:
                self._policy.add_tag(peer, "no_close")
            except Exception as e:
                self._plugin.log(f"LNPLUS: add_tag(no_close) failed for {peer}: {e}",
                                  level="warn")
            self._db.lnplus_update_swap(sid, **{flag_column: 0})
            return
        if already_tagged:
            self._db.lnplus_update_swap(sid, **{flag_column: 0})
        else:
            try:
                self._policy.add_tag(peer, "no_close")
                self._db.lnplus_update_swap(sid, **{flag_column: 1})
            except Exception as e:
                self._plugin.log(f"LNPLUS: add_tag(no_close) failed for {peer}: {e}",
                                  level="warn")

    def _release_no_close_if_ours(self, sid: str, row: Dict, peer: Optional[str],
                                  flag_column: str = "tag_added") -> None:
        """C-3 (2026-07-08 audit): unconditionally removing no_close here used
        to clobber an operator-set tag or another running contract's
        protection on the same peer. Only remove it when THIS row is the one
        that added it (flag_column == 1, stamped by _protect_peer_no_close)
        AND no OTHER 'active' row still references the same peer in EITHER
        contract role (outbound or incoming)."""
        if not peer:
            return
        if row.get(flag_column) != 1:
            return
        other_active = [r for r in self._db.lnplus_get_swaps_by_status(["active"])
                        if r.get("swap_id") != sid
                        and peer in (r.get("outbound_peer"), r.get("incoming_peer"))]
        if other_active:
            self._plugin.log(
                f"LNPLUS: swap {sid} ended but no_close on {peer} is "
                f"still held by {len(other_active)} other active contract(s) "
                "— not removing", level="info")
            return
        try:
            self._policy.remove_tag(peer, "no_close")
        except Exception as e:
            self._plugin.log(f"LNPLUS: remove_tag(no_close) failed for {peer}: {e}",
                              level="warn")

    def _incoming_channel_open(self, incoming_peer) -> Optional[bool]:
        """True/False is a genuine RPC answer ("channel open"/"no channel").
        B2: None means the RPC itself FAILED — distinct from an authoritative
        "no channel" answer, and must not be treated as a negative-rating
        signal by the caller."""
        if not incoming_peer:
            return False
        try:
            channels = self.rpc.listpeerchannels().get("channels", [])
        except Exception:
            return None
        if not isinstance(channels, list):
            return None
        return any(ch.get("peer_id") == incoming_peer and ch.get("state") in _OPEN_STATES
                   for ch in channels)

    # -- gate: pending-application timeout ----------------------------------
    def _handle_pending_timeouts(self, my: Dict) -> List[str]:
        withdrawn = []
        timeout_days = int(getattr(self._config, "lnplus_pending_timeout_days", 7) or 7)
        cutoff = int(time.time()) - timeout_days * 86400
        pending_ids = {s.get("id") for s in ((my or {}).get("pending") or [])}
        for row in self._db.lnplus_get_swaps_by_status(["applied"]):
            sid = row["swap_id"]
            applied_at = row.get("applied_at") or 0
            if applied_at > cutoff:
                continue
            if sid not in pending_ids:
                continue
            try:
                self._client.delete_application(sid)
            except LNPlusError as e:
                if not self._application_already_gone(e):
                    self._plugin.log(f"LNPLUS: delete_application failed for {sid}: {e}", level="warn")
                    continue
                self._plugin.log(
                    f"LNPLUS: delete_application for {sid} reports the application "
                    "is already gone — treating the withdrawal as complete",
                    level="info")
            except Exception as e:
                self._plugin.log(f"LNPLUS: delete_application failed for {sid}: {e}", level="warn")
                continue
            self._db.lnplus_update_swap(sid, status="withdrawn")
            withdrawn.append(sid)
        return withdrawn

    # -- status --------------------------------------------------------------
    def _poll_notifications(self) -> None:
        """Observability only: fetch LN+ notifications once per successful
        watcher pass, log them, keep the last 10 for get_status, mark read.
        No autonomous actions are taken from notifications — the polling
        phases remain the source of truth for obligations. Best-effort."""
        try:
            result = self._client.get_notifications()
            notes = result.get("notifications") or []
            if not notes:
                return
            for n in notes[:20]:
                self._plugin.log(
                    f"LNPLUS: notification [{n.get('type', '?')}] "
                    f"{str(n.get('body', ''))[:200]}", level="info")
            keep = [{"type": n.get("type"), "body": str(n.get("body", ""))[:200],
                     "created_at": n.get("created_at"), "url": n.get("url")}
                    for n in notes[:10]]
            self._recent_notifications = (keep + getattr(self, "_recent_notifications", []))[:10]
            try:
                self._client.mark_read_notifications()
            except Exception as e:
                self._plugin.log(f"LNPLUS: mark_read_notifications failed: {e}", level="debug")
        except Exception as e:
            self._plugin.log(f"LNPLUS: notification poll failed: {e}", level="debug")

    def get_status(self) -> Dict:
        recent_ended = self._db.lnplus_get_swaps_by_status(["ended"])
        # B13: recent_failed/backfill_done/last_watcher_pass — operator
        # observability into terminal-failure rows, whether adoption of
        # pre-existing LN+ state has completed, and the most recent
        # watcher pass (in-memory only, no DB).
        recent_failed = self._db.lnplus_get_swaps_by_status(
            list(_TERMINAL_PENDING_GHOST_STATUSES))
        return {
            "breaker": self.breaker_tripped(),
            "inflight": self._db.lnplus_inflight_swaps(),
            "active": self._db.lnplus_get_swaps_by_status(["active"]),
            "recent_ended": recent_ended[-10:],
            "recent_failed": recent_failed[-5:],
            "backfill_done": bool(self._db.get_config_override(_BACKFILL_FLAG)),
            "last_watcher_pass": self._last_watcher_pass,
            "recent_notifications": getattr(self, "_recent_notifications", []),
        }
