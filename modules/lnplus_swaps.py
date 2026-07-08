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

    def _check_existing_channel(self, swap) -> Optional[str]:
        """I5(a): reject a swap whose INFERRED outbound peer we already have
        a channel to. Without this an idempotency check can later (I5b)
        misread a coincidental pre-existing channel to that peer as "our
        swap channel" and never open the one the swap actually requires;
        pre-empting it here also avoids burning the one-application-per-node
        serialization slot on a swap that cannot add new capacity anyway.
        Fail-open on an RPC hiccup (matches _check_mid_contract_vanish /
        _execute_swap_open's existing-channel lookups elsewhere in this
        module) — real state is re-checked at open time regardless."""
        outbound_peer = self._infer_assignment(swap).get("outbound_peer")
        if not outbound_peer:
            return None
        try:
            channels = self.rpc.listpeerchannels(outbound_peer).get("channels", []) or []
        except Exception:
            return None
        if channels:
            return "terms:existing channel to assigned outbound peer"
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

        counterparties = [p for p in (swap.get("participants") or [])]
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
        scored.sort(key=lambda t: t[0], reverse=True)
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
_OPEN_STATES = ("OPENINGD", "CHANNELD_AWAITING_LOCKIN", "CHANNELD_NORMAL",
                "DUALOPEND_OPEN_INIT", "DUALOPEND_AWAITING_LOCKIN")
# Best-effort fallback used only when the caller does not wire the capacity
# planner's real cost/budget estimators in (estimate_open_cost_fn /
# budget_params_fn on SwapLifecycle.__init__). Still counted on the atomic
# spend rail at settle time either way.
_DEFAULT_OPEN_COST_SATS = 2500


class SwapLifecycle:
    """Obligations watcher / state machine (spec gates 10-14).

    Once an application is filled, the 48h open deadline and the eventual
    protection contract are IRREVERSIBLE COMMITMENTS. Every DB state
    transition here follows crash-safe ordering: write intent before acting
    on it externally, write outcome after the external call returns.
    Ratings and ignore_peer_fn calls are best-effort (try/except + log);
    the DB state machine transitions are not.
    """

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

    # -- breaker -------------------------------------------------------
    def breaker_tripped(self) -> Optional[str]:
        return self._db.get_config_override(_BREAKER_KEY)

    def trip_breaker(self, reason: str) -> None:
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
        my = my or {}
        pending_ids = {s.get("id") for s in (my.get("pending") or [])}
        opening_ids = {s.get("id") for s in (my.get("opening") or [])}
        completed_ids = {s.get("id") for s in (my.get("completed") or [])}
        local_inflight = self._db.lnplus_inflight_swaps()
        local_ids = {row["swap_id"] for row in local_inflight}

        ok = True
        for row in local_inflight:
            sid = row["swap_id"]
            status = row["status"]
            if status == "applied":
                compatible = sid in pending_ids or sid in opening_ids
            elif status in ("opening", "opened"):
                compatible = sid in opening_ids or sid in completed_ids
            else:
                compatible = True
            if not compatible:
                self.trip_breaker(
                    f"local swap {sid} (status {status}) missing/divergent on LN+")
                ok = False

        for sid in opening_ids:
            if sid and sid not in local_ids:
                self.trip_breaker(f"LN+ shows opening swap {sid} with no local record")
                ok = False

        # I1 fix: gate 0 was one-eyed — it only checked the opening list for
        # LN+-side entries with no local row. A pending application LN+ knows
        # about but we have no in-flight row for (applied/opening/opened) is
        # an untracked live commitment (a "ghost") that the pending-timeout
        # phase will never see (that phase only handles rows WE already know
        # about), so it must trip the breaker here instead of going unnoticed
        # forever.
        for sid in pending_ids:
            if sid and sid not in local_ids:
                self.trip_breaker(f"LN+ shows pending swap {sid} with no local record")
                ok = False
        return ok

    # -- watcher -----------------------------------------------------------
    def run_watcher_once(self) -> Dict:
        if not self._watcher_lock.acquire(blocking=False):
            return {"skipped": "watcher already running"}
        try:
            return self._run_watcher_once_locked()
        finally:
            self._watcher_lock.release()

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
            self._phase_3b(summary, set())
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
                self._execute_swap_open(row, entry)
                processed_opening_ids.add(sid)
                summary["opened"].append(sid)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error opening swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

        # Phase 3b: local opening rows not touched above this pass — the
        # funding attempt must not wait on LN+ round-trips.
        self._phase_3b(summary, processed_opening_ids)

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
                    self._check_mid_contract_vanish(row)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error processing active swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

        # Phase 6: applications that timed out still pending.
        try:
            summary["withdrawn"].extend(self._handle_pending_timeouts(my))
        except Exception as e:
            self._plugin.log(f"LNPLUS: pending timeout phase error: {e}", level="error")

        return summary

    def _phase_3b(self, summary: Dict, processed_opening_ids) -> None:
        for row in self._db.lnplus_get_swaps_by_status(["opening"]):
            sid = row["swap_id"]
            if sid in processed_opening_ids:
                continue
            if not row.get("outbound_peer") or not row.get("deadline_at"):
                continue
            try:
                self._execute_swap_open(row)
                summary["opened"].append(sid)
            except Exception as e:
                self._plugin.log(f"LNPLUS: error retrying open for swap {sid}: {e}", level="error")
                summary["errors"].append(sid)

    # -- gate 10-11: channel-open execution -------------------------------
    def _execute_swap_open(self, row: Dict, entry: Optional[Dict] = None) -> None:
        sid = row["swap_id"]
        peer = row.get("outbound_peer")
        deadline = row.get("deadline_at")
        now = int(time.time())

        if row.get("channel_funding_txid"):
            # Already funded on a prior pass — only complete_application is left.
            self._complete_and_mark_opened(sid)
            return

        try:
            channels = self.rpc.listpeerchannels().get("channels", []) or []
        except Exception as e:
            self._plugin.log(f"LNPLUS: listpeerchannels failed for swap {sid}: {e}", level="error")
            channels = []
        capacity_sats = int(row.get("capacity_sats") or 0)
        # I5(b): a channel to this peer existing is not, by itself, proof
        # it is OUR swap channel — the idempotency skip used to claim ANY
        # open channel to `peer`, including an unrelated pre-existing one
        # (e.g. from before the swap or from a regular planner open), and
        # never fund the swap's own channel at all. Only trust a match by
        # capacity (this row's committed swap terms) or because we already
        # recorded our own funding txid for it (checked above, kept here for
        # defense-in-depth against any future reordering of this method).
        existing = next(
            (ch for ch in channels
             if ch.get("peer_id") == peer and ch.get("state") in _OPEN_STATES
             and (int(ch.get("total_msat", 0) or 0) // 1000 == capacity_sats
                  or bool(row.get("channel_funding_txid")))),
            None)

        first_fund = False
        if existing is not None:
            # Idempotent-skip fundchannel; record txid if visible.
            txid = existing.get("funding_txid")
            if txid:
                self._db.lnplus_update_swap(sid, channel_funding_txid=txid, opened_at=now)
        else:
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
                self.rpc.connect(target)
            except Exception as e:
                self._plugin.log(f"LNPLUS: connect to {peer} failed for swap {sid}: {e}", level="warn")
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return

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
            reservation_id = f"lnplus-open-{sid}-{int(time.time())}"
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
                return
            if not reservation_active:
                self._plugin.log(
                    f"LNPLUS: budget reservation failed for swap {sid} — "
                    "retrying next pass", level="warn")
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return

            try:
                result = self.rpc.fundchannel(peer, int(row["capacity_sats"]), feerate=feerate)
            except Exception as e:
                self._plugin.log(f"LNPLUS: fundchannel to {peer} failed for swap {sid}: {e}", level="error")
                self._release_swap_open_reservation(reservation_id, sid)
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return
            txid = result.get("txid") if isinstance(result, dict) else None
            if not txid:
                self._plugin.log(f"LNPLUS: fundchannel for swap {sid} returned no txid", level="error")
                self._release_swap_open_reservation(reservation_id, sid)
                self._maybe_trip_deadline_miss(row, sid, deadline, now)
                return
            # Outcome, after the call.
            self._db.lnplus_update_swap(sid, channel_funding_txid=txid, opened_at=now)
            first_fund = True
            # Settle loud/bounded-retry — mirrors capacity_planner's
            # _settle_capex_reservation: the on-chain tx already committed the
            # fee, so a settle write failure must never silently release the
            # reservation and drop the spend off the rail.
            self._settle_swap_open_reservation(reservation_id, estimated_cost, sid)

        self._complete_and_mark_opened(sid)
        if first_fund:
            self._record_swap_open_planner_action(
                row, status="completed",
                reason=f"LN+ swap {sid}: channel opened to {peer}")

    def _complete_and_mark_opened(self, sid: str) -> None:
        try:
            self._client.complete_application(sid)
        except LNPlusError as e:
            self._plugin.log(f"LNPLUS: complete_application failed for {sid} (will retry): {e}",
                              level="warn")
            return
        self._db.lnplus_update_swap(sid, status="opened")

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
            try:
                self._policy.add_tag(outbound_peer, "no_close")
            except Exception as e:
                self._plugin.log(f"LNPLUS: add_tag(no_close) failed for {outbound_peer}: {e}",
                                  level="warn")

        # Outcome: contract is now protected.
        self._db.lnplus_update_swap(sid, status="active")

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

        positive = self._incoming_channel_open(incoming_peer)
        rating = "positive" if positive else "negative"

        try:
            self._client.create_rating(sid, rating)
        except Exception as e:
            self._plugin.log(f"LNPLUS: create_rating failed for swap {sid}: {e}", level="warn")

        if outbound_peer:
            try:
                self._policy.remove_tag(outbound_peer, "no_close")
            except Exception as e:
                self._plugin.log(f"LNPLUS: remove_tag(no_close) failed for {outbound_peer}: {e}",
                                  level="warn")

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

        # Outcome, after best-effort ratings/tags/ignore.
        self._db.lnplus_update_swap(sid, status="ended", outcome=rating)

    def _incoming_channel_open(self, incoming_peer) -> bool:
        if not incoming_peer:
            return False
        try:
            channels = self.rpc.listpeerchannels().get("channels", [])
        except Exception:
            return False
        if not isinstance(channels, list):
            return False
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
            except Exception as e:
                self._plugin.log(f"LNPLUS: delete_application failed for {sid}: {e}", level="warn")
                continue
            self._db.lnplus_update_swap(sid, status="withdrawn")
            withdrawn.append(sid)
        return withdrawn

    # -- status --------------------------------------------------------------
    def get_status(self) -> Dict:
        recent_ended = self._db.lnplus_get_swaps_by_status(["ended"])
        return {
            "breaker": self.breaker_tripped(),
            "inflight": self._db.lnplus_inflight_swaps(),
            "active": self._db.lnplus_get_swaps_by_status(["active"]),
            "recent_ended": recent_ended[-10:],
        }
