"""
rebalance_executor — Native rebalance engine using safe explicit routes.

Replaces sling for rebalance execution while keeping askrene and hive data on
the planning side. Fleet-planned and network-planned rebalances both execute
through a single explicit-route `getroute` + `sendpay` model with retry,
transient routing memory, and askrene result feedback.
"""

from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from modules.rebalance_memory import RebalanceRoutingMemory


class JobState(Enum):
    PENDING = "pending"
    ROUTING = "routing"
    SENDING = "sending"
    WAITING = "waiting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RebalanceResult:
    """Outcome of a rebalance execution."""
    success: bool
    fee_msat: int = 0
    fee_ppm: int = 0
    hops: int = 0
    route_type: str = ""       # "fleet" or "network"
    attempts: int = 0
    parts: int = 0             # MPP parts used
    error: Optional[str] = None


@dataclass
class RebalanceJob:
    """Tracks an in-flight rebalance."""
    job_id: str
    channel_id: str            # Destination channel SCID
    peer_id: str               # Destination peer pubkey
    amount_msat: int
    max_fee_msat: int
    route_type: str            # "fleet" or "network"
    state: JobState = JobState.PENDING
    payment_hash: str = ""
    label: str = ""            # Invoice label for cleanup
    attempts: List[Dict] = field(default_factory=list)
    start_time: int = 0
    result: Optional[RebalanceResult] = None

    def __post_init__(self):
        if not self.start_time:
            self.start_time = int(time.time())


class RebalanceExecutor:
    """Native rebalance execution using explicit circular routes.

    Fleet and network rebalances share the same safe execution path:
    build one explicit route, validate it, send it, and learn from the
    result. Askrene and hive layers still inform planning and route
    selection upstream.

    Thread-safe: jobs run in a ThreadPoolExecutor.
    """

    MAX_ATTEMPTS = 3
    MAX_CONCURRENT = 5
    FLEET_MAX_PARTS = 1
    NETWORK_MAX_PARTS = 3
    SENDPAY_TIMEOUT = 60
    INVOICE_EXPIRY = 300
    CHANNEL_BAN_TTL = 300
    FIRST_HOP_BAN_TTL = 60

    def __init__(self, plugin, config, database, hive_router=None):
        self.plugin = plugin
        self.config = config
        self.database = database
        self.hive_router = hive_router
        self._routing_memory = RebalanceRoutingMemory()
        self._our_id: Optional[str] = None
        self._pool = ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT,
                                        thread_name_prefix="rebal")
        self._jobs: Dict[str, RebalanceJob] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def _log(self, msg: str, level: str = "info") -> None:
        if self.plugin:
            self.plugin.log(f"[RebalanceExecutor] {msg}", level=level)

    def _get_our_id(self) -> Optional[str]:
        if self._our_id:
            return self._our_id
        try:
            self._our_id = self.plugin.rpc.getinfo().get("id")
        except Exception:
            pass
        return self._our_id

    # ------------------------------------------------------------------
    # Layer Selection
    # ------------------------------------------------------------------

    def _get_layers(self, route_type: str) -> List[str]:
        """Build layer list based on route type.

        NOTE: Do NOT include auto.sourcefree — it makes our outgoing
        channels appear zero-fee, causing WIRE_FEE_INSUFFICIENT when
        sendpay sends real HTLCs.  auto.localchans provides capacity
        without fee overrides.  hive-fleet layer correctly sets fleet
        member channels to 0 fee (which is the REAL fee they charge us).
        """
        layers = ["auto.localchans"]
        try:
            existing = self.plugin.rpc.call("askrene-listlayers", {})
            for l in existing.get("layers", []):
                name = l.get("layer", "")
                if route_type == "fleet":
                    if name.startswith("hive-") or name.startswith("revenue-"):
                        layers.append(name)
                else:
                    if name.startswith("revenue-"):
                        layers.append(name)
        except Exception:
            pass
        return layers

    # ------------------------------------------------------------------
    # Route Conversion
    # ------------------------------------------------------------------

    def _getroutes_to_sendpay(
        self, path: List[Dict], dest_channel: str, our_id: str,
        amount_msat: int
    ) -> List[Dict]:
        """Convert getroutes path to sendpay route format.

        getroutes: [{short_channel_id_dir, next_node_id, amount_msat, delay}]
        sendpay:   [{channel, id, amount_msat, delay}]

        Appends final hop back to ourselves for circular routing.
        """
        route = []
        for hop in path:
            scid_dir = hop.get("short_channel_id_dir", "")
            scid = scid_dir.split("/")[0] if "/" in scid_dir else scid_dir
            route.append({
                "channel": scid,
                "id": hop.get("next_node_id", ""),
                "amount_msat": hop.get("amount_msat", 0),
                "delay": hop.get("delay", 0),
            })

        # Final hop: dest peer forwards back to us via the dest channel
        route.append({
            "channel": dest_channel.replace(":", "x"),
            "id": our_id,
            "amount_msat": amount_msat,
            "delay": 18,
        })
        return route

    def _get_return_hop_policy(
        self,
        candidate,
        amount_msat: int,
        our_id: str,
    ) -> tuple[int, int]:
        """Amount and CLTV the destination peer needs to forward back to us."""
        scid = candidate.to_channel.replace(":", "x")
        base_msat = 0
        fee_ppm = 0
        cltv_delta = 6

        try:
            chans = self.plugin.rpc.listpeerchannels(candidate.to_peer_id)
            for ch in chans.get("channels", []):
                if ch.get("short_channel_id") != scid:
                    continue
                remote = ch.get("updates", {}).get("remote", {})
                fee_ppm_val = remote.get("fee_proportional_millionths")
                if fee_ppm_val is not None:
                    fee_ppm = int(fee_ppm_val or 0)
                fee_base_val = remote.get("fee_base_msat")
                if fee_base_val is not None:
                    base_msat = int(fee_base_val or 0)
                cltv_val = remote.get("cltv_expiry_delta")
                if cltv_val is not None:
                    cltv_delta = int(cltv_val or 6)
                break
        except Exception:
            pass

        if fee_ppm == 0 and base_msat == 0:
            try:
                chans = self.plugin.rpc.listchannels(source=candidate.to_peer_id)
                for ch in chans.get("channels", []):
                    if ch.get("destination") != our_id:
                        continue
                    if ch.get("short_channel_id") != scid:
                        continue
                    fee_ppm = int(ch.get("fee_per_millionth", 0) or 0)
                    base_msat = int(ch.get("base_fee_millisatoshi", 0) or 0)
                    cltv_delta = int(ch.get("delay", 6) or 6)
                    break
            except Exception:
                pass

        required_amount_msat = amount_msat + base_msat + (amount_msat * fee_ppm) // 1_000_000
        required_cltv = 18 + cltv_delta
        return required_amount_msat, required_cltv

    def _validate_sendpay_route(self, route: List[Dict], final_amount_msat: int) -> None:
        """Reject malformed routes before they can crash lightningd."""
        if not route:
            raise ValueError("empty_route")

        previous_amount = None
        for hop in route:
            amount = int(hop.get("amount_msat", 0) or 0)
            if amount <= 0:
                raise ValueError("non_positive_route_amount")
            if previous_amount is not None and amount > previous_amount:
                raise ValueError("increasing_route_amount")
            previous_amount = amount

        if int(route[0].get("amount_msat", 0) or 0) < int(final_amount_msat):
            raise ValueError("insufficient_first_hop_amount")

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _inform_channel(self, scid_dir: str, amount_msat: int, inform: str) -> None:
        """Send a single valid askrene channel update."""
        if not scid_dir or amount_msat <= 0:
            return
        if inform not in {"constrained", "unconstrained", "succeeded"}:
            raise ValueError(f"invalid askrene inform value: {inform}")
        try:
            self.plugin.rpc.call("askrene-inform-channel", {
                "layer": "revenue-local",
                "short_channel_id_dir": scid_dir,
                "amount_msat": amount_msat,
                "inform": inform,
            })
        except Exception:
            pass

    def _inform_result(self, path: List[Dict], amount_msat: int,
                       succeeded: bool) -> None:
        """Teach askrene about a successful route using per-hop amounts."""
        if not succeeded:
            raise ValueError("use _inform_failure() for failed routes")
        for hop in path:
            scid_dir = hop.get("short_channel_id_dir", "")
            hop_amount = int(hop.get("amount_msat", amount_msat) or amount_msat)
            self._inform_channel(scid_dir, hop_amount, "succeeded")

    def _inform_failure(self, path: List[Dict], failure: Dict[str, Any]) -> None:
        """Teach askrene about a routed failure using valid semantics.

        Prefix hops that definitely forwarded are marked unconstrained.
        The failing hop is marked constrained. Downstream hops are unknown.
        """
        if not path or failure.get("code") != 204:
            return

        erring_channel = failure.get("erring_channel")
        erring_direction = failure.get("erring_direction")
        fail_scid_dir = None
        if erring_channel and erring_direction is not None:
            fail_scid_dir = f"{erring_channel}/{erring_direction}"

        fail_index = None
        if fail_scid_dir:
            for idx, hop in enumerate(path):
                if hop.get("short_channel_id_dir") == fail_scid_dir:
                    fail_index = idx
                    break

        if fail_index is None:
            erring_index = int(failure.get("erring_index") or 0)
            if erring_index <= 0 or erring_index > len(path):
                return
            fail_index = erring_index - 1
            fail_scid_dir = path[fail_index].get("short_channel_id_dir", "")

        if not fail_scid_dir:
            return

        for hop in path[:fail_index]:
            scid_dir = hop.get("short_channel_id_dir", "")
            hop_amount = int(hop.get("amount_msat", 0) or 0)
            self._inform_channel(scid_dir, hop_amount, "unconstrained")

        failed_hop = path[fail_index]
        failed_amount = int(failed_hop.get("amount_msat", 0) or 0)
        self._inform_channel(fail_scid_dir, failed_amount, "constrained")

    def _path_for_inform(self, hops: List[Dict]) -> List[Dict]:
        """Normalize hop data to askrene-inform-channel format."""
        path = []
        for hop in hops:
            scid_dir = hop.get("short_channel_id_dir", "")
            if not scid_dir:
                channel = hop.get("channel", "")
                direction = hop.get("direction")
                if channel and direction is not None:
                    scid_dir = f"{channel}/{direction}"
            if scid_dir:
                path.append({
                    "short_channel_id_dir": scid_dir,
                    "amount_msat": int(hop.get("amount_msat", 0) or 0),
                })
        return path

    def _cleanup_failed_payment(self, payment_hash: str) -> None:
        """Delete a failed sendpay attempt so the hash can be retried."""
        if not payment_hash:
            return
        try:
            self.plugin.rpc.delpay(payment_hash, "failed")
        except Exception:
            pass

    def _parse_failure(self, exc: Exception) -> Dict[str, Any]:
        """Extract routing failure details from an RPC exception."""
        error = getattr(exc, "error", None)
        if not isinstance(error, dict):
            return {}
        data = error.get("data", {})
        if not isinstance(data, dict):
            data = {}
        return {
            "code": error.get("code"),
            "message": error.get("message", ""),
            "erring_index": data.get("erring_index"),
            "erring_channel": data.get("erring_channel"),
            "erring_direction": data.get("erring_direction"),
            "erring_node": data.get("erring_node"),
            "failcode": data.get("failcode", 0) or 0,
            "failcodename": data.get("failcodename", ""),
        }

    def _compute_network_route(
        self,
        job: RebalanceJob,
        candidate,
        our_id: str,
        excludes: List[str],
    ) -> tuple[List[Dict], List[Dict]]:
        """Build a safe explicit circular route using getroute."""
        source_scid = candidate.source_candidates[0] if candidate.source_candidates else ""
        source_peer = candidate.primary_source_peer_id
        if not source_scid or not source_peer:
            raise ValueError("no_source_channel")

        required_amount_msat, required_cltv = self._get_return_hop_policy(
            candidate,
            job.amount_msat,
            our_id,
        )

        getroute_kwargs = {
            "fromid": source_peer,
            "maxhops": 6,
            "fuzzpercent": 0,
        }
        merged_excludes = list(dict.fromkeys(excludes + self._routing_memory.current_excludes()))
        if merged_excludes:
            getroute_kwargs["exclude"] = merged_excludes

        route_result = self.plugin.rpc.getroute(
            candidate.to_peer_id,
            required_amount_msat,
            riskfactor=0,
            cltv=required_cltv,
            **getroute_kwargs,
        )
        route = route_result.get("route", [])
        if not route:
            raise ValueError("no_route_back")

        first_hop_scid = source_scid.replace(":", "x")
        forward_amount = route[0].get("amount_msat", job.amount_msat)

        source_fee_ppm = 0
        source_base_msat = 0
        source_cltv_delta = 6
        try:
            forward_hop = route[0]
            forward_scid = forward_hop.get("channel", "")
            forward_dir = forward_hop.get("direction")
            chans = self.plugin.rpc.listchannels(forward_scid)
            for ch in chans.get("channels", []):
                if ch.get("short_channel_id") != forward_scid:
                    continue
                if forward_dir is not None and ch.get("direction") != forward_dir:
                    continue
                fee_ppm_val = ch.get("fee_per_millionth")
                if fee_ppm_val is None:
                    fee_ppm_val = ch.get("fee_proportional_millionths", 0)
                fee_base_val = ch.get("base_fee_millisatoshi")
                if fee_base_val is None:
                    fee_base_val = ch.get("fee_base_msat", 0)
                delay_val = ch.get("delay")
                if delay_val is None:
                    delay_val = ch.get("cltv_expiry_delta", 6)
                source_cltv_delta = int(delay_val or 6)
                source_fee_ppm = int(fee_ppm_val or 0)
                source_base_msat = int(fee_base_val or 0)
                break
        except Exception:
            pass

        if source_fee_ppm == 0 and source_base_msat == 0:
            try:
                chans = self.plugin.rpc.listpeerchannels(source_peer)
                for ch in chans.get("channels", []):
                    if ch.get("short_channel_id") == first_hop_scid:
                        updates = ch.get("updates", {})
                        local = updates.get("local", {})
                        source_cltv_delta = int(local.get("cltv_expiry_delta", 6) or 6)
                        fee_ppm_val = local.get("fee_proportional_millionths")
                        if fee_ppm_val is None:
                            fee_ppm_val = ch.get("fee_proportional_millionths", 0)
                        fee_base_val = local.get("fee_base_msat")
                        if fee_base_val is None:
                            fee_base_val = ch.get("fee_base_msat", 0)
                        source_fee_ppm = int(fee_ppm_val or 0)
                        source_base_msat = int(fee_base_val or 0)
                        break
            except Exception:
                pass

        source_fee_msat = source_base_msat + (forward_amount * source_fee_ppm) // 1_000_000
        first_hop_amount = forward_amount + source_fee_msat
        first_hop_delay = route[0].get("delay", 18) + source_cltv_delta
        first_hop_direction = 1 if our_id > source_peer else 0

        first_hop = {
            "id": source_peer,
            "channel": first_hop_scid,
            "direction": first_hop_direction,
            "amount_msat": first_hop_amount,
            "delay": first_hop_delay,
            "style": "tlv",
        }
        final_hop = {
            "id": our_id,
            "channel": candidate.to_channel.replace(":", "x"),
            "amount_msat": job.amount_msat,
            "delay": 18,
            "style": "tlv",
        }
        full_route = [first_hop] + route + [final_hop]
        final_direction = 1 if candidate.to_peer_id > our_id else 0
        inform_path = self._path_for_inform([first_hop] + route)
        inform_path.append({
            "short_channel_id_dir": f"{candidate.to_channel.replace(':', 'x')}/{final_direction}",
            "amount_msat": job.amount_msat,
        })
        return full_route, inform_path

    def _learn_from_failure(
        self,
        full_route: List[Dict],
        failure: Dict[str, Any],
        exc: Exception,
    ) -> None:
        """Record short-lived routing memory from a failed attempt."""
        if not full_route:
            return

        if getattr(exc, "command", "") == "sendpay":
            direction = full_route[0].get("direction")
            if direction is not None:
                self._routing_memory.ban_channel(
                    f"{full_route[0]['channel']}/{direction}",
                    ttl_seconds=self.FIRST_HOP_BAN_TTL,
                )
            return

        code = failure.get("code")
        if code != 204:
            return

        erring_channel = failure.get("erring_channel")
        erring_direction = failure.get("erring_direction")
        erring_node = failure.get("erring_node")
        failcodename = failure.get("failcodename", "")
        erring_index = int(failure.get("erring_index") or 0)
        is_node_error = bool(int(failure.get("failcode", 0) or 0) & 0x2000)

        if is_node_error and erring_node:
            self._routing_memory.ban_node(erring_node, ttl_seconds=self.CHANNEL_BAN_TTL)

        if erring_channel and erring_direction is not None:
            scid_dir = f"{erring_channel}/{erring_direction}"
            if failcodename in {
                "WIRE_TEMPORARY_CHANNEL_FAILURE",
                "WIRE_FEE_INSUFFICIENT",
            }:
                self._routing_memory.ban_channel(scid_dir, ttl_seconds=self.CHANNEL_BAN_TTL)
            if failcodename == "WIRE_TEMPORARY_CHANNEL_FAILURE":
                amount_idx = min(max(erring_index - 1, 0), len(full_route) - 1)
                constrained_amount = max(
                    0,
                    int(full_route[amount_idx].get("amount_msat", 0) or 0) - 1,
                )
                if constrained_amount > 0:
                    self._routing_memory.constrain_channel(
                        scid_dir,
                        constrained_amount,
                        ttl_seconds=self.CHANNEL_BAN_TTL,
                    )

    def _route_constraint_excludes(
        self,
        full_route: List[Dict],
        candidate,
        our_id: str,
        amount_msat: int,
    ) -> List[str]:
        """Return hops that exceed remembered max-amount constraints."""
        excludes: List[str] = []
        for hop in full_route:
            channel = hop.get("channel")
            direction = hop.get("direction")
            if not channel or direction is None:
                continue
            scid_dir = f"{channel}/{direction}"
            max_amount = self._routing_memory.max_amount_for(scid_dir)
            if max_amount is None:
                continue
            hop_amount = int(hop.get("amount_msat", 0) or 0)
            if hop_amount > max_amount:
                excludes.append(scid_dir)

        final_direction = 1 if candidate.to_peer_id > our_id else 0
        final_scid_dir = f"{candidate.to_channel.replace(':', 'x')}/{final_direction}"
        final_max = self._routing_memory.max_amount_for(final_scid_dir)
        if final_max is not None and amount_msat > final_max:
            excludes.append(final_scid_dir)

        return list(dict.fromkeys(excludes))

    def _compute_fleet_route(
        self,
        job: RebalanceJob,
        candidate,
        our_id: str,
    ) -> tuple[List[Dict], List[Dict]]:
        """Build a circular route using getroutes for fleet rebalances.

        SAFETY: askrene has known crash vectors (see hexmem lessons 502/504).
        - Never include auto.sourcefree (WIRE_FEE_INSUFFICIENT → crash)
        - Validate all params before calling getroutes (0 amounts → crash)
        - Try/except with auto-only fallback (unknown layer TOCTOU → crash)
        """
        required_amount_msat, required_cltv = self._get_return_hop_policy(
            candidate, job.amount_msat, our_id,
        )

        # Pre-validate: askrene can crash on degenerate inputs
        if required_amount_msat <= 0 or job.max_fee_msat <= 0:
            raise ValueError("fleet_invalid_amount")
        if our_id == candidate.to_peer_id:
            raise ValueError("fleet_self_route")

        layers = self._get_layers(job.route_type)
        if "auto.no_mpp_support" not in layers:
            layers.append("auto.no_mpp_support")

        # SAFETY: never auto.sourcefree in circular routes (lesson 502)
        layers = [l for l in layers if l != "auto.sourcefree"]

        # Fleet maxfee: without auto.sourcefree, askrene includes our own
        # outgoing channel fee in the route cost.  For circular payments we
        # don't actually pay that fee (we're the sender), so the real cost
        # is lower than askrene estimates.  Use a generous 1% cap (matching
        # HiveRouter discovery) to avoid "excessive cost" rejections.
        fleet_maxfee = max(job.max_fee_msat, required_amount_msat // 100)

        params = {
            "source": our_id,
            "destination": candidate.to_peer_id,
            "amount_msat": required_amount_msat,
            "layers": layers,
            "maxfee_msat": fleet_maxfee,
            "final_cltv": required_cltv,
            "maxparts": 1,
        }
        # SAFETY: TOCTOU race on layer names can crash askrene (lesson 504).
        # Catch and retry with auto-only layers.
        try:
            result = self.plugin.rpc.call("getroutes", params)
        except Exception:
            params["layers"] = ["auto.localchans", "auto.no_mpp_support"]
            result = self.plugin.rpc.call("getroutes", params)

        routes = result.get("routes", [])
        if not routes:
            raise ValueError("no_fleet_route")
        path = routes[0].get("path", [])
        if not path:
            raise ValueError("no_fleet_route")

        full_route = self._getroutes_to_sendpay(
            path,
            candidate.to_channel,
            our_id,
            job.amount_msat,
        )

        # Strip our outgoing channel fee from the first hop.
        # Without auto.sourcefree, getroutes includes our channel's fee in
        # path[0].amount_msat.  For circular self-payments the first hop from
        # us is free (we're the sender, not a forwarder).  If we don't strip
        # it, the first peer keeps the excess as unearned profit and the
        # budget gate rejects the route as over-budget.
        if len(full_route) >= 2:
            first_scid = full_route[0].get("channel", "")
            first_peer = full_route[0].get("id", "")
            our_fee_base = 0
            our_fee_ppm = 0
            try:
                # Query only the first hop peer (not all channels)
                peer_chans = self.plugin.rpc.listpeerchannels(first_peer)
                for ch in peer_chans.get("channels", []):
                    if ch.get("short_channel_id") == first_scid:
                        local = ch.get("updates", {}).get("local", {})
                        our_fee_ppm = int(local.get(
                            "fee_proportional_millionths", 0) or 0)
                        our_fee_base = int(local.get(
                            "fee_base_msat", 0) or 0)
                        break
            except Exception:
                pass
            if our_fee_ppm > 0 or our_fee_base > 0:
                fwd_amount = int(full_route[1].get("amount_msat", 0) or 0)
                our_fee = our_fee_base + (fwd_amount * our_fee_ppm) // 1_000_000
                adjusted = full_route[0]["amount_msat"] - our_fee
                full_route[0]["amount_msat"] = max(
                    int(full_route[1]["amount_msat"]),
                    adjusted,
                )
                self._log(
                    f"Fleet first-hop fee strip: {our_fee_ppm}ppm + "
                    f"{our_fee_base}base = {our_fee}msat removed",
                    level="debug",
                )

        inform_path = list(path)
        final_direction = 1 if candidate.to_peer_id > our_id else 0
        inform_path.append({
            "short_channel_id_dir": f"{candidate.to_channel.replace(':', 'x')}/{final_direction}",
            "amount_msat": job.amount_msat,
        })
        return full_route, inform_path

    # ------------------------------------------------------------------
    # Core Execution
    # ------------------------------------------------------------------

    def _execute_single(self, job: RebalanceJob, candidate) -> RebalanceResult:
        """Execute a rebalance job.

        Uses sendpay with explicit routes.  Fleet rebalances use getroutes
        (askrene layers) for zero-fee fleet paths.  Network rebalances use
        getroute with explicit excludes and retry.
        """
        our_id = self._get_our_id()
        if not our_id:
            return RebalanceResult(success=False, error="no_node_id")

        route_type = job.route_type

        # Create self-invoice
        try:
            job.label = f"rebal-{secrets.token_hex(8)}"
            inv = self.plugin.rpc.invoice(
                amount_msat=job.amount_msat,
                label=job.label,
                description=f"{route_type} rebalance",
                expiry=self.INVOICE_EXPIRY,
            )
            job.payment_hash = inv.get("payment_hash", "")
            bolt11 = inv.get("bolt11", "")
            if not job.payment_hash or not bolt11:
                return RebalanceResult(success=False, error="invoice_failed")
        except Exception as e:
            return RebalanceResult(success=False, error=f"invoice_error: {e}")

        # Circular rebalancing via sendpay with explicit routes.
        # xpay/pay resolve self-invoices LOCALLY without sending HTLCs through
        # the network — no liquidity actually moves.  Only sendpay with an
        # explicit non-empty route forces HTLCs through intermediate nodes.

        last_error = ""
        job.state = JobState.ROUTING
        excludes: List[str] = []
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            full_route: List[Dict] = []
            inform_path: List[Dict] = []
            attempt_reserved = False
            try:
                if route_type == "fleet" and attempt == 1:
                    try:
                        full_route, inform_path = self._compute_fleet_route(
                            job, candidate, our_id,
                        )
                    except Exception:
                        full_route, inform_path = self._compute_network_route(
                            job, candidate, our_id, excludes
                        )
                else:
                    full_route, inform_path = self._compute_network_route(
                        job, candidate, our_id, excludes
                    )
                constraint_excludes = self._route_constraint_excludes(
                    full_route, candidate, our_id, job.amount_msat
                )
                if constraint_excludes:
                    excludes = list(dict.fromkeys(excludes + constraint_excludes))
                    if attempt < self.MAX_ATTEMPTS:
                        continue
                    raise ValueError("constrained_route")
                self._validate_sendpay_route(full_route, job.amount_msat)

                if self.hive_router and inform_path:
                    try:
                        attempt_reserved = bool(self.hive_router.reserve_path(inform_path))
                    except Exception:
                        attempt_reserved = False

                total_fee = max(0, full_route[0].get("amount_msat", job.amount_msat) - job.amount_msat)
                fee_ppm = (total_fee * 1_000_000) // job.amount_msat if job.amount_msat > 0 else 0

                # Budget gate: fleet routes use generous maxfee for discovery
                # (because askrene inflates cost without auto.sourcefree), but
                # we still enforce the real budget before committing sats.
                if total_fee > job.max_fee_msat:
                    raise ValueError(
                        f"route_over_budget: {fee_ppm}ppm ({total_fee}msat) "
                        f"exceeds budget {job.max_fee_msat}msat"
                    )

                self._log(
                    f"Job {job.job_id}: sendpay circular "
                    f"{len(full_route)} hops, {fee_ppm}ppm"
                )

                job.state = JobState.SENDING
                self.plugin.rpc.sendpay(
                    route=full_route,
                    payment_hash=job.payment_hash,
                    amount_msat=job.amount_msat,
                    bolt11=bolt11,
                    payment_secret=inv.get("payment_secret", ""),
                )

                job.state = JobState.WAITING
                pay_result = self.plugin.rpc.waitsendpay(
                    job.payment_hash,
                    self.SENDPAY_TIMEOUT,
                )
                status = pay_result.get("status", "")
                if status != "complete":
                    raise RuntimeError(f"waitsendpay_status={status}")

                if attempt_reserved and self.hive_router:
                    try:
                        self.hive_router.unreserve_path(inform_path)
                    except Exception:
                        pass
                self._inform_result(inform_path, job.amount_msat, succeeded=True)

                actual_sent = pay_result.get("amount_sent_msat", full_route[0].get("amount_msat", job.amount_msat))
                actual_fee = max(0, actual_sent - job.amount_msat)
                actual_ppm = (actual_fee * 1_000_000) // job.amount_msat if job.amount_msat > 0 else 0

                result = RebalanceResult(
                    success=True,
                    fee_msat=actual_fee,
                    fee_ppm=actual_ppm,
                    hops=len(full_route),
                    route_type=route_type,
                    attempts=attempt,
                    parts=1,
                )
                job.state = JobState.COMPLETE
                job.result = result

                self._log(
                    f"Job {job.job_id} SUCCESS: {candidate.to_channel} "
                    f"fee={actual_fee}msat ({actual_ppm}ppm) "
                    f"{len(full_route)} hops",
                )
                return result

            except Exception as e:
                failure = self._parse_failure(e)
                last_error = f"sendpay_error: {e}"
                if attempt_reserved and self.hive_router:
                    try:
                        self.hive_router.unreserve_path(inform_path)
                    except Exception:
                        pass
                self._learn_from_failure(full_route, failure, e)
                if inform_path:
                    self._inform_failure(inform_path, failure)
                self._cleanup_failed_payment(job.payment_hash)
                job.attempts.append({
                    "attempt": attempt,
                    "error": last_error,
                    "timestamp": int(time.time()),
                })
                self._log(f"Job {job.job_id} failed attempt {attempt}: {last_error}", level="info")

                should_retry = False
                if attempt < self.MAX_ATTEMPTS and failure.get("code") == 204:
                    erring_index = failure.get("erring_index")
                    failcode = int(failure.get("failcode", 0) or 0)
                    is_node_error = bool(failcode & 0x2000)
                    if erring_index == 0 or (erring_index == 1 and is_node_error):
                        should_retry = False
                    elif full_route and erring_index == len(full_route):
                        should_retry = False
                    else:
                        previous_excludes = list(excludes)
                        if is_node_error and failure.get("erring_node"):
                            excludes = list(dict.fromkeys(
                                excludes + [str(failure["erring_node"])]
                            ))
                        elif failure.get("erring_channel") and failure.get("erring_direction") is not None:
                            excludes = list(dict.fromkeys(excludes + [
                                f"{failure['erring_channel']}/{failure['erring_direction']}"
                            ]))
                        should_retry = excludes != previous_excludes

                if should_retry:
                    job.state = JobState.ROUTING
                    continue
                break

        # All attempts exhausted
        job.state = JobState.FAILED

        # Clean up unpaid invoice
        try:
            self.plugin.rpc.delinvoice(job.label, "unpaid")
        except Exception:
            pass

        result = RebalanceResult(
            success=False,
            route_type=route_type,
            attempts=len(job.attempts),
            error=last_error,
        )
        job.result = result

        self._log(
            f"Job {job.job_id} FAILED after {len(job.attempts)} attempts: {last_error}",
            level="warn",
        )
        return result

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def execute(self, candidate) -> RebalanceResult:
        """Execute a rebalance synchronously. Blocks until complete or failed."""
        route_type = "fleet" if candidate.hive_route_hops > 0 else "network"

        # Fleet-aware sizing
        amount_msat = candidate.amount_msat
        if route_type == "fleet" and self.hive_router:
            try:
                source_scid = candidate.source_candidates[0] if candidate.source_candidates else ""
                if source_scid:
                    channels = self.plugin.rpc.listpeerchannels()
                    for ch in channels.get("channels", []):
                        if (ch.get("short_channel_id") == source_scid
                                and ch.get("state") == "CHANNELD_NORMAL"):
                            source_peer = ch.get("peer_id", "")
                            if source_peer and self.hive_router.is_hive_member(source_peer):
                                max_through = self.hive_router.max_rebalance_through_member(source_peer)
                                if 0 < max_through < amount_msat // 1000:
                                    amount_msat = max_through * 1000
                                    self._log(
                                        f"Fleet sizing: capped to {max_through} sats "
                                        f"(peer {source_peer[:12]}...)",
                                    )
                            break
            except Exception:
                pass

        job = RebalanceJob(
            job_id=secrets.token_hex(8),
            channel_id=candidate.to_channel,
            peer_id=candidate.to_peer_id,
            amount_msat=amount_msat,
            max_fee_msat=candidate.max_budget_msat,
            route_type=route_type,
        )

        with self._lock:
            normalized = candidate.to_channel.replace(":", "x")
            if normalized in self._jobs:
                return RebalanceResult(success=False, error="job_already_active")
            self._jobs[normalized] = job

        try:
            result = self._execute_single(job, candidate)
        finally:
            with self._lock:
                self._jobs.pop(normalized, None)

        return result

    def execute_async(self, candidate,
                      callback: Callable[[RebalanceResult], None] = None) -> str:
        """Submit a rebalance job. Returns job_id."""
        route_type = "fleet" if candidate.hive_route_hops > 0 else "network"
        job = RebalanceJob(
            job_id=secrets.token_hex(8),
            channel_id=candidate.to_channel,
            peer_id=candidate.to_peer_id,
            amount_msat=candidate.amount_msat,
            max_fee_msat=candidate.max_budget_msat,
            route_type=route_type,
        )

        normalized = candidate.to_channel.replace(":", "x")
        with self._lock:
            if normalized in self._jobs:
                return ""
            self._jobs[normalized] = job

        def _run():
            try:
                result = self._execute_single(job, candidate)
                if callback:
                    callback(result)
                return result
            finally:
                with self._lock:
                    self._jobs.pop(normalized, None)
                    self._futures.pop(normalized, None)

        future = self._pool.submit(_run)
        with self._lock:
            self._futures[normalized] = future

        return job.job_id

    def cancel(self, channel_id: str) -> bool:
        """Cancel an active job."""
        normalized = channel_id.replace(":", "x")
        with self._lock:
            job = self._jobs.get(normalized)
            if not job:
                return False
            job.state = JobState.CANCELLED
            future = self._futures.get(normalized)
            if future:
                future.cancel()
        return True

    def cancel_all(self) -> int:
        """Cancel all active jobs."""
        with self._lock:
            count = 0
            for job in self._jobs.values():
                if job.state not in (JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED):
                    job.state = JobState.CANCELLED
                    count += 1
            for future in self._futures.values():
                future.cancel()
        return count

    def get_active_jobs(self) -> List[RebalanceJob]:
        """List active jobs."""
        with self._lock:
            return [j for j in self._jobs.values()
                    if j.state not in (JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED)]

    def get_job(self, channel_id: str) -> Optional[RebalanceJob]:
        """Get job for a channel."""
        normalized = channel_id.replace(":", "x")
        with self._lock:
            return self._jobs.get(normalized)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j.state not in (JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED))

    def shutdown(self) -> None:
        """Cancel all jobs and shut down thread pool."""
        self.cancel_all()
        self._pool.shutdown(wait=False)
