"""
rebalance_executor — Native rebalance engine using getroutes + sendpay.

Replaces sling for rebalance execution with full askrene layer support.
Fleet rebalances route through zero-fee fleet peers.  Network rebalances
use the best available paths with profitability biases.  Both learn from
results via askrene-inform-channel.

Supports MPP (multi-part payments) for large rebalances.
"""

from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


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
    """Native rebalance execution using getroutes + sendpay.

    Fleet rebalances use all hive-* + revenue-* askrene layers for
    zero-fee fleet routing.  Network rebalances use revenue-* layers
    for optimal path selection.  Both learn from results via
    askrene-inform-channel.

    Thread-safe: jobs run in a ThreadPoolExecutor.
    """

    MAX_ATTEMPTS = 3
    MAX_CONCURRENT = 5
    FLEET_MAX_PARTS = 1
    NETWORK_MAX_PARTS = 3
    SENDPAY_TIMEOUT = 60
    INVOICE_EXPIRY = 300

    def __init__(self, plugin, config, database, hive_router=None):
        self.plugin = plugin
        self.config = config
        self.database = database
        self.hive_router = hive_router
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

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _inform_result(self, path: List[Dict], amount_msat: int,
                       succeeded: bool) -> None:
        """Teach askrene about route success/failure."""
        inform = "succeeded" if succeeded else "failed"
        for hop in path:
            scid_dir = hop.get("short_channel_id_dir", "")
            if not scid_dir:
                continue
            try:
                self.plugin.rpc.call("askrene-inform-channel", {
                    "layer": "revenue-local",
                    "short_channel_id_dir": scid_dir,
                    "amount_msat": amount_msat,
                    "inform": inform,
                })
            except Exception:
                pass

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
                path.append({"short_channel_id_dir": scid_dir})
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
    ) -> List[Dict]:
        """Build a circular route using getroute for network rebalances."""
        source_scid = candidate.source_candidates[0] if candidate.source_candidates else ""
        source_peer = candidate.primary_source_peer_id
        if not source_scid or not source_peer:
            raise ValueError("no_source_channel")

        getroute_kwargs = {
            "fromid": source_peer,
            "maxhops": 6,
            "fuzzpercent": 0,
        }
        if excludes:
            getroute_kwargs["exclude"] = excludes

        route_result = self.plugin.rpc.getroute(
            our_id,
            job.amount_msat,
            1,
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

        return [{
            "id": source_peer,
            "channel": first_hop_scid,
            "direction": first_hop_direction,
            "amount_msat": first_hop_amount,
            "delay": first_hop_delay,
            "style": "tlv",
        }] + route

    def _compute_fleet_route(
        self,
        job: RebalanceJob,
        candidate,
        our_id: str,
    ) -> tuple[List[Dict], List[Dict]]:
        """Build a circular route using getroutes for fleet rebalances."""
        layers = self._get_layers(job.route_type)
        params = {
            "source": our_id,
            "destination": candidate.to_peer_id,
            "amount_msat": job.amount_msat,
            "layers": layers,
            "maxfee_msat": job.max_fee_msat,
            "final_cltv": 18,
        }
        try:
            result = self.plugin.rpc.call("getroutes", params)
        except Exception:
            params["layers"] = ["auto.localchans"]
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
        inform_path = list(path)
        final_direction = 1 if candidate.to_peer_id > our_id else 0
        inform_path.append({
            "short_channel_id_dir": f"{candidate.to_channel.replace(':', 'x')}/{final_direction}",
        })
        return full_route, inform_path

    # ------------------------------------------------------------------
    # Core Execution
    # ------------------------------------------------------------------

    def _execute_single(self, job: RebalanceJob, candidate) -> RebalanceResult:
        """Execute a rebalance job.

        Uses sendpay with explicit routes. Network rebalances use getroute and
        explicit excludes on retry. Fleet rebalances use layer-aware getroutes.
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

        # Circular rebalancing via getroute + sendpay.
        # xpay/pay resolve self-invoices LOCALLY without sending HTLCs through
        # the network — no liquidity actually moves.  Only sendpay with an
        # explicit non-empty route forces HTLCs through intermediate nodes.
        #
        # Pattern: getroute(fromid=dest_peer, id=our_id) finds path BACK to us.
        # We prepend our outgoing channel as the first hop to complete the circle.

        last_error = ""
        job.state = JobState.ROUTING
        excludes: List[str] = []
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            full_route: List[Dict] = []
            inform_path: List[Dict] = []
            try:
                if route_type == "fleet":
                    full_route, inform_path = self._compute_fleet_route(job, candidate, our_id)
                else:
                    full_route = self._compute_network_route(job, candidate, our_id, excludes)
                    inform_path = self._path_for_inform(full_route)

                total_fee = max(0, full_route[0].get("amount_msat", job.amount_msat) - job.amount_msat)
                fee_ppm = (total_fee * 1_000_000) // job.amount_msat if job.amount_msat > 0 else 0

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
                if inform_path:
                    self._inform_result(inform_path, job.amount_msat, succeeded=False)
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
                    elif full_route and erring_index == len(full_route) + 1:
                        should_retry = False
                    elif route_type == "network":
                        if is_node_error and failure.get("erring_node"):
                            excludes.append(str(failure["erring_node"]))
                            should_retry = True
                        elif failure.get("erring_channel") and failure.get("erring_direction") is not None:
                            excludes.append(
                                f"{failure['erring_channel']}/{failure['erring_direction']}"
                            )
                            should_retry = True
                    else:
                        should_retry = True

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

        if self.hive_router:
            try:
                self.hive_router.reserve_for_job(
                    candidate.to_channel,
                    amount_msat,
                    direction=getattr(candidate, "direction", "pull"),
                )
            except Exception:
                pass

        try:
            result = self._execute_single(job, candidate)
        finally:
            if self.hive_router:
                try:
                    self.hive_router.unreserve_for_job(
                        candidate.to_channel,
                        amount_msat,
                        direction=getattr(candidate, "direction", "pull"),
                    )
                except Exception:
                    pass
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

        if self.hive_router:
            try:
                self.hive_router.reserve_for_job(
                    candidate.to_channel,
                    candidate.amount_msat,
                    direction=getattr(candidate, "direction", "pull"),
                )
            except Exception:
                pass

        def _run():
            try:
                result = self._execute_single(job, candidate)
                if callback:
                    callback(result)
                return result
            finally:
                if self.hive_router:
                    try:
                        self.hive_router.unreserve_for_job(
                            candidate.to_channel,
                            candidate.amount_msat,
                            direction=getattr(candidate, "direction", "pull"),
                        )
                    except Exception:
                        pass
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
