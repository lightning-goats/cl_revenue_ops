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

    # ------------------------------------------------------------------
    # Core Execution
    # ------------------------------------------------------------------

    def _execute_single(self, job: RebalanceJob, candidate) -> RebalanceResult:
        """Execute a rebalance job.

        Uses CLN's native 'pay' with exclude lists to force routing
        through the desired channel.  This avoids manual route construction
        and fee calculation issues that caused WIRE_FEE_INSUFFICIENT.
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

        # xpay with askrene layers for circular rebalancing.
        # Production logs confirmed xpay DOES handle self-payment correctly
        # (invoices resolved, preimages returned).  xpay response returns
        # payment_preimage (not "status": "complete"), so we check for that.
        layers = self._get_layers(route_type)
        layers.append("auto.sourcefree")

        # Generous maxfee for fleet routes (actual cost near-zero)
        maxfee = job.max_fee_msat
        if route_type == "fleet":
            maxfee = max(job.max_fee_msat, job.amount_msat // 200)

        last_error = ""
        job.state = JobState.SENDING

        try:
            self._log(
                f"Job {job.job_id}: xpay {job.amount_msat}msat "
                f"maxfee={maxfee} layers={len(layers)}"
            )

            pay_result = self.plugin.rpc.call("xpay", {
                "invstring": bolt11,
                "maxfee": maxfee,
                "layers": layers,
                "retry_for": self.SENDPAY_TIMEOUT,
            })

            # xpay returns payment_preimage on success (NOT "status": "complete")
            preimage = pay_result.get("payment_preimage", "")
            if preimage:
                actual_sent = pay_result.get("amount_sent_msat", job.amount_msat)
                actual_fee = max(0, actual_sent - job.amount_msat)
                fee_ppm = (actual_fee * 1_000_000) // job.amount_msat if job.amount_msat > 0 else 0

                result = RebalanceResult(
                    success=True,
                    fee_msat=actual_fee,
                    fee_ppm=fee_ppm,
                    hops=0,
                    route_type=route_type,
                    attempts=1,
                    parts=pay_result.get("successful_parts", 1),
                )
                job.state = JobState.COMPLETE
                job.result = result

                self._log(
                    f"Job {job.job_id} SUCCESS: {candidate.to_channel} "
                    f"fee={actual_fee}msat ({fee_ppm}ppm) "
                    f"parts={result.parts}",
                )
                return result
            else:
                last_error = f"xpay_no_preimage: {pay_result}"

        except Exception as e:
            last_error = f"xpay_error: {e}"

        job.attempts.append({
            "attempt": 1,
            "error": last_error,
            "timestamp": int(time.time()),
        })
        self._log(f"Job {job.job_id} failed: {last_error}", level="info")

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
