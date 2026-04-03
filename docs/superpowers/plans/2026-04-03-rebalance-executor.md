# RebalanceExecutor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native rebalance execution engine using `getroutes` + `sendpay` that replaces sling, with full askrene layer support, MPP, and learning from results.

**Architecture:** Single new module `modules/rebalance_executor.py` containing `RebalanceJob`, `RebalanceResult`, and `RebalanceExecutor`. Uses `ThreadPoolExecutor` for concurrent jobs. Fleet engine uses all hive-* layers for 0-fee fleet paths. Network engine uses revenue-* layers for best available routes. Both learn from results via `askrene-inform-channel`.

**Tech Stack:** Python 3.12+, CLN RPC (getroutes, sendpay, waitsendpay, invoice, askrene-inform-channel), concurrent.futures

**Spec:** `docs/superpowers/specs/2026-04-03-rebalance-executor-design.md`

---

### File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `modules/rebalance_executor.py` | Create | RebalanceJob, RebalanceResult, RebalanceExecutor |
| `tests/test_rebalance_executor.py` | Create | Unit tests for executor |

---

### Task 1: Create dataclasses and executor skeleton

**Files:**
- Create: `/home/sat/bin/cl_revenue_ops/modules/rebalance_executor.py`

- [ ] **Step 1: Create the module with dataclasses and executor skeleton**

```python
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
        """Build layer list based on route type."""
        layers = ["auto.localchans", "auto.sourcefree"]
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
        """Execute a rebalance job (runs in thread pool)."""
        our_id = self._get_our_id()
        if not our_id:
            return RebalanceResult(success=False, error="no_node_id")

        route_type = job.route_type
        layers = self._get_layers(route_type)
        max_parts = self.FLEET_MAX_PARTS if route_type == "fleet" else self.NETWORK_MAX_PARTS

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
            payment_secret = inv.get("payment_secret", "")
            if not job.payment_hash or not payment_secret:
                return RebalanceResult(success=False, error="invoice_failed")
        except Exception as e:
            return RebalanceResult(success=False, error=f"invoice_error: {e}")

        last_error = ""
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            if job.state == JobState.CANCELLED:
                break

            job.state = JobState.ROUTING

            # Find route
            try:
                route_result = self.plugin.rpc.call("getroutes", {
                    "source": our_id,
                    "destination": job.peer_id,
                    "amount_msat": job.amount_msat,
                    "layers": layers,
                    "maxfee_msat": job.max_fee_msat,
                    "final_cltv": 18,
                    "maxparts": max_parts,
                })
            except Exception as e:
                last_error = f"getroutes_error: {e}"
                self._log(f"Job {job.job_id}: getroutes failed: {e}", level="debug")
                break  # No routes = no point retrying

            routes = route_result.get("routes", [])
            if not routes:
                last_error = "no_routes"
                break

            probability = route_result.get("probability_ppm", 0)

            job.state = JobState.SENDING

            # Execute each route part via sendpay
            all_succeeded = True
            total_fee_msat = 0
            total_hops = 0

            for part_idx, route in enumerate(routes):
                path = route.get("path", [])
                if not path:
                    continue

                dest_amount = route.get("amount_msat", 0)
                first_hop_amount = path[0].get("amount_msat", dest_amount)
                part_fee = max(0, first_hop_amount - dest_amount)

                sendpay_route = self._getroutes_to_sendpay(
                    path, candidate.to_channel, our_id, dest_amount
                )

                try:
                    sendpay_params = {
                        "route": sendpay_route,
                        "payment_hash": job.payment_hash,
                        "payment_secret": payment_secret,
                        "amount_msat": job.amount_msat,
                    }
                    if len(routes) > 1:
                        sendpay_params["partid"] = part_idx + 1
                        sendpay_params["groupid"] = 1

                    self.plugin.rpc.call("sendpay", sendpay_params)
                except Exception as e:
                    last_error = f"sendpay_error: {e}"
                    self._inform_result(path, dest_amount, succeeded=False)
                    all_succeeded = False
                    break

                total_fee_msat += part_fee
                total_hops = max(total_hops, len(path))

            if not all_succeeded:
                job.attempts.append({
                    "attempt": attempt,
                    "error": last_error,
                    "timestamp": int(time.time()),
                })
                continue

            # Wait for completion
            job.state = JobState.WAITING
            try:
                wait_params = {
                    "payment_hash": job.payment_hash,
                    "timeout": self.SENDPAY_TIMEOUT,
                }
                if len(routes) > 1:
                    wait_params["groupid"] = 1

                pay_result = self.plugin.rpc.call("waitsendpay", wait_params)
                status = pay_result.get("status", "")

                if status == "complete":
                    actual_sent = pay_result.get("amount_sent_msat", job.amount_msat)
                    actual_fee = actual_sent - job.amount_msat
                    fee_ppm = (actual_fee * 1_000_000) // job.amount_msat if job.amount_msat > 0 else 0

                    # Learn from success
                    for route in routes:
                        self._inform_result(
                            route.get("path", []),
                            route.get("amount_msat", 0),
                            succeeded=True,
                        )

                    result = RebalanceResult(
                        success=True,
                        fee_msat=actual_fee,
                        fee_ppm=fee_ppm,
                        hops=total_hops,
                        route_type=route_type,
                        attempts=attempt,
                        parts=len(routes),
                    )
                    job.state = JobState.COMPLETE
                    job.result = result

                    self._log(
                        f"Job {job.job_id} SUCCESS: {candidate.to_channel} "
                        f"fee={actual_fee}msat ({fee_ppm}ppm) "
                        f"{total_hops} hops, {len(routes)} parts, "
                        f"attempt {attempt}/{self.MAX_ATTEMPTS}",
                    )
                    return result

                else:
                    last_error = f"waitsendpay_status={status}"

            except Exception as e:
                last_error = f"waitsendpay_error: {e}"
                # Learn from failure
                for route in routes:
                    self._inform_result(
                        route.get("path", []),
                        route.get("amount_msat", 0),
                        succeeded=False,
                    )

            job.attempts.append({
                "attempt": attempt,
                "error": last_error,
                "timestamp": int(time.time()),
            })
            self._log(
                f"Job {job.job_id} attempt {attempt}/{self.MAX_ATTEMPTS} "
                f"failed: {last_error}",
                level="debug",
            )

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
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/sat/bin/cl_revenue_ops
python3 -c "import ast; ast.parse(open('modules/rebalance_executor.py').read()); print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add modules/rebalance_executor.py
git commit -m "feat: add RebalanceExecutor — native getroutes+sendpay rebalance engine"
```

---

### Task 2: Create unit tests

**Files:**
- Create: `/home/sat/bin/cl_revenue_ops/tests/test_rebalance_executor.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for RebalanceExecutor."""

import time
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, field
from typing import List
import pytest

from modules.rebalance_executor import (
    RebalanceExecutor, RebalanceJob, RebalanceResult, JobState
)


@dataclass
class MockCandidate:
    source_candidates: List[str] = field(default_factory=lambda: ["100x1x0"])
    to_channel: str = "200x1x0"
    to_peer_id: str = "dest_peer_abc"
    amount_sats: int = 500000
    amount_msat: int = 500000000
    max_budget_sats: int = 100
    max_budget_msat: int = 100000
    max_fee_ppm: int = 200
    hive_route_hops: int = 0
    direction: str = "pull"


class MockHiveRouter:
    def __init__(self, is_member=False, max_through=0):
        self.available = True
        self._is_member = is_member
        self._max = max_through

    def is_hive_member(self, pid):
        return self._is_member

    def max_rebalance_through_member(self, pid):
        return self._max


class TestRebalanceExecutorInit:
    def test_defaults(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        assert executor.active_count == 0
        assert executor.get_active_jobs() == []

    def test_with_hive_router(self):
        executor = RebalanceExecutor(
            MagicMock(), MagicMock(), MagicMock(),
            hive_router=MockHiveRouter()
        )
        assert executor.hive_router is not None


class TestLayerSelection:
    def test_fleet_layers_include_hive(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {
            "layers": [
                {"layer": "hive-fleet"},
                {"layer": "hive-reputation"},
                {"layer": "revenue-local"},
                {"layer": "unrelated"},
            ]
        }
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        layers = executor._get_layers("fleet")
        assert "auto.localchans" in layers
        assert "auto.sourcefree" in layers
        assert "hive-fleet" in layers
        assert "hive-reputation" in layers
        assert "revenue-local" in layers
        assert "unrelated" not in layers

    def test_network_layers_exclude_hive(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {
            "layers": [
                {"layer": "hive-fleet"},
                {"layer": "revenue-local"},
            ]
        }
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        layers = executor._get_layers("network")
        assert "hive-fleet" not in layers
        assert "revenue-local" in layers

    def test_layers_fallback_on_error(self):
        plugin = MagicMock()
        plugin.rpc.call.side_effect = Exception("askrene unavailable")
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        layers = executor._get_layers("fleet")
        assert layers == ["auto.localchans", "auto.sourcefree"]


class TestRouteConversion:
    def test_converts_getroutes_to_sendpay(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        path = [
            {"short_channel_id_dir": "100x1x0/1", "next_node_id": "node_a",
             "amount_msat": 501000, "delay": 42},
            {"short_channel_id_dir": "200x1x0/0", "next_node_id": "node_b",
             "amount_msat": 500000, "delay": 24},
        ]
        route = executor._getroutes_to_sendpay(path, "300x1x0", "our_id", 500000)

        assert len(route) == 3  # 2 hops + final circular hop
        assert route[0] == {"channel": "100x1x0", "id": "node_a",
                            "amount_msat": 501000, "delay": 42}
        assert route[1] == {"channel": "200x1x0", "id": "node_b",
                            "amount_msat": 500000, "delay": 24}
        assert route[2] == {"channel": "300x1x0", "id": "our_id",
                            "amount_msat": 500000, "delay": 18}


class TestRouteTypeSelection:
    def test_fleet_when_hive_route_hops(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.call.return_value = {"routes": []}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc", "payment_secret": "def"
        }
        plugin.rpc.listpeerchannels.return_value = {"channels": []}

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=2)
        result = executor.execute(candidate)

        # Should attempt fleet routing (getroutes called with fleet layers)
        assert result.route_type == "fleet"

    def test_network_when_no_hive_route(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.call.return_value = {"routes": []}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc", "payment_secret": "def"
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)
        result = executor.execute(candidate)
        assert result.route_type == "network"


class TestExecuteSuccess:
    def test_successful_single_part(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123"
        }

        def rpc_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            if method == "getroutes":
                return {
                    "probability_ppm": 950000,
                    "routes": [{
                        "amount_msat": 500000000,
                        "path": [
                            {"short_channel_id_dir": "100x1x0/1",
                             "next_node_id": "peer_a",
                             "amount_msat": 500050000,
                             "delay": 42},
                        ]
                    }]
                }
            if method == "sendpay":
                return {}
            if method == "waitsendpay":
                return {"status": "complete", "amount_sent_msat": 500050000}
            if method == "askrene-inform-channel":
                return {}
            return {}

        plugin.rpc.call.side_effect = rpc_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)
        result = executor.execute(candidate)

        assert result.success is True
        assert result.fee_msat == 50000
        assert result.fee_ppm == 100  # 50000 * 1e6 / 500000000
        assert result.attempts == 1
        assert result.route_type == "network"


class TestExecuteFailure:
    def test_no_routes(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash", "payment_secret": "secret"
        }

        def rpc_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            if method == "getroutes":
                return {"routes": []}
            return {}

        plugin.rpc.call.side_effect = rpc_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate())

        assert result.success is False
        assert result.error == "no_routes"

    def test_cleans_up_invoice_on_failure(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash", "payment_secret": "secret"
        }
        plugin.rpc.call.return_value = {"routes": [], "layers": []}

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        executor.execute(MockCandidate())

        # delinvoice should be called for cleanup
        plugin.rpc.delinvoice.assert_called_once()


class TestCancelAndShutdown:
    def test_cancel_nonexistent(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        assert executor.cancel("999x1x0") is False

    def test_cancel_all_empty(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        assert executor.cancel_all() == 0

    def test_shutdown(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        executor.shutdown()
        # Should not raise


class TestInformChannel:
    def test_informs_on_success(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {}
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())

        path = [
            {"short_channel_id_dir": "100x1x0/1", "next_node_id": "a",
             "amount_msat": 500000, "delay": 24},
        ]
        executor._inform_result(path, 500000, succeeded=True)

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert len(inform_calls) == 1
        assert inform_calls[0][0][1]["inform"] == "succeeded"

    def test_informs_on_failure(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {}
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())

        path = [{"short_channel_id_dir": "100x1x0/1", "next_node_id": "a",
                 "amount_msat": 500000, "delay": 24}]
        executor._inform_result(path, 500000, succeeded=False)

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert len(inform_calls) == 1
        assert inform_calls[0][0][1]["inform"] == "failed"
```

- [ ] **Step 2: Run tests**

```bash
cd /home/sat/bin/cl_revenue_ops
python3 -m pytest tests/test_rebalance_executor.py -v
```

Expected: All pass.

- [ ] **Step 3: Run full suite**

```bash
python3 -m pytest tests/ --ignore=tests/test_background_loops.py -x -q --tb=short
```

Expected: 932+ passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rebalance_executor.py
git commit -m "test: add RebalanceExecutor unit tests"
```

---

### Task 3: Push

- [ ] **Step 1: Push**

```bash
git push
```
