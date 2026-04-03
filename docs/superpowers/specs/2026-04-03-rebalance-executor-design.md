# RebalanceExecutor — Native Rebalance Engine

**Date:** 2026-04-03
**Status:** Approved
**Scope:** cl-revenue-ops (`modules/rebalance_executor.py`, integration with EVRebalancer)
**Sub-project:** 1 of 3 (core engine; sub-project 2 = EVRebalancer integration; sub-project 3 = sling removal)

## Problem

Sling cannot use askrene layers for pathfinding. It finds 7664 ppm routes when the fleet path costs 0 ppm. Even with restricted candidates and tight maxhops, sling's own pathfinding ignores layer overrides. The current fleet rebalance workaround (`_execute_fleet_rebalance` using getroutes + sendpay) proves the pattern works but is a bolt-on, not a proper execution engine.

## Solution

A native rebalance execution module that uses `getroutes` + `sendpay` + `waitsendpay` for ALL rebalances — fleet and non-fleet. Both types benefit from askrene layers. Supports MPP (multi-part payments) from day one via `getroutes maxparts`.

## Architecture

```
RebalanceExecutor
    ├── FleetEngine   — fleet layers, 0-fee paths, fleet-aware sizing
    └── NetworkEngine  — standard layers, best network paths
    └── Common: getroutes → sendpay → waitsendpay → inform-channel
```

### File: `modules/rebalance_executor.py`

Single new file containing:
- `RebalanceJob` dataclass (job state tracking)
- `RebalanceResult` dataclass (execution outcome)
- `RebalanceExecutor` class (unified execution interface)

### Execution Flow

```
1. execute(candidate) called by EVRebalancer
2. Determine engine: fleet (hive_route_hops > 0) or network
3. Build layer list (fleet: all hive-* + revenue-*; network: revenue-* + auto)
4. Call getroutes(source=us, destination=dest_peer, layers=..., maxparts=3)
5. For each route part returned by getroutes:
   a. Convert getroutes path to sendpay route format
   b. Append final hop back to ourselves (circular)
   c. Call sendpay with payment_hash from self-invoice
6. Call waitsendpay with timeout
7. On success: askrene-inform-channel "succeeded" for each hop
8. On failure: askrene-inform-channel "failed", retry with different route (up to 3 attempts)
9. Return RebalanceResult
```

### Route Format Conversion

getroutes returns:
```
{short_channel_id_dir: "100x1x0/1", next_node_id: "abc...", amount_msat: 500000, delay: 24}
```

sendpay expects:
```
{channel: "100x1x0", id: "abc...", amount_msat: 500000, delay: 24}
```

Plus a final hop appended for circular routing:
```
{channel: <dest_channel>, id: <our_pubkey>, amount_msat: <amount>, delay: 18}
```

### Layer Selection

**Fleet engine:**
```python
layers = ["auto.localchans", "auto.sourcefree"]
# Add all existing hive-* and revenue-* layers dynamically
for l in askrene_listlayers():
    if l.startswith("hive-") or l.startswith("revenue-"):
        layers.append(l)
```

**Network engine:**
```python
layers = ["auto.localchans", "auto.sourcefree"]
# Add revenue-local for profitability biases
for l in askrene_listlayers():
    if l.startswith("revenue-"):
        layers.append(l)
```

### MPP (Multi-Part Payments)

`getroutes` accepts `maxparts` parameter. For rebalances:
- Fleet: `maxparts=1` (fleet routes are short and reliable, MPP adds complexity)
- Network: `maxparts=3` (split across multiple paths for reliability on longer routes)

When `getroutes` returns multiple routes, execute each part as a separate `sendpay` call with the same `payment_hash` but different `partid`. `waitsendpay` tracks all parts.

### Job Model

```python
@dataclass
class RebalanceJob:
    job_id: str                    # Unique ID
    channel_id: str                # Destination channel
    peer_id: str                   # Destination peer
    amount_msat: int               # Total amount
    max_fee_msat: int              # Fee budget
    route_type: str                # "fleet" or "network"
    state: str                     # PENDING/ROUTING/SENDING/WAITING/COMPLETE/FAILED
    payment_hash: str              # From self-invoice
    label: str                     # Invoice label for cleanup
    attempts: List[Dict]           # [{route, result, fee_msat, timestamp}]
    start_time: int                # Unix timestamp
    candidate: RebalanceCandidate  # Original candidate for metadata
```

```python
@dataclass
class RebalanceResult:
    success: bool
    fee_msat: int = 0
    fee_ppm: int = 0
    hops: int = 0
    route_type: str = ""           # "fleet" or "network"
    attempts: int = 0
    parts: int = 0                 # MPP parts used
    error: Optional[str] = None
```

### Job Lifecycle

Jobs run in a `ThreadPoolExecutor(max_workers=5)`. Each job:

1. Creates self-invoice (label=`rebal-{hex}`, expiry=300s)
2. Loops up to 3 attempts:
   a. `getroutes` → build sendpay routes → `sendpay` → `waitsendpay`
   b. On success: break, report result
   c. On failure: `askrene-inform-channel` with failure, retry
3. On final failure: clean up invoice (`delinvoice label unpaid`)
4. Report result to EVRebalancer

### Learning Loop (askrene-inform-channel)

After each payment attempt, teach askrene about the result:

```python
# Success: this channel had at least this much capacity
askrene-inform-channel(layer="revenue-local", scid_dir=..., amount_msat=..., inform="succeeded")

# Failure: this channel had less than this capacity
askrene-inform-channel(layer="revenue-local", scid_dir=..., amount_msat=..., inform="failed")
```

This makes subsequent `getroutes` calls smarter — avoids paths that failed, prefers paths that worked. Over time, askrene builds an accurate capacity map.

### Fleet-Aware Sizing

For fleet rebalances, before calling `getroutes`:
1. Query `hive_router.max_rebalance_through_member()` for each fleet peer on the path
2. Cap amount to the minimum of all intermediaries' limits
3. If amount is capped below minimum viable (10k sats), skip fleet engine

### Thread Safety

- `RebalanceExecutor` owns a `ThreadPoolExecutor` and a `threading.Lock` for job tracking
- Jobs are submitted via `executor.submit()` and tracked in `_active_jobs: Dict[str, Future]`
- Job state transitions are atomic (single dict assignment under GIL)
- Invoice creation/deletion is idempotent (labels are unique per job)

### Interface

```python
class RebalanceExecutor:
    def __init__(self, plugin, config, database, hive_router=None):
        """
        Args:
            plugin: CLN plugin for RPC
            config: Runtime config
            database: For budget tracking
            hive_router: For fleet balance queries (optional)
        """

    def execute(self, candidate: RebalanceCandidate) -> RebalanceResult:
        """Execute a rebalance synchronously. Blocks until complete or failed."""

    def execute_async(self, candidate: RebalanceCandidate,
                      callback: Callable[[RebalanceResult], None] = None) -> str:
        """Submit a rebalance job. Returns job_id. Calls callback on completion."""

    def cancel(self, channel_id: str) -> bool:
        """Cancel an active job for a channel."""

    def cancel_all(self) -> int:
        """Cancel all active jobs. Returns count cancelled."""

    def get_active_jobs(self) -> List[RebalanceJob]:
        """List active jobs."""

    def get_job(self, channel_id: str) -> Optional[RebalanceJob]:
        """Get job for a specific channel."""

    def shutdown(self) -> None:
        """Cancel all jobs and shut down thread pool."""
```

### Graceful Degradation

- **askrene unavailable**: `getroutes` fails → executor returns error with `error="askrene_unavailable"`. During migration, EVRebalancer can fall back to sling.
- **Hive not available**: fleet engine disabled, all rebalances go through network engine. `hive_router is None` or `hive_router.available is False`.
- **sendpay failure**: learn from failure, retry up to 3 times with askrene-informed routes.
- **No routes found**: return immediately with `error="no_routes"`. No retry (getroutes exhaustively searches).
- **Invoice creation failure**: return with `error="invoice_failed"`.

## What Does NOT Change

- `EVRebalancer` strategy logic (candidate selection, EV analysis, source scoring)
- Budget reservation system (`database.reserve_budget` / `release_budget_reservation`)
- `RebalanceCandidate` dataclass (input to executor)
- `HiveRouter` (provides fleet balance data, not replaced)
- askrene layer management (cl-hive's AskreneLayerManager, cl-revenue-ops' HiveRouter layer logic)

## Testing

- Unit tests with mocked RPC for getroutes/sendpay/waitsendpay
- Test fleet vs network engine selection
- Test MPP route splitting
- Test retry with askrene-inform-channel learning
- Test job lifecycle (cancel, timeout, shutdown)
- Test graceful degradation (no askrene, no hive, sendpay failure)
- Full test suite regression
