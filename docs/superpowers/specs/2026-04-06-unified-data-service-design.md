# Unified Data Service — Design Spec

**Date:** 2026-04-06
**Status:** Draft
**Repo:** lightning-goats/cl_revenue_ops
**New Module:** `modules/data_service.py`
**Modified:** All modules, `cl-revenue-ops.py`, `modules/database.py`

## Problem

The plugin's data access is fragmented across three separate systems with
inconsistent patterns:

### RPC Calls — 132 invocations across 10 modules

- **30 unique RPC methods** scattered across modules with no central access point
- **RpcCache** covers only 4 methods (`listpeerchannels`, `listfunds`,
  `getinfo`, `listpeers`) with a flat 30-second TTL
- Modules bypass the cache: `capacity_planner` makes 4 direct `getinfo()` calls,
  direct `listfunds()` calls; `askrene-listlayers` called 6x across 3 modules
  without caching
- 16+ sling-related RPC calls in `rebalancer.py` are dead code (sling no longer
  used)
- Estimated 20-60 RPC calls/minute; ~85-90% cache hit ratio for the 4 cached
  methods, 0% for the other 26

### Database — 105+ methods, 5 modules bypass API

- Single `RevenueDatabase` class with well-defined API
- 5 modules execute direct SQL via `database._get_connection()`:
  - `policy_manager.py`: full CRUD on `peer_policies` (10+ SQL statements)
  - `capex_budget.py`: SUM aggregation on `rebalance_costs` + `spend_events`
  - `rebalancer.py`: orphaned rebalance cleanup
  - `fee_controller.py`: `channel_costs` SELECT
- These bypass thread-safety patterns and make the private connection accessor
  effectively public

### CLN Datastore — inconsistent IPC writes

- 6 write keys, 1 read key, scattered across 3 modules
- Missing timestamps on `revenue/status` and `revenue/liquidity-state`
- No size validation before writing (CLN has 65KB limit per key)
- Inconsistent error handling patterns across write sites

### Caching — 10+ independent caches

- `RpcCache` (30s), profitability (5min), neighbor fees (30min),
  policy (persistent), SCID-to-peer (1hr), routes (60s)
- Multiple per-cycle ephemeral caches
- Several modules independently fetch the same data with different strategies
- No cache invalidation on mutations (e.g., `setchannel()` doesn't invalidate
  `listpeerchannels` cache)

## Design

### Architecture

```
Modules (rebalancer, fee_controller, capacity_planner, etc.)
    │
    ├── self.data_service ──→ DataService (NEW)
    │                           ├── RPC Tier (tiered cache, all 30 methods)
    │                           ├── Datastore Tier (standardized write helper)
    │                           └── Internal: plugin.rpc (sole RPC caller)
    │
    └── self.database ──→ RevenueDatabase (existing, gains new methods)
```

- `DataService` (`modules/data_service.py`) is a new module that centralizes
  all CLN RPC access and datastore operations
- `RevenueDatabase` (`modules/database.py`) stays as-is, gains 8-10 new methods
  to absorb escaped SQL
- Modules receive both `data_service` and `database` via constructor injection
- `rpc_cache.py` is absorbed into DataService and deleted
- Modules stop calling `plugin.rpc.*` directly; all RPC goes through DataService
- Modules keep `self.database` for direct DB queries (105+ existing call sites
  remain untouched)

### RPC Tier — Tiered Caching

DataService replaces `RpcCache` with coverage for all RPC methods, grouped by
data volatility:

#### Forever tier (cached once at startup)

| Method | Return | Notes |
|--------|--------|-------|
| `get_node_id()` | `str` | From `getinfo().id`, immutable at runtime |
| `get_network()` | `str` | From `getinfo().network` |
| `get_node_alias()` | `str` | From `getinfo().alias` |
| `get_configs()` | `dict` | From `listconfigs()` |

#### Long tier (5-10 min TTL)

| Method | Cache Key | TTL | Notes |
|--------|-----------|-----|-------|
| `get_node_info(node_id)` | Per node_id | 10min | `listnodes` gossip is slow |
| `get_askrene_layers()` | Global | 5min | Stable within cycles |
| `get_feerates(style)` | Per style | 5min | On-chain fees change slowly |

#### Medium tier (30s TTL)

| Method | Cache Key | TTL | Notes |
|--------|-----------|-----|-------|
| `get_peer_channels(peer_id=None)` | Global or per-peer | 30s | Broadcast cached; per-peer uncached |
| `get_funds()` | Global | 30s | Wallet + channel balances |
| `get_peers()` | Global | 30s | Peer connection state |
| `get_channels(source=None, destination=None)` | Per params | 30s | Gossip graph |
| `get_forwards(status=None)` | Per status | 30s | Forward history |
| `get_closed_channels()` | Global | 30s | Closed channel history |
| `get_block_height()` | Global | 30s | From `getinfo().blockheight` |

#### Never cached (transactional)

State-changing operations pass through directly and invalidate relevant caches:

| Method | Invalidates |
|--------|-------------|
| `set_channel(**kwargs)` | `get_peer_channels` cache |
| `fund_channel(**kwargs)` | `get_funds` + `get_peer_channels` cache |
| `close_channel(**kwargs)` | `get_funds` + `get_peer_channels` cache |
| `send_pay(route, payment_hash, ...)` | Nothing (payment state is separate) |
| `wait_send_pay(payment_hash, timeout)` | Nothing |
| `create_invoice(amount_msat, label, description)` | Nothing |
| `delete_invoice(label, status)` | Nothing |
| `delete_pay(payment_hash, status)` | Nothing |
| `pay(bolt11, **kwargs)` | Nothing |
| `get_route(node_id, amount_msat, **kwargs)` | Nothing (amount-dependent) |
| `get_routes(**kwargs)` | Nothing (amount-dependent) |
| `list_pays(**kwargs)` | Nothing |
| `decode(string)` | Nothing |

Askrene mutation commands (create-layer, remove-layer, update-channel,
bias-node, bias-channel, reserve, unreserve) pass through uncached and
invalidate `get_askrene_layers()` where appropriate.

#### Thread safety

DataService uses the same `threading.Lock` pattern as the current `RpcCache`:
per-key locking, atomic check-and-fetch on cache miss. The tiered TTL is
implemented with a single `_cache` dict storing `{"value": ..., "ts": float}`
entries, checked against per-method TTL constants.

#### Cache invalidation

Mutating operations automatically invalidate relevant cached entries:

```python
def set_channel(self, **kwargs) -> dict:
    result = self._rpc.setchannel(**kwargs)
    self._invalidate("listpeerchannels")
    return result
```

Manual invalidation is also available for edge cases:

```python
data_service.invalidate("listpeerchannels")
```

### Database Tier — Absorb Escaped SQL

New methods added to `RevenueDatabase` to eliminate direct `_get_connection()`
access from other modules:

| New Method | Replaces | Module |
|------------|----------|--------|
| `get_all_policies()` | Direct SELECT on `peer_policies` | policy_manager.py |
| `get_policy(peer_id)` | Direct SELECT WHERE | policy_manager.py |
| `upsert_policy(peer_id, **fields)` | Direct INSERT OR REPLACE | policy_manager.py |
| `delete_policy(peer_id)` | Direct DELETE | policy_manager.py |
| `delete_expired_policies(now_ts)` | Direct SELECT + DELETE expired | policy_manager.py |
| `batch_delete_policies(peer_ids)` | Direct BEGIN/DELETE/COMMIT | policy_manager.py |
| `get_policies_by_tag(tag)` | Direct SELECT with tag filter | policy_manager.py |
| `get_total_capex_by_channel(since_ts)` | Direct SUM on rebalance_costs + spend_events | capex_budget.py |
| `cleanup_orphaned_rebalances(active_ids)` | Direct SELECT/DELETE on rebalances | rebalancer.py |

The `fee_controller.py` direct SQL (line 5036) for `channel_costs` lookup
already has a matching `get_channel_open_cost()` method — just update the call
site.

After this migration, `_get_connection()` is truly private to `database.py`.
No module accesses raw SQLite connections.

### Datastore Tier — Standardized Write Helper

DataService provides a uniform method for CLN datastore writes:

```python
def datastore_push(self, key: list, payload: dict) -> bool:
    """Push JSON payload to CLN datastore.

    Automatically:
    - Adds "timestamp" field (int, unix epoch) if not present
    - Validates payload is dict
    - Guards against >60KB payloads (safety margin under 65KB CLN limit)
    - Uses mode="create-or-replace"
    - Fire-and-forget: logs failures at debug level, never raises
    - Returns True on success, False on failure
    """
```

Datastore reads stay as-is — only `hive_hints.py` reads from datastore, and
its two-tier strategy (datastore read → fallback to cross-plugin RPC) is clean
and isolated.

### Sling Removal

All sling-related code is removed from `rebalancer.py`:

| Code to Remove | Description |
|---------------|-------------|
| `sling-job` / `sling-go` / `sling-stop` | Job lifecycle management |
| `sling-deletejob` | Job cleanup (3 call sites) |
| `sling-once` | Fallback execution path (2 call sites) |
| `sling-stats` | Status polling (4 call sites in loops) |
| `sling-except-peer` / `sling-except-chan` | Blocklist management (8 call sites) |
| `sling-jobsettings` | Settings query |
| Related helper methods | Sling error handling, retry logic, result parsing |

The native `RebalanceExecutor` (`rebalance_executor.py`) already handles all
rebalancing via `getroute` → `sendpay` → `waitsendpay`. Sling was the legacy
path.

### Module Migration Pattern

Each module transitions from fragmented access to unified access:

**Before:**
```python
class SomeModule:
    def __init__(self, plugin, database, ...):
        self.plugin = plugin        # for .rpc calls
        self.database = database    # for DB queries
        self.rpc_cache = None       # set later via attribute

    def some_method(self):
        channels = (self.rpc_cache.listpeerchannels()
                    if self.rpc_cache
                    else self.plugin.rpc.listpeerchannels())
        info = self.plugin.rpc.getinfo()
        node_id = info["id"]
```

**After:**
```python
class SomeModule:
    def __init__(self, plugin, database, data_service, ...):
        self.plugin = plugin              # for logging only
        self.database = database          # for DB queries (unchanged)
        self.data_service = data_service  # for all RPC + datastore

    def some_method(self):
        channels = self.data_service.get_peer_channels()
        node_id = self.data_service.get_node_id()
```

Modules keep `self.plugin` for `self.plugin.log()` calls. They stop using
`self.plugin.rpc` for data fetching. This is enforced by convention and code
review, not by making `plugin.rpc` inaccessible — the plugin object is a CLN
framework type and cannot be modified.

### DataService Construction & Injection

In `cl-revenue-ops.py`, DataService replaces RpcCache:

```python
# Before
rpc_cache = RpcCache(safe_plugin, ttl=30)
rebalancer.rpc_cache = rpc_cache
fee_controller.rpc_cache = rpc_cache
# ... etc

# After
data_service = DataService(safe_plugin)
rebalancer = EVRebalancer(plugin, database, data_service, ...)
fee_controller = FeeController(plugin, database, data_service, ...)
# ... etc
```

### CLN API Validation

Phase 7 applies corrections found from validating all 30 RPC methods against
https://docs.corelightning.org/reference. Since all RPC calls will route through
DataService by that phase, fixes are centralized in one file.

**Validated against https://docs.corelightning.org/reference (2026-04-06):**

All 30+ RPC methods validated. Only 1 issue found:

- `decodepay` — **DEPRECATED in CLN v24.11.** Use `decode` instead.
  `boltz_manager.py:457` tries `decodepay` first with `decode` fallback.
  Fix: invert the order (call `decode` first, drop `decodepay`).

Everything else confirmed correct:
- `listpeers` — NOT deprecated (only one feature string renamed in v24.08)
- `listchannels` — NOT deprecated, all params valid
- `getroute` — `maxhops` still supported (default 20)
- `feerates` — `style="perkb"` remains valid
- `askrene-bias-node` — requires CLN v25.12+
- Response field names all correct across all methods

## Migration Phases

### Phase 1: DataService foundation + RPC tier

- Create `modules/data_service.py` with tiered cache
- Implement all RPC wrapper methods with appropriate TTLs
- Implement cache invalidation on mutations
- Thread-safe implementation
- Inject DataService alongside existing RpcCache (dual availability)
- Comprehensive tests for cache behavior, TTL tiers, invalidation

### Phase 2: Database escape absorption

- Add 9 new methods to `database.py` (policy CRUD, capex aggregation,
  orphan cleanup)
- Update `policy_manager.py` to use new database methods
- Update `capex_budget.py` to use `get_total_capex_by_channel()`
- Update `rebalancer.py` to use `cleanup_orphaned_rebalances()`
- Update `fee_controller.py` cost lookup call site
- Remove all `database._get_connection()` usage outside database.py
- Tests for all new database methods

### Phase 3: Sling removal

- Delete all sling-related code from `rebalancer.py`
- Remove sling-job, sling-go, sling-stop, sling-deletejob,
  sling-once, sling-stats, sling-except-peer, sling-except-chan,
  sling-jobsettings call sites and related helpers
- Update tests that reference sling code paths
- Pure deletion — no new code

### Phase 4: Datastore tier

- Add `datastore_push()` to DataService
- Migrate `cl-revenue-ops.py` datastore writes (4 keys)
- Migrate `profitability_analyzer.py` datastore write (1 key)
- Migrate `rebalancer.py` datastore write (1 key)
- Fix missing timestamps on `revenue/status` and `revenue/liquidity-state`
- Tests for datastore_push (timestamp injection, size guard, error handling)

### Phase 5: Migrate core modules

Migrate `self.plugin.rpc.*` and `self.rpc_cache.*` calls to
`self.data_service.*` in:

- `rebalancer.py` (~15 RPC call sites after sling removal)
- `rebalance_executor.py` (~15 RPC call sites)
- `fee_controller.py` (~10 RPC call sites)

These are the highest-traffic modules. Update constructors to accept
`data_service`. Remove `self.rpc_cache` attribute.

### Phase 6: Migrate remaining modules

Same migration pattern for:

- `capacity_planner.py` (~15 RPC call sites)
- `profitability_analyzer.py` (~5 RPC call sites)
- `flow_analysis.py` (~1 RPC call site)
- `hive_router.py` (~12 RPC call sites)
- `boltz_manager.py` (~8 RPC call sites)
- `policy_manager.py` (~1 RPC call site)
- `hive_hints.py` (~2 RPC call sites)

### Phase 7: CLN API validation fixes

- Apply corrections from docs.corelightning.org audit
- Fix deprecated method usage, parameter names, response field access
- All fixes centralized in `data_service.py`

### Phase 8: Cleanup

- Delete `modules/rpc_cache.py`
- Remove `self.rpc_cache` attributes from all modules
- Remove `self.plugin.rpc` usage from all modules (except DataService)
- Update `cl-revenue-ops.py` to stop injecting RpcCache
- DataService becomes sole RPC gateway

## Testing

### DataService tests (`tests/test_data_service.py`)

- Cache tier behavior: forever, long, medium, never
- TTL expiration and refresh
- Cache invalidation on mutations
- Thread safety under concurrent access
- `datastore_push()`: timestamp injection, size guard, fire-and-forget
- `get_node_id()` cached forever (single RPC call across test)
- `get_peer_channels()` returns fresh data after `set_channel()` invalidation

### Database method tests

- Policy CRUD operations (get, upsert, delete, batch_delete, expired)
- `get_total_capex_by_channel()` aggregation correctness
- `cleanup_orphaned_rebalances()` only removes orphans

### Per-module regression

- Existing test suites pass after each phase
- No behavioral changes — same results, cleaner data access paths

## Summary

| Area | Before | After |
|------|--------|-------|
| RPC access | 10 modules call plugin.rpc directly | DataService is sole RPC gateway |
| RPC caching | 4 methods cached, flat 30s TTL | All methods covered, 4-tier TTL |
| Cache invalidation | None | Automatic on mutations |
| Database escapes | 5 modules bypass API with direct SQL | All SQL in database.py |
| Datastore writes | 3 modules, inconsistent patterns | Uniform helper with auto-timestamps |
| Sling code | 16+ dead RPC calls | Deleted |
| rpc_cache.py | Standalone module | Absorbed into DataService, deleted |
