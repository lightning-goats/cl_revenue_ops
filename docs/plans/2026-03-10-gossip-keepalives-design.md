# Gossip Keepalives Design

**Date**: 2026-03-10
**Status**: Approved

## Problem

`cl-revenue-ops` depends on fresh channel gossip in both directions:
- our `channel_update` fee changes must propagate quickly
- competitor fee changes must reach us quickly

Today the plugin tracks uptime for peers with channels, but it does not actively
maintain extra P2P-only peers that keep the node well attached to the network's
gossip backbone. If overall connectivity drops, the node's graph view can go
stale and dynamic pricing decisions degrade.

## Solution

Add an opt-in gossip keepalive subsystem that maintains a minimum number of
connected peers. The target counts all connected peers, not just P2P-only
connections. When connectivity drops below the configured target, the plugin
actively dials additional non-channel peers to restore gossip resilience.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope of target | Total connected peers | Matches approved product decision: existing channel peers already satisfy the connectivity target |
| Candidate priority | Hive members first, public hubs second | Fleet peers are known-good strategic targets; public hubs fill the rest |
| Public hub ranking | `listnodes` candidates + `listchannels` aggregation | `listnodes` alone does not expose capacity/channel count directly |
| Storage | In-memory state only | Backoff and dial attempts do not need persistence or schema changes |
| Failure handling | Non-fatal with capped exponential backoff | Avoid log spam and repeated hammering of dead peers |

## Architecture

### Main loop

A new `gossip_maintenance_loop` runs in the background from
`cl-revenue-ops.py`, following the repo's existing daemon-thread pattern:
- use `shutdown_event.wait(...)` for interruptible sleep
- read a config snapshot each cycle
- exit early if the feature is disabled

### Manager module

Add a focused helper module, `modules/gossip_keeper.py`, that owns:
- current peer inspection and filtering
- hive-priority target discovery
- graph-derived public hub scoring
- per-peer retry backoff
- conservative `connect` execution

This keeps orchestration out of the already large plugin entrypoint while still
matching the issue's requirement that the background loop live in
`cl-revenue-ops.py`.

### Hive integration

`modules/hive_bridge.py` gets a small convenience helper to return hive member
pubkeys suitable for keepalive targeting. Hive participation is opportunistic:
if `cl-hive` is unavailable or membership lookups fail, gossip keepalives fall
back to public-graph discovery with no hard dependency.

## Data Flow

On each cycle:

1. Read the config snapshot and stop if `enable_gossip_keepalives` is `false`.
2. Call `listpeers` and count all currently connected peers.
3. If connected peers already meet `target_gossip_peers`, do nothing.
4. Build exclusion sets from:
   - our own node id
   - already connected peers
   - peers with channels
   - peers currently under backoff
5. Build ranked targets:
   - hive member pubkeys first
   - then public graph candidates scored from `listnodes` + `listchannels`
6. Attempt enough `connect` calls to fill the deficit.
7. Record success/failure in memory and apply capped exponential backoff after
   failed dials.

The subsystem never opens channels and never disconnects existing peers.

## Candidate Scoring

### Hive targets

Use `hive-members` when hive mode is available. Candidate filtering excludes:
- our own node id
- already connected peers
- peers with channels

Hive members are treated as priority candidates in the returned order.

### Public graph targets

Use `listnodes` to enumerate public node ids and `listchannels` to derive hub
strength. Each candidate receives a score based on:
- number of active public edges
- aggregate public channel capacity

The scorer is intentionally simple and deterministic. No extra heuristics are
added for now beyond excluding already-connected and channel peers.

## Configuration

Add two new config fields and plugin options:

- `enable_gossip_keepalives: bool = False`
- `target_gossip_peers: int = 5`

No runtime mutability changes are required beyond standard config loading.

## Error Handling

All failures are non-fatal:
- `listpeers`, `listnodes`, `listchannels`, `hive-members`, and `connect`
  failures are logged and the cycle continues or exits safely
- per-peer connect failures enter exponential backoff
- hive lookup failures degrade to graph-only discovery
- empty candidate sets are treated as "nothing to do", not as errors

## Testing Strategy

### New tests

Create `tests/test_gossip_keeper.py` for:
- counting total connected peers
- excluding connected/channel/self peers
- hive-priority target ordering
- graph-derived scoring from `listchannels`
- connect deficit filling
- backoff after failed `connect`

### Existing test extensions

- `tests/test_hive_integrations.py`: new tests for the hive member helper
- `tests/test_plugin_audit_regressions.py` or a focused plugin test file: loop
  wiring and config parsing smoke coverage if needed

## Out of Scope

- disconnecting excess peers
- persistent dial history in SQLite
- DNS-based peer discovery
- advanced uptime or latency probing
- channel-open automation
