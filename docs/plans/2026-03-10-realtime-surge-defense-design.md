# Real-Time Surge Defense Design

## Goal

Implement issue `#67` by adding a low-latency, in-memory surge defense on the `htlc_accepted` hook that detects toxic burst flow in real time, applies a bounded temporary fee overlay, and restores the exact pre-surge baseline after cooldown.

## Context

`cl-revenue-ops` currently has:

- a pure pass-through `htlc_accepted` hook in [cl-revenue-ops.py](../../cl-revenue-ops.py)
- a centralized fee write path in `HillClimbingFeeController.set_channel_fee()`
- thread-safe RPC wrappers via `ThreadSafeRpcProxy`
- rich scheduled fee logic, but no real-time protection between controller cycles

That means the issue is not another scheduled fee feature. It needs a new hot-path-safe component that can observe bursts immediately without contaminating the existing optimizer state.

## Approaches Considered

### 1. Recommended: Dedicated real-time surge manager with strict overlay semantics

Add a new in-memory manager that:

- tracks rolling per-channel burst state from `htlc_accepted`
- queues surge and revert intents off the hook path
- applies fee overlays without changing the controller's learned baseline
- restores the exact pre-surge fee when cooldown expires

Why this wins:

- preserves hook latency
- matches the issue's explicit "instant surge + safe reversion" requirements
- avoids poisoning Thompson/AIMD or broadcast fee learning with emergency pricing
- keeps normal fee control and emergency defense conceptually separate

### 2. Lightweight globals in `cl-revenue-ops.py` with direct async `setchannel`

Add a few global dicts and directly call `safe_plugin.rpc.fire_and_forget("setchannel", ...)`.

Why not:

- weak success/failure tracking for surge and revert writes
- hard to expose accurate status/debug data
- less control over debounce and exact revert

### 3. Scheduled-loop integration only

Have `htlc_accepted` set flags that the next fee cycle consumes.

Why not:

- fails the core problem statement
- arbitrage or toxic flow can drain a channel well before the next scheduled run

## Chosen Design

### High-Level Architecture

Add a new `RealtimeSurgeDefense` component, owned by `cl-revenue-ops.py`, with three responsibilities:

1. **Fast-path accounting**
   - Update rolling burst state from `htlc_accepted`
   - Evaluate trigger, cooldown, and concentration heuristics
   - Enqueue surge/revert intents
   - Always return `{"result": "continue"}`

2. **Overlay state management**
   - Track exact baseline fee captured before surge
   - Track active surge fee, trigger reason, cooldown expiry, and debounce timestamps
   - Keep short in-memory counters for observability

3. **Background fee application**
   - Apply surge and revert writes outside the hook path
   - Bound `setchannel` update frequency per channel
   - Mark state only after successful writes

This is intentionally a strict temporary overlay. The surge fee is not promoted into the controller's normal learned state. The slow controller continues operating, but active surge overlays win until the overlay is removed and the exact baseline fee is restored.

### Hook Data Model

The hook uses only fields that are cheap and available on the hot path:

- incoming peer id
- incoming channel id / short channel id
- outgoing channel id from `forward_to`
- HTLC amount
- current timestamp

Each accepted forward contributes one event into a per-outgoing-channel rolling window.

### Rolling Burst State

Per outgoing channel, maintain an in-memory structure containing:

- a deque of recent HTLC samples inside `surge_window_seconds`
- total moved msat in the active window
- HTLC count in the window
- moved fraction of channel capacity
- top incoming peer volume share
- top incoming peer HTLC share
- active overlay state, if any
- cooldown expiry
- last `setchannel` attempt timestamp
- trigger counters for 1h and 24h windows

The first version is explicitly in-memory only. If the plugin restarts, surge state resets and normal controller behavior resumes.

### Trigger Logic

The issue asked for peer-concentration heuristics in v1, so the trigger uses both burst size and concentration.

Primary trigger:

- moved percent over `surge_window_seconds` exceeds `surge_trigger_pct`

And at least one secondary signal:

- HTLC cadence exceeds a configured count threshold
- a single incoming peer dominates burst volume
- a single incoming peer dominates burst HTLC count

This keeps v1 channel-local and low-latency while still distinguishing broad normal routing from concentrated arbitrage-style draining.

### Fee Overlay Semantics

When a trigger fires:

1. capture the current effective fee as the exact baseline
2. compute a temporary surge fee from:
   - baseline fee
   - configured multiplier range
   - moved percent severity
3. clamp the result to:
   - configured surge multiplier bounds
   - global fee controller safety maximum
4. enqueue an async `setchannel`
5. mark the channel as surge-active only after successful application

When the burst subsides and cooldown expires:

1. enqueue a revert intent
2. restore the exact captured baseline fee
3. clear overlay state only after successful revert

If the calm period arrives but debounce still blocks writes, the channel remains surge-active until the revert can be safely applied.

### Cooldown, Debounce, and Hysteresis

To avoid oscillation:

- `surge_setchannel_min_interval_seconds` limits how often a channel can be updated
- `surge_cooldown_seconds` keeps the overlay active after the last qualifying burst
- repeated toxic events during active surge extend cooldown rather than re-capturing baseline
- a new trigger while surge is active may raise the active overlay fee further, but only after debounce allows it

This creates a monotonic temporary defense during hostile flow and a single exact revert when calm returns.

### Interaction with Existing Fee Control

The scheduled fee controller remains the system of record for normal optimization, but it should not overwrite an active surge overlay.

The recommended integration is:

- the surge manager owns temporary real-time fee writes
- status/debug exposes overlay state separately from the controller's last scheduled decision
- if needed during implementation, the fee controller can read surge-active state and suppress non-emergency writes for those channels until the overlay clears

That keeps the issue scoped: real-time defense first, controller coexistence second.

## Data Flow

1. `htlc_accepted` receives a forward attempt.
2. The surge manager resolves the outgoing channel and updates the rolling channel window.
3. The manager evaluates:
   - moved percent
   - HTLC count/cadence
   - incoming peer concentration
   - current surge/cooldown/debounce state
4. If a threshold crosses:
   - enqueue a surge intent with reason and target fee
5. If calm persists past cooldown:
   - enqueue a revert intent with the stored baseline fee
6. A background worker executes `setchannel` writes.
7. `revenue-status` exposes active surge state and recent counters.

## Error Handling

- The hook must fail open: any error logs at `debug` or `warn` and still returns `{"result": "continue"}`
- Missing or malformed hook fields should skip accounting for that event rather than block forwarding
- Failed surge/revert RPCs should leave overlay state unchanged and retry only when debounce permits
- If no valid baseline fee can be determined, do not arm surge mode for that channel
- Background worker exceptions must never crash the plugin or block future intents

## Observability

Add a compact `realtime_surge_defense` section to `revenue-status` with:

- enabled/disabled
- number of active surge channels
- per-channel baseline fee and active surge fee
- cooldown remaining
- last trigger reason
- trigger counts for 1h and 24h
- last apply/revert result

The first cut should avoid new RPC surfaces unless testing shows status payloads become too large.

## Testing Strategy

Cover the feature in four layers:

1. **Manager unit tests**
   - trigger on moved-percent bursts
   - trigger on peer concentration
   - debounce prevents over-updating
   - cooldown extends during repeated bursts
   - exact baseline revert after calm

2. **Hook behavior tests**
   - `htlc_accepted` always returns continue
   - normal flow does not enqueue surge
   - malformed events fail open

3. **Plugin wiring tests**
   - config option registration and init wiring
   - `revenue-status` includes surge defense state

4. **Integration-style tests**
   - repeated hook events trigger a surge apply
   - cooldown expiry triggers revert
   - revert restores exact baseline, not a recomputed scheduled fee

## Non-Goals

- no persistent per-HTLC event storage
- no new optimizer learning rules
- no per-peer permanent sanctions or ban logic
- no replacement of the scheduled fee controller
- no broad routing-path inspection beyond the hook fields already available
