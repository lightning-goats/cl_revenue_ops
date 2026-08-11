# Peer Ban Surface Removal Design

Date: 2026-08-11

## Context

The `revenue-policy` dispatcher is documented as diagnostic for normal
operator use, while tactical policy writes require an explicit
`internal=true` or `admin=true` override. Commit `3c43f8e` added `ban` and
`unban` dispatcher actions without classifying them as tactical writes, so a
caller authorized only for the `revenue-policy` method can persistently change
peer policy.

The peer-ban feature was introduced by commit `ac4cdc2` to enforce two hard
capital gates: reject CapacityPlanner channel opens to a banned peer and veto
LN+ rings containing a banned peer. Version 3.0.0 removed CapacityPlanner,
channel-open authority, and LN+ integration. On the post-decommission source
base, no retained runtime decision path consumes `BANNED_TAG` or
`PolicyManager.is_peer_banned`.

The remaining effects of `ban_peer` are generic policy settings:
`strategy=passive` and `rebalance_mode=disabled`. Those settings already exist
as separately authorized generic policy controls. `unban_peer` is additionally
risky because it unconditionally resets both fields to `dynamic` and `enabled`
instead of restoring a known prior state.

## Decision

Remove the complete peer-ban feature instead of extending the tactical-action
allowlist.

Remove:

- `revenue-policy ban`, `unban`, and `list-banned` actions;
- `revenue-ban`, `revenue-unban`, and `revenue-list-banned` RPC methods;
- the plugin helpers used exclusively by those RPCs;
- `BANNED_TAG` and the `PolicyManager` methods `ban_peer`, `unban_peer`, and
  `is_peer_banned`;
- ban-specific tests, public documentation, RPC inventory entries, and
  compatibility claims.

Retain:

- `revenue-policy list`, `get`, `find`, and `changes` as normal diagnostic
  actions;
- the existing internal/admin gate for generic policy writes;
- generic fee strategies, rebalance modes, tags, and their retained runtime
  consumers;
- standalone operation with no Sling, Hive, Mycelium, planner, Boltz, LN+, or
  channel-open/close dependency.

## Stored Data and Upgrade Behavior

The change performs no database migration and does not rewrite existing peer
policies.

An existing policy created by `ban_peer` remains `passive` with rebalancing
`disabled`, so the plugin continues to leave that peer unmanaged. A historical
`banned` string in the policy's generic `tags` array becomes inert metadata.
An operator who intentionally wants to change such a policy must use the
separately authorized generic policy-write workflow; the plugin will not infer
or apply a replacement state during upgrade.

This avoids silently re-enabling fee or rebalance management and avoids
inventing a prior policy state that was never recorded.

## RPC and Error Semantics

The three standalone methods are removed from plugin registration and therefore
return Core Lightning's normal method-not-found response.

Within `revenue-policy`, `ban`, `unban`, and `list-banned` are ordinary unknown
actions. They return the existing unknown-action error and valid-action list.
They must not call any policy mutation helper. Malformed, absent, or unknown
actions retain the existing fail-safe error behavior.

This is an intentional compatibility break. The ban aliases had an announced
Phase C removal date, but the security boundary and removal of their original
enforcement consumers justify removing them now rather than preserving a
vulnerable or misleading compatibility surface.

## Security Invariant

A caller whose rune permits the `revenue-policy` method but who lacks the
explicit internal/admin override cannot mutate peer policy. Removed action
names must not provide an alternate mutation path, and no equivalent ban helper
may remain reachable through another public RPC.

The legitimate behavior that must remain is:

- read-only policy diagnostics work without an override;
- authorized generic policy writes still work with an override;
- passive fee policies still stop fee management;
- disabled/passive policies still exclude peers from retained rebalance paths;
- missing, malformed, and unknown actions fail without mutation.

## Test Strategy

Implementation follows red-green-refactor:

1. Add or update a regression test that expects the six ban RPC/action names to
   be absent and run it against the vulnerable base to observe the expected
   failure.
2. Add a dispatcher regression proving `ban`, `unban`, and `list-banned` return
   unknown-action errors and make zero policy-manager mutation calls; observe
   failure before production edits.
3. Remove the production surface and ban-specific unit tests.
4. Verify the focused policy, dispatcher, operator-surface, RPC-inventory, and
   architecture suites.
5. Verify syntax/import buildability, re-run the original exploit condition,
   review equivalent policy mutation sinks, and run the full repository suite.

All tests use repository fakes or mocks. No live Core Lightning action RPC,
policy mutation RPC, fee cycle, rebalance cycle, payment, open, close,
withdrawal, or swap action is permitted during validation.

## Documentation and Inventory Updates

Update the current operator documentation and compatibility catalog to remove
the ban family. Update the canonical action-RPC inventory and registered-surface
test together so the documented and executable RPC contracts agree. Historical
planning and audit records remain unchanged when they describe past behavior;
they are not current operator contracts.

## Verification and Review

This is tier-1 security and production-policy work. The owner may pass automated
criteria but may not pass the independent review criterion. A verifier other
than the owner must independently confirm:

- the original source-to-sink path no longer exists;
- all current public ban names and helpers are removed;
- the generic policy write gate remains intact;
- retained fee and rebalance policy behavior still passes;
- no Sling or retired liquidity-executor dependency returns;
- no live action RPC was invoked during testing.
