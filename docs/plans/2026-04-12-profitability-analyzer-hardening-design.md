# Profitability Analyzer Hardening Design

**Date:** 2026-04-12

**Goal:** Correct the profitability analyzer's cost attribution and aggregate reporting bugs, then land the fixes on `main` and port the relevant subset to `pure-revenue-ops` without reintroducing branch-divergence mistakes.

## Problem Summary

The audit found four real issues:

1. `BookkeeperCache` reads channel-account `bkpr-listincome` fee events with the wrong sign, so per-channel open costs can collapse to wallet-level totals instead of the channel's actual fee.
2. Aggregate profitability summaries use per-channel valuation contribution to compute total profit and ROI, even though that metric is explicitly documented as unsuitable for fleet-level revenue reporting.
3. The legacy `bkpr-listaccountevents` fallback queries the wrong account shape and is not aligned with current Core Lightning docs.
4. Peer-level profitability reporting returns the first matching channel for a peer, which is arbitrary for multi-channel peers.

## Approaches Considered

### Option 1: Fix `main` first, then port the branch-appropriate subset to `pure-revenue-ops`

This keeps one canonical implementation of the accounting fixes, verifies them on the primary branch, then ports only the relevant code to the standalone branch.

Pros:
- Lowest risk of fixing the same bug two different ways.
- Best fit for shared analyzer/database code.
- Keeps `main` as the source of truth for accounting semantics.

Cons:
- Requires a second pass to port and adapt tests for `pure-revenue-ops`.

### Option 2: Implement two first-class fixes in parallel

Treat `main` and `pure-revenue-ops` as independent products and implement both directly.

Pros:
- Each branch can optimize for its own surface immediately.

Cons:
- High chance of semantic drift.
- More review burden.
- Easy to miss shared DB or math assumptions.

### Option 3: Fix only the analyzer internals and leave operator/reporting surfaces alone

This limits the work to `modules/profitability_analyzer.py` and avoids RPC/output changes.

Pros:
- Smaller patch.

Cons:
- Leaves a user-visible bug in `revenue-profitability` and `revenue-report peer`.
- Does not actually close the audit findings end to end.

## Recommended Design

Use Option 1.

The accounting semantics should be corrected on `main` first in one canonical patch series, then the same logical fixes should be ported to `pure-revenue-ops` with only the branch-specific surface adjustments needed for that branch's standalone runtime.

## Design

### 1. Bookkeeper fee attribution

The authoritative path is `bkpr-listincome(consolidate_fees=true)` through `BookkeeperCache`.

The fix is:
- treat `onchain_fee` as a node expense from the node's perspective for both wallet and channel accounts
- compute channel-account fee as `debit_msat - credit_msat`, not `credit_msat - debit_msat`
- keep channel-account results preferred over wallet fallback when both exist

The legacy fallback path should also be corrected:
- stop querying `bkpr-listaccountevents(account=<reversed_txid>)`
- use the documented `payment_id=<funding_txid>` lookup instead
- sum matching `onchain_fee` events by txid and preserve the same debit-minus-credit semantics

This keeps both code paths aligned with current CLN docs and local `lightning-cli help`.

### 2. Aggregate profitability semantics

Per-channel valuation contribution remains useful for channel classification and protection decisions. It should continue to exist.

But aggregate summary math must distinguish:
- **real revenue:** sum of exit-channel fees earned
- **valuation contribution:** per-channel max-earned-vs-sourced metric for local channel scoring only

The fix is:
- `revenue-profitability` summary:
  - `total_revenue_sats` stays actual routed revenue
  - `total_contribution_sats` can remain as a separate informational field
  - `total_profit_sats` and `overall_roi_pct` must be computed from real revenue minus costs, not valuation contribution minus costs
- `ChannelProfitabilityAnalyzer.get_summary()` should use the same distinction

This removes the double-counted fleet-level profit/ROI bug while preserving the channel-level valuation model.

### 3. Peer-level profitability reporting

The current `get_profitability_by_peer()` helper is unsafe for peers with multiple channels because it returns the first cache hit.

The fix should make `revenue-report peer` deterministic and truthful:
- aggregate all matching channels for the peer
- return a peer-level profitability structure with:
  - `channel_count`
  - `aggregate` totals
  - per-channel entries when helpful

Avoid pretending a peer has a single profitability object when it may have several channels with different outcomes.

### 4. Branch integration strategy

Implementation order:
1. Create a dedicated `main` hardening worktree/branch.
2. Fix analyzer math and tests there.
3. Verify and merge/push to `main`.
4. Port the branch-appropriate subset to `pure-revenue-ops`.
5. Re-run that branch's full suite.
6. Push `pure-revenue-ops` separately.

Because the root `main` checkout is dirty, all integration should happen through clean worktrees, not the root checkout.

## Testing Strategy

Add or update tests for:
- docs-shaped `bkpr-listincome` channel-account fee events
- wallet-vs-channel precedence under corrected fee signs
- corrected `bkpr-listaccountevents(payment_id=...)` fallback behavior
- aggregate `revenue-profitability` summary using real revenue for total profit/ROI
- peer report behavior for multi-channel peers
- branch-specific regression checks on `pure-revenue-ops`

Full suites should pass on both branches before merge/push.

## Success Criteria

- Channel open costs are attributed from bookkeeper with correct sign semantics.
- Aggregate profitability no longer double-counts channel valuation in operator-facing totals.
- Peer-level profitability reporting is deterministic for multi-channel peers.
- `main` and `pure-revenue-ops` both receive the fixes and pass their respective full test suites.
