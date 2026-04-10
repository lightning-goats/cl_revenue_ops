# Rebalance Router V3 — Research Findings

**Date:** 2026-04-10
**Status:** In progress (source-only pass; live-node experiments deferred)
**Parent spec:** `docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md`
**Implementation plan:** `docs/superpowers/plans/2026-04-10-askrene-router-v3-research.md`
**Worktree:** `.worktrees/askrene-router-v3-20260410`
**Branch:** `feature/askrene-router-v3`

## Environment

**CLN upstream reference SHA:** `b57edd2128fa21b492f7c215d13ebfcf74bdc579`
**CLN upstream tip message:** `keysend: increase assumed final_cltv_expiry to 42 (to match LDK).`
**Upstream clone location (session-local):** `/tmp/cln-upstream` (shallow clone, depth 1)

**Live node:** No `lightningd` running in this execution environment. Live-node experiments (Tasks 3, 4, 6, 7 of the research plan) are deferred to a follow-up session with real RPC access. Sections that depend on live-node data are marked `DEFERRED: needs live node` and list the exact commands to run when the follow-up session begins.

**Citation format:** `ElementsProject/lightning@b57edd21:<path>#L<start>-L<end>`

All claims in completed sections have either a source citation or a direct file excerpt. No claim is based on memory or docs.corelightning.org; only the upstream source at the pinned SHA is authoritative.

---

## 1. getroutes Contract

_PENDING: Task 1_

## 2. Layer Lifecycle

_PENDING: Task 2_

## 3. Layer Semantics Under Pair Pinning

**DEFERRED: needs live node.**

This section requires running `getroutes` against real peers with and without `hive-fleet` layers to measure whether layers actually bias middle-hop selection under pair pinning. The exact commands to execute in the follow-up session are listed in the research plan at `docs/superpowers/plans/2026-04-10-askrene-router-v3-research.md` under Task 3.

## 4. Exclude-Via-Layer Pattern

**DEFERRED: needs live node.**

Requires live `askrene-create-layer`/`askrene-remove-layer` benchmarks to measure whether per-retry layer cost is under the 50ms threshold from the parent spec. Commands in plan Task 4.

## 5. xpay API Surface

_PENDING: Task 5_

## 6. xpay vs sendpay+waitsendpay Behavior Diff For Circular Self-Pays

**DEFERRED: needs live node.**

Requires actually sending 1000 sats around the node twice (once with `sendpay`, once with `xpay`) and diffing the RPC transcripts. Commands in plan Task 6.

## 7. setconfig Runtime-Switch Verification

**DEFERRED: needs live node.**

Requires live `setconfig` on a dynamic key with plugin log observation to confirm pyln-client's notification model. Commands in plan Task 7.

## 8. Failure-Mode Taxonomy

_PENDING: Task 8_

## 9. Decision Records

_PENDING: Task 9 (partial — decisions that depend on deferred live-node data will be marked "pending experiment")_
