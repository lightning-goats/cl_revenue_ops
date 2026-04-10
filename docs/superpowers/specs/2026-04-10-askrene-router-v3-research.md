# Rebalance Router V3 — Research Findings

**Date:** 2026-04-10
**Status:** In progress
**Parent spec:** `docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md`
**Implementation plan:** `docs/superpowers/plans/2026-04-10-askrene-router-v3-research.md`
**Worktree:** `.worktrees/askrene-router-v3-20260410`
**Branch:** `feature/askrene-router-v3`

## Environment

**CLN upstream reference SHA:** `b57edd2128fa21b492f7c215d13ebfcf74bdc579`
**CLN upstream tip message:** `keysend: increase assumed final_cltv_expiry to 42 (to match LDK).`
**Upstream clone location (session-local):** `/tmp/cln-upstream` (shallow clone, depth 1)

**Live node** (accessed via `ssh lnnode 'lightning-cli …'`):

```json
{
  "version": "v25.12.1",
  "network": "bitcoin",
  "id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
  "blockheight": 944486,
  "num_peers": 63,
  "num_active_channels": 46
}
```

**Live askrene layers observed:** `hive-fleet`, `hive-reputation`, `hive-corridors`, `hive-traffic` (all four cl-hive layers), `revenue-local`, `xpay` (automatic xpay-internal layer).

**Citation format:** `ElementsProject/lightning@b57edd21:<path>#L<start>-L<end>`

All claims have either a source citation or a captured live-node RPC transcript. No claim is based on memory or docs.corelightning.org; only the upstream source at the pinned SHA and real RPC transcripts are authoritative.

---

## 1. getroutes Contract

_PENDING: Task 1_

## 2. Layer Lifecycle

_PENDING: Task 2_

## 3. Layer Semantics Under Pair Pinning

_PENDING: Task 3_

## 4. Exclude-Via-Layer Pattern

_PENDING: Task 4_

## 5. xpay API Surface

_PENDING: Task 5_

## 6. xpay vs sendpay+waitsendpay Behavior Diff For Circular Self-Pays

_PENDING: Task 6_

## 7. setconfig Runtime-Switch Verification

_PENDING: Task 7_

## 8. Failure-Mode Taxonomy

_PENDING: Task 8_

## 9. Decision Records

_PENDING: Task 9 (partial — decisions that depend on deferred live-node data will be marked "pending experiment")_
