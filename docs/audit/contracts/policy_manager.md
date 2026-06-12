# Intent Contract: modules/policy_manager.py

Tier 1 (deep treatment). Authored 2026-06-12 from code + entry-point wiring. No outcome data was
consulted; hypotheses are pre-registered.

## 1. Purpose

PolicyManager is the declarative per-peer control plane: it stores and serves a `PeerPolicy`
(fee strategy DYNAMIC/STATIC/PASSIVE, rebalance mode ENABLED/DISABLED/SOURCE_ONLY/SINK_ONLY, tags,
optional fee-multiplier bounds, optional expiry) that gates what the automated engines may do to each
peer (policy_manager.py:58-100, 152-178). It replaced the legacy `ignored_peers` mechanism; the
legacy RPCs are now compatibility shims over policies. It is deliberately not an optimizer: it
validates, rate-limits, caches (write-through), persists, and notifies subscribers of changes
(callbacks, :219-253) so the fee controller wakes affected channels immediately
(fee_controller.py:6751-6795). It also runs two advisory/automation layers: profitability-driven
policy *suggestions* (bleeders/zombies/high-velocity sources — suggestions only, never auto-applied,
:898-1041) and hive-corridor auto-policies that tag corridor owners and reconcile role loss while
refusing to touch operator-owned policies (:1047-1139). Docstring drift, noted: the
`apply_corridor_policies` docstring says fleet members get "dedicated hint-driven 0-fee handling in
the fee controller" (:1056-1057, :1085-1087), but the current fee controller treats hive membership
as advisory and keeps DTS/PID active (fee_controller.py:4692-4694); the 0-PPM forced path no longer
exists, so that comment describes a previous regime.

## 2. Inputs / Outputs

Inputs (consumed):
- `policies` table via Database: `get_all_policies`, `upsert_policy`, `upsert_policies_batch`,
  `delete_policy`, `delete_expired_policies`, `get_policy_changes_since`,
  `get_last_policy_change_timestamp` (:346, :663-667, :714, :1275-1279, :1316, :497, :527).
- ProfitabilityAnalyzer (optional, for suggestions): `identify_bleeders`,
  rebalance success-rate trend queries (:924, :950-951); `channel_states` for high-velocity source
  detection (:1005).
- HiveHints (injected, cl-revenue-ops.py:2059-2060): `is_hive_member`, `get_corridor_role` (:1081-1082).
- data_service `listpeerchannels` for the corridor sweep (:1068).

Outputs / consumers:
- FeeController: `get_policy()` consulted per channel each fee cycle — PASSIVE skips, STATIC applies
  fixed fee, DYNAMIC proceeds (fee_controller.py:4635-4690); initial-fee path (fee_controller.py:7380-7401);
  gossip prefetch skips PASSIVE peers (fee_controller.py:4441-4446). Change callback wakes sleeping
  channels of the affected peer (registered fee_controller.py:2563-2565).
- Rebalancer and planners: `should_rebalance(peer_id, as_destination)` direction gating (:840-861),
  `should_manage_fees` (:825-838), `get_fee_multiplier_bounds` (:879-892).
- RPC surfaces (entry point): `revenue-policy` (list/get/set/delete/find/changes/batch,
  cl-revenue-ops.py:4007-4140), `revenue-ignore`/`revenue-unignore`/`revenue-list-ignored` legacy shims
  (cl-revenue-ops.py:3883-3985).
- Periodic `cleanup_expired_policies` from the flow-analysis loop (cl-revenue-ops.py:2172-2174);
  `apply_corridor_policies` before each fee cycle (cl-revenue-ops.py:2520-2523, `_refresh_fee_cycle_hive_inputs`).
- Datastore/files: none directly; policy effects surface through fee-change records and stable fees.

## 3. Invariants

- PM-I1 **Peer-id validation.** Every write path validates a 66-char hex peer_id and raises ValueError
  otherwise (:300-313, used at :572, :712, :1179). Checkable: tests/test_database_policies.py,
  test_operator_surface.py.
- PM-I2 **STATIC requires a target — single-set path only.** `set_policy(strategy=static)` without
  `fee_ppm_target` raises (:606-609). VERIFIED GAP: `set_policies_batch` performs no such check
  (its per-entry validation at :1184-1207 covers strategy/mode/fee bounds/tags/multipliers/expiry but
  not the static-target pairing), so a batch update can persist STATIC with fee_ppm_target=None —
  in which case the fee controller silently falls through to dynamic management, the exact failure
  this invariant exists to prevent. The only backstops are consumer-side null-checks
  (fee_controller.py:4644 and :7391 proceed past STATIC when the target is None). Needs a code fix
  to restore batch/single parity.
- PM-I3 **Fee target bounds.** `fee_ppm_target` must be a non-negative int <= 100000 (:601-605,
  batch :1203-1207). Note: the *applied* static fee is additionally clamped to configured economic
  bounds by the fee controller (fee_controller.py:4649-4652).
- PM-I4 **Multiplier bounds are sane.** Per-policy fee multiplier min/max are validated into
  [0.1, 5.0], min <= max rejected at write time (:621-640, batch :1214-1234), and read-side
  `get_fee_multiplier_bounds` re-enforces global bounds and swaps inversions defensively (:112-129).
- PM-I5 **Expiry is bounded and honored.** `expires_in_hours` is capped at 30 days (:648-651);
  expired policies are never served — `get_policy` returns the DYNAMIC/ENABLED default and deletes
  the stale row (:106-110, :441-468); cache load skips expired rows (:350-353).
- PM-I6 **Rate limit counts only committed changes (single-set path).** Max 10 policy changes/minute/
  peer; exceeding raises RuntimeError. In `set_policy` the counter increments only after a successful
  DB write, so failed validation or DB errors never consume budget (:259-298, :655-670). Verified
  asymmetry: the batch variant records timestamps after validation and the read-only rate check but
  BEFORE the DB write (:1253-1272 records, :1275 writes), so a failed batch DB write does consume
  budget — failed validation does not.
- PM-I7 **Default is permissive.** A peer with no stored policy gets DYNAMIC + ENABLED + no tags
  (:460-468, DEFAULT_POLICY :182-188) — absence of policy must never silently freeze a peer.
- PM-I8 **Automation never overwrites operator intent.** `apply_corridor_policies` skips
  'manual'-tagged policies (:1076-1079) and refuses to modify stored policies it did not create:
  corridor-owner writes and role-loss deletions touch only `auto_corridor`-tagged rows (:1091-1129),
  and the hive-membership branch additionally deletes legacy `auto_fleet`-tagged rows (:1084-1090) —
  not only `auto_corridor` as previously stated. Caveat: the auto_fleet deletion keys on tag
  presence alone; a stored policy carrying that tag is deleted even if an operator has since
  modified other fields (only the 'manual' tag protects it).
- PM-I9 **Every committed change notifies subscribers — except lazy expiry.** set/delete/batch/
  periodic-expiry-cleanup all fire registered callbacks after persistence (:696, :723-729, :1298,
  :1327-1332), so the fee controller's wake behavior (test_policy_change_wake.py) is reachable from
  those mutation paths. Verified exception: the lazy-expiry path inside `get_policy` (:446-458,
  :470-476) evicts the cached policy and deletes expired DB rows WITHOUT notifying; expiry
  notifications are only guaranteed via the periodic `cleanup_expired_policies` loop
  (cl-revenue-ops.py:2172-2174).
- PM-I10 **Batch is validate-first.** `set_policies_batch` validates every entry (and checks rate
  limits read-only) before any DB write; a single invalid entry fails the whole batch with no
  counter pollution and no partial persistence (:1173-1279). Two verified deviations from
  set_policy parity: the STATIC-requires-target check is missing (see PM-I2) and rate-limit
  timestamps are recorded before the DB write (see PM-I6).
- PM-I11 **Legacy semantics are narrower than PASSIVE.** `is_peer_ignored` returns True only for
  PASSIVE *and* DISABLED (:1346-1358), while fee management stops at PASSIVE alone (:825-838) —
  consumers must not treat these as equivalent.
- PM-I12 **Batch size is bounded.** `set_policies_batch` rejects batches over 100 entries before any
  validation work (:1162-1165).
- PM-I13 **Corrupt rows degrade to safe defaults.** Unparseable tags JSON, an unknown strategy, or an
  unknown rebalance_mode in a stored row are logged and replaced by [] / DYNAMIC / ENABLED during row
  decode rather than raising (:362-418) — a corrupted row can neither brick policy reads nor silently
  freeze a peer (consistent with PM-I7's permissive default).

## 4. Revenue role

Indirect — a guardrail and division-of-labor layer rather than an earner. Causal paths to net revenue:
(a) PASSIVE/STATIC let the operator pin pricing where the optimizer is known to misprice (or where a
business relationship fixes the fee), preventing revenue-destroying experimentation on those peers;
(b) rebalance modes stop the rebalancer from spending sats refilling channels that never pay back
(bleeders) or draining channels that earn as sources — the suggestions engine exists precisely to
convert profitability losses into DISABLED/source_only policies; (c) corridor auto-policies keep
hive corridor-owner channels under active management and tagged for protection, supporting
fleet-level routing income; (d) expiry makes tactical overrides self-cleaning so stale manual
interventions cannot quietly suppress optimization forever. The module's own writes cost nothing;
its revenue contribution is the avoided losses and preserved optimizer coverage.

## 5. Pre-registered hypotheses

- PM-H1 **Disabling rebalance on flagged bleeders improves net PnL.** Metric: per-channel net margin
  = routing fees earned (listforwards-window) minus attributed rebalance spend
  (revenue-spend-ledger / revenue-profitability), 14 days before vs 14 days after the policy change.
  Population: peers whose rebalance_mode transitioned to disabled/source_only (policy timestamps via
  revenue-policy changes / fee-change records). Control: contemporaneous bleeder-like channels
  (negative net margin in the before-window) with unchanged policy. Direction: treated channels'
  net margin improves more than controls'. Test: difference-in-differences with paired bootstrap,
  95% CI excluding zero.
- PM-H2 **Policy gating actually binds (compliance precondition for any revenue claim).** Metric:
  count of automated fee changes (revenue-history entries with manual=false and reason_code !=
  policy_static) on peers while PASSIVE, and advertised-fee deviations from fee_ppm_target while
  STATIC (listpeerchannels). Baseline: expected zero. Direction/test: exact binomial against zero
  tolerance; any violation falsifies the contract (this is a conformance hypothesis: if it fails,
  PM-H1's causal attribution is void). Caveats: (a) revenue-history records are written best-effort
  post-RPC (fee_controller.py:7077-7089), so absence of a record is weaker evidence than presence —
  cross-check against listpeerchannels fee transitions; (b) STATIC compliance must use the
  controller-clamped value clamp(fee_ppm_target, min_fee, max_fee), not the raw target
  (fee_controller.py:4649-4652).
- PM-H3 **Expiring tactical overrides recover optimizer upside.** Metric: fees/day in the 7 days
  after a policy expiry event (peer reverts to DYNAMIC) vs the last 7 days under the override.
  Population: expired policies observed in the corpus window. Direction: fees/day does not decline
  (and median improves) after reversion to DYNAMIC. Test: Wilcoxon signed-rank, p < 0.05; report CI.

## 6. Observable surface (hermes corpus)

- `revenue-history.json` — fee-change records: reason_code `policy_static` marks STATIC enforcement;
  absence of automated changes on PASSIVE peers (PM-H2); change timestamps bracket policy transitions.
- `listpeerchannels.json` — hourly advertised fees: STATIC pin compliance, PASSIVE fee immobility,
  before/after trajectories for PM-H1/PM-H3.
- `listforwards-window.json.gz` — earnings for PM-H1/PM-H3 metrics.
- `revenue-spend-ledger.json`, `revenue-profitability.json` — rebalance spend attribution (PM-H1)
  and bleeder identification baselines.
- `revenue-hive-hints-status.json` — membership/corridor roles to verify PM-I8 sweep behavior.
- `revenue-status.json` — fee_decision dominant skip reasons (`policy_passive`, `policy_static`
  counters from fee cycles, fee_controller.py:4464-4476) showing the gate firing.
- Note: there is no hourly artifact dumping the policies table itself; policy state must be inferred
  from effects plus any operator-run `revenue-policy list` captures.

## 7. Uncertainties

- No corpus artifact captures `revenue-policy list` periodically; reconstructing policy transition
  times (needed for PM-H1/PM-H3 windows) may be imprecise. Operator: can the hermes collector add an
  hourly `revenue-policy list` (and `revenue-policy changes`) capture for Phase 2?
- How many explicit policies exist in production, and are any STATIC/PASSIVE currently set? If zero,
  PM-H1/PM-H2/PM-H3 are untestable on this corpus and should be deferred or seeded deliberately.
- Are policy *suggestions* (`get_policy_suggestions`) surfaced anywhere operationally (RPC consumer,
  dashboard, hive)? I found the generator but no caller in the entry point besides potential RPC
  paths — if nothing consumes them, the bleeder-mitigation revenue path is currently theoretical.
- PM-1 known limitation (:173-175): write-through cache can briefly diverge from DB under concurrent
  writers. Single-process plugin threading makes this unlikely; confirm no external writer touches the
  policies table directly.
- The corridor sweep treats `current.tags and "manual" in current.tags` as operator ownership
  (:1076-1079) but operators are not documented anywhere as needing to add a 'manual' tag; the
  stronger guard is the stored-policy/auto_corridor check (:1098-1106). Is the 'manual' tag convention
  documented for operators anywhere?
