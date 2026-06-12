# Intent Contract: modules/hive_hints.py

Tier 1 — deep treatment. Authored 2026-06-12 from code at commit 9f8f219.
Cross-checked against docs/contracts/HIVE_HINTS_CONTRACT.md and docs/audits/CL_REVENUE_OPS_HINT_ADAPTER_AUDIT.md era fixes.

## 1. Purpose

HiveHintAdapter is the *sole* integration boundary between cl_revenue_ops and the cl_hive / cl-mycelium fleet coordinator (module docstring, :1-11). It polls a producer-controlled hint snapshot — preferring the CLN datastore key ["hive","hints"] and falling back to the `hive-export-hints` RPC (:282-348) — validates it (strict JSON rejecting NaN/Inf literals :152-166; required generated_at + hints map :391-404), enforces freshness/TTL with clock-skew and ceiling clamps (:426-483), enforces M2 privacy scoping consumer-side (:496-615), and exposes the result only as *bounded, advisory* lookups: capped multiplicative biases for fees/rebalancing/corridor utilization, validated membership, open/closure hints, route-segment intelligence, and metabolic/immune influence that explicitly grants no execution authority (:1939-1959). The design intent, faithfully implemented, is that a malicious or corrupt fleet payload can shade decisions by at most a few percent and can never crash, unbound, or authorize anything. One docstring/code divergence: the header claims missing/stale hints make "all lookups silently return 1.0" (:9-10), but the stale-fallback machinery (default policy "bounded_bias") deliberately keeps serving fee_bias and rebalance_bias from a stale snapshot for up to 48h while neutralizing everything else (:54-73, :173-180, :617-633) — the docstring oversimplifies real behavior.

## 2. Inputs / Outputs

**Inputs (transports):** CLN datastore key ["hive","hints"] via listdatastore (:217-254); `hive-export-hints` RPC exposed by cl_hive (:256-273). poll() is driven by modules/hive_runtime.py:30 (refresh_hive_runtime), called from the main plugin's timer loops (cl-revenue-ops.py:2234, :2516, :2626).

**Constructor config:** revenue-ops-hive-hints-enabled / -ttl / allow_all_hints_m2_scope (cl-revenue-ops.py:1772-1776, :2042-2051). stale_fallback_policy is a constructor parameter defaulting to "bounded_bias" (:59, :199-204) but is NOT wired to any CLN option — the main plugin never passes it (cl-revenue-ops.py:2044-2049), so production always runs bounded_bias; the diagnostics_only / full_legacy_fallback policies are dead configuration today.

**RPC exposed:** none of its own; revenue-hive-hints-status (cl-revenue-ops.py:2792 → get_status :2297 / refresh_status_for_debug :2086) is the diagnostic surface.

**Consumers (all read-only adapters into other modules' decisions):**
- fee_controller: get_fee_bias (:2681), corridor/centrality/elasticity/peak-hours (:2875-2917), get_fleet_fee_prior as an absolute Thompson prior (:7187), is_hive_member for role classification (:2791, :7407).
- rebalancer: is_hive_member (:585, :1551), get_rebalance_bias (:595).
- capacity_planner: get_open_candidates (:1906), member protection (:898, :1067), is_closure_recommended_fresh (:924-927), get_member_peer_ids/get_fleet_topology (:2040-2074), corridor/reputation/metabolic/immune open biases (:2339-2442), get_rebalance_bias in open-EV (:3095), get_metabolic_status (:2453).
- capex_budget: is_hive_member / get_corridor_role (:429-431); policy_manager (:1081-1082); profitability_analyzer (:2695-2697); hive_router: is_hive_member / get_fleet_balance (:159, :421).

**Outputs:** no datastore writes, no database writes, no RPC actions — pure in-memory cached snapshot plus bounded getters. (Compare: hive-export-hints.json and listdatastore segment-observations in the corpus are produced by cl_hive, not by this module.)

## 3. Invariants

- **HH-I1 (TTL):** Hints older than the effective TTL are never served by `*_fresh` getters; effective TTL is the operator ttl_override when set, else min(producer ttl_seconds, 86400), default 900s — the override is clamped to the same 86400 ceiling — and a generated_at more than 300s in the future invalidates the snapshot (:184-185, :426-453, :711-725, :798-815).
- **HH-I2 (stale-fallback bound):** Under the default bounded_bias policy, a stale snapshot may serve *only* fee_bias and rebalance_bias, only while its age ≤ max(6h, min(24×TTL, 48h)) — the 48h ceiling is absolute — and recency is re-checked at every read, not just at poll time (:60-73, :173-180, :459-479, :617-633).
- **HH-I3 (bounded biases):** get_fee_bias ∈ [0.9, 1.1], get_rebalance_bias ∈ [0.85, 1.15], get_corridor_utilization_bias ∈ [0.9, 1.1], regardless of payload content (:21-23, :882-904, :910-938, :944-972).
- **HH-I4 (neutral on absence):** Missing, invalid, stale-beyond-window, or out-of-scope hints yield neutral values: 1.0 biases, {} hints, [] sections, reputation 50, quality 0.5, confidence 0.0 (:695-725, :778-815, and every getter's fallback branch).
- **HH-I5 (non-finite rejection):** Non-finite numerics are rejected in depth, with two verified gaps. The strict JSON parse rejecting NaN/Inf literals (:152-166) covers only the *datastore* transport (:242, :247); the hive-export-hints RPC fallback returns pre-parsed data through pyln (:266) with no equivalent literal rejection. Snapshot validation checks only generated_at/ttl_seconds finiteness (:391-404). Per-field, the bias/score getters and `_clamp_float` reject non-finite values via `_finite_number`/isfinite (:377-389, :888-893, :916-921, :949-955, :1488-1499) — but the segment-score and segment-observation validators (:1215-1281) use bare float() with two-sided min/max clamps, where a NaN input clamps to the *upper* bound (e.g. NaN confidence or net_utility becomes 1.0) instead of being rejected. All outputs remain bounded (nothing can unbound or crash), but "NaN never influences a decision" does not hold for segment intelligence reached via the RPC transport.
- **HH-I6 (M2 scope):** When a snapshot carries M2 markers, M2-sensitive per-peer fields and sections are served only for peers within the normalized scope; scope "all_hints" is honored only with explicit operator opt-in (allow_all_hints_m2_scope), otherwise demoted to channel_and_fleet_peers (:47-53, :514-524, :550-615, :695-725).
- **HH-I7 (open candidates fresh-only):** get_open_candidates returns [] unless the snapshot is currently fresh, and every returned hint is re-validated through get_channel_open_hint (enumerated open_preference/reason/size vocab, clamped confidence) (:1405-1457).
- **HH-I8 (absolute values rejected, not clamped):** get_fleet_fee_prior returns None unless the value is finite and within [1, 10000] ppm — a poisoned absolute prior is dropped, never clamped into use (:186-193, :1965-1978). Likewise get_fleet_balance rejects available > capacity or capacity > 21M BTC (:1085-1112).
- **HH-I9 (no authority):** Metabolic/immune influence never grants permission: get_metabolic_action_constraints always reports additional_permission=False with execution/budget authority "cl_revenue_ops"; deltas are capped (fee ±5%, rebalance ±15%, open −15%/+10%, closure-watch ±15%) and require confidence ≥ 0.50 (:26-44, :1939-1959).
- **HH-I10 (closure hints):** closure_recommended is neutralized under stale fallback (it is in STALE_FALLBACK_NEUTRALIZED_FIELDS :61-73), and the capacity planner's escalation path prefers the fresh-only variant (capacity_planner.py:924-927); a stale snapshot therefore cannot drive a close.
- **HH-I11 (snapshot atomicity):** All reads take a consistent snapshot under the RLock; poll() replaces or clears the snapshot atomically, so a getter can never observe a half-updated snapshot (:206, :275-280, :350-355, :1439-1444).
- **HH-I12 (transport preference):** The live hive-export-hints RPC is invoked only when the datastore payload is missing, invalid, or stale (:293-319); a fresh datastore snapshot never triggers cross-plugin RPC.

## 4. Revenue role

Entirely indirect: this module earns nothing and spends nothing; it shades other modules' decisions by bounded amounts. The intended causal chain is (a) fleet-aggregated traffic/corridor knowledge nudges fee_controller toward fees the corridor will bear (±10% max) and seeds new-channel Thompson priors with the fleet fee median, shortening costly fee exploration; (b) rebalance bias (±15%) steers paid rebalancing toward peers the fleet observes as sinks/quality peers, raising the hit rate per sat of rebalance spend; (c) open hints and member topology give the capacity planner demand evidence it cannot see locally, improving open EV; (d) member protection prevents revenue-destroying closes of fleet corridors. Because every effect is capped at single-digit-to-15-percent multipliers, the expected revenue effect per decision is small by design; the value thesis is many small corrections compounding, plus avoidance of catastrophic actions (closing fleet channels, absurd fees on new channels).

## 5. Pre-registered hypotheses

- **HH-H1 (freshness correlates with revenue):** Node-hours where revenue-hive-hints-status.json reports a fresh, usable snapshot have higher routing fee income (hourly fee delta derived from revenue-history.json / listforwards-window.json.gz) than node-hours where hints are stale or missing, within the same node. Direction: fresh > not-fresh. Test: one-sided Mann-Whitney U per node on hourly fee income, combined across nodes via Fisher's method. Honest caveat: confounded (hive outages correlate with other problems); a null here is weak evidence, a reversal would be alarming.
- **HH-H2 (fleet fee prior reduces exploration waste):** Channels opened while the fleet prior was *available* reach a stable fee faster than channels opened during hint outages. Identification caveat: the chosen prior source ("fleet" vs "network") is emitted only in debug logs (fee_controller.py:7186-7193, :7426-7430) and is not persisted in any corpus artifact, so direct seeded/non-seeded labelling is impossible; use intention-to-treat assignment instead — a channel counts as "prior-available" when the revenue-hive-hints-status.json snapshot nearest its open time reports a fresh snapshot AND hive-export-hints.json carries a fleet_fee_median for that peer. Metric: days from open to fee stability (no >25% fee change for 7 days, from hourly revenue-status.json fee fields). Direction: prior-available < prior-unavailable. Test: one-sided log-rank on time-to-stability; fall back to Mann-Whitney on observed durations if censoring is rare. ITT dilution biases toward the null, so a positive result is conservative.
- **HH-H3 (corridor-owner bias is earned):** Peers marked corridor_role="owner" in hive-export-hints.json carry higher forward volume per capacity through their channels than non-corridor peers on the same node (the hint should reflect — and its +3% fee / +10% utilization bias exploit — real traffic). Direction: owner > none. Test: one-sided Mann-Whitney U on daily forwarded sats per capacity per channel, pooled across snapshot-hours with the role held constant ≥ 24h.

## 6. Observable surface

- **revenue-hive-hints-status.json** — the adapter's own status: freshness, source (datastore / export / stale-fallback), generation, m2 scope debug, stale-fallback field lists, segment score summaries (:2086-2297, cl-revenue-ops.py:2792).
- **hive-export-hints.json** — the producer payload itself (what the adapter would ingest); lets the corpus replay validation/scoping decisions offline.
- **listdatastore segment-observations** — the segment-observation coordination channel adjacent to the hint snapshot.
- **revenue-planner-candidates.json** — candidates with source="hive" show the open-hint pathway working end to end.
- **revenue-status.json / revenue-fee surfaces inside it** — hive bias and fleet-prior contributions to fee decisions (fee_controller debug fields).
- **revenue-history.json, listforwards-window.json.gz, listpeerchannels.json** — outcome data for HH-H1..H3.

## 7. Uncertainties

- Which producer runs in production — legacy cl-hive (schema_version 1) or cl-mycelium with explicit m2_scope — and what scope is actually requested? The adapter's behavior differs materially (legacy snapshots bypass M2 scoping entirely, :551-552, :597-599).
- Actual configured values of revenue-ops-hive-hints-ttl (ttl_override) and whether allow_all_hints_m2_scope was ever enabled.
- How often the stale-fallback path activates in practice, and whether the bounded-bias-while-stale behavior (HH-I2) has ever influenced a fee during a coordinator outage — the corpus status artifact should answer this.
- The membership grace-cache coupling: get_membership_status deliberately reports usable=False during bounded_bias fallback so fee_controller's `_cached_hive_membership_active` path keeps members at member fees (:826-872). I did not verify fee_controller's side of that contract; it is documented only in this adapter's docstring.
- get_drain_direction is documented as intentionally unused by the fee controller (askrene layer avoids double-counting, :1014-1025) — whether any consumer actually reads it today is unclear.
- Whether hive-export-hints RPC timeouts (the original motivation for datastore-first, :282-289) still occur on production nodes.
- Exploitability of the HH-I5 gap: whether lightningd's own JSON handling would actually deliver NaN/Infinity literals from a cl_hive RPC response through to pyln (Python json.loads accepts them; lightningd's C parser may not). If lightningd rejects them, the RPC-transport gap is theoretical and only the segment-validator clamp-to-upper-bound behavior remains.
