# Production validation checkpoints — fee/capex/rebalance changes

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` to work task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. **Read every section before starting —
> context-free execution of individual tasks will miss important caveats.**

**Goal:** Determine whether the code changes merged via PRs #87 / #88 / #89 are
working as intended on two mainnet production nodes. Two hard checkpoints
(T+14 and T+28 days from deploy), plus a continuous rollback-watch.

**Scope:** Two production routing nodes running a 2-node cl-hive fleet with
the three merged PRs. Deploy time is referred to as **T0** (operator will
supply the actual timestamp). Analyst should NOT run experiments; this is a
pure observability and decision plan.

---

## Context — what was deployed

Three independent PRs on `lightning-goats/cl_revenue_ops`:

| PR | Branch | Summary |
|----|--------|---------|
| #87 | `fee-controller-improvements-2026-04-23` | 9 fee-setting algorithm improvements (intra-fleet ppm=1, competition_aware mode, variance-continuous blend, CLN-default filter, variance-gated undercut, rebalance-floor widening, percentile preserve, threshold config) |
| #88 | `capacity-planner-dynamic-close-cost-2026-04-23` | Dynamic close-cost estimation from `feerates` RPC instead of static 3000-sat constant |
| #89 | `rebalance-coordination-reserved-slots-2026-04-23` | Coordination pairs from cl-hive's hive-export-hints can bypass the planner `max_pairs` cap via 2 reserved slots (default) |

**Related memory files (analyst should read these):**

- `project_analyzer_opener_bug_2026_04_22.md` — past profit analyses mis-charged remote-opened channels to the wrong cohort
- `project_hive_topology_dependence_2026_04_23.md` — hive's value prop is sparse-network-regime-specific; dense networks see F3p ≈ 0.29× even with perfect algorithm
- `project_lab_gossip_density_blindspot_2026_04_22.md` — lab cannot exercise market-fee-mode; mainnet is the actual test
- `project_competition_aware_unreachable_2026_04_23.md` — in low-median-fee topologies the compaware preserve branch may not fire

---

## Known limitations that shape interpretation

1. **Small fleet (2 nodes).** Some new code paths activate rarely:
   - `rebalance_coordination_reserved_slots` only matters when cl-hive publishes >`max_pairs` coordination candidates at once — unlikely on a 2-node hive.
   - Fleet-wide gossip effects (fleet fee priors, corridor coordination) have minimal surface area.
   - Most gains come from #87's single-node mechanisms (variance blend, CLN filter, undercut gating).

2. **No pre-deploy baseline snapshot captured for profitability comparison.** We will work with T0+{14,28} vs T0-{14,28} retrospective comparisons using wallet history — imperfect but usable.

3. **Mainnet is stochastic.** A single 28-day window carries event-level noise (large-payment routings, peer closures, random mempool spikes). Treat effect sizes smaller than ~10% as inconclusive.

---

## File Structure

| File | Repo | Action | Responsibility |
|------|------|--------|----------------|
| `docs/plans/2026-04-23-production-validation-checkpoints.md` | cl_revenue_ops | This plan | Handoff reference |
| `docs/reports/2026-04-DD-production-t14-findings.md` | cl_revenue_ops | Analyst produces at T+14 | Week-2 report |
| `docs/reports/2026-04-DD-production-t28-findings.md` | cl_revenue_ops | Analyst produces at T+28 | Week-4 report (go/no-go/iterate decision) |

---

## Task 0: Establish T0 and data-collection scripts

- [ ] Operator supplies exact deploy timestamp(s). If the two nodes deployed at different times, track them separately. Record in the T+14 report header.
- [ ] Capture current state of each node as T0+ε snapshot (wallet, channels, fees, recent forwards/pays). This is the earliest possible baseline.

**Per-node collection script** (run once, save output):
```bash
# For each production node, from its operator shell:
mkdir -p ~/deploy-T0
lightning-cli listpeerchannels > ~/deploy-T0/listpeerchannels.json
lightning-cli listforwards > ~/deploy-T0/listforwards.json
lightning-cli listpays > ~/deploy-T0/listpays.json
lightning-cli revenue-config > ~/deploy-T0/revenue-config.json
lightning-cli revenue-profitability > ~/deploy-T0/revenue-profitability.json 2>/dev/null || true
lightning-cli hive-members > ~/deploy-T0/hive-members.json 2>/dev/null || true
lightning-cli feerates perkb > ~/deploy-T0/feerates.json
```

**Expected:** analyst has JSON snapshots for both nodes. If operator already captured these at deploy, great; if not, do it now — later snapshots use these as reference.

---

## Task 1 — Continuous rollback watch (ongoing from T0)

Even before checkpoint day, the analyst should flag these and notify operator immediately:

### Red flags (recommend rollback)
- [ ] **Plugin crash/restart** more than 3× in a 24h window (grep `plugin-cl-revenue-ops.py` restart events in `debug.log`)
- [ ] **Fees at 0 ppm on any non-hive channel** — never intended; if observed, likely bug in the intra-fleet classifier
- [ ] **Fees at `max_fee_ppm` ceiling on any normally-priced channel** — variance-continuous blend shouldn't produce sustained ceiling hits
- [ ] **Rebalance success rate < 50% over 7-day window** when it was historically > 80%
- [ ] **Routing revenue drop > 25% vs pre-deploy 14-day trailing average** (adjust for mempool/seasonality)

### Yellow flags (investigate, don't roll back yet)
- [ ] Plugin error logs containing `TypeError`, `AttributeError`, or `KeyError` → grep for tracebacks in `debug.log`; a handful of recovered-from errors is normal, >10/day is concerning
- [ ] `REBALANCE_FLOOR` log line firing > 100× per day (may indicate the widened ROUTER floor is noisy)
- [ ] `competition_aware preserve` or `competition_aware undercut` logs showing the same channel oscillating back and forth cycle-to-cycle (would indicate a sort-order bug, not a principled fee move)

**Collection command** for rollback watch:
```bash
# Ship to analyst periodically or provide tail access:
docker exec <node> tail -100000 /home/clightning/.lightning/debug.log | \
  grep -E "FEE:|REBALANCE_FLOOR|competition_aware|INITIAL_FEE|Hive member|Traceback|Error"
```

---

## Task 2 — T+14 checkpoint: direction signal

**Purpose:** confirm that new code paths are firing and no regressions are visible. Economics at 14 days will be noisy; focus on *behavior*, not *revenue*.

### Task 2.1: Confirm new code paths are alive per-node

For each production node:

- [ ] **Intra-fleet ppm=1 is applied.** On hive-member channels, `fee_proportional_millionths` should be 1.
  ```bash
  lightning-cli listpeerchannels | jq '.channels[] | select(.fee_proportional_millionths == 1) | .peer_id'
  ```
  Correlate with `hive-members` output: every peer in hive-members should appear with ppm=1. If a hive member has ppm != 1, investigate — likely `hive_hints` isn't reaching the fee controller.

- [ ] **INITIAL_FEE log shows `Hive member: 1-PPM fleet policy`.**
  ```bash
  grep "Hive member: 1-PPM fleet policy" debug.log | head -5
  ```
  Expect at least 1-2 entries per day per hive channel over the first days.

- [ ] **Variance-continuous blend fires.** Grep FEE lines for blend ratios other than 0.20. Look for any of {0.30, 0.45, 0.60}:
  ```bash
  grep "FEE:.*blend:" debug.log | grep -vE "blend:0\.(15|20)" | wc -l
  ```
  Zero matches over 14 days = posteriors never tightened (may be OK, may be a bug — investigate).

- [ ] **CLN-default filter active.** After 14 days of gossip, neighbor_median calls should be producing values, and undercut targets should NOT collapse to floor_ppm against dormant competitors:
  ```bash
  grep "competitive undercut" debug.log | awk '{print $NF}' | head -10
  ```
  The `(market=X, undercut=Y%)` values should show medians > 10 ppm for most channels.

- [ ] **Dynamic close-cost estimator is active.** When planner runs its loop:
  ```bash
  grep "estimated_closure_cost\|_estimate_close_cost" debug.log | tail -5
  ```
  Expect cost values that track mempool: cheap at low feerate days, expensive at high.

- [ ] **Rebalance coordination reserved slots — OPTIONAL on 2-node hive.** Likely never triggers; confirm config is read:
  ```bash
  grep "coordination_reserved_slots\|rebalance_coordination" debug.log | head
  ```
  No-op is expected on small fleets.

### Task 2.2: Compute behavior summary

For each node, compute over the 14-day window:

- [ ] Fee-change event count (how many FEE: log lines)
- [ ] Rebalance attempt count and success rate (from `listpays` — use the analyzer script pattern at `/tmp/polar-mcp-integration-2026-04-21/scripts/analyze_framings.py` which has a `cln_payment_costs` helper that returns success/fail/pending counts; adapt for production)
- [ ] Forward count and total fees earned
- [ ] Channel count delta (any unexpected closures?)

### Task 2.3: Go/no-go decision gate at T+14

- [ ] **All red-flag items from Task 1 are absent.** If any red flag fired → pause analysis, notify operator, consider rollback.
- [ ] **All "new paths alive" checks (Task 2.1) pass.** Intra-fleet ppm=1 is the only MUST item; others are nice-to-have on a small fleet.
- [ ] **Forwards are still happening** — not zero.

### Task 2.4: Produce T+14 report

Write `docs/reports/2026-04-DD-production-t14-findings.md` with:
1. Deploy timestamp confirmation
2. Red-flag status (green/red)
3. Per-node behavior summary table (fees, forwards, rebalances)
4. New-path activation summary (which fired, how often)
5. Any yellow flags observed + recommendation (continue / investigate / pause)
6. Specific data/commands the operator should pull next

---

## Task 3 — T+28 checkpoint: economic signal

**Purpose:** confirm (or reject) that the changes are improving economics. 28 days is still noisy, but should be enough to see a 10%+ effect if present.

### Task 3.1: Confirm all Task 1 and Task 2 items continue to hold

- [ ] Rollback-watch flags: still green
- [ ] New paths: still firing

### Task 3.2: Economic comparison

For each node:

- [ ] **Revenue trend:** compare trailing-28-day fees earned against the 28 days prior to T0 (if wallet history permits).
  ```bash
  # Fees earned since T0
  lightning-cli listforwards | jq "[.forwards[] | select(.status==\"settled\" and .received_time >= $T0_UNIX) | .fee_msat|tonumber] | add / 1000"
  ```
- [ ] **Per-peer effective ppm:** for each external (non-hive) channel, compute `fees_earned / msat_forwarded` over the 28-day window. Compare to pre-T0 values if possible.
- [ ] **Rebalance cost per success:** aggregate from listpays. Should be stable or improving vs pre-T0 distribution.
- [ ] **Amortized on-chain capex:** if any channels opened/closed during the window, use analyze_framings.py's amortization formula:
  `amortized_cost = actual_cost × (28 days / 180 days) ≈ actual_cost × 0.156`
- [ ] **Profit = fees − rebalance − amortized_capex.** Compare 28-day pre vs 28-day post.

### Task 3.3: Specific per-PR hypotheses to test

- [ ] **#87 intra-fleet ppm=1:** hive-member channels' aggregated sats_routed × 1 ppm / 1e6 = expected new revenue on internal leakage recapture. Check `listforwards` for forwards where the outgoing channel is a hive peer and compute.
- [ ] **#87 competition_aware:** grep count of `competition_aware preserve` vs `competition_aware undercut`. If preserve count > 0, that's evidence the DTS-preserve mechanism fired on a real competitor distribution (unlike in our lab where it couldn't).
- [ ] **#87 variance-gated undercut:** count of `undercut explore: X preserved (posterior_std=...)` log lines. Nonzero = DTS got room to learn on high-variance channels.
- [ ] **#88 dynamic close cost:** if any closes happened, compare estimated cost at decision time vs actual paid. Error < 20% is success.
- [ ] **#89 coordination reserved slots:** likely no-op on 2-node hive; confirm via "no coord-preempted logs would have been prevented by reserved slots" reasoning.

### Task 3.4: Go/iterate/rollback decision at T+28

Decision tree:

| Outcome | Action |
|---------|--------|
| Green across the board; revenue up or flat; all paths firing as expected | **SHIP** — leave code in place, monitor quarterly |
| Revenue up, some paths silent on small fleet | **SHIP with notes** — document which features are dormant on 2-node fleet; queue for larger-fleet validation |
| Revenue flat within 10%, all paths firing | **SHIP** — within noise, no evidence of harm, some evidence of correct activation |
| Revenue down 10-20% with unclear cause | **INVESTIGATE** — bisect; likely one specific feature is net-negative on this operator's topology |
| Revenue down >20% | **ROLLBACK** — revert PRs in reverse order (#89 → #88 → #87); most disruptive first |
| Red flag fired at any point | **ROLLBACK** immediately, post-mortem |

### Task 3.5: Produce T+28 report

Write `docs/reports/2026-04-DD-production-t28-findings.md` with:
1. Executive summary (ship / iterate / rollback)
2. Revenue / profit comparison, honest confidence bounds
3. Per-PR hypothesis status (confirmed / inconclusive / refuted)
4. Amortized capex treatment
5. Yellow-flag events observed during the 28 days
6. Recommendations for next iteration (if any)

---

## Task 4 — Decision & handoff

- [ ] Analyst delivers T+28 report to operator
- [ ] Operator acts on decision (ship / iterate / rollback)
- [ ] If ship: schedule quarterly review and move on
- [ ] If iterate: open issue with specific hypothesis and measurement plan
- [ ] If rollback: revert PRs, open issue with post-mortem

---

## Notes for the analyst

**Things you don't need to do:**
- Don't run lab experiments — we've established lab ≠ mainnet for this class of change.
- Don't re-implement changes or tune parameters on the fly.
- Don't chase sub-10% effects — they're noise in a 28-day mainnet window.

**Things to be careful about:**
- **Analyzer opener-bug:** any profit analysis that diffs channel counts over time must only count channels where `opener == "local"`. Remote-opened channels cost the receiving node 0 sats. See `project_analyzer_opener_bug_2026_04_22.md`.
- **Amortization:** don't charge 100% of a channel-open cost against a 28-day window. Use the 180-day amortization formula.
- **Multi-channel peers:** `fees_per_peer` aggregation can mislead if a peer has multiple channels; use per-scid aggregation for algorithm-level questions, per-peer for economic views.

**When to escalate to the operator immediately, not at checkpoint:**
- Any red-flag condition from Task 1
- Unusual on-chain tx activity (unexpected closes, transfers)
- Plugin stops responding to RPC

**What to do if a metric is inconclusive:**
- Log it honestly. "No signal" is a legitimate finding. Do not manufacture conclusions.
- If the lab had clearer signal than production (e.g., competition_aware fired 40% in our dense-lab run but 0% on production), that's a data point worth reporting — tells us the deployed topology doesn't exercise that feature and we should either accept it or target a denser fleet.
