# Operator-Surface Reduction Proposal — 2026-08-01

Analyzed at commit 88f168c against 4 months of production evidence (lnnode:
complete `config_overrides` table at 94 versions, plus the live lightningd
config file). Goal: fewer knobs, fewer commands, one authoritative runtime
surface, and set-time rejection of known operator mistakes.

Status: PROPOSAL — phases require the contract-compatibility announcement
process (docs/refactor/phase0/contract-compatibility-policy.md).

Note: the "Boltz auto-cycle effectively OFF" discrepancy flagged in §2e was
resolved the same day (DB override flipped back to true, config v95, after
the July deadlock reason — lesson 620 — was dissolved by the budget retune).
The single-surface recommendation that discrepancy motivates stands.

## Headline

| Surface | Today | Proposed |
|---|---|---|
| RPC methods | 69 | ~24 |
| CLN options | 121 | ~15–20 (deployment plumbing only) |
| revenue-config public keys | 62 | ~30 |
| Values settable on TWO surfaces | 46 | 0 |

Production evidence: only ~33 knobs were EVER touched in 4 months; 12 of them
are econ_* rollout flags at stable end-state values. The double-surface
(CLN option + revenue-config key for the same value) has already caused two
documented live incidents (2026-07-08 refresh-loop stomps) and one live
config-file lie (boltz-auto-cycle-enabled).

## 0. Verified surface counts

- 69 `@plugin.method` (cl-revenue-ops.py:3820–10636)
- 121 `plugin.add_option` (21 dynamic) (cl-revenue-ops.py:1150–1998)
- 62 `PUBLIC_RUNTIME_KEYS` (modules/config.py:27–148)
- 46 double-surfaced values (init mapping cl-revenue-ops.py:2524–2700)
- Already single-surfaced (the newer pattern): `paused`,
  `capex_probability_budget_bonus`, `authority_level`, `risk_profile`, and the
  12 `econ_*` flags.
- Precedence machinery is itself an incident source:
  `_refresh_dynamic_config()` (cl-revenue-ops.py:6741–6860) needs three
  hand-written "DB override wins" guards, each added after the 2026-07-08
  production incident where the refresh loop stomped an operator override.

## 1. RPC classification (69 → ~24)

### KEEP-CORE (9)
`revenue-status`, `revenue-fee-debug`, `revenue-rebalance-debug`,
`revenue-dashboard`, `revenue-config`, `revenue-profitability`,
`revenue-health`, `revenue-policy`, `revenue-profile-preview`.

### KEEP-RARE (12) — recovery/ops with no substitute
`revenue-set-fee`, `revenue-rebalance`, `revenue-fee-authority-status` (the
only method the runbooks depend on), `revenue-lnplus-breaker-clear`,
`revenue-lnplus-abandon`, `revenue-lnplus-backfill`, `revenue-lnplus-status`,
`revenue-cleanup-closed`, `revenue-clear-reservations`,
`revenue-spend-release-stale`, `revenue-boltz-refund`, `revenue-boltz-claim`.

### MERGE
1. Ban trio → `revenue-policy action=ban|unban|list-banned` (−3).
2. Manual cycle triggers → `revenue-cycle <fees|rebalance|flow|planner|boltz|all>`
   replacing `revenue-fee-cycle`, `revenue-rebalance-cycle`, `revenue-analyze`,
   `revenue-wake-all`, `revenue-planner-execute`,
   `revenue-boltz-auto-cycle-run-now` (−6→1).
3. Planner reads → `revenue-planner <status|candidates|sources|history|report>`
   (−4→1, absorbs `revenue-capacity-report`).
4. Budget views → one `revenue-budget` (absorbs `revenue-total-cost-budget`,
   `revenue-capex-status`, `revenue-boltz-budget`, `revenue-spend-ledger`)
   (−4→1).
5. `revenue-report` → folded into dashboard/policy/history (−1).
6. Boltz family (22) → `revenue-boltz <verb>` dispatcher (−21→1); all bodies
   are 4–10-line manager wrappers, CLI namespacing already implies it.
7. Spend family → `revenue-spend <ledger|reserve|release|release-stale|settle>`
   or demote reserve/release/settle as internal (−4).

### DEMOTE (internal/diagnostic)
`revenue-econ-*` (3, retire with econ flag removal),
`revenue-hot-channel-protection-peers`, `revenue-boltz-auto-cycle-status`,
`revenue-boltz-external-pay-ignores`, balance/treasury recommendations+status,
`revenue-spend-reserve/-release/-settle`.

### REMOVE
`revenue-ignore`/`revenue-unignore`/`revenue-list-ignored` — deprecated since
v1.4, already locked behind `_policy_write_override` (cl-revenue-ops.py:
5131–5146); no operator path exists. Needs an announced window.

## 2. Knob classification

### (a) Never touched in production → hardcode default, delete knob (~63 options + config twins)
- Deprecated no-ops (already announced 2026-08-12): 6 fee-market-boundary,
  `rebalance-min-profit`.
- Self-described inert: growth `experiment-fraction`/`max-extra-sats`,
  `base-fee-policy` (both values equivalent), `rebalance-router` (only 'v3').
- Algorithm internals never touched: rebalance tuning block (10), cooldown
  overrides (2), engine misc (6: congestion threshold, reputation ×2,
  kelly, vegas ×2), intervals (flow/rebalance/fee → 3600/900/1800),
  `flow-window-days`=7, `neighbor-median-min-competitors`=2,
  `min-fee-ppm-saturated`=0, `rpc-timeout-seconds`=15,
  `reservation-timeout-hours`=4, `diagnostic-rebalance-max-fee-sats`=400,
  `allow-zero-cost-auto-rebalance-when-budget-zero`=false,
  hot-channel protection (7), boltz auto-cycle timing (3: 15/1/120),
  boltz misc (timeout 60, routing-fee-limit-ppm 0, max-withdraw 10M —
  hardcoding the last strengthens a safety rail), treasury tuning (4:
  BTC/1/80/true), htlcmax pcts (0.50/0.25/0.45), LN+ tuning (10 of 12 —
  LN+ runs live entirely on defaults), `market-fee-mode`=undercut,
  `min-wallet-reserve`=1,000,000.

### (b) Deployment-specific → keep as CLN options (~12)
`db-path`, `boltz-enabled`, `boltz-cli-path`, `boltz-datadir`,
`boltz-use-sudo`, `boltz-sudo-user`, `boltz-btc-wallet`, `boltz-lbtc-wallet`,
`askrene-layers`, `dry-run`, `fee-authority-enabled` (runbook-governed),
`fee-replay-capture-enabled`.

### (c/d) Actively tuned economics → keep, single-surfaced on revenue-config
Fee rails, budgets, fee_profile, growth budget, planner gates/caps/ROI,
boltz auto-cycle enable + structural envelope, receivable targets, drain
bias/discount, dynamic htlcmax, LN+ enable/execute, paused, authority_level,
risk_profile. Promote into PUBLIC_RUNTIME_KEYS as their options are removed:
`planner_min/max_channel_sats`, `planner_max_fee_rate_sat_vb`,
`planner_interval`, treasury enable/target/min-deficit,
`rebalance_hold_margin`.

**Winning surface: revenue-config (DB).** DB already beats the file by
design; the file line is at best dead and at worst a lie (the live
boltz-auto-cycle case); the refresh-loop guard complexity exists only
because of the double surface.

### (e) econ_* rollout flags — retirement
Production end-state: 10 of 12 True and stable; the two boltz flags False
only because Boltz automation was off (now re-enabled).
- Step 1 (additive, now): flip all 12 defaults to True — fresh nodes should
  run the governed paths production actually tests, not the abandoned legacy.
- Step 2 (announced window, after one stable release): remove the 12 keys,
  delete flags + dead legacy branches + `revenue-econ-*` diagnostics.

### Risk-profile coverage
PROFILE_BUNDLES cover 11 economic keys (2 inert ones must leave the
bundles). Post-reduction, README quick-start should present `risk_profile` +
rails (`paused`, fee rails, `authority_level`) as the entire day-1 surface —
which is what production behavior confirms actually gets touched.

## 3. Default changes (fresh-deployment-safe, informed by production)

| Key | Current → Proposed | Evidence |
|---|---|---|
| econ_* ×12 | False → True | 10 stable-True for months; 2 no-op-True |
| min_fee_ppm | 10 → 50 | operator file value; 10 invites unprofitable forwards |
| planner_min_channel_sats | 500k → 1M | prod picked 2M; sub-1M rarely clears chain-cost ROI |
| enable_dynamic_htlcmax | False → True | prod-proven, bounded valve |
| capex_probability_budget_bonus | drop from public keys | rationale obsolete (v3-only router) |
| max_fee_ppm, weekly/daily budgets, ROI, drain bias/discount, structural envelope, automation enables | keep current defaults | production values are node-scale tuning; automation and spend-uplift stay opt-in on fresh nodes |

## 4. Migration phases

- **A — 2026-08-12 (already announced):** delete fee_market_boundary_* and
  rebalance_min_profit (options, fields, keys, gates, snapshot fields);
  schema/table items per policy; run tools/deprecation_scan.py; regenerate
  config/cl-revenue-ops.conf.full.
- **B — additive, immediate:** §3 default flips; startup warning for unknown
  DB override rows (today silently skipped — the lnplus_fleet_pubkeys case).
- **C — new announced window (announce ~2026-08-05, remove ~2026-09-05):**
  class-(a) option removals (warn-only no-ops during window); the 46 mirrored
  options (revenue-config becomes sole surface); RPC merges/removals with
  alias-forwarding during the window; PUBLIC_RUNTIME_KEYS promotions.
- **D — after one stable release at econ-default-True:** econ flag + legacy
  branch + revenue-econ-* removal (announce at C).

### Live-node cleanup (lnnode)
- Delete file lines restating defaults: fee-interval, boltz-enforce-budget,
  planner-dry-run, planner-max-opens-per-cycle, planner-execute-closes,
  planner-min-annual-roi-pct.
- Delete file lines shadowed by DB overrides: min/max-fee-ppm,
  daily/weekly-budget-sats, boltz-auto-cycle-enabled.
- Delete DB rows: `lnplus_fleet_pubkeys` (dangling since v2.17.0),
  `capex_probability_budget_bonus` (restates default), and
  restated-default planner rows.

## 5. Error-proofing the remaining surface

1. `daily_budget_sats > weekly_budget_sats`: promote from boot-warn to
   set-time rejection (profile preview already treats it as contradiction).
2. `planner_min_channel_sats` vs `planner_max_channel_sats`: no cross-check
   anywhere; crossed pair silently disables all opens. Add set-time + boot.
3. htlcmax pct ordering (sink ≤ balanced ≤ source): moot if hardcoded.
4. Set-time shadowed-gate warnings on `revenue-config set` (e.g.
   `boltz_auto_cycle_enabled` gates `boltz_structural_budget_sats_per_day` —
   the exact shadow found in production).
5. Unknown-override startup warning (Phase B).
6. Profile-first onboarding in README quick-start.
