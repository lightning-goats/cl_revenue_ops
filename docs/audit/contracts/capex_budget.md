# Intent Contract: modules/capex_budget.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`CapexBudgetEngine` (modules/capex_budget.py:117) is the unified capital-expenditure
budget calculator: from the profitability analyzer cache, 30-day spend history
(`rebalance_costs` + `spend_events`), config rates, and optional hive/capital-efficiency
inputs it computes per-channel rebalance budgets with tier classification
(proven/active/bootstrap/fleet/blocked), a fleet exploration budget (channel opens), a
tactical budget (Boltz treasury), and a fleet priority class
(defensive/preservation/operational/growth). Pure calculation layer — no CLN RPC calls
(:118–122); all arithmetic in msat with sats only at reporting boundaries. It also
records Boltz swap fees into the spend ledger so its own budgets self-deplete.

## Inputs / Outputs

- **Constructed at** cl-revenue-ops.py:2076–2084 with profitability analyzer, database,
  config, hive_hints, capital_efficiency, and a hive-member check.
- **Reads**: `profitability.analyze_all_channels()` (:152), bleeder status (:190),
  `database.get_total_capex_by_channel(30)` (:665–670),
  `database.get_spend_ledger_summary(window_hours=720)` (:672–677),
  `database.get_channel_rebalance_success_rate` (:488, diagnostics only),
  `database.get_confirmed_onchain_sats` (:626–631),
  `capital_efficiency.analyze()` (:159–163), hive hints member/corridor role (:427–434).
- **Writes**: Boltz fees via `database.record_spend_event(category="boltz", ...)`
  (`record_boltz_spend`, :352–395).
- **Consumers**: `revenue-capex-status` RPC (cl-revenue-ops.py:5876, pushes
  `["revenue","capex-summary"]` datastore key at :5928); `revenue-total-cost-budget`
  (cl-revenue-ops.py:5867); rebalancer (`set_capex_engine`,
  modules/rebalancer.py:1209) and v2 engine (budgets become
  `ChannelState.remaining_budget_sats`, modules/rebalance_state_v2.py:267);
  capacity_planner (`set_capex_engine`, modules/capacity_planner.py:151); boltz_manager.

## Invariants (budget ceilings)

- **CB-1** Global envelope is a hard ceiling: if `capex_global_envelope_sats > 0`, the sum
  of all channel budgets + exploration + tactical is proportionally scaled down to fit it
  (:261–280). With no configured envelope, the envelope equals the raw total (no-op).
- **CB-2** Emergency overrides tighten, never loosen: `daily_budget_sats*30` and
  `weekly_budget_sats*(30/7)` each `min()` against the envelope when set (:267–272).
- **CB-3** Budgets are funded and debited on the same 30d window (audit F1): proven
  budget = `max(0, contribution_30d * reinvestment_rate - capex_spent_30d)` (:504–507),
  and the proven gate itself requires >100 sats earned in the last 30 days, not lifetime
  (:516–523). Fleet revenue funding uses `fees_earned_30d_msat` when available (:180–184).
  Caveat: every "30d" input flows through `_windowed_msat` (:44–57, :416–418), which
  silently falls back to the **lifetime** value when the prof object lacks
  `window_30d_available=True` — under that fallback the funded/debited windows mismatch
  again (lifetime funding vs 30d debits), reopening the exact F1 failure mode for
  producers that don't prefetch the 30d P&L.
- **CB-4** Category budgets self-deplete: exploration and tactical budgets subtract both
  spent and actively-reserved sats for their category (`channel_open`, `boltz`) and floor
  at 0 (`_apply_category_spend_remaining`). DB errors fail CLOSED: if either spend-history
  read (`_get_total_capex_by_channel`, `_get_spend_ledger_summary`) raises, the wrapper
  returns None with a warning log and `compute_allocations` zeroes ALL channel, exploration
  and tactical budgets for the cycle (spend denied), flagging `CapexAllocations.db_degraded=True`
  — it never re-grants budgets as if nothing was spent.
- **CB-5** Blocked means zero: zombies, hard bleeders, and in-grace zero-contribution
  channels return tier `blocked` with `budget_msat=0` (:457–471, :538–541); hive-member
  channels bypass these gates into the `fleet` tier capped at min(50 bps of capacity,
  200 sats) with a 10-sat floor (:441–455).
- **CB-6** Multipliers are bounded: ROI multiplier clamped to [0.25, 1.5] and neutral
  when unreliable (:498–502); hive multiplier ∈ {1.0, 1.5, 2.0} (:426–434); efficiency
  multiplier ∈ [0, 1.5] with dead capital zeroed unless gateway value floors it at 0.25
  (`_get_efficiency_multiplier`, :558–586).
- **CB-7** One small bleeder cannot flip the fleet defensive: fleet-significant bleeding
  requires >1 hard bleeder or hard-bleeder capacity >10% of fleet capacity (:215–222);
  priority class ordering is defensive > preservation > operational > growth (:605–618).
- **CB-8** Boltz spend recording is idempotent and validated: event id `boltz:<swap_id>`,
  positive integer fee required, `INSERT OR REPLACE` downstream (:374–395).

## Revenue role

Direct lever on net revenue: it decides how many sats may be burned re-acquiring
liquidity per channel and fleet-wide. Too generous → bleeders; too tight → starved
earners. The ceiling claims (CB-1..CB-4) are the plugin's spend-control backbone.

## Observable surface

`revenue-capex-status.json` in the hermes corpus (RPC output + `["revenue","capex-summary"]`
datastore key); `revenue-spend-ledger.json` reflects its Boltz spend events and the
category depletion it reads back; cl-hive's metabolism ledger lists
`revenue_capex_status` and `revenue_total_cost_budget` as sources
(cl-hive/modules/organism/runtime.py:2303–2310).

## Uncertainties

- `attribute_boltz_cost` 50/50 channel/tactical split (:333–350) is a stipulated policy,
  not derived from evidence; no caller-side verification was done here.
- Wallet-backed bootstrap exploration (:633–644) intentionally skips spend-ledger
  debiting (double-count argument in the docstring); whether reservations alone are
  sufficient protection during concurrent opens is untested.
- `success_rate_30d` is carried as a diagnostic only (audit F6); confirm no downstream
  consumer still multiplies by it.
