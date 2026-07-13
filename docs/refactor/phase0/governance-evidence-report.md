# Governance Evidence Report (PR 10, Phase G — 2026-07-13)

Direct evidence for the provisionally-met architecture claims
(gap-closure Gap 8). Each claim cites code, pins, or production
telemetry; weaknesses are stated, not smoothed.

## 1. All actions are typed intents

| Mutation category | Intent path | Gate | Evidence |
|---|---|---|---|
| Fee changes | SET_FEE via `_governed_authorize_fee_broadcast` (fail-closed) + shadow recording | `econ_governor_fees_enabled` (live) | fee_controller.py:7509; tests/test_governed_fees.py |
| htlc_max | SAME path — htlcmax broadcasts ride `set_channel_fee` (`rpc_params["htlcmax"]`, fee_controller.py:7771) through the same governed gate to `data_service.set_channel` | same | verified this audit; arbiter stage 3 handles SET_HTLC_MAX contradictions |
| Rebalances | REBALANCE envelopes at batch arbitration + governed reservation | `econ_governor_rebalance_enabled`, `econ_cycle_rebalance_enabled` (live) | rebalance_engine_v2.py:2065; 208+ live authorizations |
| Opens | OPEN_CHANNEL at planner governed reserve | `econ_governor_planner_enabled` (live) | capacity_planner.py:3575 |
| Closes | CLOSE_CHANNEL at selection-time batch arbitration + governed reserve | `econ_cycle_planner_enabled` (live) | capacity_planner.py:3419 |
| Swap-in/out | SWAP_IN/SWAP_OUT at Boltz batch + manager governed reservation | `econ_governor_boltz_enabled`, `econ_cycle_boltz_enabled` (live) | boltz_manager.py:1690; cl-revenue-ops.py:9562 |
| LN+ application/obligation | OPEN_CHANNEL with `CONTRACT_OBLIGATION` at governed reserve; application decisions are pre-intent gates (no mutation) | `econ_governor_lnplus_enabled` (live) | lnplus_swaps.py:732 |
| Reserve maintenance (treasury) | treasury swaps execute through the Boltz swap path above (same reservation + intent machinery) | same as Boltz | capex_budget.py reserve_boltz_swap_budget |
| External rating/release (LN+ ratings, gossip refresh) | NOT intent-typed — DELIBERATE: zero-economic-cost external courtesy actions (ratings) and gossip maintenance carry no capital/fee authority; they are audit-logged but below the intent boundary | n/a | documented here; adding intents for them is possible but adds no governance value |
| Reconciliation/recovery | ledger-only mutations (`reconciliation_completed`, quarantine) — they CORRECT records, never spend; startup reservation cleanup releases only | n/a (recovery class) | econ_reconcile.py; tests/test_econ_reconcile.py |
| Manual operator RPCs | bypass EV, never bypass audit/ledger conventions; explicitly operator-directed | operator | EV matrix exception class 5 |

Backstop: the spender-site guard (`tests/test_all_spenders_atomic.py`)
pins every (file, function, callee) that can reach a spend primitive —
an unclassified spender fails CI. The mutation-path inventory
(`docs/refactor/phase0/mutation-paths.md` + its enforcement pin) covers
non-spend mutations.

## 2. One global arbiter — conflict-matrix disposition

One arbitration authority exists in two coordinated layers sharing one
vocabulary and (live) one registry: the pure batch `arbitrate()` and
the shared `ActiveIntentRegistry` consulted by EVERY
`GovernorFacade.authorize()` across all five paths (single instance via
`EconShadow.arbitration_registry`).

| Spec conflict | Disposition | Where | Corpus |
|---|---|---|---|
| Close vs rebalance | IMPLEMENTED (batch + live) | econ_arbiter stage 4 + registry | 18 |
| Close vs protection | ASSIGNED: protection authority vetoes at SELECTION — protected channels never emit close intents; an arbiter rule would be dead code | protection_service (3C), goldened | 19 |
| Open vs LN+ | IMPLEMENTED (PR 10, batch + live): `CONFLICT_DUPLICATE_OPEN` — both paths emit OPEN_CHANNEL to the peer; higher priority (LN+ 80) wins in batch, first-registered wins live | econ_arbiter, flag `econ_conflict_rules_extended` | 20 |
| Rebalance vs structural swap | IMPLEMENTED (PR 10): `CONFLICT_REBALANCE_SWAP` — batch: SWAP_OUT outranks; live: either blocks the other | econ_arbiter, same flag | 21 |
| Contradictory fee changes | IMPLEMENTED (batch, pre-existing): stage 3 resolves SET_FEE/SET_HTLC_MAX per target, best-sorted wins | econ_arbiter stage 3 | 35 (ordering) |
| Fee reduction vs depletion protection | ASSIGNED: rails-stage ownership — the saturated min-fee carve-out with flow-aware exemption IS the depletion protection, applied inside the single fee path (no second authority to conflict with) | fee_controller floors, goldened | 8 |
| Duplicate intents | IMPLEMENTED (batch + live): `INTENT_SUPERSEDED` | stage 2 + registry | 31 |
| Stale intents | IMPLEMENTED: batch `INTENT_STALE`; governor `STALE` fail-closed | stage 1 + facade | 30, 34 |
| Contractual obligation vs lower priority | IMPLEMENTED structurally: obligations carry priority 80 (J3 sorts first) + `CONTRACT_OBLIGATION` code + authority exemption; duplicate-open rule now makes the obligation WIN against a competing planner open | J3 ladder + PR 10 | 29, 20 |

Score: 6 implemented in the arbiter (3 original + stage-3 contradictions
+ 2 new behind `econ_conflict_rules_extended`), 3 assigned with named
owners and golden coverage. No rule is unowned.

## 3. One governor — LN+ semantics

Every mutation authorization flows through `GovernorFacade.authorize()`
(paused → authority → registry → stale → reserve, fail-closed). LN+
obligation fulfillment is EXEMPT from `paused` and `authority_level`
(invariant 6: an accepted swap is a debt) but NOT from the governor:
the intent is authorized, reserved under the caller's reservation_id,
and ledgered with `CONTRACT_OBLIGATION` + `canonical_snapshot_id`
evidence. Pins: tests/test_governed_lnplus.py (pause exemption +
ledger trail), tests/test_authority_levels.py (structural pin that the
LN+ facade carries no authority_check), corpus scenario 29 (blocked-if-
gated vs ungated-by-design decision pair).

## 4. Executor and adapter coverage

- Mutating CLN RPC inventory: `docs/refactor/phase0/mutation-paths.md`
  (Phase 0) — every mutating verb behind `modules/data_service.py`
  typed methods (21 verbs) with the raw-RPC fallbacks enumerated;
  enforcement pin fails on new unlisted mutation sites.
- Boltz writes: `boltzcli` SUBPROCESS adapter (contradiction #3) —
  all invocations inside boltz_manager; guard-tested (3E).
- LN+ writes: HTTP client class inside lnplus_swaps; connect/fund via
  data_service adapter with raw fallback (2F).
- Executor→intent mapping: section 1 table above.
- Adapter ownership: 3E adapter-isolation guard tests pin that no
  policy module talks to external systems directly (URL/subprocess
  literals scoped to adapters).
- No-unclassified-mutation guard: spender-site ALLOWLIST
  (test_all_spenders_atomic) + mutation-path pin + snapshot-dependency
  pins (PR 2) — three independent tripwires.

## 5. Test-gate evidence by category

3,700 tests (unit) at this commit; category signals below are measured
per test FILE (non-exclusive; heuristic where noted):

| Category | Evidence | Count |
|---|---|---|
| Golden behavioral | tests/golden/ (byte-pinned, GOLDEN_UPDATE recording policy) | 88 tests / 7 suites, zero unexplained changes across the program |
| Invariants/property | monotonicity (econ_ev), fail-closed pins, oversubscription, checked-int semantics | 32 files carry invariant signals |
| Failure injection | side_effect-raising doubles across governor/shadow/adapters (fail-open vs fail-closed contracts) | 74 files |
| Ledger replay | test_econ_ledger + corpus 25/26/40 | 12 files |
| Restart & reconciliation | restart-survival, hydration, reconciler spec, corpus 24 | 37 files |
| RPC compatibility | rpc-surface pin (68), operator-surface pins (62 keys), explainability | 4 pin suites |
| Datastore compatibility | table-inventory pin, schema validity | 2 pin suites |
| Conformance | 40-scenario corpus, standalone validator, byte-identical regeneration pin | 2 suites + 42 payloads |
| CLN integration (supported minimum) | **HONEST GAP**: pyln.testing not installed in this environment — tests/test_pyln_integration.py SKIPS. tests/integration exists but is excluded from unit runs | not exercised here |
| CLN integration (current production) | operational: ~25 uneventful deploys on lnnode's production CLN through this program, each with post-restart verification | operational evidence |
| Bookkeeper present/absent | profitability fallback path + corpus 39 (contract record) | unit + corpus |
| Plugin stop/start | every deploy (~25) exercised stop/start with state rehydration checks | operational |
| Full daemon restart | NOT exercised during this program (node uptime preserved); restart-survival is covered at the plugin/database layer | honest gap — schedule with next node maintenance |
| Production rollout gates | per-flag staged rollout, DB backup before every deploy, reconciliation sweep + completeness detector post-deploy | procedure + evidence throughout |

## Verdict for the completion review

- DoD 2 (typed intents): **met** — one deliberate, documented
  sub-intent class (zero-cost external courtesy actions), everything
  economic is enveloped and gated.
- DoD 3 (one arbiter): **met-conditional on flag flip** — 6 rules
  implemented (2 awaiting the `econ_conflict_rules_extended` flip), 3
  assigned with owners; one shared registry + one pure batch authority.
- DoD 4 (one governor): **met** — LN+ exemption is from authority/pause
  dimensions only, evidenced.
- DoD 8 (adapters): **met** — inventories + three tripwires.
- DoD 13 (test gates): **partial** — two honest gaps: pyln-based
  minimum-CLN integration not exercised in this environment; full
  daemon restart not exercised during the program.
