# Refactor Phase 1 — Common Structures (Foundations Tranche) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the additive foundations of `docs/planning/refactor.md` Phase 1 — checked domain types, stable reason codes, cycle context, canonical snapshot types, typed intents with structured explanations and deterministic idempotency keys, published schemas + conformance fixtures, an append-only ledger module, and a governor facade that delegates to current checks — WITHOUT modifying any existing production file.

**Architecture:** Every deliverable is a NEW module under `modules/` (flat, matching repo convention per Phase 0 contradiction #2) plus tests, schemas, and conformance fixtures. Nothing imports into `cl-revenue-ops.py` or existing modules yet; wiring (RPC projections, shadow comparison, ledger persistence in the production DB, governor routing) is the next tranche, after operator review. Semantics follow `docs/refactor/phase0/wire-contract-spec.md` exactly.

**Tech Stack:** Python 3.12 dataclasses (frozen), sqlite3, hashlib/sha256, JSON Schema 2020-12 via `jsonschema` (tests only), pytest.

## Global Constraints

- Phase 1 exit gate (refactor.md line 787): golden tests show semantic parity; **no new component has sole live authority**. This tranche keeps parity trivially: existing files are UNTOUCHED (`git diff` on pre-existing files stays empty; verify each task).
- New modules must not be imported by `cl-revenue-ops.py` or any pre-existing module in this tranche (verify: `grep -l "econ_types\|econ_snapshot\|econ_intents\|econ_ledger\|reason_codes\|cycle_context\|governor_facade" cl-revenue-ops.py modules/*.py` returns only the new files themselves).
- Numeric semantics per `docs/refactor/phase0/wire-contract-spec.md`: integer msat, unsigned range [0, 2^63−1], signed range [−2^63, 2^63−1], overflow fails closed, micro fixed-point (denominator 1,000,000) for ratios/confidence, msat→sat ceil for fees/budgets/costs/revenue-reporting, floor for balances, toward-zero for signed deltas.
- Determinism per J3: canonical JSON = sorted keys, `separators=(",", ":")`, UTF-8; idempotency key = sha256 of canonical JSON of the envelope subset (intent_type, target, amount_msat, snapshot_id, budget_bucket); no wall-clock or randomness inside the new modules — time and seed always injected.
- Enum wire values: stable UPPER_SNAKE strings (J1).
- AGENTS.md rules apply (no action RPCs in tests; no Sling; no hive).
- Branch: `worktree-refactor`; one commit per task; full suite must stay green after every task (`python3 -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider`, baseline this tranche: 3212 passed).
- Do not re-record any Phase 0 golden fixture.

## Spec coverage (refactor.md Phase 1 bullets → tasks)

| Phase 1 bullet | Task |
|---|---|
| Checked monetary/fixed-point domain types | 1 |
| Stable reason codes | 2 |
| (J3 clock/seed injection prerequisite) | 3 |
| Canonical snapshots | 4 |
| Typed intents + structured explanations | 5 |
| Versioned schemas + portable fixture layout | 6 |
| Append-only ledger alongside legacy persistence | 7 |
| Governor facade delegating to current checks | 8 |
| RPC output from projections; live shadow comparison | EXCLUDED — next tranche (touches production files; needs operator review of these foundations first) |

---

### Task 1: Checked domain types — `modules/econ_types.py`

**Files:** Create `modules/econ_types.py`, `tests/test_econ_types.py`.

**Interfaces (produced, consumed by Tasks 3–8):**
- `EconArithmeticError(ArithmeticError)` — every checked failure raises this (fail closed).
- `U63_MAX = 2**63 - 1`, `I64_MIN = -(2**63)`.
- `Msat(value: int)` frozen dataclass; validates `0 <= value <= U63_MAX`, rejects bool/non-int. Methods: `add(o: Msat) -> Msat`, `sub(o: Msat) -> Msat` (raises `EconArithmeticError` if result negative), `diff(o: Msat) -> SignedMsat`, `to_sats_ceil() -> Sat`, `to_sats_floor() -> Sat`, classmethod `from_sats(sats: int) -> Msat` (checked ×1000). NO arithmetic operator overloading with plain ints — explicit methods only, so units cannot mix silently.
- `SignedMsat(value: int)` — range [I64_MIN, U63_MAX]; `to_sats_toward_zero() -> int`.
- `Sat(value: int)` — `0 <= value <= U63_MAX`; `to_msat() -> Msat` (checked).
- `Ppm(value: int)` — `0 <= value <= 10_000_000`; `fee_ceil(amount: Msat) -> Msat` = ceil(amount×ppm/1e6), `fee_floor(amount: Msat) -> Msat`.
- `Micro(value: int)` — fixed-point ratio 0..1_000_000 (confidence); classmethod `from_float_clamped(f: float) -> Micro` (non-authoritative ingestion helper, clamps then rounds half-even is NOT used — use `round()` banker's? No: use `int(f * 1_000_000 + 0.5)` floor-half-up documented).
- `UnixTime(value: int)` — 0..U63_MAX; `plus_seconds(s: int) -> UnixTime` checked.
- `ChannelId(value: str)` — must match `^[0-9]+x[0-9]+x[0-9]+$`.
- `PeerId(value: str)` — must match `^0[23][0-9a-f]{64}$`.
- `IntentId(value: str)` — non-empty, `^[a-z0-9-]{1,64}$`.

- [ ] Write `tests/test_econ_types.py` first: construction bounds (0, U63_MAX pass; −1, U63_MAX+1, bool, float, str raise `EconArithmeticError`); `Msat.add` overflow at U63_MAX raises; `sub` below zero raises; `diff` returns correct sign; conversion rules `Msat(1500).to_sats_ceil() == Sat(2)`, `.to_sats_floor() == Sat(1)`, `SignedMsat(-1500).to_sats_toward_zero() == -1`; `Ppm(250).fee_ceil(Msat(1_000_000)) == Msat(250)`; `Ppm` on odd amounts rounds up (ceil) vs down (floor); `Micro` bounds; `ChannelId("123x456x0")` ok, `ChannelId("bogus")` raises; `PeerId` regex; frozen-ness (`dataclasses.FrozenInstanceError` on assignment).
- [ ] Run: `python3 -m pytest tests/test_econ_types.py -q` → FAIL (module missing).
- [ ] Implement `modules/econ_types.py`. Every validator raises `EconArithmeticError` with the offending value in the message. Module docstring cites wire-contract-spec.md.
- [ ] Run to green; run full suite; commit `feat(refactor): checked economic domain types (Phase 1)`.

### Task 2: Stable reason-code catalog — `modules/reason_codes.py`

**Files:** Create `modules/reason_codes.py`, `tests/test_reason_codes.py`.

**Interfaces:**
- `ReasonCode` — a frozen dataclass `(code: str, layer: str, kind: str)`; module-level constants for the wire-contract-spec v0 table (BUDGET_EXHAUSTED … SCHEMA_INVALID) **plus** `PAUSED` (governor, rejection — needed by Task 8; add it to `docs/refactor/phase0/wire-contract-spec.md` catalog table in the same commit).
- `CATALOG: dict[str, ReasonCode]` — all codes by string; `LAYERS = {"policy","arbiter","governor","executor","reconciliation","any"}`, `KINDS = {"hold","rejection","deferral","clamp","failure","unknown"}`.
- `is_valid_code(code: str) -> bool`.

- [ ] Write tests: every catalog entry has valid layer/kind; codes are UPPER_SNAKE (`^[A-Z][A-Z0-9_]*$`); `CATALOG` keys equal each entry's `.code`; the 15 spec codes + PAUSED all present; `is_valid_code`.
- [ ] Run FAIL → implement → run green → full suite → update wire-contract-spec.md table with PAUSED → commit `feat(refactor): stable reason-code catalog v0 (Phase 1)`.

### Task 3: Cycle context — `modules/cycle_context.py`

**Files:** Create `modules/cycle_context.py`, `tests/test_cycle_context.py`.

**Interfaces:**
- `CycleContext(cycle_id: str, cycle_time: UnixTime, seed: int, snapshot_id: str)` frozen; validates cycle_id/snapshot_id non-empty, seed `0 <= seed <= U63_MAX`.
- `rng(self) -> random.Random` — returns a NEW `random.Random(self.seed)` each call (repeatable); the ONLY sanctioned randomness source for future policies.
- `derive_seed(self, component: str) -> int` — sha256(seed || component) → int in [0, U63_MAX], so components get independent-but-deterministic streams.

- [ ] Tests: same context → `rng().random()` sequences identical across calls and processes (fixed literal expected value for seed 42's first draw); `derive_seed` deterministic, differs by component; frozen; validation.
- [ ] FAIL → implement → green → full suite → commit `feat(refactor): deterministic cycle context (Phase 1)`.

### Task 4: Canonical snapshot types — `modules/econ_snapshot.py`

**Files:** Create `modules/econ_snapshot.py`, `tests/test_econ_snapshot.py`.

**Interfaces:**
- `Protection(reason: str, owner: str, expires_at: Optional[UnixTime])` frozen.
- `ChannelSnapshot(...)` frozen — fields exactly per `schemas/economic_snapshot.v0.schema.json` channel_snapshot ($defs), typed with Task 1 wrappers (`Msat`, `SignedMsat` for net_value, `ChannelId`, `PeerId`, `Micro` for confidence). `role`/`lifecycle` are plain strings validated against the schema enums (module-level `ROLES`/`LIFECYCLES` frozensets copied from the schema).
- `NodeState(...)` frozen — per schema `node` object (daily_budget as `BudgetState(cap_msat, reserved_msat, spent_msat)` frozen dataclass).
- `EconomicSnapshot(schema_name="economic_snapshot", schema_version=0, snapshot_id, observed_at: UnixTime, evidence_window_seconds: int, node: NodeState, channels: tuple[ChannelSnapshot, ...])` frozen; channels sorted by channel_id at construction (J3 stable ordering).
- `to_wire(snap) -> dict` — plain JSON-safe dict matching the schema (ints unwrapped).
- `canonical_json(obj: Any) -> str` — sorted keys, `(",", ":")` separators (lives here; Task 5 imports it).
- `build_channel_snapshot(*, channel: dict, prof: Any = None, flow_confidence: Optional[float] = None, role: str = "UNKNOWN", lifecycle: str = "PRODUCTIVE", protections: tuple = ()) -> ChannelSnapshot` — pure mapper from a `listpeerchannels`-shaped dict (`short_channel_id`, `peer_id`, `total_msat`, `to_us_msat`, `spendable_msat`, `receivable_msat` as ints) + optional profitability object (duck-typed: `revenue.fees_earned_msat`, `revenue.sourced_fee_contribution_msat`, `revenue.volume_routed_msat`, `revenue.forward_count`, `costs.rebalance_cost_sats`, `costs.open_cost_sats`, `net_profit_sats`, `sourced_forward_count_30d`) per `docs/refactor/phase0/snapshot-mapping.md`. Missing prof → zeros with `confidence` unchanged (missing evidence lowers confidence at the CALLER's discretion, never invents cost — invariant 7 note in docstring).

- [ ] Tests: construction + immutability; channels auto-sorted by channel_id regardless of input order; `to_wire` output validates against `schemas/economic_snapshot.v0.schema.json` (jsonschema importorskip); `canonical_json` insensitive to dict insertion order (build two dicts in different orders → identical string); builder maps a realistic listpeerchannels dict + `_make_prof`-style object (copy the helper shape from `tests/golden/test_golden_profitability.py`) to expected msat values (hand-computed: fees_earned 2000 sats → exit_revenue_msat 2_000_000); remote = capacity − local; builder with prof=None yields zero-valued economics.
- [ ] FAIL → implement → green → full suite → commit `feat(refactor): canonical EconomicSnapshot types and builder (Phase 1)`.

### Task 5: Typed intents — `modules/econ_intents.py`

**Files:** Create `modules/econ_intents.py`, `tests/test_econ_intents.py`.

**Interfaces:**
- `INTENT_TYPES = ("SET_FEE","SET_HTLC_MAX","REBALANCE","OPEN_CHANNEL","CLOSE_CHANNEL","SWAP_IN","SWAP_OUT","JOIN_LIQUIDITY_SWAP","MAINTAIN_ONCHAIN_RESERVE")` (spec Workstream B minimum set, stable strings).
- `Explanation(kind: str, components: tuple[tuple[str, Any], ...])` frozen; `render() -> str` produces `"kind: name=value, ..."` — human text derived FROM structure (spec rule).
- `IntentEnvelope(...)` frozen: `intent_id: IntentId`, `intent_type: str` (validated ∈ INTENT_TYPES), `idempotency_key: str`, `snapshot_id: str`, `schema_name="intent"`, `schema_version=0`, `created_at: UnixTime`, `expires_at: UnixTime` (must be > created_at), `target: str` (channel/peer/onchain identifier), `amount_msat: Optional[Msat]`, `expected_benefit_msat: SignedMsat`, `max_cost_msat: Msat`, `capital_committed_msat: Msat`, `confidence_micro: Micro`, `reason_codes: tuple[str, ...]` (each `is_valid_code` OR intent-local lowercase evidence tags — decide: require catalog codes only, empty tuple allowed), `explanation: Explanation`, `preconditions: tuple[str, ...]`, `priority: int` (0–100), `budget_bucket: str`, `origin_policy: str`, `reversible: bool`.
- `compute_idempotency_key(intent_type, target, amount_msat: Optional[int], snapshot_id, budget_bucket) -> str` — sha256 hexdigest of `canonical_json` (import from econ_snapshot) of exactly those five fields.
- `make_intent(**fields) -> IntentEnvelope` — computes idempotency key + `intent_id = IntentId("int-" + key[:16])`.
- `is_expired(env, now: UnixTime) -> bool`.
- `to_wire(env) -> dict` / `from_wire(d: dict) -> IntentEnvelope` — exact round-trip.

- [ ] Tests: valid construction for each of the 9 intent types (parametrized); invalid type raises; `expires_at <= created_at` raises; negative/oversized amounts impossible via domain types (assert `EconArithmeticError` propagates); idempotency key STABLE (fixed literal sha256 hex for a fixed input — compute once and pin) and unchanged when kwargs are supplied in different order; two intents differing only in amount get different keys; `is_expired` boundary (`now == expires_at` → expired); `to_wire→from_wire` round-trip equality; `Explanation.render()` exact string; unknown reason code raises.
- [ ] FAIL → implement → green → full suite → commit `feat(refactor): typed intent envelope with deterministic idempotency keys (Phase 1)`.

### Task 6: Intent schema + conformance fixtures

**Files:** Create `schemas/intent.v0.schema.json`; create `tests/conformance/scenarios/routine-cycle-smoke/expected-intents.json`; Modify `tests/test_econ_intents.py` (add schema-validation test).

- [ ] Write `schemas/intent.v0.schema.json`: JSON Schema 2020-12, `$id` an absolute https URI (jsonschema 4.10 requirement learned in Phase 0), `schema_name` const "intent", `schema_version` const 0, required = every envelope field, enums for intent_type, `additionalProperties: true` (draft rule), pattern for idempotency_key `^[0-9a-f]{64}$`, and ≥1 embedded example produced by `to_wire(make_intent(...))`.
- [ ] Add test to `tests/test_econ_intents.py`: `to_wire` output of a sample intent validates against the schema (importorskip jsonschema). `tests/test_schema_validity.py` picks the new schema up automatically (glob) — run it.
- [ ] Write `expected-intents.json`: a JSON array of one wire-form intent (schema_name/schema_version fields make the standalone validator accept it — NOTE: the validator validates top-level objects; wrap as `{"schema_name":"intent","schema_version":0, ...}` single object OR extend nothing: store a single intent object, not an array).
- [ ] Run `python3 tools/conformance/validate_fixtures.py` → exit 0 (both snapshot.json and expected-intents.json OK); run schema + intent tests; full suite; commit `feat(refactor): intent v0 schema and conformance fixture (Phase 1)`.

### Task 7: Append-only ledger — `modules/econ_ledger.py`

**Files:** Create `modules/econ_ledger.py`, `tests/test_econ_ledger.py`.

**Interfaces:**
- `EVENT_TYPES` — exactly the spec Workstream E vocabulary: intent_proposed, intent_rejected, intent_deferred, intent_authorized, budget_reserved, execution_started, execution_succeeded, execution_failed, execution_outcome_unknown, cost_recorded, reservation_released, reconciliation_completed.
- `EconLedger(path: str)` — owns its own sqlite table `econ_ledger_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, intent_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, cycle_id TEXT NOT NULL, at INTEGER NOT NULL, amounts_json TEXT NOT NULL, details_json TEXT NOT NULL)`; created via CREATE TABLE IF NOT EXISTS on init. NOT wired into `modules/database.py` in this tranche (production DB untouched; persistence pin unaffected — scans database.py only). Wiring note goes in the module docstring.
- `append(*, event_type, intent_id, idempotency_key, cycle_id, at: int, amounts: dict[str, int] = {}, details: dict = {}) -> int` — validates event_type ∈ EVENT_TYPES, amounts values are ints within signed-64 (fail closed via econ_types range check); INSERT only. No update/delete methods exist.
- `events(since_id: int = 0) -> list[dict]` — ordered by event_id (explicit ORDER BY, J3).
- `replay() -> LedgerState` where `LedgerState(reserved_msat: dict[str, int]` (by idempotency_key)`, spent_msat: dict[str, int]` (by bucket from amounts["bucket"]? — no: spent keyed by idempotency_key too, plus `total_spent_msat: int`), `terminal: dict[str, str]` (idempotency_key → final intent state)`)`. State machine: budget_reserved adds reservation (duplicate budget_reserved for same key is idempotent — second ignored); cost_recorded moves min(cost, reserved) to spent and releases remainder ONLY when followed by reservation_released? Keep simple + spec-true: `budget_reserved` → reserved[key] = amount; `cost_recorded` → spent[key] += amount, reserved[key] -= amount (floor 0, missing reservation → spent anyway and flagged in `anomalies: tuple[str,...]` — missing cost record must not create free budget, and unexpected cost must not crash replay); `reservation_released` → reserved[key] = 0; `execution_succeeded/failed/outcome_unknown` and `intent_rejected/deferred` set terminal state (first terminal wins; DUPLICATES HARMLESS — later duplicates ignored).
- [ ] Tests: append + ordered read-back; invalid event type raises; replay reconstructs reserved/spent across the reserve→execute→cost→release lifecycle (hand-computed numbers); duplicate execution_succeeded callbacks harmless (same terminal state, no double spend); cost without reservation lands in spent AND anomalies (never negative reserved); replay of empty ledger; append-only surface (no public method mutates existing rows — assert via `dir()` no update/delete and sqlite trigger? simpler: test that events(since_id) after more appends shows original rows byte-identical).
- [ ] FAIL → implement → green → full suite → commit `feat(refactor): append-only economic ledger with replay (Phase 1)`.

### Task 8: Governor facade — `modules/governor_facade.py`

**Files:** Create `modules/governor_facade.py`, `tests/test_governor_facade.py`.

**Interfaces:**
- `AuthorizationToken(token_id: str, intent_id: str, reservation_id: str, reserved_msat: int, budget_bucket: str, issued_at: int)` frozen.
- `GovernorDecision(authorized: bool, token: Optional[AuthorizationToken], reason_code: str)`.
- `GovernorFacade(*, reserve_spend: Callable, release_spend: Callable, is_paused: Callable[[], bool], ledger: Optional[EconLedger] = None)` — DELEGATES to injected callables (production will pass `Database.reserve_spend` etc.; tests pass a real `Database` on a temp file). Facade adds NO new authority (Phase 1 rule).
- `authorize(self, env: IntentEnvelope, now: int) -> GovernorDecision`:
  1. `is_paused()` → PAUSED rejection.
  2. `is_expired(env, UnixTime(now))` → INTENT_STALE rejection.
  3. `reserve_spend(reservation_id=env.idempotency_key, amount_sats=ceil-msat→sat of env.max_cost_msat, category=env.budget_bucket)` — worst-case cost reserved BEFORE execution (spec reservation rule); delegate returns falsy → BUDGET_EXHAUSTED.
  4. Success → token (token_id = `"auth-" + env.idempotency_key[:16]`), ledger `intent_authorized` + `budget_reserved` events appended when a ledger is provided.
- `release(self, token, now)` → delegates + ledger `reservation_released`.

- [ ] Tests: paused → PAUSED, no reservation attempted (delegate MagicMock not called); stale intent → INTENT_STALE; reserve failure → BUDGET_EXHAUSTED with no token; success path returns token and (with a real `EconLedger` temp file) appends intent_authorized + budget_reserved; **concurrency test**: real `modules.database.Database` on temp path with a small cap, N=8 threads authorizing intents whose max costs would oversubscribe → sum of granted reservations ≤ cap (delegates to the existing atomic `reserve_spend` — this PROVES the facade inherits the current guarantee); msat→sat ceil conversion asserted (max_cost_msat 1500 reserves 2 sats).
- [ ] FAIL → implement → green → full suite → commit `feat(refactor): governor facade delegating to existing budget checks (Phase 1)`.

### Task 9: Tranche wrap-up

**Files:** Modify `docs/refactor/phase0/README.md` (add Phase 1 tranche status section); Modify `docs/refactor/phase0/pr-sequence.md` (mark items 1–6 landed as commits).

- [ ] Verification checklist: full suite green (> 3212); `git diff 5e8f747 --stat -- cl-revenue-ops.py` empty AND `git diff 5e8f747 --name-only -- modules/ | grep -v "econ_\|reason_codes\|cycle_context\|governor_facade"` empty (no pre-existing module modified); no new module imported by production (grep check from Global Constraints); `python3 tools/conformance/validate_fixtures.py` exit 0; goldens untouched (`git diff 5e8f747 --name-only -- tests/golden/fixtures/` shows only Phase-0-added files, none modified after their creating commit).
- [ ] Update docs; commit `docs(refactor): Phase 1 foundations tranche status`.
- [ ] Report to operator: foundations ready; next tranche = wiring (shadow snapshot emission, RPC projections, ledger in production DB, governor routing) which starts modifying production files and needs explicit go-ahead.

## Self-review notes

- Spec coverage: all Phase 1 bullets mapped except RPC-projection/shadow-comparison — explicitly deferred with rationale (production-file changes; operator gate), documented in Task 9 report.
- Type consistency: `canonical_json` lives in econ_snapshot (Task 4), imported by econ_intents (Task 5); `EconLedger` (Task 7) consumed by facade (Task 8); reason code `PAUSED` added in Task 2 because Task 8 needs it.
- No placeholders: signatures, validation rules, event vocabulary, and test lists are exact; literal sha256 pin in Task 5 is computed at implementation time and then frozen in the test (a characterization pin, same policy as goldens).
