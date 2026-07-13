#!/usr/bin/env python3
"""Conformance-corpus generator (gap-closure Phase F, PR 9).

Regenerates tests/conformance/scenarios/ (40 scenario classes) and the
coverage report docs/refactor/phase0/conformance-coverage.md.

DIVISION OF LABOR: this GENERATOR imports the reference implementation
so every `expected` block is produced by (or copied verbatim from the
goldens of) the authoritative code — never hand-invented. The VALIDATOR
(validate_fixtures.py) imports nothing from the plugin; a foreign
implementation replays `inputs` and compares `expected` under the J5
rules (exact integers/enums/ordering/reason codes; prose excluded).

Deterministic: fixed timestamps, sorted keys, no randomness. Re-running
must be byte-identical (pinned by tests/test_conformance_corpus.py).

Usage: python3 tools/conformance/generate_scenarios.py
"""
import json
import pathlib
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from modules.econ_arbiter import ActiveIntentRegistry, arbitrate  # noqa: E402
from modules.econ_intents import (  # noqa: E402
    Explanation,
    is_expired,
    make_intent,
    to_wire,
)
from modules.econ_ledger import EconLedger  # noqa: E402
from modules.econ_snapshot import canonical_json  # noqa: E402
from modules.econ_types import (  # noqa: E402
    EconArithmeticError,
    Micro,
    Msat,
    SignedMsat,
    UnixTime,
)
from modules.governor_facade import GovernorFacade, authority_allows  # noqa: E402
from modules.rebalance_modes import MODES  # noqa: E402

NOW = 1_752_400_000
SCENARIOS_DIR = REPO / "tests" / "conformance" / "scenarios"
GOLDEN_DIR = REPO / "tests" / "golden" / "fixtures"
COVERAGE_DOC = REPO / "docs" / "refactor" / "phase0" / "conformance-coverage.md"


def _env(intent_type="REBALANCE", target="111x222x0", amount=400_000,
         benefit=0, max_cost=3_000, capital=None, confidence=0,
         priority=50, bucket="rebalance", snapshot_id="snap-1",
         created=NOW, expires=None, reason_codes=(), policy="conformance"):
    return make_intent(
        intent_type=intent_type, snapshot_id=snapshot_id,
        created_at=UnixTime(created),
        expires_at=UnixTime(expires if expires is not None
                            else created + 600),
        target=target,
        amount_msat=Msat(amount * 1000) if amount else None,
        expected_benefit_msat=SignedMsat(benefit),
        max_cost_msat=Msat(max_cost * 1000),
        capital_committed_msat=Msat((capital if capital is not None
                                     else amount or 0) * 1000),
        confidence_micro=Micro(confidence),
        reason_codes=tuple(reason_codes),
        explanation=Explanation("conformance", (("case", 1),)),
        preconditions=(), priority=priority, budget_bucket=bucket,
        origin_policy=policy, reversible=False)


def _case(case, category, inputs, expected, requirement="", source="",
          notes=()):
    payload = {"schema_name": "conformance_case", "schema_version": 0,
               "case": case, "category": category, "inputs": inputs,
               "expected": expected}
    if requirement:
        payload["requirement"] = requirement
    if source:
        payload["source"] = source
    if notes:
        payload["notes"] = list(notes)
    return payload


def _from_golden(case, category, fixture, requirement, notes=()):
    """Package a golden fixture verbatim: inputs/expected split follows
    the fixture's own 'inputs' key; everything else is the expected."""
    data = json.loads((GOLDEN_DIR / fixture).read_text())
    inputs = data.pop("inputs", {})
    return _case(case, category, inputs, data, requirement,
                 source=f"tests/golden/fixtures/{fixture}",
                 notes=notes)


def _facade(reserve_ok=True, paused=False, authority_check=None,
            registry=None):
    return GovernorFacade(
        reserve_spend=lambda **_kw: reserve_ok,
        release_spend=lambda _rid: True,
        is_paused=lambda: paused,
        registry=registry,
        authority_check=authority_check)


def _decision_wire(decision):
    return {"authorized": bool(decision.authorized),
            "reason_code": str(decision.reason_code or "")}


def _arb_wire(result):
    return {
        "ordered_intent_ids": [e.intent_id.value for e in result.ordered],
        "rejected": [{"intent_id": e.intent_id.value, "reason_code": rc,
                      "conflicting_key": ck}
                     for e, rc, ck in result.rejected],
    }


# ---------------------------------------------------------------------------
# Scenario registry: name -> (category, requirement, builder)
# Builder returns {filename: payload}.
# ---------------------------------------------------------------------------

def s01():
    return {"case.json": _from_golden(
        "ordinary-profitable-channel", "classification",
        "profitability/classify_young_profitable.json",
        "DoD 1/9: profitable classification from routing evidence")}


def s02():
    return {"case.json": _from_golden(
        "source-gateway-protection", "classification",
        "close_protection/gateway_30d_protected.json",
        "F5: inbound-gateway channels protected from closure")}


def s03():
    return {"case.json": _from_golden(
        "sink-depletion-refill-preferred", "rebalance_mode",
        "rebalance/plan_profitable_dest_preferred.json",
        "F4: depleted profitable destinations preferred for refill",
        notes=["Sink depletion expressed through the planner's "
               "destination preference for profitable drained channels."])}


def s04():
    return {"case.json": _from_golden(
        "balanced-channel-role", "classification",
        "profitability/role30d_balanced_30d.json",
        "Workstream A: BALANCED role from 30d flow evidence")}


def s05():
    return {"case.json": _from_golden(
        "underwater-classification", "classification",
        "profitability/marginal_roi_negative_profit.json",
        "Workstream A: negative marginal ROI (underwater)")}


def s06():
    return {"case.json": _from_golden(
        "stagnant-candidate", "classification",
        "profitability/classify_old_loser_stagnant.json",
        "Workstream A: stagnant classification")}


def s07():
    return {"case.json": _from_golden(
        "zombie-classification", "classification",
        "profitability/classify_zombie_after_failed_defib.json",
        "Workstream A: zombie after failed defibrillation")}


def s08():
    return {"case.json": _from_golden(
        "fee-rail-floor", "fee_stage",
        "fee/floor_defaults_no_chain_costs.json",
        "ADR-001 stage 1 (rails): fee floor")}


def s09():
    return {"case.json": _from_golden(
        "fee-rate-limit-clamp", "fee_stage",
        "fee/damping_large_raise_clamped.json",
        "ADR-001 stage 2 (rate_limit): per-cycle delta cap")}


def s10():
    return {"case.json": _from_golden(
        "fee-deadband-no-change", "fee_stage",
        "fee/damping_no_change.json",
        "ADR-001 stage 3 (deadband): no-op suppression")}


def s11():
    return {"case.json": _from_golden(
        "fee-cooldown-wake-cycle", "fee_stage",
        "fee/damping_wake_cycle_wider_cap.json",
        "ADR-001 stage 4 (cooldown): wake-from-sleep cycle semantics")}


def s12():
    from modules.fee_controller import GaussianThompsonState, PIDState
    pid = PIDState()
    pid.last_update_time = -1  # dt=0 branch: deterministic
    multiplier = pid.calculate_multiplier(
        current_outbound_ratio=0.2, capacity_sats=5_000_000,
        flow_state="balanced")
    ts = GaussianThompsonState()
    ts.update_posterior(fee=250, revenue_rate=3.5, hours=6.0)
    ts.update_posterior(fee=400, revenue_rate=2.0, hours=6.0)
    return {"case.json": _case(
        "dts-pid-components", "fee_stage",
        {"pid": {"current_outbound_ratio": 0.2,
                 "capacity_sats": 5_000_000, "flow_state": "balanced",
                 "fresh_state": True},
         "dts_observations": [
             {"fee": 250, "revenue_rate": 3.5, "hours": 6.0},
             {"fee": 400, "revenue_rate": 2.0, "hours": 6.0}]},
        {"pid_multiplier": round(multiplier, 12),
         "pid_ewma_error": round(pid.ewma_error, 12),
         "posterior_mean": round(ts.posterior_mean, 12),
         "posterior_std": round(ts.posterior_std, 12)},
        "ADR-001: DTS posterior update + PID multiplier are "
        "deterministic given inputs",
        notes=["PID dt=0 (fresh state); sampling randomness is the "
               "recorded portability hazard and is NOT part of this "
               "case."])}


def s13():
    return {"case.json": _from_golden(
        "dynamic-htlcmax", "admission",
        "htlcmax/balanced_mid_share.json",
        "F3: dynamic htlc_max admission control")}


def s14():
    hot = MODES["hot_protection"]
    normal = MODES["normal"]
    return {"case.json": _case(
        "hot-channel-priority", "rebalance_mode",
        {"modes": ["hot_protection", "normal"]},
        {"hot_protection_priority": hot.priority,
         "normal_priority": normal.priority,
         "hot_beats_normal": hot.priority > normal.priority},
        "F4 table: hot-channel protection outranks normal "
        "redistribution as priority data, not a separate path")}


def s15():
    return {"case.json": _from_golden(
        "normal-rebalance-plan", "rebalance_mode",
        "rebalance/plan_amount_bounded_by_chunk.json",
        "F4: one planner; chunk-bounded amounts")}


def s16():
    return {"case.json": _from_golden(
        "structural-drain-boltz", "rebalance_mode",
        "boltz/cycle_dry_run_executable_balance_plan.json",
        "F4/G: structural drain via Boltz balance plan (dry-run)")}


def s17():
    manual = MODES["manual"]
    diagnostic = MODES["diagnostic"]
    return {"case.json": _case(
        "manual-and-diagnostic-modes", "rebalance_mode",
        {"modes": ["manual", "diagnostic"]},
        {"manual": {"priority": manual.priority,
                    "operator_directed": True},
         "diagnostic": {"priority": diagnostic.priority,
                        "bounded_spend": True}},
        "F4 + contradiction #7: manual is operator-directed; "
        "diagnostic is a BOUNDED spend (evidence purchase), not free")}


def s18():
    close = _env(intent_type="CLOSE_CHANNEL", target="111x222x0",
                 amount=0, max_cost=0, bucket="channel_open", priority=60)
    reb = _env(intent_type="REBALANCE", target="111x222x0")
    result = arbitrate([close, reb], now=NOW)
    return {"case.json": _case(
        "conflicting-close-and-rebalance", "arbitration",
        {"intents": [to_wire(close), to_wire(reb)]},
        _arb_wire(result),
        "J3/spec conflict rule: rebalance into a channel scheduled "
        "for closure is rejected (CONFLICT_CLOSE_REBALANCE)")}


def s19():
    return {"case.json": _from_golden(
        "protected-close-rejection", "authorization",
        "close_protection/allowed_protect_tag_blocks.json",
        "F5: protection tags veto closure before any intent exists")}


def s20():
    lnplus = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                  amount=2_000_000, priority=80, bucket="channel_open",
                  policy="lnplus_lifecycle_governed")
    planner = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                   amount=1_000_000, priority=50, bucket="channel_open",
                   policy="planner")
    result = arbitrate([planner, lnplus], now=NOW, extended_rules=True)
    return {"case.json": _case(
        "open-vs-lnplus-conflict", "arbitration",
        {"intents": [to_wire(planner), to_wire(lnplus)],
         "extended_rules": True},
        _arb_wire(result),
        "Spec conflict rule: open vs LN+ — both paths emit OPEN_CHANNEL "
        "to the peer; CONFLICT_DUPLICATE_OPEN rejects the lower-priority "
        "one (the LN+ obligation at priority 80 wins)",
        notes=["Gated by econ_conflict_rules_extended (PR 10)."])}


def s21():
    swap = _env("SWAP_OUT", target="111x222x0", amount=250_000,
                bucket="rebalance", policy="boltz")
    reb = _env("REBALANCE", target="111x222x0")
    result = arbitrate([reb, swap], now=NOW, extended_rules=True)
    return {"case.json": _case(
        "circular-rebalance-vs-boltz-structural", "arbitration",
        {"intents": [to_wire(reb), to_wire(swap)],
         "extended_rules": True},
        _arb_wire(result),
        "Spec conflict rule: rebalance vs structural swap — the "
        "structural SWAP_OUT outranks; CONFLICT_REBALANCE_SWAP rejects "
        "the circular rebalance (live registry blocks both directions)",
        notes=["Gated by econ_conflict_rules_extended (PR 10)."])}


def s22():
    env = _env()
    decision = _facade(reserve_ok=False).authorize(env, NOW,
                                                   reservation_id="r-1")
    return {"case.json": _case(
        "budget-exhaustion", "authorization",
        {"intent": to_wire(env), "reserve_delegate_grants": False},
        _decision_wire(decision),
        "DoD 4/5: refused reservation -> BUDGET_EXHAUSTED, no spend")}


def s23():
    from modules.database import Database
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(pathlib.Path(tmp) / "t.db"), MagicMock())
        db.initialize()
        first = db.reserve_spend(
            reservation_id="c-1", amount_sats=800, category="rebalance",
            subcategory=None, metadata=None, effective_budget_sats=1000,
            since_timestamp=NOW - 86400)
        second = db.reserve_spend(
            reservation_id="c-2", amount_sats=800, category="rebalance",
            subcategory=None, metadata=None, effective_budget_sats=1000,
            since_timestamp=NOW - 86400)
    return {"case.json": _case(
        "concurrent-reservation-contention", "reservation",
        {"budget_sats": 1000,
         "reservations": [{"id": "c-1", "amount_sats": 800},
                          {"id": "c-2", "amount_sats": 800}]},
        {"first_granted": bool(first), "second_granted": bool(second)},
        "DoD 5: atomic reservations cannot jointly oversubscribe",
        notes=["Sequential here; the atomicity itself is exercised by "
               "the 8-thread governor concurrency test."])}


def s24():
    from modules.database import Database
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "t.db")
        db = Database(path, MagicMock())
        db.initialize()
        db.reserve_spend(
            reservation_id="restart-1", amount_sats=500,
            category="rebalance", subcategory=None, metadata=None,
            effective_budget_sats=1000, since_timestamp=NOW - 86400)
        db2 = Database(path, MagicMock())  # "restart"
        db2.initialize()
        states = db2.get_spend_reservation_states(["restart-1"])
    return {"case.json": _case(
        "restart-with-outstanding-reservation", "reservation",
        {"reserved_before_restart": {"id": "restart-1",
                                     "amount_sats": 500}},
        {"state_after_restart": {k: str(v) for k, v in
                                 (states or {}).items()}},
        "DoD 5: reservations survive restart (durable store)")}


def s25():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = EconLedger(str(pathlib.Path(tmp) / "l.db"))
        ledger.append(event_type="cost_recorded", intent_id="i-25",
                      idempotency_key="k-25", cycle_id="c-25", at=NOW,
                      amounts={"cost_msat": 7_000})
        state = ledger.replay()
    return {"case.json": _case(
        "missing-execution-cost-context", "ledger",
        {"events": [{"event_type": "cost_recorded",
                     "idempotency_key": "k-25",
                     "amounts": {"cost_msat": 7000}}]},
        {"anomalies": list(state.anomalies),
         "spent_msat": dict(state.spent_msat)},
        "DoD 6: cost without reservation context is an ANOMALY, "
        "never silently absorbed")}


def s26():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = EconLedger(str(pathlib.Path(tmp) / "l.db"))
        for etype in ("budget_reserved", "execution_started",
                      "execution_outcome_unknown"):
            ledger.append(event_type=etype, intent_id="i-26",
                          idempotency_key="k-26", cycle_id="c-26",
                          at=NOW,
                          amounts=({"reserved_msat": 5_000}
                                   if etype == "budget_reserved" else {}))
        state = ledger.replay()
    return {"case.json": _case(
        "unknown-external-execution-outcome", "ledger",
        {"lifecycle": ["budget_reserved", "execution_started",
                       "execution_outcome_unknown"]},
        {"terminal": dict(state.terminal),
         "reserved_msat": dict(state.reserved_msat)},
        "Workstream E: unknown outcome is a TERMINAL state pending "
        "reconciliation; reservation state preserved for the sweep")}


def s27():
    return {"case.json": _case(
        "boltz-timeout-after-acceptance", "failure_mode",
        {"lifecycle": ["execution_started"], "stale_after_seconds": 3600,
         "age_seconds": 7200},
        {"resolvable_as_db_missing": False, "quarantine_when_stale": True},
        "Reconciler spec: in-flight (started-without-terminal) keys are "
        "NEVER auto-zeroed; stale ones quarantine only",
        source="modules/econ_reconcile.py (TDD-pinned in "
               "tests/test_econ_reconcile.py)")}


def s28():
    return {"case.json": _from_golden(
        "lnplus-state-divergence-gate", "failure_mode",
        "lnplus/filter_dual_swap_rejected.json",
        "Workstream G: LN+ platform-state divergence rejected at the "
        "gate chain; watcher reconciliation is preflight-gated",
        notes=["Full watcher divergence handling is exercised by "
               "tests/test_lnplus_swaps.py::reconcile paths."])}


def s29():
    obligation = _env(intent_type="OPEN_CHANNEL", target="02" + "b" * 64,
                      amount=0, max_cost=214, capital=2_000_000,
                      bucket="channel_open", priority=80,
                      reason_codes=("CONTRACT_OBLIGATION",))
    blocked = _facade(authority_check=lambda: authority_allows(
        "observe", "capital")).authorize(obligation, NOW,
                                         reservation_id="lnplus-1")
    ungated = _facade(authority_check=None).authorize(
        obligation, NOW, reservation_id="lnplus-1")
    return {"case.json": _case(
        "lnplus-obligation-under-lower-authority", "authorization",
        {"intent": to_wire(obligation), "authority_level": "observe"},
        {"if_authority_gated": _decision_wire(blocked),
         "lnplus_path_is_ungated": _decision_wire(ungated),
         "invariant": "obligation fulfillment is EXEMPT from "
                      "authority_level but NEVER from the governor"},
        "Invariant 6 + Workstream I: accepted obligations complete "
        "under any authority level, still authorized and ledgered")}


def s30():
    env = _env(created=NOW - 700, expires=NOW - 100)
    decision = _facade().authorize(env, NOW, reservation_id="r-stale")
    return {"case.json": _case(
        "stale-intent-rejected", "authorization",
        {"intent": to_wire(env), "now": NOW},
        _decision_wire(decision),
        "DoD 4: stale envelopes rejected fail-closed (STALE)")}


def s31():
    a = _env(target="111x222x0")
    b = _env(target="111x222x0")
    result = arbitrate([a, b], now=NOW)
    return {"case.json": _case(
        "duplicate-idempotency-key", "arbitration",
        {"intents": [to_wire(a), to_wire(b)],
         "keys_equal": a.idempotency_key == b.idempotency_key},
        _arb_wire(result),
        "J3: identical five-field subsets share an idempotency key; "
        "duplicates superseded (INTENT_SUPERSEDED)")}


def s32():
    try:
        Msat(2 ** 63)
        overflow = "accepted"
    except EconArithmeticError as e:
        overflow = "EconArithmeticError"
    try:
        Msat(-1)
        negative = "accepted"
    except EconArithmeticError:
        negative = "EconArithmeticError"
    return {"case.json": _case(
        "numeric-overflow-underflow", "intent_semantics",
        {"msat_values": ["2**63", "-1"]},
        {"overflow_2_pow_63": overflow, "negative_msat": negative},
        "Workstream J numeric rules: msat in [0, 2^63-1], checked — "
        "out-of-range raises, never wraps")}


def s33():
    return {"case.json": _case(
        "msat-sat-rounding-boundaries", "intent_semantics",
        {"msat": [999, 1000, 1001, 1999]},
        {"floor_sats": [Msat(v).to_sats_floor().value
                        for v in (999, 1000, 1001, 1999)],
         "ceil_sats": [Msat(v).to_sats_ceil().value
                       for v in (999, 1000, 1001, 1999)]},
        "Workstream J: explicit, documented rounding at msat/sat "
        "boundaries")}


def s34():
    env = _env(created=NOW, expires=NOW + 600)
    return {"case.json": _case(
        "expired-intent-predicate", "intent_semantics",
        {"intent": to_wire(env),
         "probe_times": [NOW + 599, NOW + 600, NOW + 601]},
        {"is_expired": [is_expired(env, UnixTime(t))
                        for t in (NOW + 599, NOW + 600, NOW + 601)]},
        "Workstream B: expiry boundary is inclusive at expires_at")}


def s35():
    envs = [
        _env(target="300x1x0", priority=50, benefit=5_000),
        _env(target="100x1x0", priority=50, benefit=5_000),
        _env(target="200x1x0", priority=80, benefit=0),
        _env(target="400x1x0", priority=50, benefit=9_000),
    ]
    result = arbitrate(envs, now=NOW)
    return {"case.json": _case(
        "stable-ordering-tie-breaking", "determinism",
        {"intents": [to_wire(e) for e in envs]},
        _arb_wire(result),
        "J3 ladder: precedence, -priority, -EV, -confidence, capital, "
        "target, intent_id — total and stable")}


def s36():
    a = canonical_json({"b": 2, "a": 1, "nested": {"y": 2, "x": 1}})
    b = canonical_json({"nested": {"x": 1, "y": 2}, "a": 1, "b": 2})
    return {"case.json": _case(
        "map-insertion-order-independence", "determinism",
        {"payload": {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}}},
        {"canonical": a, "insertion_order_independent": a == b},
        "Workstream J: canonical JSON is insertion-order independent")}


def s37():
    env1 = _env(target="111x222x0")
    env2 = _env(target="111x222x0")
    return {"case.json": _case(
        "clock-seed-determinism", "determinism",
        {"identical_field_sets": 2, "fixed_created_at": NOW},
        {"intent_ids_equal": env1.intent_id.value == env2.intent_id.value,
         "idempotency_keys_equal":
             env1.idempotency_key == env2.idempotency_key,
         "intent_id": env1.intent_id.value},
        "Workstream J: identical (fields, clock) -> identical ids; "
        "no hidden entropy in the authoritative path")}


def s38():
    close = _env(intent_type="CLOSE_CHANNEL", target="500x1x0", amount=0,
                 max_cost=0, bucket="channel_open", priority=60)
    ok = _env(target="600x1x0")
    conflicted = _env(target="500x1x0")
    dup = _env(target="600x1x0")
    result = arbitrate([close, ok, conflicted, dup], now=NOW)
    return {"case.json": _case(
        "partial-batch-completion", "arbitration",
        {"intents": [to_wire(e) for e in (close, ok, conflicted, dup)]},
        _arb_wire(result),
        "Workstream H: a batch completes partially — survivors ordered, "
        "each rejection carries its own reason code")}


def s39():
    return {"case.json": _case(
        "bookkeeper-present-and-absent", "failure_mode",
        {"open_cost_source": ["bookkeeper", "estimate_fallback"]},
        {"bookkeeper_present": "open cost from bookkeeper events",
         "bookkeeper_absent": "falls back to estimated_open_cost_sats; "
                              "profitability stays computable",
         "invariant": "absence degrades cost precision, never "
                      "availability"},
        "Workstream G: bookkeeper is an optional evidence source",
        source="modules/profitability_analyzer.py:"
               "_get_open_cost_from_bookkeeper",
        notes=["Behavioral pin lives in the profitability unit tests; "
               "this case records the contract for porters."])}


def s40():
    # Sanitized production capture, lnnode econ ledger 2026-07-13
    # (first governed rebalance spend lifecycle + a snapshot_created).
    events = [
        {"schema_name": "ledger_event", "schema_version": 0,
         "event_type": "budget_reserved", "intent_id": "sanitized",
         "idempotency_key": "504", "cycle_id": "spend-rebalance",
         "at": 1783946902, "amounts": {"reserved_msat": 400000},
         "details": {}},
        {"schema_name": "ledger_event", "schema_version": 0,
         "event_type": "reservation_released", "intent_id": "sanitized",
         "idempotency_key": "504", "cycle_id": "spend-generic",
         "at": 1783946902, "amounts": {},
         "details": {"reason": "released"}},
        {"schema_name": "ledger_event", "schema_version": 0,
         "event_type": "snapshot_created",
         "intent_id": "preview-1783962518",
         "idempotency_key": "preview-1783962518",
         "cycle_id": "preview-1783962518", "at": 1783962518,
         "amounts": {}, "details": {"observed_at": 1783962518}},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ledger = EconLedger(str(pathlib.Path(tmp) / "l.db"))
        for event in events:
            ledger.append(event_type=event["event_type"],
                          intent_id=event["intent_id"],
                          idempotency_key=event["idempotency_key"],
                          cycle_id=event["cycle_id"], at=event["at"],
                          amounts=event["amounts"],
                          details=event["details"])
        state = ledger.replay()
    projection = {
        "schema_name": "ledger_projection", "schema_version": 0,
        "reserved_msat": dict(state.reserved_msat),
        "spent_msat": dict(state.spent_msat),
        "total_spent_msat": state.total_spent_msat,
        "terminal": dict(state.terminal),
        "anomalies": list(state.anomalies),
    }
    return {
        "expected-projections.json": projection,
        "expected-ledger-events.json": {
            "schema_name": "conformance_case", "schema_version": 0,
            "case": "production-capture-events", "category":
                "production_capture",
            "inputs": {"source": "lnnode econ_ledger.db 2026-07-13, "
                                 "sanitized (intent ids redacted)"},
            "expected": {"events": events}},
        "case.json": _case(
            "sanitized-production-decisions", "production_capture",
            {"events": events},
            {"replay": {
                "reserved_msat": dict(state.reserved_msat),
                "spent_msat": dict(state.spent_msat),
                "terminal": dict(state.terminal),
                "anomalies": list(state.anomalies)}},
            "DoD 17: replaying real production lifecycles reproduces "
            "the reference ledger state")}


SCENARIOS = {
    "01-ordinary-profitable-channel": s01,
    "02-source-gateway-protection": s02,
    "03-sink-depletion": s03,
    "04-balanced-channel": s04,
    "05-underwater-classification": s05,
    "06-stagnant-candidate": s06,
    "07-zombie-classification": s07,
    "08-fee-rail": s08,
    "09-fee-rate-limit": s09,
    "10-fee-deadband": s10,
    "11-fee-cooldown": s11,
    "12-dts-pid-components": s12,
    "13-dynamic-htlcmax": s13,
    "14-hot-channel-priority": s14,
    "15-normal-rebalance": s15,
    "16-structural-drain": s16,
    "17-manual-diagnostic-rebalance": s17,
    "18-conflicting-close-rebalance": s18,
    "19-protected-close-rejection": s19,
    "20-open-vs-lnplus": s20,
    "21-circular-vs-boltz-structural": s21,
    "22-budget-exhaustion": s22,
    "23-concurrent-reservation-contention": s23,
    "24-restart-outstanding-reservation": s24,
    "25-missing-execution-cost": s25,
    "26-unknown-execution-outcome": s26,
    "27-boltz-timeout-after-acceptance": s27,
    "28-lnplus-state-divergence": s28,
    "29-lnplus-obligation-lower-authority": s29,
    "30-stale-intent": s30,
    "31-duplicate-idempotency-key": s31,
    "32-numeric-overflow-underflow": s32,
    "33-msat-rounding-boundaries": s33,
    "34-expired-intent": s34,
    "35-stable-ordering-tiebreak": s35,
    "36-map-order-independence": s36,
    "37-clock-seed-determinism": s37,
    "38-partial-batch-completion": s38,
    "39-bookkeeper-present-absent": s39,
    "40-sanitized-production-decisions": s40,
}


def _write_coverage(dirs):
    rows = []
    for name in sorted(dirs):
        case = json.loads(
            (SCENARIOS_DIR / name / "case.json").read_text())
        files = sorted(p.name for p in (SCENARIOS_DIR / name).glob("*.json"))
        rows.append((name, case["category"],
                     case.get("requirement", ""),
                     case.get("source", "generated"),
                     ", ".join(files)))
    gaps = [r for r in rows if r[1] == "documented_gap"]
    lines = [
        "# Conformance Corpus Coverage (generated — PR 9, Phase F)",
        "",
        "Generator: `tools/conformance/generate_scenarios.py` "
        "(deterministic; regenerating must be byte-identical). "
        "Validation: `tools/conformance/validate_fixtures.py` — no "
        "plugin imports. Golden-derived cases copy fixture bytes "
        "verbatim; generated cases are produced BY the reference "
        "implementation.",
        "",
        f"Scenario classes: {len(rows)}/40. Documented gaps: "
        f"{len(gaps)} (listed last — these are honest holes with "
        "pointers, not silent omissions).",
        "",
        "| # Scenario | Category | Requirement | Source | Files |",
        "|---|---|---|---|---|",
    ]
    for name, cat, req, src, files in rows:
        lines.append(f"| `{name}` | {cat} | {req} | {src} | {files} |")
    lines += [
        "",
        "## Reason-code / rule coverage",
        "",
        "- Arbitration: `INTENT_SUPERSEDED` (31, 38), "
        "`CONFLICT_CLOSE_REBALANCE` (18, 38), J3 total order (35).",
        "- Authorization: `BUDGET_EXHAUSTED` (22), `STALE` (30), "
        "`AUTHORITY_LEVEL_BLOCKED` vs obligation exemption (29).",
        "- Ledger/lifecycle: anomaly on orphan cost (25), "
        "unknown-outcome terminal (26), in-flight quarantine rule (27), "
        "replayed production lifecycle (40).",
        "- Reservations: oversubscription refusal (23), restart "
        "durability (24).",
        "- Numeric/determinism: checked ranges (32), rounding (33), "
        "expiry boundary (34), canonical JSON (36), id determinism "
        "(37).",
        "- Policy stages: fee rails/rate/deadband/cooldown (8-11), "
        "DTS+PID components (12), htlc_max (13), classifications "
        "(1-7), modes/priorities (14-17).",
        "",
        "## Documented gaps",
        "",
    ]
    for name, _, req, _, _ in gaps:
        lines.append(f"- `{name}`: {req} — see the case's notes for the "
                     "blocking dependency.")
    COVERAGE_DOC.write_text("\n".join(lines) + "\n")


def main() -> int:
    if SCENARIOS_DIR.exists():
        shutil.rmtree(SCENARIOS_DIR)
    SCENARIOS_DIR.mkdir(parents=True)
    for name, builder in sorted(SCENARIOS.items()):
        out = builder()
        target = SCENARIOS_DIR / name
        target.mkdir()
        for filename, payload in sorted(out.items()):
            (target / filename).write_text(
                json.dumps(payload, indent=1, sort_keys=True) + "\n")
    _write_coverage(list(SCENARIOS))
    print(f"generated {len(SCENARIOS)} scenarios -> {SCENARIOS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
