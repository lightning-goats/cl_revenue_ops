"""Deterministic economic cycle, shadow stage (refactor Workstream H).

The spec's cycle: collect → close window → snapshot → generate intents →
arbitrate → authorize → execute → record → publish. This module delivers
the DETERMINISTIC CORE of that cycle running in shadow:

- one collection pass (adapter reads, done by the caller/collector),
- a canonical versioned EconomicSnapshot under an injected CycleContext
  (clock + seed — no wall-clock or unseeded randomness inside the core),
- pure intent generation (v0: REBALANCE intents from the pure planner
  candidates; SET_FEE generation joins when the fee policy migrates in
  Phase 4 — the DTS sampler is not yet seed-injected),
- BATCH arbitration via econ_arbiter.arbitrate (the pure J3-ordered
  ranking, activated over a real cycle's intent set),
- a ledgered, publishable result. NO execution: the shadow cycle holds
  no authority (execution cutover is per-loop, later tranches).

Determinism contract (spec acceptance): identical inputs + identical
CycleContext produce byte-identical wire output, regardless of input
ordering. Pinned by tests/test_econ_cycle.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .cycle_context import CycleContext
from .econ_arbiter import ArbitrationResult, arbitrate
from .econ_intents import Explanation, IntentEnvelope, make_intent, to_wire
from .econ_snapshot import build_channel_snapshot, canonical_json
from .econ_types import Micro, Msat, SignedMsat, UnixTime


@dataclass(frozen=True)
class CycleResult:
    context: CycleContext
    channel_count: int
    intents: Tuple[IntentEnvelope, ...]
    arbitration: ArbitrationResult

    def to_wire(self) -> dict:
        return {
            "schema_name": "econ_cycle_result",
            "schema_version": 0,
            "cycle_id": self.context.cycle_id,
            "cycle_time": self.context.cycle_time.value,
            "seed": self.context.seed,
            "snapshot_id": self.context.snapshot_id,
            "channel_count": self.channel_count,
            "intents_proposed": len(self.intents),
            "ordered": [to_wire(e) for e in self.arbitration.ordered],
            "rejected": [
                {"intent": to_wire(env), "reason_code": code,
                 "detail": detail}
                for env, code, detail in self.arbitration.rejected
            ],
            "superseded": dict(self.arbitration.superseded),
        }

    def canonical(self) -> str:
        return canonical_json(self.to_wire())


def rebalance_intent_pairs(pairs: List[Any], ctx: CycleContext,
                           ev_enabled: bool = False,
                           ) -> List[Tuple[IntentEnvelope, Any]]:
    """Like rebalance_intents_from_pairs but keeps the (envelope, pair)
    association — the execution cutover needs to map arbitrated
    envelopes back to their candidates.

    ev_enabled (PR 6, econ_ev_populated): populate the envelope's
    EV/confidence from pair.score_decomposition (final_score_sats /
    p_success). Default False keeps the byte-pinned zeros."""
    out = []
    for pair in sorted(pairs, key=lambda p: (str(p.dest_channel_id),
                                             str(p.source_channel_id))):
        env = _rebalance_intent(pair, ctx, ev_enabled=ev_enabled)
        out.append((env, pair))
    return out


def rebalance_intents_from_pairs(pairs: List[Any],
                                 ctx: CycleContext) -> List[IntentEnvelope]:
    """Map pure-planner PairCandidates to REBALANCE intent envelopes.
    Deterministic: envelope fields derive only from pair data + ctx."""
    return [env for env, _pair in rebalance_intent_pairs(pairs, ctx)]


def _rebalance_intent(pair: Any, ctx: CycleContext,
                      ev_enabled: bool = False) -> IntentEnvelope:
    amount = int(getattr(pair, "amount_sats", 0) or 0)
    max_fee = int(getattr(pair, "pair_budget_sats", 0) or 0)
    benefit, confidence = SignedMsat(0), Micro(0)
    if ev_enabled:
        from .econ_ev import benefit_msat_from_sats, confidence_micro
        decomp = getattr(pair, "score_decomposition", None) or {}
        if "final_score_sats" in decomp:
            benefit = benefit_msat_from_sats(decomp.get("final_score_sats"))
            confidence = confidence_micro(decomp.get("p_success"))
    return make_intent(
        intent_type="REBALANCE",
        snapshot_id=ctx.snapshot_id,
        created_at=ctx.cycle_time,
        expires_at=ctx.cycle_time.plus_seconds(600),
        target=str(pair.dest_channel_id),
        amount_msat=Msat(amount * 1000),
        expected_benefit_msat=benefit,
        max_cost_msat=Msat(max_fee * 1000),
        capital_committed_msat=Msat(amount * 1000),
        confidence_micro=confidence,
        reason_codes=(),
        explanation=Explanation("cycle_rebalance", (
            ("source", str(pair.source_channel_id)),
            ("dest", str(pair.dest_channel_id)),
            ("amount_sats", amount),
            ("score", round(float(getattr(pair, "score", 0.0) or 0.0), 6)),
        )),
        preconditions=(),
        priority=50,
        budget_bucket="rebalance",
        origin_policy="econ_cycle_shadow",
        reversible=False,
    )


def plan_cycle(*, pairs: List[Any], ctx: CycleContext,
               channel_count: int = 0) -> CycleResult:
    """The pure cycle core: intents from policy outputs, then batch
    arbitration under the J3 ladder. No I/O, no clock, no execution."""
    intents = rebalance_intents_from_pairs(pairs, ctx)
    arbitration = arbitrate(list(intents), now=ctx.cycle_time.value)
    return CycleResult(
        context=ctx,
        channel_count=int(channel_count),
        intents=tuple(intents),
        arbitration=arbitration,
    )


def run_shadow_cycle(*, rebalance_engine: Any, econ_shadow: Any,
                     now: int, cycle_seq: int) -> Optional[dict]:
    """Collector + core + publisher for one shadow cycle. Fail-open:
    returns the wire result or None; never raises into the caller.

    Collection reuses the engine's read-only candidate planning (one
    pass, no execution); the CycleContext seed derives deterministically
    from the cycle identity so replays reproduce byte-identical output.
    """
    try:
        if rebalance_engine is None or econ_shadow is None:
            return None
        pairs = rebalance_engine.find_candidates() or []
        cycle_id = f"econ-cycle-{int(now)}-{int(cycle_seq)}"
        base_ctx = CycleContext(
            cycle_id=cycle_id,
            cycle_time=UnixTime(int(now)),
            seed=0,
            snapshot_id=cycle_id,
        )
        ctx = CycleContext(
            cycle_id=cycle_id,
            cycle_time=UnixTime(int(now)),
            seed=base_ctx.derive_seed("econ-cycle"),
            snapshot_id=cycle_id,
        )
        result = plan_cycle(pairs=pairs, ctx=ctx,
                            channel_count=len(pairs))
        ledger = econ_shadow.ledger_for_reconciliation()
        if ledger is not None:
            for env in result.intents:
                try:
                    ledger.append(
                        event_type="intent_proposed",
                        intent_id=env.intent_id.value,
                        idempotency_key=env.idempotency_key,
                        cycle_id=cycle_id,
                        at=int(now),
                        details={"target": env.target, "shadow_cycle": True,
                                 "explanation": env.explanation.render()},
                    )
                except Exception:
                    pass
        return result.to_wire()
    except Exception:
        return None
