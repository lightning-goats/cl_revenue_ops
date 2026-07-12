"""Econ shadow assembler (refactor Phase 1 wiring tranche).

Records the plugin's LIVE decisions as typed intents in the append-only
econ ledger (observe-mode emission, Workstream B acceptance) and builds
on-demand canonical-snapshot previews (Workstream A) — without holding
any authority over execution.

FAIL-OPEN CONTRACT: no method of this class may raise into a caller. A
failure disables the affected operation, logs once at warn (then debug),
and production cycles continue untouched. Gated by the runtime flag
``econ_shadow_enabled`` (default False): with the flag off this module
records nothing and creates no files.

The ledger lives in its OWN sqlite file (econ_ledger.db beside
revenue_ops.db) — the production database schema is untouched; see
docs/planning/2026-07-12-refactor-phase1-wiring.md.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .econ_intents import Explanation, make_intent
from .econ_ledger import EconLedger
from .econ_snapshot import (
    ROLES,
    BudgetState,
    EconomicSnapshot,
    NodeState,
    build_channel_snapshot,
    to_wire,
)
from .econ_types import Micro, Msat, SignedMsat, UnixTime


class EconShadow:
    def __init__(self, plugin, config, ledger_path: Optional[str] = None):
        self._plugin = plugin
        self._config = config
        self._ledger: Optional[EconLedger] = None
        self._ledger_failed = False
        self._ledger_path = ledger_path or self._default_ledger_path()
        self.intents_recorded_total = 0

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _default_ledger_path(self) -> str:
        try:
            db_path = os.path.expanduser(
                str(getattr(self._config, "db_path", "")) or
                "~/.lightning/revenue_ops.db")
            return os.path.join(os.path.dirname(db_path), "econ_ledger.db")
        except Exception:
            return "econ_ledger.db"

    def _log(self, message: str, level: str = "debug") -> None:
        try:
            self._plugin.log(f"ECON-SHADOW: {message}", level=level)
        except Exception:
            pass

    def enabled(self) -> bool:
        try:
            cfg = self._config.snapshot() \
                if hasattr(self._config, "snapshot") else self._config
            raw = getattr(cfg, "econ_shadow_enabled", False)
            if isinstance(raw, str):
                return raw.strip().lower() in ("true", "1", "yes", "on")
            return raw is True
        except Exception:
            return False

    def _get_ledger(self) -> Optional[EconLedger]:
        if self._ledger is not None:
            return self._ledger
        if self._ledger_failed:
            return None
        try:
            self._ledger = EconLedger(self._ledger_path)
            return self._ledger
        except Exception as e:
            self._ledger_failed = True
            self._log(f"ledger unavailable ({e}) — shadow recording "
                      f"disabled for this session", level="warn")
            return None

    # ------------------------------------------------------------------
    # observe-mode intent emission
    # ------------------------------------------------------------------
    def record_fee_intents(self, adjustments: List[Any], now: int) -> int:
        """Record this fee cycle's applied adjustments as SET_FEE intent
        proposals. Never raises; returns the count recorded."""
        try:
            if not self.enabled() or not adjustments:
                return 0
            ledger = self._get_ledger()
            if ledger is None:
                return 0
            recorded = 0
            for adj in adjustments:
                try:
                    env = make_intent(
                        intent_type="SET_FEE",
                        snapshot_id=f"fee-cycle-{int(now)}",
                        created_at=UnixTime(int(now)),
                        expires_at=UnixTime(int(now) + 3600),
                        target=str(adj.channel_id),
                        amount_msat=None,
                        expected_benefit_msat=SignedMsat(0),
                        max_cost_msat=Msat(0),
                        capital_committed_msat=Msat(0),
                        confidence_micro=Micro(0),
                        reason_codes=(),
                        explanation=Explanation("fee_adjustment", (
                            ("old_fee_ppm", int(adj.old_fee_ppm)),
                            ("new_fee_ppm", int(adj.new_fee_ppm)),
                            ("controller_reason_code",
                             str(getattr(adj, "reason_code", ""))),
                        )),
                        preconditions=(),
                        priority=50,
                        budget_bucket="fees",
                        origin_policy="fee_controller_shadow",
                        reversible=True,
                    )
                    ledger.append(
                        event_type="intent_proposed",
                        intent_id=env.intent_id.value,
                        idempotency_key=env.idempotency_key,
                        cycle_id=env.snapshot_id,
                        at=int(now),
                        details={"explanation": env.explanation.render()},
                    )
                    recorded += 1
                except Exception as e:
                    self._log(f"adjustment skipped: {e}")
            self.intents_recorded_total += recorded
            return recorded
        except Exception as e:
            self._log(f"record_fee_intents failed open: {e}")
            return 0

    # ------------------------------------------------------------------
    # legacy spend-path journal (Phase 2 pilot A)
    #
    # The generic spend lifecycle (Database.reserve_spend / settle /
    # release) predates typed intents, so reservation_id doubles as both
    # intent_id and idempotency_key. Same fail-open + flag contract as
    # everything else in this class. Timestamps are audit stamps
    # (int(time.time())), not decision inputs.
    # ------------------------------------------------------------------
    def _journal(self, event_type: str, reservation_id: Any,
                 category: str = "", amounts: Optional[dict] = None,
                 details: Optional[dict] = None) -> None:
        try:
            if not self.enabled():
                return
            rid = str(reservation_id or "").strip()
            if not rid:
                return
            ledger = self._get_ledger()
            if ledger is None:
                return
            import time as _time
            ledger.append(
                event_type=event_type,
                intent_id=rid,
                idempotency_key=rid,
                cycle_id=f"spend-{str(category or 'generic')}",
                at=int(_time.time()),
                amounts=amounts or {},
                details=details or {},
            )
        except Exception as e:
            self._log(f"spend journal skipped ({event_type}): {e}")

    def note_spend_reserved(self, reservation_id: Any, amount_sats: Any,
                            category: str = "") -> None:
        try:
            amounts = {"reserved_msat": int(amount_sats) * 1000}
        except (TypeError, ValueError):
            self._log(f"note_spend_reserved skipped: bad amount "
                      f"{amount_sats!r}")
            return
        self._journal("budget_reserved", reservation_id, category,
                      amounts=amounts)

    def note_spend_settled(self, reservation_id: Any,
                           actual_spent_sats: Any,
                           category: str = "") -> None:
        try:
            amounts = {"cost_msat": int(actual_spent_sats) * 1000}
        except (TypeError, ValueError):
            self._log(f"note_spend_settled skipped: bad amount "
                      f"{actual_spent_sats!r}")
            return
        self._journal("cost_recorded", reservation_id, category,
                      amounts=amounts)
        self._journal("execution_succeeded", reservation_id, category)
        # DB semantics: settle is terminal for the whole reservation —
        # the unused remainder is released (spec reservation machine:
        # reserved -> spent, unused portion -> released).
        self._journal("reservation_released", reservation_id, category,
                      details={"reason": "settled"})

    def note_spend_released(self, reservation_id: Any,
                            reason: str = "released") -> None:
        self._journal("reservation_released", reservation_id,
                      details={"reason": str(reason)})

    def ledger_for_reconciliation(self) -> Optional[EconLedger]:
        """The lazy ledger, for the reconciliation sweep (Phase 2
        pilot B). None when disabled or unavailable."""
        if not self.enabled():
            return None
        return self._get_ledger()

    # ------------------------------------------------------------------
    # on-demand snapshot preview
    # ------------------------------------------------------------------
    def build_snapshot_preview(self, *, channels: Any,
                               profitability: Dict[str, Any],
                               budget: Dict[str, Any], now: int,
                               receivable_ratio_target: float = 0.0,
                               ) -> Tuple[Optional[dict], List[str]]:
        """Assemble a canonical snapshot from live caches. Placeholder
        fields are DECLARED in the returned approximations list — missing
        evidence is labeled, never silently invented (invariant 7)."""
        approximations = [
            "lifecycle=PRODUCTIVE for all channels (lifecycle model is "
            "Workstream F5)",
            "confidence_micro=0 (flow confidence not wired yet)",
            "onchain_confirmed_msat=0, reserved_msat=0, "
            "sourced_volume_msat=0 (not wired yet)",
            "protections not populated (policy tags not wired yet)",
        ]
        try:
            channel_snaps = []
            for channel in channels:
                try:
                    scid = str(channel["short_channel_id"])
                    prof = (profitability or {}).get(scid)
                    role = "UNKNOWN"
                    role_30d = getattr(prof, "role_30d", None)
                    name = getattr(role_30d, "name", None)
                    if name in ROLES:
                        role = name
                    channel_snaps.append(build_channel_snapshot(
                        channel=channel, prof=prof, role=role))
                except Exception as e:
                    approximations.append(
                        f"channel skipped ({e.__class__.__name__}): "
                        f"{str(channel)[:80]}")
            total_local = sum(c.local_msat.value for c in channel_snaps)
            total_remote = sum(c.remote_msat.value for c in channel_snaps)
            total_capacity = sum(
                c.capacity_msat.value for c in channel_snaps)
            budget = budget or {}
            node = NodeState(
                total_local_msat=Msat(total_local),
                total_remote_msat=Msat(total_remote),
                receivable_objective_msat=Msat(int(
                    total_capacity * max(0.0, min(1.0, float(
                        receivable_ratio_target or 0.0))))),
                onchain_confirmed_msat=Msat(0),
                reserved_msat=Msat(0),
                daily_budget=BudgetState(
                    cap_msat=Msat.from_sats(int(
                        budget.get("cap_sats", 0) or 0)),
                    reserved_msat=Msat.from_sats(int(
                        budget.get("reserved_sats", 0) or 0)),
                    spent_msat=Msat.from_sats(int(
                        budget.get("spent_sats", 0) or 0)),
                ),
            )
            snap = EconomicSnapshot(
                snapshot_id=f"preview-{int(now)}",
                observed_at=UnixTime(int(now)),
                evidence_window_seconds=30 * 86400,
                node=node,
                channels=tuple(channel_snaps),
            )
            return to_wire(snap), approximations
        except Exception as e:
            self._log(f"snapshot preview failed open: {e}")
            return None, [f"preview failed: {e.__class__.__name__}: {e}"]
