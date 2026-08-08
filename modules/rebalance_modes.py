"""Unified rebalance-mode descriptors (refactor Phase 3D, Workstream F4).

The heavy F4 unification already exists in this repo: every rebalance —
auto cycle, hot-channel protection, structural drain, and manual
— prices and executes through the ONE v2 engine pipeline
(planner → router → executor → rebalance_history), and hot-channel
protection is already a pair-level priority/budget modifier inside the
auto cycle rather than an independent subsystem (the spec's requested
end state).

What was missing is the spec's table: mode differences expressed as
DATA (priority, budget allocation, deadline, authority) instead of
scattered boolean kwargs (`reserve_budget`, `account_costs`,
`enforce_budget`). This module is that table; call sites route their
engine kwargs through it so a mode's semantics live in exactly one
place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RebalanceMode:
    name: str
    # Requested priority for arbitration (J3 ladder; operator=100).
    priority: int
    # Budget allocation per the spec table.
    budget_bucket: str
    # Engine kwargs (the previously-scattered booleans).
    reserve_on_rail: bool   # atomic reservation on the unified rail
    account_costs: bool     # engine records settled fee into rebalance_costs
    # Typical deadline class (documentation + future arbiter input).
    deadline: str
    # Who owns accounting when not on the rail.
    accounting_owner: str


MODES: Dict[str, RebalanceMode] = {
    # Auto cycle (includes hot-channel protection as pair-level
    # priority/budget boosts and structural-drain demand computation —
    # the engine reserves and accounts internally via
    # _reserve_execution_budget/_finish_execution_budget).
    "normal": RebalanceMode(
        name="normal", priority=50, budget_bucket="maintenance",
        reserve_on_rail=True, account_costs=True, deadline="normal",
        accounting_owner="engine",
    ),
    "hot_protection": RebalanceMode(
        name="hot_protection", priority=90,
        budget_bucket="revenue_protection",
        reserve_on_rail=True, account_costs=True, deadline="short",
        accounting_owner="engine",
    ),
    "structural_drain": RebalanceMode(
        name="structural_drain", priority=20, budget_bucket="structural",
        reserve_on_rail=True, account_costs=True, deadline="long",
        accounting_owner="engine",
    ),
    # Operator-explicit: owns override semantics and its own accounting
    # (P4-020: engine reservation deliberately skipped).
    "manual": RebalanceMode(
        name="manual", priority=100, budget_bucket="operator_explicit",
        reserve_on_rail=False, account_costs=False, deadline="immediate",
        accounting_owner="caller",
    ),
}


def engine_kwargs(mode_name: str) -> dict:
    """The engine execute_candidate kwargs a mode implies."""
    mode = MODES[mode_name]
    return {
        "reserve_budget": mode.reserve_on_rail,
        "account_costs": mode.account_costs,
    }
