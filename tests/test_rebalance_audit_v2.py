from unittest.mock import MagicMock

from modules.rebalance_audit_v2 import RebalanceAudit


# ------------------------------------------------------------------
# Pure formatter tests
# ------------------------------------------------------------------

def test_format_pick_includes_all_fields():
    msg = RebalanceAudit.format_pick(
        source_channel_id="111x1x0",
        dest_channel_id="222x2x0",
        amount_sats=50_000,
        route_cost_sats=120,
        value_score=0.87,
    )
    assert "REBAL_PICK" in msg
    assert "source=111x1x0" in msg
    assert "dest=222x2x0" in msg
    assert "amount=50000" in msg
    assert "route_cost_sats=120" in msg
    assert "value_score=0.87" in msg


def test_format_skip_includes_reason_and_budget():
    msg = RebalanceAudit.format_skip(
        channel_id="333x3x0",
        reason="skipped_no_budget",
        value_class="hive",
        remaining_budget_sats=0,
    )
    assert "REBAL_SKIP" in msg
    assert "channel=333x3x0" in msg
    assert "reason=skipped_no_budget" in msg
    assert "value_class=hive" in msg
    assert "budget=0" in msg


def test_format_skip_omits_route_cost_when_zero():
    msg = RebalanceAudit.format_skip(
        channel_id="444x4x0",
        reason="skipped_cooldown",
        value_class="active",
        remaining_budget_sats=1000,
        route_cost_sats=0,
    )
    assert "route_cost=" not in msg


def test_format_skip_with_detail():
    msg = RebalanceAudit.format_skip(
        channel_id="555x5x0",
        reason="skipped_policy",
        value_class="profitable",
        remaining_budget_sats=500,
        route_cost_sats=200,
        detail="peer_policy=passive",
    )
    assert "route_cost=200" in msg
    assert "detail=peer_policy=passive" in msg


# ------------------------------------------------------------------
# Plugin logging integration tests
# ------------------------------------------------------------------

def test_log_pick_calls_plugin_log():
    plugin = MagicMock()
    audit = RebalanceAudit(plugin)

    audit.log_pick(
        source_channel_id="111x1x0",
        dest_channel_id="222x2x0",
        amount_sats=50_000,
        route_cost_sats=120,
        value_score=0.87,
    )

    plugin.log.assert_called_once()
    logged_msg = plugin.log.call_args[0][0]
    assert "REBAL_PICK" in logged_msg
    assert "source=111x1x0" in logged_msg
    assert plugin.log.call_args[1]["level"] == "debug"


def test_log_skip_calls_plugin_log():
    plugin = MagicMock()
    audit = RebalanceAudit(plugin)

    audit.log_skip(
        channel_id="333x3x0",
        reason="skipped_no_route",
        value_class="hive",
        remaining_budget_sats=2000,
    )

    plugin.log.assert_called_once()
    logged_msg = plugin.log.call_args[0][0]
    assert "REBAL_SKIP" in logged_msg
    assert "reason=skipped_no_route" in logged_msg
    assert plugin.log.call_args[1]["level"] == "debug"


def test_log_cycle_summary_format():
    plugin = MagicMock()
    audit = RebalanceAudit(plugin)

    audit.log_cycle_summary(
        selected_count=3,
        skipped_count=7,
        total_valuable=10,
        total_channels=25,
        total_budget_sats=15_000,
    )

    plugin.log.assert_called_once()
    logged_msg = plugin.log.call_args[0][0]
    assert "REBAL_CYCLE" in logged_msg
    assert "selected=3" in logged_msg
    assert "skipped=7" in logged_msg
    assert "valuable=10/25" in logged_msg
    assert "budget_remaining=15000" in logged_msg
    assert plugin.log.call_args[1]["level"] == "debug"


# ------------------------------------------------------------------
# RA2-1: skip-reason vocabulary drift guard
# ------------------------------------------------------------------
#
# VALID_SKIP_REASONS is the canonical bucketing vocabulary for log
# consumers. Phase 2 of the audit found >= 10 production skip reasons
# missing from it (RA2-1). The scraper below walks every module's AST and
# collects the reason strings that can actually reach a SkipRecord, so the
# vocabulary can never silently diverge from the emitters again.

import ast
from pathlib import Path

MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"

# Functions whose returned strings flow (via variables) into SkipRecord
# reasons even though they do not construct SkipRecord themselves.
INDIRECT_REASON_PRODUCERS = {
    "_source_eligibility",           # rebalance_state_v2: source gate reasons
    "_destination_eligibility",      # rebalance_state_v2: dest gate reasons
    "_translate_getroutes_error",    # rebalance_router_v3: askrene error map
    "_validate_getroutes_middle_path",  # rebalance_router_v3: path validation
    "_executor_unavailable_reason",  # rebalance_engine_v2: native_unavailable (DEF-077)
}


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _collect_string_literals(node, out: set) -> None:
    """Collect string constants from a reason expression, including
    ``x or "fallback"`` and conditional expressions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value:
            out.add(node.value)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _collect_string_literals(value, out)
    elif isinstance(node, ast.IfExp):
        _collect_string_literals(node.body, out)
        _collect_string_literals(node.orelse, out)


def collect_emitted_skip_reasons() -> set:
    reasons: set = set()
    emitter_functions: set = set()
    trees = {
        path: ast.parse(path.read_text())
        for path in sorted(MODULES_DIR.glob("*.py"))
    }

    # Pass 1: SkipRecord construction sites. Within any function that
    # builds a SkipRecord, also collect assignments to a local `reason`
    # variable (planner-style ``reason, detail = "...", "..."``) and the
    # function's own `reason` parameter default (overlay-style helpers).
    for tree in trees.values():
        for func in (n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            builds_skip_record = False
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and _call_name(node) == "SkipRecord":
                    builds_skip_record = True
                    for kw in node.keywords:
                        if kw.arg == "reason":
                            _collect_string_literals(kw.value, reasons)
            if not builds_skip_record:
                continue
            emitter_functions.add(func.name)
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "reason":
                        _collect_string_literals(node.value, reasons)
                    elif (isinstance(target, ast.Tuple)
                          and isinstance(node.value, ast.Tuple)
                          and len(target.elts) == len(node.value.elts)):
                        for elt, value in zip(target.elts, node.value.elts):
                            if isinstance(elt, ast.Name) and elt.id == "reason":
                                _collect_string_literals(value, reasons)
            args = func.args
            positional = args.args[len(args.args) - len(args.defaults):]
            for arg, default in (
                list(zip(positional, args.defaults))
                + list(zip(args.kwonlyargs, args.kw_defaults))
            ):
                if arg.arg == "reason" and default is not None:
                    _collect_string_literals(default, reasons)

    # Pass 2: `reason=` keyword literals at call sites of skip-emitting
    # helpers (e.g. suppress_leased_pairs(..., reason="fleet_lease_held")).
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in emitter_functions:
                for kw in node.keywords:
                    if kw.arg == "reason":
                        _collect_string_literals(kw.value, reasons)

    # Pass 3: indirect producers — string constants in their return tuples
    # become SkipRecord reasons downstream. These producers return either
    # ``(bool, reason)`` (state-layer eligibility) or ``(reason, detail)``
    # (engine executor-unavailable). A bucketing reason is always a single
    # grep-friendly token, so drop any whitespace-bearing string (the human-
    # readable detail leg) to avoid polluting the vocabulary. (DEF-077)
    for tree in trees.values():
        for func in (n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name in INDIRECT_REASON_PRODUCERS):
            for node in ast.walk(func):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                    tuple_strings: set = set()
                    for elt in node.value.elts:
                        _collect_string_literals(elt, tuple_strings)
                    reasons |= {
                        s for s in tuple_strings
                        if s and not any(ch.isspace() for ch in s)
                    }

    return reasons


def test_scraper_finds_known_production_emitters():
    """Sanity guard for the scraper itself: if AST collection silently
    breaks, this pins a spread of known reasons from every emitter layer."""
    scraped = collect_emitted_skip_reasons()
    expected_sample = {
        # planner
        "inside_band", "no_partner", "source_ineligible", "dest_ineligible",
        # coordination overlay
        "not_designated_executor", "coordination_unavailable",
        "coordination_unresolvable_endpoint", "coordination_preempted",
        "lease_conflict",
        # engine
        "hive_equalization_cooldown", "pair_cooldown", "pair_futility",
        "below_hold_margin", "cycle_already_running", "fleet_lease_held",
        "max_pairs_reached", "route_over_budget", "no_route",
        "native_unavailable",
        # state-layer eligibility strings
        "cooldown", "not_valuable", "no_budget",
        # v3 router
        "unknown_source_node", "unknown_layer", "askrene_child_died",
        "path_loops_through_us",
    }
    missing = expected_sample - scraped
    assert not missing, f"scraper no longer sees known emitters: {sorted(missing)}"


def test_all_emitted_skip_reasons_are_in_vocabulary():
    """RA2-1: every reason a production SkipRecord can carry must be a
    member of VALID_SKIP_REASONS, or log consumers lose bucketing."""
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS

    scraped = collect_emitted_skip_reasons()
    unknown = scraped - VALID_SKIP_REASONS
    assert not unknown, (
        "SkipRecord emitters use reasons missing from VALID_SKIP_REASONS "
        f"(vocabulary drift): {sorted(unknown)}"
    )


def test_native_unavailable_is_covered_by_drift_guard():
    """DEF-077: 'native_unavailable' (emitted by the v3 engine when the
    native executor RPC surface is down) must be both scraped by the drift
    guard AND present in VALID_SKIP_REASONS. The drift guard now covers it
    via the _executor_unavailable_reason producer, so removing it from
    VALID_SKIP_REASONS would make test_all_emitted_skip_reasons_are_in_
    vocabulary fail (it no longer passes silently)."""
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS

    scraped = collect_emitted_skip_reasons()
    assert "native_unavailable" in scraped, (
        "drift guard does not see native_unavailable — _executor_unavailable_"
        "reason missing from INDIRECT_REASON_PRODUCERS"
    )
    assert "native_unavailable" in VALID_SKIP_REASONS


def test_non_actionable_reasons_are_subset_of_vocabulary():
    from modules.rebalance_audit_v2 import (
        NON_ACTIONABLE_SKIP_REASONS,
        VALID_SKIP_REASONS,
    )

    assert NON_ACTIONABLE_SKIP_REASONS <= VALID_SKIP_REASONS
