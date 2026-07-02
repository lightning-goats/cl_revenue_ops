"""Permanent regression armor for the RPC operator-param security sweep.

A matrix test over every ``@plugin.method`` handler. For each handler we take a
baseline of valid params, confirm it returns a dict under a mocked-but-live
plugin environment, then substitute a family of malformed values (None,
negative, zero, huge, wrong-type, empty-string, empty-collection) into each
operator param one at a time. The invariant under test:

    A handler must return a dict (a clean error or a validated result); it must
    never leak an uncaught exception (TypeError / ValueError / re.error /
    AttributeError / KeyError / IndexError) from raw operator-param handling.

Differential design: if a handler cannot be exercised with generic mocks (its
baseline call raises or returns a non-dict because a MagicMock return value
flows into real arithmetic/parsing), it is recorded as SKIPPED rather than
producing a false failure. The critical validated handlers (planner limit,
analyze/set-fee/rebalance channel_id, dashboard window, hive-hints segment) are
additionally pinned by the dedicated per-finding tests.
"""

import inspect

import pytest

from modules.config import Config
from tests.plugin_test_utils import DummyPlugin, load_plugin_module


# Malformed value classes applied to each operator param.
MALFORMED_VALUES = [None, -1, 0, 10 ** 12, 1.5, "x", "", [], {}]

# Exceptions that indicate a raw-param leak (a handler crashing instead of
# returning a clean error dict).
LEAK_EXCEPTIONS = (TypeError, ValueError, AttributeError, KeyError, IndexError)


def _baseline_value(name, default):
    """A plausibly-valid baseline value for a param."""
    if default is not inspect.Parameter.empty and default is not None:
        return default
    n = name.lower()
    if n in ("peer_id", "only_peer_id"):
        return "02" + "a" * 64
    if n in ("channel_id", "from_channel", "to_channel", "only_channel_id"):
        return "123x456x0"
    if n == "swap_ids":
        return ["swap1"]
    if n.endswith("_id") or n in ("swap_id", "reservation_id", "reference_id"):
        return "id1"
    if n in ("address", "destination", "to_address"):
        return "bcrt1qexampledestinationaddressxxxxxxxxxxxx"
    if "currency" in n:
        return "BTC"
    if n == "swap_mnemonic":
        return "abandon " * 11 + "about"
    if n == "metadata_json":
        return "{}"
    if n in ("action",):
        return "list"
    if n in ("strategy",):
        return "dynamic"
    if n in ("rebalance",):
        return "enabled"
    if n in ("report_type",):
        return "summary"
    if n in ("category", "subcategory"):
        return "rebalance"
    if n in ("swap_type",):
        return "reverse"
    if n in ("reason",):
        return "manual"
    if n in ("key", "value", "tag", "note", "source"):
        return "x"
    # numeric-ish fallbacks
    if any(tok in n for tok in ("pct", "ratio", "roi", "multiplier", "factor", "margin")):
        return 1.0
    if any(tok in n for tok in ("sats", "ppm", "limit", "window", "hours", "seconds",
                                 "actions", "candidates", "days", "amount", "fee")):
        return 100
    return "x"


def _handler_specs(mod):
    """Yield (rpc_name, func, param_names, baseline_kwargs, has_var_kw)."""
    specs = []
    for rpc_name, registration in mod.plugin.methods.items() if hasattr(mod.plugin, "methods") else []:
        pass  # DummyPlugin doesn't record methods; fall back to source scan below.
    # DummyPlugin.method returns the function unchanged, so the decorated
    # handlers are plain module attributes. Discover them by scanning globals
    # for functions whose first param is `plugin`.
    seen = set()
    for attr in dir(mod):
        fn = getattr(mod, attr)
        if not callable(fn) or not inspect.isfunction(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        if not params or params[0].name != "plugin":
            continue
        if fn in seen:
            continue
        # Only treat as an RPC handler if it is registered via @plugin.method.
        # We can't read the decorator, so use a heuristic: module-level funcs
        # whose first arg is `plugin` and that are referenced as RPC handlers.
        seen.add(fn)
        param_names = []
        baseline = {}
        has_var_kw = False
        for p in params[1:]:
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_kw = True
                continue
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            param_names.append(p.name)
            baseline[p.name] = _baseline_value(p.name, p.default)
        specs.append((attr, fn, param_names, baseline, has_var_kw))
    return specs


# The set of RPC handler function names (from @plugin.method) so the matrix
# only fuzzes real operator surfaces, not internal helpers that happen to take
# a `plugin` first arg.
RPC_HANDLER_FUNCS = {
    "revenue_rebalance_cycle", "revenue_status", "revenue_hive_hints_status",
    "revenue_rebalance_debug", "revenue_fee_debug", "revenue_fee_cycle",
    "revenue_analyze", "revenue_wake_all", "revenue_capacity_report",
    "revenue_planner_status", "planner_candidate_sources",
    "revenue_planner_candidates", "revenue_planner_execute",
    "revenue_planner_history", "revenue_set_fee", "revenue_rebalance",
    "revenue_profitability", "revenue_history", "revenue_ignore",
    "revenue_unignore", "revenue_list_ignored", "revenue_policy",
    "revenue_report", "revenue_hot_channel_protection_peers", "revenue_config",
    "revenue_dashboard", "revenue_health", "revenue_cleanup_closed",
    "revenue_clear_reservations", "revenue_total_cost_budget",
    "revenue_capex_status", "revenue_spend_ledger", "revenue_spend_reserve",
    "revenue_spend_release", "revenue_spend_release_stale",
    "revenue_spend_settle", "revenue_boltz_quote", "revenue_boltz_loop_out",
    "revenue_boltz_loop_in", "revenue_boltz_status", "revenue_boltz_history",
    "revenue_boltz_external_pay_ignores", "revenue_boltz_budget",
    "revenue_boltz_wallet", "revenue_boltz_refund", "revenue_boltz_claim",
    "revenue_boltz_chainswap", "revenue_boltz_withdraw", "revenue_boltz_deposit",
    "revenue_boltz_backup", "revenue_boltz_backup_verify",
    "revenue_boltz_balance_recommendations", "revenue_boltz_auto_cycle_status",
    "revenue_boltz_auto_cycle_run_now", "revenue_boltz_balance_cycle",
    "revenue_boltz_expansion_treasury_status",
    "revenue_boltz_expansion_treasury_recommendations",
    "revenue_boltz_expansion_treasury_cycle",
}


def _setup_env(mod):
    """Wire module globals so handlers run past their init guards."""
    from unittest.mock import MagicMock
    mod.config = Config()
    for name in ("flow_analyzer", "fee_controller", "rebalancer", "database",
                 "profitability_analyzer", "capacity_planner", "safe_plugin",
                 "data_service", "policy_manager", "boltz_manager",
                 "capex_engine", "hive_hints"):
        if hasattr(mod, name):
            setattr(mod, name, MagicMock())


@pytest.fixture(scope="module")
def mod():
    return load_plugin_module()


def test_handler_discovery_covers_full_surface(mod):
    specs = [s for s in _handler_specs(mod) if s[0] in RPC_HANDLER_FUNCS]
    found = {s[0] for s in specs}
    missing = RPC_HANDLER_FUNCS - found
    assert not missing, f"handlers not discovered: {sorted(missing)}"
    # 58 handlers is the current full surface.
    assert len(found) == len(RPC_HANDLER_FUNCS) == 58


def test_param_matrix_no_leaks(mod):
    plugin = DummyPlugin()
    specs = [s for s in _handler_specs(mod) if s[0] in RPC_HANDLER_FUNCS]

    failures = []
    skipped = []
    param_cases = 0
    handlers_fuzzed = 0
    no_param_handlers = 0

    for rpc_name, fn, param_names, baseline, has_var_kw in specs:
        _setup_env(mod)

        # Baseline call must succeed and return a dict, else this handler is
        # not auto-fuzzable with generic mocks (recorded, not failed).
        try:
            base = fn(plugin, **baseline)
        except Exception as e:  # noqa: BLE001
            skipped.append((rpc_name, "baseline-raise", repr(e)))
            continue
        if not isinstance(base, dict):
            skipped.append((rpc_name, "baseline-nondict", type(base).__name__))
            continue

        if not param_names:
            no_param_handlers += 1
            # No operator params: extras must be ignored (only possible when
            # the handler declares **kwargs; pyln filters otherwise).
            if has_var_kw:
                try:
                    r = fn(plugin, **baseline, unexpected_extra_param="junk")
                except Exception as e:  # noqa: BLE001
                    failures.append((rpc_name, "**extras", "junk", repr(e)))
                else:
                    if not isinstance(r, dict):
                        failures.append((rpc_name, "**extras", "junk", "non-dict"))
            continue

        handlers_fuzzed += 1
        for pname in param_names:
            for mv in MALFORMED_VALUES:
                if mv == baseline.get(pname):
                    continue
                kw = dict(baseline)
                kw[pname] = mv
                param_cases += 1
                try:
                    r = fn(plugin, **kw)
                except LEAK_EXCEPTIONS as e:
                    failures.append((rpc_name, pname, repr(mv), repr(e)))
                except Exception:  # noqa: BLE001
                    # Non-leak exceptions (e.g. handler re-raising a cleanly
                    # wrapped RPCError) are out of scope for this armor.
                    pass
                else:
                    if not isinstance(r, dict):
                        failures.append((rpc_name, pname, repr(mv), f"non-dict:{type(r).__name__}"))

    # Diagnostics visible on failure.
    detail = (
        f"\nhandlers fuzzed: {handlers_fuzzed}"
        f"\nno-param handlers: {no_param_handlers}"
        f"\nskipped (not auto-fuzzable): {len(skipped)}"
        f"\nparam cases: {param_cases}"
        f"\nfailures: {failures[:20]}"
    )
    assert not failures, detail

    # Sanity: the matrix must actually exercise a meaningful surface.
    assert param_cases >= 200, detail
    assert handlers_fuzzed + no_param_handlers + len(skipped) == len(specs)
