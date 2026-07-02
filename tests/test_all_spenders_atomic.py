"""PERMANENT ENUMERATION GUARD: every autonomous money-committing site is
atomic-or-rail-counted, and a NEW or moved spender trips this test.

Three deep-audit refutation passes each found ONE MORE autonomous spender that
bypassed the atomic cross-category budget (boltz -> P4-014, capex -> P4-018,
defibrillation -> P4-020). The atomic BEGIN IMMEDIATE cross-category sum itself
is verified exact; the recurring hole was always a spender that did not RESERVE.
This guard converts "the refuter keeps finding one more spender" into a standing
invariant: it AST-scans modules/ + cl-revenue-ops.py for every money-committing
call site and asserts each is present in an explicit ALLOWLIST with a coverage
classification. It FAILS if:

  * the scan finds a money-committing call NOT in the allowlist (a new or moved
    spender), or
  * any allowlisted site is classified with a class OUTSIDE the covered set
    (i.e. a daemon-reachable spender left uncovered) -- which forces an operator
    ruling instead of silently shipping a hole.

Modelled on the RA2-1 skip-reason drift guard
(tests/test_rebalance_audit_v2.py / tests/test_architecture_guard.py).
"""

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = REPO_ROOT / "modules"
MAIN_PLUGIN = REPO_ROOT / "cl-revenue-ops.py"


# ---------------------------------------------------------------------------
# Money-committing markers. Each occurrence in production source is a candidate
# spend site that MUST be enumerated in the allowlist.
# ---------------------------------------------------------------------------
# Callee-name (func.attr / func.id) markers.
_CALLEE_MARKERS = {
    "reserve_spend": "reserve_spend",        # generic ledger atomic reserve
    "reserve_budget": "reserve_budget",      # rebalance atomic reserve
    "execute_candidate": "execute_candidate",  # rebalance execution dispatch
    "_rpc_fundchannel": "fundchannel_rpc",   # planner channel-open wrapper call
    "fund_channel": "fund_channel",          # data_service fundchannel wrapper
    "_rpc_close": "close_rpc",               # planner channel-close wrapper call
    "close_channel": "close_channel",        # data_service close wrapper
}
# rpc.call("<method>", ...) string markers (the raw on-chain RPCs).
_RPC_CALL_STRINGS = {"fundchannel": "rpc_fundchannel", "close": "rpc_close"}
# boltz CLI swap-create command literals.
_SWAP_LITERALS = {"createswap": "boltz_swap_create", "createreverseswap": "boltz_swap_create"}


def _callee_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _enclosing_func_resolver(tree: ast.AST):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def resolve(node):
        n = node
        while n is not None:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return n.name
            n = parent.get(n)
        return "<module>"

    return resolve


def _scan_tree(fname: str, tree: ast.AST, sites: set):
    encl = _enclosing_func_resolver(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            cn = _callee_name(node)
            if cn in _CALLEE_MARKERS:
                sites.add((fname, encl(node), _CALLEE_MARKERS[cn]))
            if cn == "call" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and a0.value in _RPC_CALL_STRINGS:
                    sites.add((fname, encl(node), _RPC_CALL_STRINGS[a0.value]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _SWAP_LITERALS:
                sites.add((fname, encl(node), _SWAP_LITERALS[node.value]))


def collect_spend_sites(extra_sources=None) -> set:
    """Return the set of (filename, enclosing_function, marker) tuples for every
    money-committing call site in production source.

    ``extra_sources``: optional list of (filename, source) pairs, used by the
    test-the-test to prove an injected new spender is detected.
    """
    sites: set = set()
    files = sorted(MODULES_DIR.glob("*.py")) + [MAIN_PLUGIN]
    for path in files:
        _scan_tree(path.name, ast.parse(path.read_text(), filename=path.name), sites)
    for fname, source in (extra_sources or []):
        _scan_tree(fname, ast.parse(source, filename=fname), sites)
    return sites


# ---------------------------------------------------------------------------
# COVERAGE CLASSIFICATION.
# A site classified with a class NOT in COVERED_CLASSES (e.g. the sentinel
# "daemon-reachable-uncovered") FAILS the guard on purpose: a spender that
# cannot be made atomic must be escalated to an operator ruling, never shipped.
# ---------------------------------------------------------------------------
ATOMIC_RESERVE = "atomic-reserve"           # reserves inside BEGIN IMMEDIATE cross-cat sum
RAIL_COUNTED_COST = "rail-counted-cost"     # the committing action; its cost is reserved+settled on the rail
OPERATOR_ONLY = "operator-only-non-daemon"  # reachable only from an operator RPC, not an autonomous daemon
NOT_A_SPEND = "not-a-spend"                 # pure RPC transport/plumbing; reserving caller owns the budget

COVERED_CLASSES = frozenset({ATOMIC_RESERVE, RAIL_COUNTED_COST, OPERATOR_ONLY, NOT_A_SPEND})

# Explicit allowlist of EVERY known money-committing site with its coverage
# classification and a justification (finding citation). Keyed by
# (filename, enclosing_function, marker).
ALLOWLIST = {
    # --- Daemon spender 1/5: rebalance auto-cycle (T3) ----------------------
    ("rebalancer.py", "execute_rebalance", "reserve_budget"): (
        ATOMIC_RESERVE,
        "EVRebalancer auto-cycle (T3) reserves the fee cap via reserve_budget -> "
        "_reserve_budget_atomic BEGIN IMMEDIATE, full effective budget (P4-016/017/DD1).",
    ),
    ("rebalance_engine_v2.py", "_reserve_execution_budget", "reserve_budget"): (
        ATOMIC_RESERVE,
        "Shared v2 engine reserve for the auto cycle (T3) AND the defibrillation "
        "diagnostic shock (T7, via execute_candidate(reserve_budget=True), P4-020); "
        "full effective budget inside BEGIN IMMEDIATE (P4-016).",
    ),
    # --- Daemon spender 5/5: defibrillation diagnostic shock (T7) -----------
    ("rebalancer.py", "_execute_candidate_v2", "execute_candidate"): (
        ATOMIC_RESERVE,
        "Dispatches one candidate to the engine. The defibrillation daemon path "
        "passes reserve_budget=True so the shock reserves atomically via "
        "_reserve_execution_budget (P4-020); manual/explicit callers pass False and "
        "own their own accounting (operator-only).",
    ),
    # --- Daemon spender 2/5: boltz auto-cycle (T6) --------------------------
    ("capex_budget.py", "reserve_boltz_swap_budget", "reserve_spend"): (
        ATOMIC_RESERVE,
        "Boltz swap pre-create reserve (T6 auto-cycle + manual), category='boltz', "
        "effective_budget passed -> atomic cross-category rejection (P4-014/DD1).",
    ),
    ("boltz_manager.py", "loop_in", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "boltzcli 'createswap' (submarine/loop-in). The swap fee is reserved "
        "atomically by reserve_boltz_swap_budget BEFORE creation (P4-014) and "
        "settled loud/retry (P4-019); the CLI call itself commits the reserved cost.",
    ),
    ("boltz_manager.py", "_loop_out_locked", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "boltzcli 'createreverseswap' (loop-out, external-pay). Reserved atomically "
        "by reserve_boltz_swap_budget (P4-014), settled loud/retry (P4-019).",
    ),
    ("boltz_manager.py", "_build_args", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "Nested arg-builder for 'createreverseswap' inside _loop_out_locked; same "
        "boltz reserve/settle rail (P4-014/P4-019).",
    ),
    # --- Daemon spender 3/5: capacity-planner channel OPEN (T7) -------------
    ("capacity_planner.py", "_execute_open", "reserve_spend"): (
        ATOMIC_RESERVE,
        "Planner open reserve (T7), category='channel_open', effective_budget passed "
        "-> atomic cross-category rejection before fundchannel (P4-018/DD1).",
    ),
    ("capacity_planner.py", "_execute_open", "fundchannel_rpc"): (
        RAIL_COUNTED_COST,
        "The on-chain channel open. Its fee is reserved atomically by the "
        "reserve_spend in the same method (P4-018) and settled loud/retry + "
        "protected from the stale sweep (P4-021).",
    ),
    # --- Daemon spender 4/5: capacity-planner channel CLOSE (T7) ------------
    ("capacity_planner.py", "_execute_close", "reserve_spend"): (
        ATOMIC_RESERVE,
        "Planner close reserve (T7), category='channel_close', reserved BEFORE the "
        "on-chain close, effective_budget passed (P4-018/DD1).",
    ),
    ("capacity_planner.py", "_execute_close", "close_rpc"): (
        RAIL_COUNTED_COST,
        "The on-chain channel close. Fee reserved atomically by the reserve_spend in "
        "the same method (P4-018), settled loud/retry + sweep-protected (P4-021).",
    ),
    # --- Operator-only RPC spender (T0 dispatch; other plugins / operator) --
    ("cl-revenue-ops.py", "revenue_spend_reserve", "reserve_spend"): (
        OPERATOR_ONLY,
        "revenue-spend-reserve RPC (T0 pyln dispatch). Atomic (effective_budget "
        "passed, force_fresh gate, P2-011/DD1). Reachable only by an operator / "
        "sibling plugin, not an autonomous daemon.",
    ),
    # --- Pure RPC transport wrappers (no independent budget decision) -------
    ("capacity_planner.py", "_rpc_fundchannel", "fund_channel"): (
        NOT_A_SPEND,
        "Transport wrapper: forwards to data_service.fund_channel. Only "
        "daemon-reachable via _execute_open, which reserves atomically first (P4-018).",
    ),
    ("capacity_planner.py", "_rpc_fundchannel", "rpc_fundchannel"): (
        NOT_A_SPEND,
        "rpc.call('fundchannel') fallback inside the _rpc_fundchannel transport "
        "wrapper; reserving caller is _execute_open (P4-018).",
    ),
    ("capacity_planner.py", "_rpc_close", "close_channel"): (
        NOT_A_SPEND,
        "Transport wrapper: forwards to data_service.close_channel. Reserving caller "
        "is _execute_close (P4-018).",
    ),
    ("capacity_planner.py", "_rpc_close", "rpc_close"): (
        NOT_A_SPEND,
        "rpc.call('close') fallback inside the _rpc_close transport wrapper; "
        "reserving caller is _execute_close (P4-018).",
    ),
    ("data_service.py", "fund_channel", "rpc_fundchannel"): (
        NOT_A_SPEND,
        "DataService RPC transport (rpc.call('fundchannel')). Reached only via the "
        "planner open path, which reserves atomically first (P4-018).",
    ),
    ("data_service.py", "close_channel", "rpc_close"): (
        NOT_A_SPEND,
        "DataService RPC transport (rpc.call('close')). Reached only via the planner "
        "close path, which reserves atomically first (P4-018).",
    ),
}


# ---------------------------------------------------------------------------
# Guard tests.
# ---------------------------------------------------------------------------
def test_scanner_finds_the_known_five_daemon_spenders():
    """Sanity guard for the scanner itself: the five known autonomous daemon
    spenders (and the operator RPC reserve) must all be seen."""
    sites = collect_spend_sites()
    required = {
        ("rebalancer.py", "execute_rebalance", "reserve_budget"),                 # rebalance auto
        ("rebalance_engine_v2.py", "_reserve_execution_budget", "reserve_budget"),  # engine/defib reserve
        ("rebalancer.py", "_execute_candidate_v2", "execute_candidate"),          # defibrillation dispatch
        ("capex_budget.py", "reserve_boltz_swap_budget", "reserve_spend"),        # boltz
        ("capacity_planner.py", "_execute_open", "reserve_spend"),                # capex open
        ("capacity_planner.py", "_execute_close", "reserve_spend"),               # capex close
        ("cl-revenue-ops.py", "revenue_spend_reserve", "reserve_spend"),          # operator RPC
    }
    missing = required - sites
    assert not missing, f"scanner no longer sees known spend sites: {sorted(missing)}"


def test_every_money_committing_site_is_allowlisted():
    """The core invariant: a NEW or MOVED money-committing call site that is not
    in the allowlist fails the build (so the next refuter's 'one more spender'
    cannot ship silently). A stale allowlist entry also fails, keeping the
    allowlist an honest mirror of the code."""
    scanned = collect_spend_sites()
    allow = set(ALLOWLIST.keys())

    unaccounted = scanned - allow
    assert not unaccounted, (
        "NEW/moved money-committing spend site(s) not in ALLOWLIST -- classify "
        "each (atomic-reserve / rail-counted-cost / operator-only-non-daemon / "
        f"not-a-spend) and cite its coverage: {sorted(unaccounted)}"
    )

    stale = allow - scanned
    assert not stale, (
        "ALLOWLIST entries no longer present in the source (moved/removed) -- "
        f"update the allowlist so it stays an honest mirror: {sorted(stale)}"
    )


def test_no_allowlisted_spender_is_uncovered():
    """Every allowlisted site must carry a COVERED classification. A site marked
    with anything else (e.g. the 'daemon-reachable-uncovered' sentinel) fails
    here, forcing an operator ruling rather than shipping a budget hole."""
    uncovered = {
        site: klass
        for site, (klass, _why) in ALLOWLIST.items()
        if klass not in COVERED_CLASSES
    }
    assert not uncovered, (
        "money-committing site(s) classified uncovered -- escalate for an "
        f"operator ruling: {uncovered}"
    )
    # Every entry must also carry a non-empty justification.
    unjustified = [site for site, (_k, why) in ALLOWLIST.items() if not str(why).strip()]
    assert not unjustified, f"allowlist entries missing a justification: {unjustified}"


def test_the_test_detects_an_injected_new_spender():
    """test-the-test: an injected NEW spender (a rogue reserve_spend in a fresh
    function) is picked up by the scanner and, being absent from the allowlist,
    would fail test_every_money_committing_site_is_allowlisted."""
    rogue_source = (
        "class Rogue:\n"
        "    def _rogue_autonomous_spender(self, db, amt):\n"
        "        return db.reserve_spend(reservation_id='x', amount_sats=amt,\n"
        "                                category='rogue')\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_daemon.py", rogue_source)])
    injected = ("rogue_daemon.py", "_rogue_autonomous_spender", "reserve_spend")
    assert injected in sites, "scanner failed to detect an injected new spender"
    assert injected not in ALLOWLIST, (
        "sanity: the injected rogue spender must not already be allowlisted"
    )
    # Prove the guard assertion would fire on it.
    unaccounted = sites - set(ALLOWLIST.keys())
    assert injected in unaccounted, (
        "the enumeration guard would NOT trip on a new spender -- guard is toothless"
    )


def test_the_test_detects_an_injected_new_onchain_spender():
    """test-the-test (on-chain variant): an injected raw rpc.call('fundchannel')
    in a fresh function is also detected."""
    rogue_source = (
        "class RogueChain:\n"
        "    def _rogue_open(self, peer, amt):\n"
        "        return self.plugin.rpc.call('fundchannel', {'id': peer, 'amount': amt})\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_chain.py", rogue_source)])
    injected = ("rogue_chain.py", "_rogue_open", "rpc_fundchannel")
    assert injected in sites
    assert injected not in ALLOWLIST
