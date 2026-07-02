"""PERMANENT ENUMERATION GUARD: every autonomous money-committing site is
atomic-or-rail-counted, and a NEW or moved spender trips this test.

FOUR deep-audit refutation passes each found ONE MORE autonomous spender that
bypassed the atomic cross-category budget (boltz -> P4-014, capex -> P4-018,
defibrillation -> P4-020, chainswap -> P4-023). Pass 4 also proved the guard
ITSELF was unsound (P4-024): its markers were an allowlist of already-known
literals (createswap/createreverseswap; fundchannel/close only). It never
emitted a site for ``createchainswap`` and was structurally blind to
computed-method ``rpc.call(method, ...)`` (rebalance_native_executor_v2._rpc_call
dispatches ``sendpay`` through a variable) and to the rest of the money-RPC
family. That gives FALSE assurance of completeness — the exact "one more
spender" failure it was built to prevent.

This rewrite makes the scanner a GENUINE enumeration, not an allowlist of known
literals:

  (a) Boltz swap-create: ANY argv-list whose element 0 is a ``create*`` command
      string (createswap, createreverseswap, createchainswap, and any FUTURE
      create*), plus each swap method's ``_open_swap_budget_reservation`` call
      (the atomic pre-create reserve). createchainswap now produces a site.
  (b) RPC money calls: the FULL money-RPC method set via ``rpc.call("<method>")``
      (fundchannel/multifundchannel/close/splice_*/withdraw/txprepare/txsend/
      txdiscard/sendpay/sendonion/pay/keysend/fundpsbt/signpsbt/sendpsbt), AND
      computed-method ``rpc.call(method, ...)`` where the method is NOT a
      constant -> a "dynamic-method" site, so a variable-dispatched money RPC
      cannot hide.
  (c) Every emitted site MUST be in the ALLOWLIST with a coverage class in
      {atomic-reserve, rail-counted-cost, operator-only-non-daemon, not-a-spend,
      dynamic-method-reviewed}; the guard FAILs on any unaccounted site, any
      daemon-reachable-uncovered site, and any dynamic-method site not
      explicitly reviewed.

It FAILS if:
  * the scan finds a money-committing call NOT in the allowlist (a new or moved
    spender), or a stale allowlist entry no longer in the source, or
  * any allowlisted site carries a class OUTSIDE the covered set, or
  * a dynamic-method site is not classified ``dynamic-method-reviewed``.

The test-the-test injects a rogue createchainswap, a rogue rpc.call('sendpay'),
and a rogue rpc.call(computed_method,...) and proves the scanner emits a site
and the guard trips for each — if the scanner cannot see one, the guard is still
unsound.

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
# Callee-name (func.attr / func.id) markers: a call to one of these names is a
# reservation / execution / RPC-wrapper commit site.
_CALLEE_MARKERS = {
    "reserve_spend": "reserve_spend",                  # generic ledger atomic reserve
    "reserve_budget": "reserve_budget",                # rebalance atomic reserve
    "execute_candidate": "execute_candidate",          # rebalance execution dispatch
    "_open_swap_budget_reservation": "boltz_swap_reserve",  # boltz atomic pre-create reserve
    "_rpc_fundchannel": "fundchannel_rpc",             # planner channel-open wrapper call
    "fund_channel": "fund_channel",                    # data_service fundchannel wrapper
    "_rpc_close": "close_rpc",                          # planner channel-close wrapper call
    "close_channel": "close_channel",                  # data_service close wrapper
}

# The FULL money-committing RPC method set. Any ``rpc.call("<method>", ...)`` with
# one of these CONSTANT method names is an on-chain / payment commit; the marker
# is "rpc_<method>". A NEW money RPC added to this set (or a new call site of an
# existing one) surfaces automatically.
_MONEY_RPC_METHODS = frozenset({
    "fundchannel", "multifundchannel", "close",
    "splice_init", "splice_update", "splice_signed",
    "withdraw", "txprepare", "txsend", "txdiscard",
    "sendpay", "sendonion", "pay", "keysend",
    "fundpsbt", "signpsbt", "sendpsbt",
})

# Marker emitted for a computed-method ``rpc.call(method, ...)`` where args[0] is
# NOT a constant string. A variable-dispatched money RPC (e.g.
# rebalance_native_executor_v2._rpc_call passing "sendpay") would otherwise be
# invisible; this forces an explicit dynamic-method-reviewed classification.
_DYNAMIC_METHOD_MARKER = "dynamic-method-rpc"

# Marker for a boltz swap-create argv list (element 0 is a create* command).
_BOLTZ_CREATE_MARKER = "boltz_swap_create"


def _callee_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _first_elt_create_command(node: ast.AST):
    """If ``node`` is a list literal whose first element is a string constant
    beginning with 'create' (a boltzcli swap-create command), return that
    command string; else None. This matches createswap, createreverseswap,
    createchainswap, AND any future create* without an allowlist of literals."""
    if isinstance(node, ast.List) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value.startswith("create"):
                return first.value
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
            if cn == "call":
                # ``<rpc>.call(<method>, ...)`` — money-RPC (constant) or
                # dynamic-method (computed) dispatch.
                a0 = node.args[0] if node.args else None
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    if a0.value in _MONEY_RPC_METHODS:
                        sites.add((fname, encl(node), f"rpc_{a0.value}"))
                elif a0 is not None:
                    # First arg present but NOT a constant string -> the method
                    # is computed at runtime; a money RPC could hide here.
                    sites.add((fname, encl(node), _DYNAMIC_METHOD_MARKER))
        # Boltz swap-create argv lists (element 0 is a create* command).
        cmd = _first_elt_create_command(node)
        if cmd is not None:
            sites.add((fname, encl(node), _BOLTZ_CREATE_MARKER))


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
DYNAMIC_METHOD_REVIEWED = "dynamic-method-reviewed"  # computed-method rpc.call, reviewed + classified

COVERED_CLASSES = frozenset({
    ATOMIC_RESERVE, RAIL_COUNTED_COST, OPERATOR_ONLY, NOT_A_SPEND,
    DYNAMIC_METHOD_REVIEWED,
})

# Explicit allowlist of EVERY known money-committing site with its coverage
# classification and a justification (finding citation). Keyed by
# (filename, enclosing_function, marker).
ALLOWLIST = {
    # --- Daemon spender 1/6: rebalance auto-cycle (T3) ----------------------
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
    # --- Daemon spender 5/6: defibrillation diagnostic shock (T7) -----------
    ("rebalancer.py", "_execute_candidate_v2", "execute_candidate"): (
        ATOMIC_RESERVE,
        "Dispatches one candidate to the engine. The defibrillation daemon path "
        "passes reserve_budget=True so the shock reserves atomically via "
        "_reserve_execution_budget (P4-020), and account_costs=True so the fee is "
        "recorded before the reservation is marked spent (P4-025); manual/explicit "
        "callers pass False and own their own accounting (operator-only).",
    ),
    # --- Daemon spender 2/6: boltz auto-cycle (T6) --------------------------
    ("capex_budget.py", "reserve_boltz_swap_budget", "reserve_spend"): (
        ATOMIC_RESERVE,
        "Boltz swap pre-create reserve (T6 auto-cycle + manual), category='boltz', "
        "effective_budget passed -> atomic cross-category rejection (P4-014/DD1). "
        "Reached from every boltz swap-create path incl. chainswap (P4-023).",
    ),
    # boltz atomic pre-create reserves (loop_in / loop_out / chainswap) -------
    ("boltz_manager.py", "loop_in", "boltz_swap_reserve"): (
        ATOMIC_RESERVE,
        "loop_in (createswap) reserves the swap fee atomically via "
        "_open_swap_budget_reservation BEFORE creation (P4-014/DD1).",
    ),
    ("boltz_manager.py", "_loop_out_locked", "boltz_swap_reserve"): (
        ATOMIC_RESERVE,
        "loop_out (createreverseswap) reserves the swap fee atomically via "
        "_open_swap_budget_reservation BEFORE creation (P4-014/DD1).",
    ),
    ("boltz_manager.py", "chainswap", "boltz_swap_reserve"): (
        ATOMIC_RESERVE,
        "chainswap (createchainswap) NEW atomic pre-create reserve via "
        "_open_swap_budget_reservation BEFORE createchainswap; rejects the swap "
        "when the unified budget would be exceeded (P4-023/DD1).",
    ),
    # boltz swap-create argv lists (the committing CLI action) ----------------
    ("boltz_manager.py", "loop_in", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "boltzcli 'createswap' (submarine/loop-in). Reserved atomically by "
        "_open_swap_budget_reservation BEFORE creation (P4-014), settled "
        "loud/retry (P4-019); the CLI call itself commits the reserved cost.",
    ),
    ("boltz_manager.py", "_loop_out_locked", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "boltzcli 'createreverseswap --external-pay' (loop-out). Reserved "
        "atomically (P4-014), settled loud/retry (P4-019).",
    ),
    ("boltz_manager.py", "_build_args", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "Nested arg-builder for 'createreverseswap' inside _loop_out_locked; same "
        "boltz reserve/settle rail (P4-014/P4-019).",
    ),
    ("boltz_manager.py", "chainswap", "boltz_swap_create"): (
        RAIL_COUNTED_COST,
        "boltzcli 'createchainswap' (the 6th swap-create). Now reserved "
        "atomically by _open_swap_budget_reservation BEFORE creation and settled "
        "loud/retry on the boltz rail (P4-023/DD1).",
    ),
    # boltz loop-out first-hop invoice payment (rail-counted) -----------------
    ("boltz_manager.py", "_pay_invoice_via_first_hop", "rpc_pay"): (
        RAIL_COUNTED_COST,
        "rpc.call('pay') fallback that settles the reverse-swap invoice inside "
        "_loop_out_locked. The swap fee is reserved atomically by "
        "_open_swap_budget_reservation BEFORE the swap is created (P4-014); this "
        "pay commits the reserved cost.",
    ),
    # --- Daemon spender 3/6: capacity-planner channel OPEN (T7) -------------
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
    # --- Daemon spender 4/6: capacity-planner channel CLOSE (T7) ------------
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
    ("data_service.py", "pay", "rpc_pay"): (
        NOT_A_SPEND,
        "DataService RPC transport (rpc.call('pay')). Only production caller is the "
        "boltz loop-out first-hop pay, whose swap fee is reserved atomically on the "
        "boltz rail first (P4-014).",
    ),
    # --- Dynamic-method rpc.call sites (computed method) --------------------
    ("rebalance_native_executor_v2.py", "_rpc_call", DYNAMIC_METHOD_REVIEWED): (
        DYNAMIC_METHOD_REVIEWED,
        "NativeRebalanceExecutor._rpc_call dispatches a computed method (incl. the "
        "money RPC 'sendpay'/'sendonion') via rpc.call(method, ...). Rail-counted: "
        "the rebalance engine reserves the fee cap atomically (reserve_budget inside "
        "_reserve_budget_atomic's BEGIN IMMEDIATE, P4-016/020) BEFORE the executor "
        "runs, and records/settles the actual fee (record_rebalance_cost / "
        "mark_budget_spent, P4-025). The variable method cannot hide a NEW money "
        "RPC because the reserving caller owns the whole reserve+pay window.",
    ),
    ("cl-revenue-ops.py", "_run", DYNAMIC_METHOD_REVIEWED): (
        DYNAMIC_METHOD_REVIEWED,
        "ThreadSafeRpcProxy.fire_and_forget._run is the generic async RPC "
        "transport (rpc.call(method_name, ...)). Not a spend decision: the money "
        "decision and its atomic reservation are owned by the calling site that "
        "names the concrete method; this proxy only forwards it off-thread.",
    ),
}


# ---------------------------------------------------------------------------
# Guard tests.
# ---------------------------------------------------------------------------
def test_scanner_finds_the_known_six_daemon_spenders():
    """Sanity guard for the scanner itself: the six known autonomous daemon
    spenders (and the operator RPC reserve) must all be seen, including the
    chainswap atomic reserve + createchainswap surfaced in pass 4."""
    sites = collect_spend_sites()
    required = {
        ("rebalancer.py", "execute_rebalance", "reserve_budget"),                 # rebalance auto
        ("rebalance_engine_v2.py", "_reserve_execution_budget", "reserve_budget"),  # engine/defib reserve
        ("rebalancer.py", "_execute_candidate_v2", "execute_candidate"),          # defibrillation dispatch
        ("capex_budget.py", "reserve_boltz_swap_budget", "reserve_spend"),        # boltz
        ("capacity_planner.py", "_execute_open", "reserve_spend"),                # capex open
        ("capacity_planner.py", "_execute_close", "reserve_spend"),               # capex close
        ("cl-revenue-ops.py", "revenue_spend_reserve", "reserve_spend"),          # operator RPC
        # pass-4 surfaced sites:
        ("boltz_manager.py", "chainswap", "boltz_swap_create"),                   # createchainswap
        ("boltz_manager.py", "chainswap", "boltz_swap_reserve"),                  # chainswap atomic reserve
        ("rebalance_native_executor_v2.py", "_rpc_call", _DYNAMIC_METHOD_MARKER),  # computed sendpay
    }
    missing = required - sites
    assert not missing, f"scanner no longer sees known spend sites: {sorted(missing)}"


def test_every_money_committing_site_is_allowlisted():
    """The core invariant: a NEW or MOVED money-committing call site that is not
    in the allowlist fails the build (so the next refuter's 'one more spender'
    cannot ship silently). A stale allowlist entry also fails, keeping the
    allowlist an honest mirror of the code.

    Dynamic-method sites are normalised: the scanner emits the raw
    ``dynamic-method-rpc`` marker; the allowlist keys them under
    ``dynamic-method-reviewed`` in the same (file, func)."""
    scanned = _normalise_dynamic(collect_spend_sites())
    allow = set(ALLOWLIST.keys())

    unaccounted = scanned - allow
    assert not unaccounted, (
        "NEW/moved money-committing spend site(s) not in ALLOWLIST -- classify "
        "each (atomic-reserve / rail-counted-cost / operator-only-non-daemon / "
        "not-a-spend / dynamic-method-reviewed) and cite its coverage: "
        f"{sorted(unaccounted)}"
    )

    stale = allow - scanned
    assert not stale, (
        "ALLOWLIST entries no longer present in the source (moved/removed) -- "
        f"update the allowlist so it stays an honest mirror: {sorted(stale)}"
    )


def _normalise_dynamic(sites):
    """Map each raw dynamic-method-rpc site to its reviewed allowlist key so the
    single allowlist can carry the justification once."""
    out = set()
    for f, fn, m in sites:
        if m == _DYNAMIC_METHOD_MARKER:
            out.add((f, fn, DYNAMIC_METHOD_REVIEWED))
        else:
            out.add((f, fn, m))
    return out


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


def test_every_dynamic_method_site_is_explicitly_reviewed():
    """Every emitted dynamic-method rpc.call site (computed method) MUST be
    classified 'dynamic-method-reviewed' -- a variable-dispatched money RPC that
    is merely allowlisted under some other class (or missing) fails here, so it
    can never hide."""
    dynamic_sites = {s for s in collect_spend_sites() if s[2] == _DYNAMIC_METHOD_MARKER}
    assert dynamic_sites, "scanner emitted NO dynamic-method sites -- it is blind"
    for site in dynamic_sites:
        key = (site[0], site[1], DYNAMIC_METHOD_REVIEWED)
        assert key in ALLOWLIST, (
            f"dynamic-method rpc.call site not reviewed: {site} -- a computed "
            "money RPC must be explicitly classified dynamic-method-reviewed"
        )
        klass, _why = ALLOWLIST[key]
        assert klass == DYNAMIC_METHOD_REVIEWED, (
            f"dynamic-method site {key} classified {klass!r}, must be "
            "dynamic-method-reviewed"
        )


def test_scanner_marker_and_allowlist_marker_agree_for_dynamic_sites():
    """The scanner emits _DYNAMIC_METHOD_MARKER; the allowlist keys such sites
    under DYNAMIC_METHOD_REVIEWED. Prove the (file, func) pairs line up so no
    dynamic site is left unaccounted after normalisation."""
    scanned = collect_spend_sites()
    scanned_dyn = {(f, fn) for (f, fn, m) in scanned if m == _DYNAMIC_METHOD_MARKER}
    allow_dyn = {(f, fn) for (f, fn, m) in ALLOWLIST if m == DYNAMIC_METHOD_REVIEWED}
    assert scanned_dyn == allow_dyn, (
        f"dynamic-method site mismatch: scanned-only={sorted(scanned_dyn - allow_dyn)} "
        f"allow-only={sorted(allow_dyn - scanned_dyn)}"
    )


# ---------------------------------------------------------------------------
# TEST-THE-TEST: an unsound scanner would silently miss these. Each injection
# MUST produce a site AND trip the guard (be unaccounted).
# ---------------------------------------------------------------------------
def _guard_would_trip(sites):
    """Helper: True iff the enumeration guard would fail on ``sites``."""
    return bool(_normalise_dynamic(sites) - set(ALLOWLIST.keys()))


def test_the_test_detects_an_injected_rogue_chainswap():
    """(i) A rogue createchainswap (the pass-4 miss) is emitted and trips the
    guard -- the exact spender the OLD literal allowlist was blind to."""
    rogue = (
        "class RogueChain:\n"
        "    def _rogue_chainswap(self, amt):\n"
        "        args = ['createchainswap', '--json', '--', str(amt)]\n"
        "        return self._run_json(args)\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_chain.py", rogue)])
    injected = ("rogue_chain.py", "_rogue_chainswap", _BOLTZ_CREATE_MARKER)
    assert injected in sites, "scanner is blind to an injected createchainswap"
    assert injected not in ALLOWLIST
    assert _guard_would_trip({injected}), "guard would NOT trip on a rogue chainswap"


def test_the_test_detects_an_injected_rogue_sendpay():
    """(ii) A rogue rpc.call('sendpay', ...) -- a money RPC the OLD scanner
    (fundchannel/close only) could not see -- is emitted and trips the guard."""
    rogue = (
        "class RoguePay:\n"
        "    def _rogue_sendpay(self, route, ph):\n"
        "        return self.plugin.rpc.call('sendpay', {'route': route, 'payment_hash': ph})\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_pay.py", rogue)])
    injected = ("rogue_pay.py", "_rogue_sendpay", "rpc_sendpay")
    assert injected in sites, "scanner is blind to an injected rpc.call('sendpay')"
    assert injected not in ALLOWLIST
    assert _guard_would_trip({injected}), "guard would NOT trip on a rogue sendpay"


def test_the_test_detects_an_injected_rogue_dynamic_method():
    """(iii) A rogue rpc.call(computed_method, ...) -- variable dispatch, the
    exact way rebalance_native_executor hides sendpay -- is emitted as a
    dynamic-method site and trips the guard."""
    rogue = (
        "class RogueDyn:\n"
        "    def _rogue_dispatch(self, method, params):\n"
        "        return self.plugin.rpc.call(method, params)\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_dyn.py", rogue)])
    injected = ("rogue_dyn.py", "_rogue_dispatch", _DYNAMIC_METHOD_MARKER)
    assert injected in sites, (
        "scanner is blind to a computed-method rpc.call -- a variable-dispatched "
        "money RPC could hide (guard still unsound)"
    )
    # After normalisation the injected dynamic site is unaccounted -> trips.
    assert ("rogue_dyn.py", "_rogue_dispatch", DYNAMIC_METHOD_REVIEWED) not in ALLOWLIST
    assert _guard_would_trip({injected}), "guard would NOT trip on a rogue dynamic dispatch"


def test_the_test_detects_an_injected_new_reserve_spender():
    """test-the-test: an injected NEW reserve_spend spender in a fresh function
    is picked up and, being absent from the allowlist, trips the guard."""
    rogue = (
        "class Rogue:\n"
        "    def _rogue_autonomous_spender(self, db, amt):\n"
        "        return db.reserve_spend(reservation_id='x', amount_sats=amt,\n"
        "                                category='rogue')\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_daemon.py", rogue)])
    injected = ("rogue_daemon.py", "_rogue_autonomous_spender", "reserve_spend")
    assert injected in sites, "scanner failed to detect an injected new spender"
    assert injected not in ALLOWLIST
    assert _guard_would_trip({injected}), "the enumeration guard is toothless on a new spender"
