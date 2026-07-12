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
family. That gives FALSE assurance of completeness -- the exact "one more
spender" failure it was built to prevent.

Two further gaps were then closed to make this a TRULY complete mirror:

  * P4-026 (attribute-style RPC blindness): the scanner modelled
    ``rpc.call("method")`` and ``rpc.call(computed_var)`` but NOT pyln ATTRIBUTE
    dispatch ``rpc.<method>(...)``. The real instance is
    data_service.send_pay -> ``self._plugin.rpc.sendpay(route, payment_hash)``
    (modules/data_service.py). A money RPC issued as an attribute call was
    invisible. The scanner now flags ``<rpc-handle>.<money-method>(...)`` for the
    full money-RPC method set (marker ``rpc_attr_<method>``). The receiver is
    required to be an rpc handle (a name ``rpc`` or an attribute chain ending in
    ``.rpc`` -- so ``rpc.sendpay``, ``self.rpc.sendpay`` and
    ``self._plugin.rpc.sendpay`` all trip) so the detector models pyln dispatch
    rather than colliding with unrelated same-named methods (a SQLite
    ``conn.close()`` or a wrapper ``data_service.pay(...)`` is NOT an rpc money
    call and must not pollute the mirror).

  * P4-027 (set-collapse granularity): sites used to be keyed on
    (filename, enclosing_function, marker) as a SET, so a SECOND same-marker
    money call inside an allowlisted function was silently absorbed. Sites are
    now COUNTED per (filename, enclosing_function, marker) and the allowlist
    declares the expected count, so an ADDED same-marker call surfaces as a
    count mismatch and trips the guard.

This makes the scanner a GENUINE enumeration, not an allowlist of known
literals:

  (a) Boltz swap-create: ANY argv-list whose element 0 is a ``create*`` command
      string (createswap, createreverseswap, createchainswap, and any FUTURE
      create*), plus each swap method's ``_open_swap_budget_reservation`` call
      (the atomic pre-create reserve). createchainswap now produces a site.
  (b) RPC money calls: the FULL money-RPC method set via ``rpc.call("<method>")``
      (fundchannel/multifundchannel/close/splice_*/withdraw/txprepare/txsend/
      txdiscard/sendpay/sendonion/pay/keysend/fundpsbt/signpsbt/sendpsbt), AND
      the SAME method set issued as pyln attribute dispatch
      ``<rpc-handle>.<method>(...)`` (marker ``rpc_attr_<method>``), AND
      computed-method ``rpc.call(method, ...)`` where the method is NOT a
      constant -> a "dynamic-method" site, so a variable-dispatched money RPC
      cannot hide.
  (c) Every emitted site MUST be in the ALLOWLIST with a coverage class in
      {atomic-reserve, rail-counted-cost, operator-only-non-daemon, not-a-spend,
      dynamic-method-reviewed} AND with the correct occurrence count; the guard
      FAILs on any unaccounted site, any count mismatch, any
      daemon-reachable-uncovered site, and any dynamic-method site not
      explicitly reviewed.

It FAILS if:
  * the scan finds a money-committing call NOT in the allowlist (a new or moved
    spender), or a stale allowlist entry no longer in the source, or an added
    same-marker call (declared-count mismatch), or
  * any allowlisted site carries a class OUTSIDE the covered set, or
  * a dynamic-method site is not classified ``dynamic-method-reviewed``.

The test-the-test injects a rogue createchainswap, a rogue rpc.call('sendpay'),
a rogue rpc.call(computed_method,...), a rogue ATTRIBUTE-form
``self.rpc.sendpay(...)``, and a DUPLICATE same-marker call inside an already
allowlisted function, and proves the scanner emits a site and the guard trips
for each -- if the scanner cannot see one, the guard is still unsound.

Modelled on the RA2-1 skip-reason drift guard
(tests/test_rebalance_audit_v2.py / tests/test_architecture_guard.py).
"""

import ast
import os
import sys
from collections import Counter
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
# one of these CONSTANT method names is an on-chain / payment commit (marker
# ``rpc_<method>``); the SAME method issued as pyln attribute dispatch
# ``<rpc-handle>.<method>(...)`` is a commit too (marker ``rpc_attr_<method>``).
# A NEW money RPC added to this set (or a new call site of an existing one)
# surfaces automatically.
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


def _is_rpc_handle(node: ast.AST) -> bool:
    """True iff ``node`` is a pyln rpc handle expression: the bare name ``rpc``
    or any attribute chain whose final component is ``.rpc``. This matches
    ``rpc``, ``self.rpc``, ``self._plugin.rpc``, ``plugin.rpc`` -- i.e. the
    receiver of a ``rpc.<method>(...)`` pyln attribute dispatch -- while
    deliberately NOT matching unrelated receivers that merely expose a
    same-named method (``conn.close()``, ``self.data_service.pay(...)``,
    ``_require_boltz_manager().withdraw(...)`` are not rpc money calls)."""
    if isinstance(node, ast.Name):
        return node.id == "rpc"
    if isinstance(node, ast.Attribute):
        return node.attr == "rpc"
    return False


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


def _scan_tree(fname: str, tree: ast.AST, counts: Counter):
    """Increment ``counts`` (keyed (filename, enclosing_function, marker)) once
    per money-committing call site in ``tree``. COUNTED, not set-collapsed, so a
    second same-marker call in the same function surfaces (P4-027)."""
    encl = _enclosing_func_resolver(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            cn = _callee_name(node)
            if cn in _CALLEE_MARKERS:
                counts[(fname, encl(node), _CALLEE_MARKERS[cn])] += 1
            # Pyln attribute dispatch: ``<rpc-handle>.<money-method>(...)`` (e.g.
            # self._plugin.rpc.sendpay(route, hash)). Receiver must be an rpc
            # handle so we model pyln dispatch, not same-named methods on other
            # objects (P4-026).
            f = node.func
            if (isinstance(f, ast.Attribute)
                    and f.attr in _MONEY_RPC_METHODS
                    and _is_rpc_handle(f.value)):
                counts[(fname, encl(node), f"rpc_attr_{f.attr}")] += 1
            if cn == "call":
                # ``<rpc>.call(<method>, ...)`` -- money-RPC (constant) or
                # dynamic-method (computed) dispatch.
                a0 = node.args[0] if node.args else None
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    if a0.value in _MONEY_RPC_METHODS:
                        counts[(fname, encl(node), f"rpc_{a0.value}")] += 1
                elif a0 is not None:
                    # First arg present but NOT a constant string -> the method
                    # is computed at runtime; a money RPC could hide here.
                    counts[(fname, encl(node), _DYNAMIC_METHOD_MARKER)] += 1
        # Boltz swap-create argv lists (element 0 is a create* command).
        cmd = _first_elt_create_command(node)
        if cmd is not None:
            counts[(fname, encl(node), _BOLTZ_CREATE_MARKER)] += 1


def collect_spend_site_counts(extra_sources=None) -> Counter:
    """Return a Counter of (filename, enclosing_function, marker) -> occurrence
    count for every money-committing call site in production source.

    ``extra_sources``: optional list of (filename, source) pairs, used by the
    test-the-test to prove an injected new/duplicate spender is detected.
    Injected counts are ADDED to the real-tree counts for the same key, so a
    duplicate same-marker call in an existing allowlisted function surfaces as a
    count mismatch."""
    counts: Counter = Counter()
    files = sorted(MODULES_DIR.glob("*.py")) + [MAIN_PLUGIN]
    for path in files:
        _scan_tree(path.name, ast.parse(path.read_text(), filename=path.name), counts)
    for fname, source in (extra_sources or []):
        _scan_tree(fname, ast.parse(source, filename=fname), counts)
    return counts


def collect_spend_sites(extra_sources=None) -> set:
    """Return the SET of (filename, enclosing_function, marker) keys for every
    money-committing call site (identity only, counts collapsed)."""
    return set(collect_spend_site_counts(extra_sources).keys())


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
# classification, its expected occurrence COUNT, and a justification (finding
# citation). Keyed by (filename, enclosing_function, marker); value is
# (coverage_class, expected_count, justification). A SECOND same-marker call in
# any of these functions raises the scanned count above the declared count and
# trips the guard (P4-027).
ALLOWLIST = {
    # --- Daemon spender 1/6: rebalance auto-cycle (T3) ----------------------
    ("rebalancer.py", "execute_rebalance", "reserve_budget"): (
        ATOMIC_RESERVE, 1,
        "EVRebalancer auto-cycle (T3) reserves the fee cap via reserve_budget -> "
        "_reserve_budget_atomic BEGIN IMMEDIATE, full effective budget (P4-016/017/DD1).",
    ),
    ("rebalance_engine_v2.py", "_reserve_execution_budget", "reserve_budget"): (
        ATOMIC_RESERVE, 1,
        "Shared v2 engine reserve for the auto cycle (T3) AND the defibrillation "
        "diagnostic shock (T7, via execute_candidate(reserve_budget=True), P4-020); "
        "full effective budget inside BEGIN IMMEDIATE (P4-016).",
    ),
    ("rebalance_engine_v2.py", "_reserve_delegate", "reserve_budget"): (
        ATOMIC_RESERVE, 1,
        "Phase 2D governed entry (2026-07-12): GovernorFacade delegate closure "
        "inside _governed_reserve_execution_budget — the SAME atomic "
        "reserve_budget call as the legacy site above, reserved under the "
        "caller's reservation_id; active only when "
        "econ_governor_rebalance_enabled is true.",
    ),
    ("capacity_planner.py", "_planner_reserve_delegate", "reserve_spend"): (
        ATOMIC_RESERVE, 1,
        "Phase 2E governed entry (2026-07-12): GovernorFacade delegate closure "
        "inside _governed_reserve_spend — the SAME atomic reserve_spend call "
        "as the legacy _execute_open/_execute_close sites, same "
        "reservation_id/kwargs; active only when "
        "econ_governor_planner_enabled is true.",
    ),
    # --- Daemon spender 5/6: defibrillation diagnostic shock (T7) -----------
    ("rebalancer.py", "_execute_candidate_v2", "execute_candidate"): (
        ATOMIC_RESERVE, 1,
        "Dispatches one candidate to the engine. The defibrillation daemon path "
        "passes reserve_budget=True so the shock reserves atomically via "
        "_reserve_execution_budget (P4-020), and account_costs=True so the fee is "
        "recorded before the reservation is marked spent (P4-025); manual/explicit "
        "callers pass False and own their own accounting (operator-only).",
    ),
    # --- Daemon spender 2/6: boltz auto-cycle (T6) --------------------------
    ("capex_budget.py", "reserve_boltz_swap_budget", "reserve_spend"): (
        ATOMIC_RESERVE, 1,
        "Boltz swap pre-create reserve (T6 auto-cycle + manual), category='boltz', "
        "effective_budget passed -> atomic cross-category rejection (P4-014/DD1). "
        "Reached from every boltz swap-create path incl. chainswap (P4-023).",
    ),
    # boltz atomic pre-create reserves (loop_in / loop_out / chainswap) -------
    ("boltz_manager.py", "loop_in", "boltz_swap_reserve"): (
        ATOMIC_RESERVE, 1,
        "loop_in (createswap) reserves the swap fee atomically via "
        "_open_swap_budget_reservation BEFORE creation (P4-014/DD1).",
    ),
    ("boltz_manager.py", "_loop_out_locked", "boltz_swap_reserve"): (
        ATOMIC_RESERVE, 1,
        "loop_out (createreverseswap) reserves the swap fee atomically via "
        "_open_swap_budget_reservation BEFORE creation (P4-014/DD1).",
    ),
    ("boltz_manager.py", "chainswap", "boltz_swap_reserve"): (
        ATOMIC_RESERVE, 1,
        "chainswap (createchainswap) NEW atomic pre-create reserve via "
        "_open_swap_budget_reservation BEFORE createchainswap; rejects the swap "
        "when the unified budget would be exceeded (P4-023/DD1).",
    ),
    # boltz swap-create argv lists (the committing CLI action) ----------------
    ("boltz_manager.py", "loop_in", "boltz_swap_create"): (
        RAIL_COUNTED_COST, 1,
        "boltzcli 'createswap' (submarine/loop-in). Reserved atomically by "
        "_open_swap_budget_reservation BEFORE creation (P4-014), settled "
        "loud/retry (P4-019); the CLI call itself commits the reserved cost.",
    ),
    ("boltz_manager.py", "_loop_out_locked", "boltz_swap_create"): (
        RAIL_COUNTED_COST, 1,
        "boltzcli 'createreverseswap --external-pay' (loop-out). Reserved "
        "atomically (P4-014), settled loud/retry (P4-019).",
    ),
    ("boltz_manager.py", "_build_args", "boltz_swap_create"): (
        RAIL_COUNTED_COST, 1,
        "Nested arg-builder for 'createreverseswap' inside _loop_out_locked; same "
        "boltz reserve/settle rail (P4-014/P4-019).",
    ),
    ("boltz_manager.py", "chainswap", "boltz_swap_create"): (
        RAIL_COUNTED_COST, 1,
        "boltzcli 'createchainswap' (the 6th swap-create). Now reserved "
        "atomically by _open_swap_budget_reservation BEFORE creation and settled "
        "loud/retry on the boltz rail (P4-023/DD1).",
    ),
    # boltz loop-out first-hop invoice payment (rail-counted) -----------------
    ("boltz_manager.py", "_pay_invoice_via_first_hop", "rpc_pay"): (
        RAIL_COUNTED_COST, 1,
        "rpc.call('pay') fallback that settles the reverse-swap invoice inside "
        "_loop_out_locked. The swap fee is reserved atomically by "
        "_open_swap_budget_reservation BEFORE the swap is created (P4-014); this "
        "pay commits the reserved cost.",
    ),
    # --- Daemon spender 3/6: capacity-planner channel OPEN (T7) -------------
    ("capacity_planner.py", "_execute_open", "reserve_spend"): (
        ATOMIC_RESERVE, 1,
        "Planner open reserve (T7), category='channel_open', effective_budget passed "
        "-> atomic cross-category rejection before fundchannel (P4-018/DD1).",
    ),
    ("capacity_planner.py", "_execute_open", "fundchannel_rpc"): (
        RAIL_COUNTED_COST, 1,
        "The on-chain channel open. Its fee is reserved atomically by the "
        "reserve_spend in the same method (P4-018) and settled loud/retry + "
        "protected from the stale sweep (P4-021).",
    ),
    # --- Daemon spender 4/6: capacity-planner channel CLOSE (T7) ------------
    ("capacity_planner.py", "_execute_close", "reserve_spend"): (
        ATOMIC_RESERVE, 1,
        "Planner close reserve (T7), category='channel_close', reserved BEFORE the "
        "on-chain close, effective_budget passed (P4-018/DD1).",
    ),
    ("capacity_planner.py", "_execute_close", "close_rpc"): (
        RAIL_COUNTED_COST, 1,
        "The on-chain channel close. Fee reserved atomically by the reserve_spend in "
        "the same method (P4-018), settled loud/retry + sweep-protected (P4-021).",
    ),
    # --- Operator-only RPC spender (T0 dispatch; other plugins / operator) --
    ("cl-revenue-ops.py", "revenue_spend_reserve", "reserve_spend"): (
        OPERATOR_ONLY, 1,
        "revenue-spend-reserve RPC (T0 pyln dispatch). Atomic (effective_budget "
        "passed, force_fresh gate, P2-011/DD1). Reachable only by an operator / "
        "sibling plugin, not an autonomous daemon.",
    ),
    # --- Pure RPC transport wrappers (no independent budget decision) -------
    ("capacity_planner.py", "_rpc_fundchannel", "fund_channel"): (
        NOT_A_SPEND, 1,
        "Transport wrapper: forwards to data_service.fund_channel. Only "
        "daemon-reachable via _execute_open, which reserves atomically first (P4-018).",
    ),
    ("capacity_planner.py", "_rpc_fundchannel", "rpc_fundchannel"): (
        NOT_A_SPEND, 1,
        "rpc.call('fundchannel') fallback inside the _rpc_fundchannel transport "
        "wrapper; reserving caller is _execute_open (P4-018).",
    ),
    ("capacity_planner.py", "_rpc_close", "close_channel"): (
        NOT_A_SPEND, 1,
        "Transport wrapper: forwards to data_service.close_channel. Reserving caller "
        "is _execute_close (P4-018).",
    ),
    ("capacity_planner.py", "_rpc_close", "rpc_close"): (
        NOT_A_SPEND, 1,
        "rpc.call('close') fallback inside the _rpc_close transport wrapper; "
        "reserving caller is _execute_close (P4-018).",
    ),
    ("data_service.py", "fund_channel", "rpc_fundchannel"): (
        NOT_A_SPEND, 1,
        "DataService RPC transport (rpc.call('fundchannel')). Reached only via the "
        "planner open path, which reserves atomically first (P4-018).",
    ),
    ("data_service.py", "close_channel", "rpc_close"): (
        NOT_A_SPEND, 1,
        "DataService RPC transport (rpc.call('close')). Reached only via the planner "
        "close path, which reserves atomically first (P4-018).",
    ),
    ("data_service.py", "pay", "rpc_pay"): (
        NOT_A_SPEND, 1,
        "DataService RPC transport (rpc.call('pay')). Only production caller is the "
        "boltz loop-out first-hop pay, whose swap fee is reserved atomically on the "
        "boltz rail first (P4-014).",
    ),
    # --- Attribute-style money RPC dispatch (rpc.<method>(...), P4-026) -----
    ("data_service.py", "send_pay", "rpc_attr_sendpay"): (
        NOT_A_SPEND, 1,
        "DataService.send_pay is the pyln attribute-dispatch transport "
        "self._plugin.rpc.sendpay(route, payment_hash) -- surfaced by the P4-026 "
        "attribute-dispatch detector. Not a spend decision and has NO live daemon "
        "caller today: the autonomous rebalance payment goes through "
        "rebalance_native_executor_v2._rpc_call (the covered dynamic-method site, "
        "which the engine reserves atomically before, P4-016/020/025). If a daemon "
        "path ever routes a payment through this wrapper it must reserve on a rail "
        "first exactly like the native executor does.",
    ),
    # --- Dynamic-method rpc.call sites (computed method) --------------------
    ("rebalance_native_executor_v2.py", "_rpc_call", DYNAMIC_METHOD_REVIEWED): (
        DYNAMIC_METHOD_REVIEWED, 3,
        "NativeRebalanceExecutor._rpc_call dispatches a computed method (incl. the "
        "money RPC 'sendpay'/'sendonion') via rpc.call(method, ...); the three "
        "occurrences are the dispatch/fallback paths in this one wrapper. "
        "Rail-counted: the rebalance engine reserves the fee cap atomically "
        "(reserve_budget inside _reserve_budget_atomic's BEGIN IMMEDIATE, "
        "P4-016/020) BEFORE the executor runs, and records/settles the actual fee "
        "(record_rebalance_cost / mark_budget_spent, P4-025). The variable method "
        "cannot hide a NEW money RPC because the reserving caller owns the whole "
        "reserve+pay window. Count is declared (3) so an ADDED dispatch call in "
        "this wrapper trips the guard (P4-027).",
    ),
    ("cl-revenue-ops.py", "_run", DYNAMIC_METHOD_REVIEWED): (
        DYNAMIC_METHOD_REVIEWED, 1,
        "ThreadSafeRpcProxy.fire_and_forget._run is the generic async RPC "
        "transport (rpc.call(method_name, ...)). Not a spend decision: the money "
        "decision and its atomic reservation are owned by the calling site that "
        "names the concrete method; this proxy only forwards it off-thread.",
    ),
    # --- Daemon spender: LN+ liquidity-swap channel OPEN (Task 6, hardened) --
    ("lnplus_swaps.py", "_execute_swap_open", "reserve_spend"): (
        ATOMIC_RESERVE, 1,
        "SwapLifecycle._execute_swap_open reserves atomically (category="
        "'channel_open', subcategory='lnplus_swap') via db.reserve_spend "
        "BEGIN IMMEDIATE, IMMEDIATELY before self.rpc.fundchannel, mirroring "
        "capacity_planner._execute_open/_execute_close (P4-018/DD1 parity). "
        "The reservation_id is unique per attempt (swap_id + int(time.time())) "
        "so reserve_spend's terminal-state guard (refusing to resurrect a "
        "spent/released id) can never block a retry after a failed attempt. "
        "When the caller wires estimate_open_cost_fn/budget_params_fn (the "
        "capacity planner's _estimate_open_cost / _unified_reserve_budget_"
        "params) this is a real cross-category-enforced reserve identical to "
        "the planner's own opens; absent that wiring it falls back to a fixed "
        "_DEFAULT_OPEN_COST_SATS best-effort reservation (effective_budget_"
        "sats=None -> reserve_spend skips enforcement but the amount is still "
        "counted on the rail at settle). A reservation failure (False or "
        "raise) aborts BEFORE fundchannel is called and logs a warning for "
        "the hourly watcher to retry.",
    ),
    ("lnplus_swaps.py", "_execute_swap_open", "rpc_attr_fundchannel"): (
        RAIL_COUNTED_COST, 1,
        "SwapLifecycle._execute_swap_open's self.rpc.fundchannel(...) -- the "
        "on-chain channel open that fulfils an LN+ liquidity-swap application. "
        "Reserved atomically by the reserve_spend call in the same method "
        "IMMEDIATELY before this call (see the paired allowlist entry above), "
        "mirroring capacity_planner._execute_open's reserve-then-fundchannel "
        "pairing (P4-018 parity). On success the reservation is settled loud/ "
        "bounded-retry via _settle_swap_open_reservation (mirrors "
        "capacity_planner._settle_capex_reservation: 3 attempts calling "
        "mark_spend_reservation_spent, and on persistent failure the "
        "reservation is left active and logged at error rather than released, "
        "so a committed fee always stays counted on the unified rail). On a "
        "fundchannel exception or a missing txid the reservation is released "
        "best-effort via _release_swap_open_reservation and the attempt is "
        "retried next watcher pass. Independently, capacity is also reserved "
        "intent-first at the ledger level: SwapEvaluator._select_and_apply "
        "writes the lnplus_swaps row (status='applied', capacity_sats) BEFORE "
        "create_application is even sent, and capacity_planner.py subtracts "
        "db.lnplus_reserved_sats() (sum of capacity across applied/opening/"
        "opened rows) from its own available on-chain funds before its "
        "reserve_spend-gated opens (capacity_planner.py ~line 515) -- "
        "cross-category protection against the planner double spending the "
        "same coins. SwapEvaluator additionally serializes to AT MOST ONE "
        "swap in flight (has_inflight() gate) and re-checks confirmed "
        "on-chain funds >= capacity+fees at apply time (gate 7), so this rail "
        "never carries more than one reserved capacity at once. The "
        "fundchannel call itself is ALSO idempotency-guarded: a "
        "channel_funding_txid already on the row short-circuits to a "
        "complete_application retry, and a listpeerchannels existing-channel "
        "check (OPENINGD/CHANNELD_AWAITING_LOCKIN/CHANNELD_NORMAL/DUALOPEND_*) "
        "skips a duplicate fundchannel on retry. run_watcher_once holds a "
        "non-blocking single-flight lock so two watcher passes can never race "
        "this call. Row-level settlement is loud too: the row is written "
        "status='opening'/outcome='fundchannel attempt' BEFORE the call "
        "(intent) and channel_funding_txid/opened_at AFTER it returns "
        "(outcome); a miss of the 48h deadline trips the circuit breaker and "
        "records a failed swap_open planner action, blocking further LN+ "
        "applications until an operator clears it.",
    ),
}


# ---------------------------------------------------------------------------
# Normalisation helpers: the scanner emits the raw ``dynamic-method-rpc`` marker;
# the allowlist keys such sites under ``dynamic-method-reviewed`` in the same
# (file, func). Provide both a key-set and a count-aware normaliser.
# ---------------------------------------------------------------------------
def _normalise_dynamic(sites):
    """Map each raw dynamic-method-rpc site key to its reviewed allowlist key so
    the single allowlist can carry the justification once (key-set form)."""
    out = set()
    for f, fn, m in sites:
        if m == _DYNAMIC_METHOD_MARKER:
            out.add((f, fn, DYNAMIC_METHOD_REVIEWED))
        else:
            out.add((f, fn, m))
    return out


def _normalise_dynamic_counts(counts: Counter) -> Counter:
    """Count-aware analogue of _normalise_dynamic: fold dynamic-method-rpc counts
    into the reviewed key (summing occurrences in the same file+func)."""
    out: Counter = Counter()
    for (f, fn, m), c in counts.items():
        if m == _DYNAMIC_METHOD_MARKER:
            out[(f, fn, DYNAMIC_METHOD_REVIEWED)] += c
        else:
            out[(f, fn, m)] += c
    return out


def _guard_failures(extra_sources=None):
    """Return (unaccounted, stale, count_mismatch) for the enumeration guard over
    the real tree plus optional ``extra_sources``:
      * unaccounted  -- scanned keys not in the allowlist (new/moved spender)
      * stale        -- allowlist keys no longer in the source
      * count_mismatch -- key -> (scanned_count, declared_count) where an added
                          (or removed) same-marker call diverges from the
                          declared count (P4-027).
    """
    counts = _normalise_dynamic_counts(collect_spend_site_counts(extra_sources))
    scanned = set(counts)
    allow = set(ALLOWLIST)
    unaccounted = scanned - allow
    stale = allow - scanned
    count_mismatch = {
        key: (counts[key], ALLOWLIST[key][1])
        for key in (scanned & allow)
        if counts[key] != ALLOWLIST[key][1]
    }
    return unaccounted, stale, count_mismatch


# ---------------------------------------------------------------------------
# Guard tests.
# ---------------------------------------------------------------------------
def test_scanner_finds_the_known_six_daemon_spenders():
    """Sanity guard for the scanner itself: the six known autonomous daemon
    spenders (and the operator RPC reserve) must all be seen, including the
    chainswap atomic reserve + createchainswap surfaced in pass 4 and the
    attribute-dispatch sendpay surfaced by P4-026."""
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
        # P4-026 surfaced site (pyln attribute dispatch):
        ("data_service.py", "send_pay", "rpc_attr_sendpay"),                      # self._plugin.rpc.sendpay
    }
    missing = required - sites
    assert not missing, f"scanner no longer sees known spend sites: {sorted(missing)}"


def test_every_money_committing_site_is_allowlisted():
    """The core invariant: a NEW or MOVED money-committing call site that is not
    in the allowlist fails the build (so the next refuter's 'one more spender'
    cannot ship silently). A stale allowlist entry also fails, and an ADDED
    same-marker call (declared-count mismatch, P4-027) fails, keeping the
    allowlist an honest mirror of the code.

    Dynamic-method sites are normalised: the scanner emits the raw
    ``dynamic-method-rpc`` marker; the allowlist keys them under
    ``dynamic-method-reviewed`` in the same (file, func)."""
    unaccounted, stale, count_mismatch = _guard_failures()

    assert not unaccounted, (
        "NEW/moved money-committing spend site(s) not in ALLOWLIST -- classify "
        "each (atomic-reserve / rail-counted-cost / operator-only-non-daemon / "
        "not-a-spend / dynamic-method-reviewed) and cite its coverage: "
        f"{sorted(unaccounted)}"
    )

    assert not stale, (
        "ALLOWLIST entries no longer present in the source (moved/removed) -- "
        f"update the allowlist so it stays an honest mirror: {sorted(stale)}"
    )

    assert not count_mismatch, (
        "money-committing site(s) whose scanned occurrence count diverges from "
        "the declared count -- an ADDED same-marker money call must be reviewed "
        "and the declared count updated (P4-027): "
        f"{dict(sorted(count_mismatch.items()))}"
    )


def test_no_allowlisted_spender_is_uncovered():
    """Every allowlisted site must carry a COVERED classification. A site marked
    with anything else (e.g. the 'daemon-reachable-uncovered' sentinel) fails
    here, forcing an operator ruling rather than shipping a budget hole."""
    uncovered = {
        site: klass
        for site, (klass, _count, _why) in ALLOWLIST.items()
        if klass not in COVERED_CLASSES
    }
    assert not uncovered, (
        "money-committing site(s) classified uncovered -- escalate for an "
        f"operator ruling: {uncovered}"
    )
    # Every entry must also carry a non-empty justification.
    unjustified = [
        site for site, (_k, _count, why) in ALLOWLIST.items() if not str(why).strip()
    ]
    assert not unjustified, f"allowlist entries missing a justification: {unjustified}"
    # Every declared count must be a positive integer.
    bad_counts = {
        site: count
        for site, (_k, count, _why) in ALLOWLIST.items()
        if not isinstance(count, int) or count < 1
    }
    assert not bad_counts, f"allowlist entries with a non-positive count: {bad_counts}"


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
        klass, _count, _why = ALLOWLIST[key]
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


def test_attribute_dispatch_detector_ignores_non_rpc_receivers():
    """The P4-026 attribute detector models pyln dispatch, not any same-named
    method. A SQLite ``conn.close()``, a wrapper ``data_service.pay(...)`` and a
    boltz-manager ``_require_boltz_manager().withdraw(...)`` are NOT rpc money
    calls and must not emit rpc_attr_* sites (else the mirror is polluted and the
    guard fires on unrelated code)."""
    noise = (
        "class Noise:\n"
        "    def _do(self, conn, ds):\n"
        "        conn.close()\n"
        "        ds.data_service.pay(x=1)\n"
        "        get_mgr().withdraw(amount_sats=1)\n"
        "        cur.close()\n"
    )
    sites = collect_spend_sites(extra_sources=[("noise_mod.py", noise)])
    attr = {s for s in sites if s[0] == "noise_mod.py" and s[2].startswith("rpc_attr_")}
    assert not attr, f"non-rpc receiver wrongly flagged as attribute money RPC: {sorted(attr)}"


# ---------------------------------------------------------------------------
# TEST-THE-TEST: an unsound scanner would silently miss these. Each injection
# MUST produce a site AND trip the guard (be unaccounted, or a count mismatch).
# ---------------------------------------------------------------------------
def _guard_would_trip(sites):
    """Helper: True iff the enumeration guard would fail on ``sites`` (a key set)
    on the unaccounted axis alone."""
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


def test_the_test_detects_an_injected_attribute_style_sendpay():
    """(iv) P4-026: a rogue pyln ATTRIBUTE-form ``self.rpc.sendpay(route, hash)``
    -- the exact shape of the real data_service.send_pay call, which the OLD
    scanner (rpc.call only) could not see -- is emitted as an rpc_attr_sendpay
    site and trips the guard. An unsound scanner would silently miss it."""
    rogue = (
        "class RogueAttr:\n"
        "    def _rogue_attr_sendpay(self, route, ph):\n"
        "        return self.rpc.sendpay(route, ph)\n"
    )
    sites = collect_spend_sites(extra_sources=[("rogue_attr.py", rogue)])
    injected = ("rogue_attr.py", "_rogue_attr_sendpay", "rpc_attr_sendpay")
    assert injected in sites, (
        "scanner is blind to an attribute-style rpc.sendpay -- a pyln "
        "attribute-dispatch money RPC could hide (P4-026)"
    )
    assert injected not in ALLOWLIST
    assert _guard_would_trip({injected}), "guard would NOT trip on a rogue attribute sendpay"


def test_the_test_detects_a_duplicate_same_marker_call():
    """(v) P4-027: a SECOND same-marker money call injected into an ALREADY
    allowlisted function raises the scanned occurrence count above the declared
    count and trips the guard on the count axis -- the exact absorption the old
    set-collapse allowed. Here two extra reserve_spend calls are injected into
    capacity_planner._execute_open (declared count 1); the guard must report a
    count mismatch for that key."""
    rogue = (
        "class Dup:\n"
        "    def _execute_open(self, db, amt):\n"
        "        db.reserve_spend(reservation_id='a', amount_sats=amt, category='x')\n"
        "        db.reserve_spend(reservation_id='b', amount_sats=amt, category='x')\n"
    )
    unaccounted, stale, count_mismatch = _guard_failures(
        extra_sources=[("capacity_planner.py", rogue)]
    )
    key = ("capacity_planner.py", "_execute_open", "reserve_spend")
    assert key in count_mismatch, (
        "a SECOND same-marker call in an allowlisted function was absorbed -- the "
        f"guard did not surface a count mismatch (mismatches={count_mismatch})"
    )
    scanned_count, declared_count = count_mismatch[key]
    assert scanned_count > declared_count, (
        f"expected scanned count ({scanned_count}) to exceed declared "
        f"({declared_count}) for the injected duplicate"
    )


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


# ===========================================================================
# RPC-HANDLE-ALIAS FORBID-PATTERN LINT (P4-028).
#
# The enumeration scanner above only SEES a money RPC when it is issued
# directly on a syntactic rpc handle -- a bare name ``rpc`` or an attribute
# chain ending ``.rpc`` (``rpc.call('sendpay')``, ``self.rpc.sendpay(...)``,
# ``self._plugin.rpc.sendpay(...)``). An ALIASED handle evades it silently and
# emits NO site:
#
#     h = self.rpc;                 h.sendpay(route, ph)          # (1) local alias
#     getattr(self.rpc, 'sendpay')(route, ph)                    # (2) getattr dispatch
#     m = self.rpc.sendpay;         m(...)                        # (3) stashed bound method
#     p = self._plugin.rpc;         p.pay(params)                # (4) attr alias
#
# No live instance of any of these exists today (the P4-028 refuter confirmed
# zero live aliases), so this is a THEORETICAL gap -- but the guard's whole
# purpose is to stop a FUTURE silent spender. Rather than track dataflow, this
# is a complete FORBID-PATTERN lint: it FAILS the build if any runtime source
# file binds an rpc handle to a non-canonical name (or stashes a bound money
# method, or getattrs a handle) in a way that would let a money RPC be issued
# off a name the scanner cannot recognise. The author must EITHER call the money
# RPC directly on the rpc handle (so the scanner sees it) OR add an explicit,
# justified RPC_ALIAS_ALLOWLIST entry.
#
# A canonical REBIND (``self.rpc = rpc`` -- target is itself a canonical handle)
# is NOT an evasion: the scanner still sees ``self.rpc.<method>(...)``. Only
# binding to a NON-canonical name hides the handle, so only that is flagged.
# ---------------------------------------------------------------------------

# Aliasing forms the lint recognises.
FORM_ASSIGN_HANDLE = "alias-assign-rpc-handle"                 # x = self.rpc
FORM_GETATTR_HANDLE = "alias-getattr-rpc-handle"              # getattr(self.rpc, ...)
FORM_ASSIGN_BOUND_METHOD = "alias-assign-bound-money-method"  # self._sp = self.rpc.sendpay

# Accepted rpc-handle-alias sites. Keyed (filename, enclosing_function, form,
# detail); value is a justification. Near-empty by design: the ONLY accepted
# alias is the single generic pyln transport proxy.
RPC_ALIAS_ALLOWLIST = {
    ("cl-revenue-ops.py", "__init__", FORM_ASSIGN_HANDLE, "_rpc"): (
        "ThreadSafeRpcProxy.__init__ binds ``self._rpc = plugin_instance.rpc`` -- "
        "the ONE generic pyln transport proxy. Every money RPC dispatched through it "
        "is forwarded BY NAME via ThreadSafeRpcProxy.__getattr__/call (which does "
        "``getattr(self._rpc, name)`` -- not flagged because ``self._rpc`` is not a "
        "canonical handle), and the concrete-method spend sites are enumerated on the "
        "``self.rpc`` proxy handle every caller actually uses (self.rpc = "
        "ThreadSafeRpcProxy(...)). This alias is pure transport plumbing, not a spend "
        "decision, and cannot hide a money RPC from the scanner (P4-028)."
    ),
}


def _is_canonical_rpc_target(t: ast.AST) -> bool:
    """True iff assignment target ``t`` is itself a canonical rpc handle name
    (bare ``rpc`` or an attribute ending ``.rpc``). Binding the handle to such a
    name (``self.rpc = rpc``) is a legitimate rebind -- the scanner still sees
    money calls on it -- so it is NOT an evasion and is not flagged."""
    if isinstance(t, ast.Name):
        return t.id == "rpc"
    if isinstance(t, ast.Attribute):
        return t.attr == "rpc"
    return False


def _is_bound_money_method(v: ast.AST) -> bool:
    """True iff ``v`` is ``<rpc-handle>.<money-method>`` -- a bound money-RPC
    method being stashed (e.g. ``self._sp = self.rpc.sendpay``)."""
    return (isinstance(v, ast.Attribute)
            and v.attr in _MONEY_RPC_METHODS
            and _is_rpc_handle(v.value))


def _target_detail(t: ast.AST) -> str:
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    return "<expr>"


def _scan_rpc_aliases(fname: str, tree: ast.AST):
    """Yield (filename, enclosing_function, form, detail, lineno) for every
    rpc-handle-alias construct in ``tree`` that could hide a money RPC from the
    enumeration scanner."""
    encl = _enclosing_func_resolver(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            # Pair each target with its value (elementwise for a shape-matching
            # tuple/list unpack, else each target with the whole value).
            if (isinstance(value, (ast.Tuple, ast.List)) and len(targets) == 1
                    and isinstance(targets[0], (ast.Tuple, ast.List))
                    and len(targets[0].elts) == len(value.elts)):
                pairs = list(zip(targets[0].elts, value.elts))
            else:
                pairs = [(t, value) for t in targets]
            for tgt, val in pairs:
                # (1)/(4): assignment VALUE is an rpc handle bound to a
                # non-canonical name -> the handle is hidden behind that name.
                if _is_rpc_handle(val) and not _is_canonical_rpc_target(tgt):
                    yield (fname, encl(node), FORM_ASSIGN_HANDLE,
                           _target_detail(tgt), node.lineno)
                # (3): assignment VALUE stashes a bound money method.
                if _is_bound_money_method(val):
                    yield (fname, encl(node), FORM_ASSIGN_BOUND_METHOD,
                           val.attr, node.lineno)
        # (2): getattr(<rpc-handle>, ...) -- dynamic method resolution off a
        # canonical handle can dispatch any money RPC by string.
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and node.args and _is_rpc_handle(node.args[0])):
            yield (fname, encl(node), FORM_GETATTR_HANDLE, "getattr", node.lineno)


def collect_rpc_alias_sites(extra_sources=None):
    """Return the list of rpc-handle-alias sites (with line numbers) across
    production source. ``extra_sources`` (list of (filename, source)) lets the
    test-the-test inject each evasion form and assert the lint trips."""
    sites = []
    files = sorted(MODULES_DIR.glob("*.py")) + [MAIN_PLUGIN]
    for path in files:
        sites.extend(_scan_rpc_aliases(
            path.name, ast.parse(path.read_text(), filename=path.name)))
    for fname, source in (extra_sources or []):
        sites.extend(_scan_rpc_aliases(fname, ast.parse(source, filename=fname)))
    return sites


def _rpc_alias_violations(extra_sources=None):
    """Alias sites NOT in RPC_ALIAS_ALLOWLIST (keyed on
    (filename, enclosing_function, form, detail))."""
    return [
        s for s in collect_rpc_alias_sites(extra_sources)
        if (s[0], s[1], s[2], s[3]) not in RPC_ALIAS_ALLOWLIST
    ]


def _format_alias_violations(violations):
    return "\n".join(
        f"  {f}:{lineno} [{form}] alias detail={detail!r} in {func}()"
        for (f, func, form, detail, lineno) in sorted(violations)
    )


def test_no_aliased_rpc_handle_hides_a_money_rpc():
    """P4-028 forbid-pattern lint: the real tree must be GREEN. Any assignment
    that binds an rpc handle to a NON-canonical name, any stashed bound money
    method, or any getattr on an rpc handle -- unless explicitly justified in
    RPC_ALIAS_ALLOWLIST -- fails here, because it could issue a money RPC off a
    name the enumeration scanner cannot see (a silent future spender)."""
    violations = _rpc_alias_violations()
    assert not violations, (
        "rpc-handle-alias construct(s) that could hide a money RPC from the "
        "enumeration scanner -- EITHER call the money RPC directly on the rpc "
        "handle (so the scanner sees the site) OR add a justified "
        "RPC_ALIAS_ALLOWLIST entry:\n" + _format_alias_violations(violations)
    )


def test_alias_lint_allowlists_only_the_known_generic_proxy():
    """The alias allowlist must stay near-empty: the ONLY accepted alias is the
    single generic ThreadSafeRpcProxy transport, and every entry must carry a
    non-empty justification. A stale entry (no longer in source) also fails so
    the allowlist stays an honest mirror."""
    scanned_keys = {(s[0], s[1], s[2], s[3]) for s in collect_rpc_alias_sites()}
    stale = set(RPC_ALIAS_ALLOWLIST) - scanned_keys
    assert not stale, f"stale RPC_ALIAS_ALLOWLIST entries no longer in source: {sorted(stale)}"
    unjustified = [k for k, why in RPC_ALIAS_ALLOWLIST.items() if not str(why).strip()]
    assert not unjustified, f"alias allowlist entries missing justification: {unjustified}"


def test_alias_lint_ignores_canonical_rpc_rebind():
    """A canonical rebind (``self.rpc = rpc`` -- target is itself a canonical
    handle) is NOT an evasion and must not be flagged: the scanner still sees
    money calls on ``self.rpc``. (BoltzCliManager.__init__ does exactly this.)"""
    rebind = (
        "class Mgr:\n"
        "    def __init__(self, plugin, rpc):\n"
        "        self.rpc = rpc\n"
    )
    violations = _rpc_alias_violations(extra_sources=[("rebind_mod.py", rebind)])
    assert not any(v[0] == "rebind_mod.py" for v in violations), (
        f"canonical rpc rebind wrongly flagged as an alias evasion: "
        f"{[v for v in violations if v[0] == 'rebind_mod.py']}"
    )


# ---------------------------------------------------------------------------
# TEST-THE-TEST for the alias lint: each of the four evasion forms, injected
# into a scratch fixture, MUST trip the lint (be a violation). An incomplete
# lint would silently miss one and leave the aliasing evasion open.
# ---------------------------------------------------------------------------
def test_the_test_alias_lint_trips_on_local_handle_alias():
    """(1) ``h = self.rpc; h.sendpay(route, ph)`` -- local alias of the handle."""
    rogue = (
        "class RogueLocal:\n"
        "    def _rogue(self, route, ph):\n"
        "        h = self.rpc\n"
        "        return h.sendpay(route, ph)\n"
    )
    violations = _rpc_alias_violations(extra_sources=[("rogue_local.py", rogue)])
    hit = [v for v in violations if v[0] == "rogue_local.py"]
    assert any(v[2] == FORM_ASSIGN_HANDLE and v[3] == "h" for v in hit), (
        f"alias lint blind to a local rpc-handle alias (h = self.rpc): {hit}"
    )


def test_the_test_alias_lint_trips_on_getattr_dispatch():
    """(2) ``getattr(self.rpc, 'sendpay')(route, ph)`` -- getattr dispatch."""
    rogue = (
        "class RogueGetattr:\n"
        "    def _rogue(self, route, ph):\n"
        "        return getattr(self.rpc, 'sendpay')(route, ph)\n"
    )
    violations = _rpc_alias_violations(extra_sources=[("rogue_getattr.py", rogue)])
    hit = [v for v in violations if v[0] == "rogue_getattr.py"]
    assert any(v[2] == FORM_GETATTR_HANDLE for v in hit), (
        f"alias lint blind to a getattr(rpc-handle, ...) dispatch: {hit}"
    )


def test_the_test_alias_lint_trips_on_stashed_bound_method():
    """(3) ``m = self.rpc.sendpay; m(...)`` -- stashed bound money method."""
    rogue = (
        "class RogueBound:\n"
        "    def _rogue(self):\n"
        "        m = self.rpc.sendpay\n"
        "        return m\n"
    )
    violations = _rpc_alias_violations(extra_sources=[("rogue_bound.py", rogue)])
    hit = [v for v in violations if v[0] == "rogue_bound.py"]
    assert any(v[2] == FORM_ASSIGN_BOUND_METHOD and v[3] == "sendpay" for v in hit), (
        f"alias lint blind to a stashed bound money method (m = self.rpc.sendpay): {hit}"
    )


def test_the_test_alias_lint_trips_on_plugin_rpc_attr_alias():
    """(4) ``p = self._plugin.rpc; p.pay(params)`` -- attribute-chain alias."""
    rogue = (
        "class RoguePlugin:\n"
        "    def _rogue(self, params):\n"
        "        p = self._plugin.rpc\n"
        "        return p.pay(params)\n"
    )
    violations = _rpc_alias_violations(extra_sources=[("rogue_plugin.py", rogue)])
    hit = [v for v in violations if v[0] == "rogue_plugin.py"]
    assert any(v[2] == FORM_ASSIGN_HANDLE and v[3] == "p" for v in hit), (
        f"alias lint blind to a self._plugin.rpc alias (p = self._plugin.rpc): {hit}"
    )
