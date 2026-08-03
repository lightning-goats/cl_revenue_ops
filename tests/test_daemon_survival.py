"""
Daemon-survival fault-injection tests (deep-audit Phase 2C).

These tests document finding **P1-010** (daemon-survival dimension,
``docs/audit/deep/findings-ledger.md``):

    Per-iteration sleep/interval/config.snapshot() tail of every daemon loop is
    OUTSIDE the loop-body try/except; an exception there kills the thread
    permanently and silently (daemon=True, no respawn, no heartbeat).
    Pattern repeats across every loop.

The 7 daemon loops started as threads in ``cl-revenue-ops.py`` (see the
``threading.Thread(...).start()`` block, currently near line 2799):

    flow-analysis      -> flow_analysis_loop
    fee-adjustment     -> fee_adjustment_loop
    rebalance-check    -> rebalance_check_loop
    startup-snapshot   -> snapshot_peers_delayed   (ONE-SHOT, no while loop)
    financial-snapshot -> financial_snapshot_loop  (inverted: sleep-tail is at
                                                    the TOP of the while body)
    capacity-planner   -> capacity_planner_loop

Testing strategy
----------------
The loop functions are *closures* defined inside the plugin ``init`` handler,
so they cannot be imported. Instead we extract each loop's **real source** from
``cl-revenue-ops.py`` via ``ast.get_source_segment`` and exec it in a controlled
namespace where every closed-over name (``shutdown_event``, ``config``, the work
function, ``random``, ...) is a mock we drive. This runs the REAL loop bytecode,
so the placement of the try/except (and the unguarded tail) is exactly as it
ships in production -- these are not reconstructions.

Two exception-injection positions per loop, one raise each:
  (a) INSIDE the try-guarded body  -> loop MUST survive to the next iteration.
  (b) In the TAIL outside the try (interval/jitter computation: ``random.randint``
      is in the unguarded sleep-computation tail of every loop) -> the thread
      dies silently.

The tail-death tests assert the CURRENT (defective) behavior so they PASS today
and document the bug. **When the P1-010 fix lands (per-iteration try/except that
wraps the whole body incl. the tail), these assertions must be FLIPPED to assert
survival** -- the tests are written so the failure points you straight here.

A third group asserts the operator has NO heartbeat/per-thread-liveness surface
(``revenue-health`` reports no loop liveness), so a dead loop is undetectable.

NOTHING in the plugin is modified by this phase; the fix is BEHAVIORAL-HOLD for
the operator ruling.
"""

import ast
import textwrap
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = REPO_ROOT / "cl-revenue-ops.py"
_SOURCE = MAIN_FILE.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE, filename="cl-revenue-ops.py")


# ---------------------------------------------------------------------------
# Injected exception markers (distinct so we can attribute a thread death)
# ---------------------------------------------------------------------------
class InjectedBodyError(RuntimeError):
    """Raised from inside the try-guarded loop body."""


class InjectedTailError(RuntimeError):
    """Raised from the unguarded interval/sleep computation tail (P1-010)."""


# ---------------------------------------------------------------------------
# Loop registry
# ---------------------------------------------------------------------------
# work_name: the callable driven as the loop's unit of work. A string is a
#   global-scope name; a (obj, attr) tuple is a method on a namespace mock.
# has_while: False for the one-shot startup-snapshot thread.
LOOP_SPECS = {
    "flow_analysis_loop":     {"thread": "flow-analysis",      "work": "run_flow_analysis",         "has_while": True},
    "fee_adjustment_loop":    {"thread": "fee-adjustment",     "work": "run_fee_adjustment",        "has_while": True},
    "rebalance_check_loop":   {"thread": "rebalance-check",    "work": "run_rebalance_check",       "has_while": True},
    "capacity_planner_loop":  {"thread": "capacity-planner",   "work": ("capacity_planner", "execute_cycle"), "has_while": True},
    "financial_snapshot_loop":{"thread": "financial-snapshot", "work": "_take_financial_snapshot",  "has_while": True},
    "snapshot_peers_delayed": {"thread": "startup-snapshot",   "work": "_snapshot_peers_once",      "has_while": False},
}

WHILE_LOOPS = [name for name, s in LOOP_SPECS.items() if s["has_while"]]
ALL_LOOPS = list(LOOP_SPECS)


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------
def _find_funcdef(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"loop function {name!r} not found in cl-revenue-ops.py")


def _extract_callable(name, namespace):
    """Compile the real source of loop ``name`` into ``namespace`` and return it.

    The def is extracted in isolation, so every closed-over name compiles to a
    global lookup resolved from ``namespace`` at call time.
    """
    node = _find_funcdef(name)
    src = textwrap.dedent(ast.get_source_segment(_SOURCE, node))
    code = compile(src, filename="cl-revenue-ops.py::" + name, mode="exec")
    exec(code, namespace)  # noqa: S102 - executing our own vendored source, controlled ns
    return namespace[name]


# ---------------------------------------------------------------------------
# Drivers / doubles
# ---------------------------------------------------------------------------
class FakeShutdownEvent:
    """threading.Event stand-in we drive deterministically.

    ``wait()`` never blocks. ``auto_set_after`` is a safety escape hatch so the
    loop always terminates even if a future fix makes the tail non-fatal (keeps
    the flipped test from hanging).
    """

    def __init__(self, auto_set_after=8):
        self._set = False
        self.wait_calls = 0
        self._auto_set_after = auto_set_after

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._auto_set_after is not None and self.wait_calls >= self._auto_set_after:
            self._set = True
        return self._set


class WorkController:
    """Callable standing in for a loop's unit of work.

    raise_first  -> raise InjectedBodyError on the first call (body injection).
    stop_after   -> once call count reaches this, set the shutdown event so the
                    loop exits cleanly on its next wait().
    """

    def __init__(self, event, raise_first=False, stop_after=None):
        self.event = event
        self.raise_first = raise_first
        self.stop_after = stop_after
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.raise_first and self.calls == 1:
            raise InjectedBodyError("P1-010 body-injection: must be caught by loop try/except")
        if self.stop_after is not None and self.calls >= self.stop_after:
            self.event.set()
        return {}  # dict is safe for callers that do result.get(...)


def _build_namespace(event, work_name, work_callable, randint_mock):
    """A controlled global namespace covering every name the loops close over."""
    ns = {}
    ns["shutdown_event"] = event
    ns["plugin"] = MagicMock()
    ns["time"] = __import__("time")
    ns["traceback"] = __import__("traceback")
    ns["random"] = types.SimpleNamespace(randint=randint_mock)

    # Except-clause classes must be real exception types.
    ns["RPCTimeoutError"] = type("RPCTimeoutError", (Exception,), {})
    ns["RPCBreakerOpen"] = type("RPCBreakerOpen", (Exception,), {})

    # Collaborators.
    database = MagicMock()
    database.get_planner_actions.return_value = []
    ns["database"] = database
    ns["policy_manager"] = MagicMock()
    ns["capacity_planner"] = MagicMock()
    ns["capacity_planner"].execute_cycle.return_value = {}
    ns["hive_hints"] = MagicMock()
    ns["hive_router"] = MagicMock()

    boltz_manager = MagicMock()
    boltz_manager.enabled = True
    ns["boltz_manager"] = boltz_manager

    # config: real numeric attrs (MagicMock attrs break max()/int()).
    config = MagicMock()
    config.planner_enabled = True
    config.planner_interval = 600
    config.flow_window_days = 7
    config.boltz_auto_cycle_startup_delay_seconds = 0
    config.boltz_auto_cycle_interval_minutes = 15
    config.snapshot.return_value = types.SimpleNamespace(
        flow_interval=60,
        fee_interval=60,
        rebalance_interval=60,
        planner_interval=600,
        flow_window_days=7,
    )
    ns["config"] = config

    # Helper functions closed over by the loops.
    for helper in (
        "_refresh_fee_cycle_hive_inputs",
        "_refresh_dynamic_config",
        "_boltz_auto_cycle_mark_state",
        "_take_financial_snapshot",
        "_snapshot_peers_once",
        "_run_boltz_auto_cycle_once",
        "run_flow_analysis",
        "run_fee_adjustment",
        "run_rebalance_check",
        # DD5 / P1-010: the canonical guard records a per-thread heartbeat.
        "_record_loop_heartbeat",
    ):
        ns[helper] = MagicMock(return_value={})

    # DD5 / P1-010: bounded backoff constant used by the canonical guard's
    # except-clause (shutdown_event.wait(_LOOP_BACKOFF_SECONDS)).
    ns["_LOOP_BACKOFF_SECONDS"] = 0

    # Wire the work callable into its slot.
    if isinstance(work_name, tuple):
        obj_name, attr = work_name
        setattr(ns[obj_name], attr, work_callable)
    else:
        ns[work_name] = work_callable

    return ns


def _run_loop_thread(loop_fn, timeout=5.0):
    """Run ``loop_fn`` as a daemon thread; capture any uncaught exception.

    Returns (still_alive, captured_exception). A daemon loop that dies from an
    unguarded exception surfaces it only via ``threading.excepthook`` (i.e.
    stderr) -- the plugin itself never learns the thread is gone. We capture
    that hook to prove the silent death.
    """
    captured = {}

    def hook(args):
        captured["exc"] = args.exc_value

    original_hook = threading.excepthook
    threading.excepthook = hook
    try:
        t = threading.Thread(target=loop_fn, daemon=True, name="p1010-test")
        t.start()
        t.join(timeout)
        alive = t.is_alive()
    finally:
        threading.excepthook = original_hook
    return alive, captured.get("exc")


# ===========================================================================
# GROUP 1 -- Static structural proof of P1-010 against the REAL source
# ===========================================================================
def _while_node(func_node):
    for node in ast.walk(func_node):
        if isinstance(node, ast.While):
            # test is `not shutdown_event.is_set()`
            return node
    return None


@pytest.mark.parametrize("loop_name", WHILE_LOOPS)
def test_p1010_tail_is_inside_canonical_guard(loop_name):
    """DD5 / P1-010 FIXED: the per-iteration sleep/interval tail now sits INSIDE
    the canonical guard.

    Structural assertion on the shipped source: each `while not
    shutdown_event.is_set()` body wraps its ENTIRE iteration (work AND the
    interval/sleep tail) in one try/except, so there is NO unguarded
    sleep-computation sibling at the top level of the while body. An exception
    anywhere in the iteration is therefore caught and the daemon survives.

    (Flipped from the pre-fix assertion that required the tail to be unguarded.)
    """
    func = _find_funcdef(loop_name)
    wnode = _while_node(func)
    assert wnode is not None, f"{loop_name}: expected a `while not shutdown_event.is_set()` loop"

    body = wnode.body

    # A Try guards the iteration.
    has_try = any(isinstance(n, ast.Try) for s in body for n in ast.walk(s))
    assert has_try, f"{loop_name}: expected a try/except guarding the loop iteration"

    def _mentions_tail(stmt):
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                # random.randint(...) (interval/jitter) or shutdown_event.wait(...)
                if n.func.attr in {"randint", "wait"}:
                    return True
        return False

    # The fix: the interval/sleep tail is NO LONGER a top-level sibling of the
    # guard -- it lives inside the canonical try/except.
    unguarded_tail = [
        s for s in body if not isinstance(s, ast.Try) and _mentions_tail(s)
    ]
    assert not unguarded_tail, (
        f"{loop_name}: the interval/sleep tail (random.randint / "
        f"shutdown_event.wait) must live INSIDE the canonical per-iteration "
        f"try/except -- found an unguarded tail statement (P1-010 regression)."
    )


def test_p1010_startup_snapshot_is_one_shot_no_tail():
    """P1-010 N/A shape: startup-snapshot (snapshot_peers_delayed) is a ONE-SHOT.

    It has no `while` loop and thus no per-iteration tail, so the specific
    P1-010 tail pattern does not apply. It is still an unguarded daemon (any
    exception kills the single run with no retry / no heartbeat) -- exercised by
    test_body_exception_* below.
    """
    func = _find_funcdef("snapshot_peers_delayed")
    assert _while_node(func) is None, "startup-snapshot is expected to be a one-shot (no while loop)"


# ===========================================================================
# GROUP 2a -- Body-exception: loop MUST survive to the next iteration
# ===========================================================================
@pytest.mark.parametrize("loop_name", WHILE_LOOPS)
def test_body_exception_loop_survives(loop_name):
    """P1-010 baseline: an exception INSIDE the try-guarded body is caught and
    the loop continues to the next iteration.

    This is the correct, desired behavior and should keep passing after the fix.
    The work double raises once (iteration 1) then, on its second call, sets the
    shutdown event; survival is proven by the work being invoked >= 2 times with
    no exception escaping the thread.
    """
    spec = LOOP_SPECS[loop_name]
    event = FakeShutdownEvent()
    randint = MagicMock(return_value=1)  # tail behaves normally here
    work = WorkController(event, raise_first=True, stop_after=2)
    ns = _build_namespace(event, spec["work"], work, randint)
    loop_fn = _extract_callable(loop_name, ns)

    alive, exc = _run_loop_thread(loop_fn)

    assert not alive, f"{loop_name}: loop thread hung"
    assert exc is None, (
        f"{loop_name}: a body exception escaped the loop's try/except and killed "
        f"the thread (got {exc!r}); the body IS supposed to be guarded."
    )
    assert work.calls >= 2, (
        f"{loop_name}: loop did not survive the body exception to a 2nd iteration "
        f"(work called {work.calls}x)."
    )


def test_body_exception_startup_snapshot_is_now_guarded():
    """DD5 / P1-010 FIXED -- the one-shot startup-snapshot's single run is now
    guarded, so an exception in it no longer kills the thread silently.

    (Flipped from the pre-fix assertion that the one-shot dies on an unguarded
    exception; DD5 explicitly guards the single run.)
    """
    event = FakeShutdownEvent()
    randint = MagicMock(return_value=1)
    work = MagicMock(side_effect=InjectedBodyError("startup snapshot boom"))
    ns = _build_namespace(event, "_snapshot_peers_once", work, randint)
    loop_fn = _extract_callable("snapshot_peers_delayed", ns)

    alive, exc = _run_loop_thread(loop_fn)

    assert not alive
    assert exc is None, (
        "startup-snapshot one-shot should now swallow its unguarded exception "
        f"(captured {exc!r}); DD5 guards the single run."
    )
    assert work.called, "the one-shot work should still have been invoked once"


# ===========================================================================
# GROUP 2b -- Tail-exception: thread DIES (documents the P1-010 defect)
# ===========================================================================
@pytest.mark.parametrize("loop_name", WHILE_LOOPS)
def test_tail_exception_no_longer_kills_thread(loop_name):
    """DD5 / P1-010 FIXED -- the canonical guard swallows a tail exception and the
    loop SURVIVES to the next iteration.

    An exception in the interval/sleep tail (here: random.randint) is now caught
    by the per-iteration try/except that wraps the whole body incl. the tail, so
    it never escapes to threading.excepthook and never kills the daemon thread.
    The loop keeps ticking (work runs each iteration) until the shutdown event's
    escape hatch terminates it cleanly.

    (Flipped from test_tail_exception_kills_thread_CURRENT_DEFECT, which asserted
    the pre-fix silent-death behavior.)
    """
    spec = LOOP_SPECS[loop_name]
    event = FakeShutdownEvent(auto_set_after=8)  # escape hatch keeps a fixed loop from hanging
    randint = MagicMock(side_effect=InjectedTailError(f"P1-010 tail in {loop_name}"))
    work = WorkController(event, raise_first=False, stop_after=None)  # body succeeds
    ns = _build_namespace(event, spec["work"], work, randint)
    loop_fn = _extract_callable(loop_name, ns)

    alive, exc = _run_loop_thread(loop_fn)

    assert not alive, f"{loop_name}: thread neither survived-then-stopped nor terminated within timeout"
    assert exc is None, (
        f"{loop_name}: a tail exception escaped the canonical guard and killed "
        f"the thread (got {exc!r}); the whole iteration is supposed to be guarded."
    )
    assert work.calls >= 1, f"{loop_name}: work should have run at least once while the loop survived the tail errors"


# ===========================================================================
# GROUP 3 -- No heartbeat / per-thread liveness surface (the detection gap)
# ===========================================================================
def _health_func_source():
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "revenue_health":
            return ast.get_source_segment(_SOURCE, node)
    raise AssertionError("revenue_health handler not found")


def test_revenue_health_reports_thread_liveness():
    """DD5 / P1-010 FIXED: revenue-health now exposes per-thread/loop liveness so
    a dead or stalled daemon loop is operator-detectable.

    (Flipped from the pre-fix assertion that no liveness surface existed.)
    """
    src = _health_func_source()
    lowered = src.lower()
    liveness_markers = [
        "heartbeat",
        "liveness",
        "last_tick",
        "loops",
        "stalled",
    ]
    present = [m for m in liveness_markers if m in lowered]
    assert present, (
        "revenue-health must expose a per-thread loop liveness surface "
        "(e.g. result['loops'] with per-thread heartbeat age / alive-stalled)."
    )


def test_heartbeat_mechanism_present_in_plugin():
    """DD5 / P1-010 FIXED (whole-file): the plugin now records a per-loop
    heartbeat that the canonical guard updates every iteration, and surfaces it.

    (Flipped from the pre-fix assertion that no heartbeat mechanism existed.)
    """
    lowered = _SOURCE.lower()
    assert "heartbeat" in lowered, (
        "expected a per-loop heartbeat mechanism in cl-revenue-ops.py (DD5)."
    )
    assert "_record_loop_heartbeat" in _SOURCE, (
        "expected the canonical guard to record a per-thread heartbeat via "
        "_record_loop_heartbeat(...)."
    )
