#!/usr/bin/env python3
"""Generate startup-hydration parity fixtures from the Python implementation.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_hydration_fixtures.py <output.json>

The output feeds crates/revops-db/tests/notifications.rs in the Rust port
(cl-revenue-ops-r), pinning `_compute_forward_hydration_start`
(cl-revenue-ops.py:602-625) input/output pairs. Python is the source of
truth: nothing here is hand-computed, everything comes from calling the
real function via a `pyln`-mocked import of cl-revenue-ops.py (same
mocking pattern as gen_schema_fixture.sh).
"""
import importlib.util
import json
import os
import sys
from unittest.mock import MagicMock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_ROOT)

mock = MagicMock()
mock.Plugin = MagicMock
mock.RpcError = Exception
sys.modules.setdefault("pyln", mock)
sys.modules.setdefault("pyln.client", mock)
sys.path.insert(0, REPO_ROOT)

spec = importlib.util.spec_from_file_location("revops_main", "cl-revenue-ops.py")
revops_main = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(revops_main)
    fn = revops_main._compute_forward_hydration_start
except Exception as e:  # pragma: no cover - fallback path, see plan Step 1 note
    print(f"Full-module import failed ({e}); falling back to AST extraction", file=sys.stderr)
    import ast

    tree = ast.parse(open("cl-revenue-ops.py").read())
    func_node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_compute_forward_hydration_start"
    )
    const_node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FORWARD_HYDRATION_EVENT_JITTER_SECONDS" for t in n.targets)
    )
    namespace = {"time": __import__("time"), "Optional": None}
    exec(compile(ast.Module(body=[const_node, func_node], type_ignores=[]), "<extracted>", "exec"), namespace)
    fn = namespace["_compute_forward_hydration_start"]

NOW = 1_800_000_000
CASES = [
    {"last_forward_ts": None, "flow_window_days": 7, "now": NOW},
    {"last_forward_ts": None, "flow_window_days": 30, "now": NOW},
    {"last_forward_ts": NOW - 100, "flow_window_days": 7, "now": NOW},       # within jitter -> None
    {"last_forward_ts": NOW - 300, "flow_window_days": 7, "now": NOW},       # exactly at boundary -> None
    {"last_forward_ts": NOW - 301, "flow_window_days": 7, "now": NOW},       # just over -> backfill
    {"last_forward_ts": NOW - 10 * 86400, "flow_window_days": 7, "now": NOW},
    {"last_forward_ts": NOW - 100 * 86400, "flow_window_days": 7, "now": NOW},  # floor clamps it
]
out = [{**c, "result": fn(c["last_forward_ts"], c["flow_window_days"], c["now"])} for c in CASES]

if len(sys.argv) < 2:
    print("usage: gen_hydration_fixtures.py <output.json>", file=sys.stderr)
    sys.exit(1)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"wrote {len(out)} hydration cases -> {sys.argv[1]}")
