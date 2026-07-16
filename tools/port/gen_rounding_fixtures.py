#!/usr/bin/env python3
"""Generate cross-language rounding fixtures from the Python implementation.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_rounding_fixtures.py <output.json>

The output feeds crates/revops-core/tests/rounding_parity.rs in the Rust
port (cl-revenue-ops-r). Python is the source of truth for every expected
value below: nothing here is hand-computed, everything is produced by
calling the real modules.utils functions.

Scope note: Python's parse_msat() also accepts pyln Millisatoshi-like
objects via a `.millisatoshis` attribute duck-type check. That form has no
JSON representation and is intentionally excluded from this fixture and
from the Rust port at this layer.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.utils import base_to_sats_ceil, base_to_sats_floor, parse_msat, sats_to_base

CASES = [0, 1, 999, 1000, 1001, 1500, 1999, 2000, 49_999, 50_013_000,
         999_999_999_999, 2**53 - 1]

# String-form parse_msat inputs, as accepted by the Python implementation
# ("Nmsat" / "N", surrounding whitespace trimmed).
STR_CASES = ["0msat", "1msat", "1001msat", "50013000msat", "123", "  42msat "]

# Extra parse_msat inputs added per task-2 brief: negative int passthrough,
# float truncation toward zero (both signs), bool (must be 0, U-1 fix),
# None (must be 0), and an unparseable string (must be 0). Expected values
# are NOT hand-written here -- they come straight out of parse_msat() below,
# which is the whole point of this generator (Python truth is the arbiter).
EXTRA_PARSE_CASES = [-1500, 1.9, -1.9, True, None, "garbage"]

out = {
    "ceil": [{"msat": c, "sats": base_to_sats_ceil(c)} for c in CASES],
    "floor": [{"msat": c, "sats": base_to_sats_floor(c)} for c in CASES],
    "to_base": [{"sats": c, "msat": sats_to_base(c)} for c in CASES],
    "parse": (
        [{"input": s, "msat": parse_msat(s)} for s in STR_CASES]
        + [{"input": c, "msat": parse_msat(c)} for c in CASES]
        + [{"input": c, "msat": parse_msat(c)} for c in EXTRA_PARSE_CASES]
    ),
}

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
    sys.exit(1)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]}")
