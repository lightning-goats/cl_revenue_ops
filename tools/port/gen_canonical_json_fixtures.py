#!/usr/bin/env python3
"""Generate canonical JSON fixtures for cross-language byte parity testing.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_canonical_json_fixtures.py <output.json>

The output feeds crates/revops-core/tests/canonical_parity.rs in the Rust
port (cl-revenue-ops-r). Python's json.dumps(sort_keys=True,
separators=(",", ":"), ensure_ascii=False) is the source of truth for canonical
JSON encoding.
"""
import json
import sys

CASES = [
    {"b": 1, "a": [3, 2, {"z": None, "y": True}], "μ": "ünïcode"},
    {"nested": {"c": {"b": {"a": 0}}}, "empty": {}, "list": []},
    {"floats_forbidden_note": "ints only in canonical payloads", "n": 2**53 - 1},
    {"s": "quote\" backslash\\ newline\n tab\t"},
    {},
]

out = [{"value": c,
        "canonical": json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}
       for c in CASES]

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
    sys.exit(1)

json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(f"wrote {sys.argv[1]}")
