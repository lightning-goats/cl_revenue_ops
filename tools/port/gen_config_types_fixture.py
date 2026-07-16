#!/usr/bin/env python3
"""AST-extract the `Config` dataclass's field types plus its public/deprecated
runtime-key classification sets from modules/config.py.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_config_types_fixture.py <output.json>

The output feeds crates/revops/src/config_types.rs in the Rust port
(cl-revenue-ops-r), which loads it to know each `Config` field's declared
Python type (so `revenue-r-config get` can convert a resolved CLN option
value to the JSON scalar shape Python would emit) and to replicate
`Config.classify_runtime_key` (PUBLIC_RUNTIME_KEYS / DEPRECATED_RUNTIME_KEYS).

Only `int`/`float`/`bool`/`str` fields with a plain `Name` annotation are
captured -- this naturally excludes the dataclass's non-configurable,
non-JSON-scalar fields (`_lock: threading.Lock`, `_override_warnings: list`),
which use non-`Name` or out-of-map annotations and are never set via a CLN
option in the first place.
"""
import ast
import json
import sys

TYPE_MAP = {"int": "int", "float": "float", "bool": "bool", "str": "string"}


def literal_or_call_arg(node):
    """`ast.literal_eval`, unwrapping a single-arg constructor call first
    (e.g. `frozenset({...})` -> the `{...}` Set literal) since
    `literal_eval` can't evaluate a `Call` node directly."""
    if isinstance(node, ast.Call):
        node = node.args[0]
    return ast.literal_eval(node)


tree = ast.parse(open("modules/config.py").read())

fields = {}
public_keys = None
deprecated_keys = None

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "Config":
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                ann = stmt.annotation
                if isinstance(ann, ast.Name) and ann.id in TYPE_MAP:
                    fields[stmt.target.id] = TYPE_MAP[ann.id]
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "PUBLIC_RUNTIME_KEYS":
                public_keys = sorted(literal_or_call_arg(node.value))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id == "DEPRECATED_RUNTIME_KEYS":
            deprecated_keys = sorted(literal_or_call_arg(node.value))

if public_keys is None:
    sys.exit("PUBLIC_RUNTIME_KEYS not found in modules/config.py")
if deprecated_keys is None:
    sys.exit("DEPRECATED_RUNTIME_KEYS not found in modules/config.py")
if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
    sys.exit(1)

out = {
    "fields": fields,
    "public_keys": public_keys,
    "deprecated_keys": deprecated_keys,
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(
    f"{len(fields)} fields, {len(public_keys)} public keys, "
    f"{len(deprecated_keys)} deprecated keys -> {sys.argv[1]}"
)
