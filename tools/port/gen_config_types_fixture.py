#!/usr/bin/env python3
"""AST-extract the `Config` dataclass's field types, its public/deprecated
runtime-key classification sets, and its override-validation tables
(CONFIG_FIELD_RANGES, STRING_ENUM_VALID_VALUES) from modules/config.py.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_config_types_fixture.py <output.json>

The output feeds crates/revops/src/config_types.rs in the Rust port
(cl-revenue-ops-r), which loads it to know each `Config` field's declared
Python type (so `revenue-r-config get` can convert a resolved CLN option
value to the JSON scalar shape Python would emit) and to replicate
`Config.classify_runtime_key` (PUBLIC_RUNTIME_KEYS / DEPRECATED_RUNTIME_KEYS).
`ranges`/`enums` feed `revops::config_resolve::validate_override`, the Rust
port of `Config._apply_override`'s range/enum gate
(modules/config.py:1015-1047) -- CONFIG_FIELD_RANGES has 96 entries as of
this writing (large enough that hand-transcription risks drift), so both
tables are AST-extracted here rather than transcribed, even though
STRING_ENUM_VALID_VALUES (5 entries) alone would have been small enough to
transcribe by hand with line citations.

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
ranges = None
enums = None

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
        elif node.target.id == "CONFIG_FIELD_RANGES":
            raw_ranges = literal_or_call_arg(node.value)
            ranges = {k: list(v) for k, v in raw_ranges.items()}
        elif node.target.id == "STRING_ENUM_VALID_VALUES":
            raw_enums = literal_or_call_arg(node.value)
            enums = {k: list(v) for k, v in raw_enums.items()}

# ---------------------------------------------------------------------------
# Per-field bool STARTUP casts (2026-07-22 Rust audit M2): the
# `Config(**config_kwargs)` construction in cl-revenue-ops.py converts each
# bool option with one of two inconsistent expressions —
# `.lower() == 'true'` (strict) or `.lower() in ('true', '1', 'yes')`
# (tolerant, note: NO 'on') — and `_apply_override`'s generic
# `('true','1','yes','on')` parser applies ONLY to DB override rows. The
# Rust layer-(b) (listconfigs) conversion must mirror the per-field startup
# cast, so AST-extract it here rather than hand-transcribe 23 fields.
# fee_replay_capture_enabled wraps its cast in str(...).strip(); the strip
# only matters for whitespace-padded values of that one field and is folded
# into "eq_true" (documented divergence: Rust does not strip).
# ---------------------------------------------------------------------------


def classify_bool_cast(node):
    """Return "eq_true" / "in_true_1_yes" for a recognized startup bool
    cast expression, else None."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    op, comp = node.ops[0], node.comparators[0]
    if isinstance(op, ast.Eq):
        if isinstance(comp, ast.Constant) and comp.value == "true":
            return "eq_true"
        return None
    if isinstance(op, ast.In):
        try:
            values = tuple(ast.literal_eval(comp))
        except (ValueError, SyntaxError):
            return None
        if values == ("true", "1", "yes"):
            return "in_true_1_yes"
        return None
    return None


plugin_tree = ast.parse(open("cl-revenue-ops.py").read())
bool_casts = {}
for node in ast.walk(plugin_tree):
    # The construction is `config_kwargs = dict(<field>=<cast expr>, ...)`
    # (cl-revenue-ops.py:2413), then `Config(**{k: v ...})` at 2605.
    if not (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "config_kwargs"
                    for t in node.targets)
            and isinstance(node.value, ast.Call)
            and len(node.value.keywords) > 20):
        continue
    for kw in node.value.keywords:
        if kw.arg is None or fields.get(kw.arg) != "bool":
            continue
        cast = classify_bool_cast(kw.value)
        if cast is None:
            # Unwrap one parenthesized/str(...)-wrapped level (the
            # fee_replay_capture_enabled shape) before giving up.
            for inner in ast.walk(kw.value):
                cast = classify_bool_cast(inner)
                if cast is not None:
                    break
        if cast is None:
            sys.exit(
                f"bool field {kw.arg!r} in the Config(...) construction has "
                "an unrecognized startup cast expression — extend "
                "classify_bool_cast (do NOT let it silently default)"
            )
        bool_casts[kw.arg] = cast
if not bool_casts:
    sys.exit("no Config(...) bool casts extracted from cl-revenue-ops.py")

if public_keys is None:
    sys.exit("PUBLIC_RUNTIME_KEYS not found in modules/config.py")
if deprecated_keys is None:
    sys.exit("DEPRECATED_RUNTIME_KEYS not found in modules/config.py")
if ranges is None:
    sys.exit("CONFIG_FIELD_RANGES not found in modules/config.py")
if enums is None:
    sys.exit("STRING_ENUM_VALID_VALUES not found in modules/config.py")
if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
    sys.exit(1)

out = {
    "fields": fields,
    "public_keys": public_keys,
    "deprecated_keys": deprecated_keys,
    "ranges": ranges,
    "enums": enums,
    "bool_casts": bool_casts,
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(
    f"{len(fields)} fields, {len(public_keys)} public keys, "
    f"{len(deprecated_keys)} deprecated keys, {len(ranges)} ranges, "
    f"{len(enums)} enums, {len(bool_casts)} bool casts -> {sys.argv[1]}"
)
