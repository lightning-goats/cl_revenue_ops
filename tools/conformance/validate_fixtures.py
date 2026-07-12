#!/usr/bin/env python3
"""Standalone conformance-fixture validator (refactor Phase 0, J5).

Validates every *.json under the given scenarios directory against the
schema named by its `schema_name`/`schema_version` fields, loaded from
schemas/. Deliberately imports NOTHING from the plugin: a Rust
implementation must be able to reimplement this file from the schemas
alone.

Usage: python3 tools/conformance/validate_fixtures.py [scenarios_dir]
Exit 0 = all valid; 1 = any invalid/unknown-schema payload.
"""
import json
import pathlib
import sys

try:
    import jsonschema
except ImportError:
    sys.exit("jsonschema is required: pip install jsonschema")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = REPO / "schemas"


def load_schemas():
    schemas = {}
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text())
        key = (schema["properties"]["schema_name"]["const"],
               schema["properties"]["schema_version"]["const"])
        schemas[key] = schema
    return schemas


def main(scenarios_dir: str) -> int:
    schemas = load_schemas()
    failures = 0
    payloads = sorted(pathlib.Path(scenarios_dir).rglob("*.json"))
    if not payloads:
        print(f"no payloads found under {scenarios_dir}")
        return 1
    for payload_path in payloads:
        try:
            payload = json.loads(payload_path.read_text())
            key = (payload.get("schema_name"),
                   payload.get("schema_version"))
            if key not in schemas:
                raise ValueError(f"unknown schema {key}")
            jsonschema.validate(payload, schemas[key])
            print(f"OK   {payload_path}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {payload_path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "tests" / "conformance" / "scenarios")
    sys.exit(main(target))
