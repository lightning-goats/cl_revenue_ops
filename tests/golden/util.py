"""Golden characterization-test harness (refactor Phase 0).

golden_check(name, actual):
  - normal mode: compare `actual` (after jsonify canonicalization) to
    tests/golden/fixtures/<name>.json; exact equality required.
  - GOLDEN_UPDATE=1: (re)write the fixture from `actual`.

POLICY (docs/planning/refactor.md, Test strategy): never re-record a
fixture just to make a failing test pass. An intentional behavior change
needs a dedicated test and a rationale in the commit message.
"""
import dataclasses
import enum
import json
import os
import pathlib

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


def jsonify(obj):
    """Canonicalize domain objects to JSON-safe structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonify(dataclasses.asdict(obj))
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        return [jsonify(v) for v in seq]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def golden_check(name: str, actual) -> None:
    # Late-bind FIXTURE_DIR through the module so tests can monkeypatch it.
    import tests.golden.util as _self
    path = _self.FIXTURE_DIR / f"{name}.json"
    canonical = jsonify(actual)
    if os.environ.get("GOLDEN_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(canonical, sort_keys=True, indent=2) + "\n"
        )
        return
    assert path.exists(), (
        f"golden fixture missing: {path}\n"
        f"Record it with: GOLDEN_UPDATE=1 python3 -m pytest <this test> "
        f"— then REVIEW the recorded values for plausibility before "
        f"committing."
    )
    expected = json.loads(path.read_text())
    assert canonical == expected, (
        f"golden mismatch for {name!r}.\n"
        f"expected: {json.dumps(expected, sort_keys=True)[:2000]}\n"
        f"actual:   {json.dumps(canonical, sort_keys=True)[:2000]}\n"
        f"If this change is INTENTIONAL, re-record with GOLDEN_UPDATE=1 "
        f"and justify in the commit message; never re-record merely to "
        f"go green."
    )
