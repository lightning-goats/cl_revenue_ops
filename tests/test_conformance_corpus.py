"""PR 9 (gap-closure Phase F): conformance-corpus integrity.

- 40 scenario classes exist and every payload validates against the
  schemas via the STANDALONE validator (no plugin imports there);
- regenerating the corpus is byte-identical (the generator is
  deterministic — drift means the reference behavior changed and must
  be a conscious commit);
- the coverage report exists, maps every scenario, and documented gaps
  are visible, not silent.
"""
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCENARIOS = REPO / "tests" / "conformance" / "scenarios"
GENERATOR = REPO / "tools" / "conformance" / "generate_scenarios.py"
VALIDATOR = REPO / "tools" / "conformance" / "validate_fixtures.py"
COVERAGE = REPO / "docs" / "refactor" / "phase0" / "conformance-coverage.md"


def test_forty_scenario_classes_present():
    dirs = sorted(d.name for d in SCENARIOS.iterdir() if d.is_dir())
    assert len(dirs) == 40
    assert dirs[0].startswith("01-") and dirs[-1].startswith("40-")


def test_standalone_validator_accepts_corpus():
    pytest.importorskip("jsonschema")
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(SCENARIOS)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_case_has_required_shape():
    for case_path in sorted(SCENARIOS.glob("*/case.json")):
        case = json.loads(case_path.read_text())
        assert case["schema_name"] == "conformance_case"
        assert case["inputs"] is not None
        assert case["expected"], f"{case_path} has empty expected"


def test_documented_gaps_are_explicit_not_silent():
    gaps = []
    for case_path in sorted(SCENARIOS.glob("*/case.json")):
        case = json.loads(case_path.read_text())
        if case["category"] == "documented_gap":
            gaps.append(case_path.parent.name)
            assert case.get("notes"), \
                f"{case_path}: gap without explanation"
            assert case["expected"] == {"implemented": False}
    # PR 10 closed both former gaps (open-vs-LN+, rebalance-vs-swap)
    # behind econ_conflict_rules_extended — no documented gaps remain.
    assert gaps == []


def test_coverage_report_maps_all_scenarios():
    text = COVERAGE.read_text()
    for d in sorted(d.name for d in SCENARIOS.iterdir() if d.is_dir()):
        assert f"`{d}`" in text, f"coverage report missing {d}"
    assert "Documented gaps" in text


def test_regeneration_is_byte_identical(tmp_path):
    """The corpus is a deterministic function of the reference
    implementation. Drift = behavior change = must be reviewed."""
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(SCENARIOS, baseline)
    result = subprocess.run([sys.executable, str(GENERATOR)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    fresh = sorted(SCENARIOS.rglob("*.json"))
    old = sorted(baseline.rglob("*.json"))
    assert [p.relative_to(SCENARIOS) for p in fresh] == \
        [p.relative_to(baseline) for p in old]
    for new_path, old_path in zip(fresh, old):
        assert new_path.read_text() == old_path.read_text(), \
            f"non-deterministic or drifted: {new_path.name}"
