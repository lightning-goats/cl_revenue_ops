"""Phase 0: the standalone conformance validator accepts the seed corpus
and rejects invalid payloads. The validator must import nothing from
modules/ or cl-revenue-ops.py (cross-language portability requirement)."""
import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("jsonschema")

REPO = pathlib.Path(__file__).resolve().parent.parent
VALIDATOR = REPO / "tools" / "conformance" / "validate_fixtures.py"
CORPUS = REPO / "tests" / "conformance" / "scenarios"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        capture_output=True, text=True, cwd=REPO,
    )


def test_validator_passes_seed_corpus():
    res = _run(str(CORPUS))
    assert res.returncode == 0, res.stdout + res.stderr


def test_validator_rejects_bad_payload(tmp_path):
    bad = tmp_path / "scenarios" / "broken"
    bad.mkdir(parents=True)
    (bad / "snapshot.json").write_text(json.dumps({
        "schema_name": "economic_snapshot",
        "schema_version": 0,
        "snapshot_id": "x",
        # missing required fields entirely
    }))
    res = _run(str(tmp_path / "scenarios"))
    assert res.returncode != 0
    assert "broken" in res.stdout + res.stderr


def test_validator_imports_no_plugin_code():
    text = VALIDATOR.read_text()
    assert "import modules" not in text
    assert "from modules" not in text
    assert "cl-revenue-ops" not in text
