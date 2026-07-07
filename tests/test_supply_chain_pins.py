"""Phase 3C supply-chain guards.

Covers the three Phase 3C deliverables that live in-tree:
  * requirements.txt is fully, exactly pinned and matches the environment
    (via tools/audit/check_pins.py, the CI gate).
  * tools/audit/gen_sbom.py emits a well-formed CycloneDX 1.5 document.
  * the plugin's non-fatal version-floor helpers behave.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.plugin_test_utils import load_plugin_module

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "audit"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


check_pins = _load("check_pins_mod", TOOLS / "check_pins.py")
gen_sbom = _load("gen_sbom_mod", TOOLS / "gen_sbom.py")


# --------------------------------------------------------------------------
# requirements.txt shape + installed-match
# --------------------------------------------------------------------------

def test_requirements_txt_is_fully_pinned_shape():
    req = ROOT / "requirements.txt"
    checked, problems = check_pins.check(str(req), check_installed=False)
    assert problems == [], problems
    assert checked >= 3  # pyln-client, PyYAML, numpy


def test_requirements_txt_matches_installed_environment():
    """This is what the CI gate asserts; the passing suite proves the env."""
    req = ROOT / "requirements.txt"
    _, problems = check_pins.check(str(req), check_installed=True)
    assert problems == [], problems


def test_check_pins_rejects_loose_specifiers(tmp_path):
    loose = tmp_path / "requirements.txt"
    loose.write_text("numpy>=1.0\nPyYAML\npyln-client==26.4\n")
    checked, problems = check_pins.check(str(loose), check_installed=False)
    # Only the exact pin counts as checked; the two loose lines are problems.
    assert checked == 1
    assert len(problems) == 2
    joined = "\n".join(problems)
    assert "numpy" in joined and "PyYAML" in joined


def test_check_pins_flags_drift(tmp_path):
    drift = tmp_path / "requirements.txt"
    drift.write_text("numpy==0.0.1-doesnotexist\n")
    _, problems = check_pins.check(str(drift), check_installed=True)
    assert len(problems) == 1
    assert "drift" in problems[0]


def test_check_pins_rejects_include_directives(tmp_path):
    inc = tmp_path / "requirements.txt"
    inc.write_text("-r other.txt\nnumpy==2.4.2\n")
    _, problems = check_pins.check(str(inc), check_installed=False)
    assert any("directive" in p for p in problems)


def test_check_pins_cli_exit_zero_on_repo_requirements():
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "check_pins.py"), "--no-installed-check"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.returncode == 0, proc.stdout


def test_check_pins_cli_exit_nonzero_on_loose(tmp_path):
    loose = tmp_path / "requirements.txt"
    loose.write_text("numpy\n")
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "check_pins.py"),
         "--requirements", str(loose), "--no-installed-check"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.returncode == 1, proc.stdout


# --------------------------------------------------------------------------
# no undeclared / forbidden runtime imports
# --------------------------------------------------------------------------

def test_bech32_base58_not_imported_anywhere():
    """Phase 1 reimplemented address validation inline to avoid these deps."""
    for py in ROOT.rglob("*.py"):
        if ".worktrees" in py.parts or ".venv" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import bech32"), py
            assert not stripped.startswith("from bech32"), py
            assert not stripped.startswith("import base58"), py
            assert not stripped.startswith("from base58"), py


# --------------------------------------------------------------------------
# SBOM generator
# --------------------------------------------------------------------------

def test_gen_sbom_is_valid_cyclonedx():
    bom = gen_sbom.build_bom(str(ROOT))
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.5"
    assert bom["serialNumber"].startswith("urn:uuid:")
    assert isinstance(bom["components"], list) and bom["components"]
    names = {c["name"].lower() for c in bom["components"]}
    assert {"numpy", "pyyaml", "pyln-client"} <= names
    for c in bom["components"]:
        assert c["type"] == "library"
        assert c["purl"].startswith("pkg:pypi/")
        assert c["version"]
    # deterministic serial: same env -> same serialNumber
    assert gen_sbom.build_bom(str(ROOT))["serialNumber"] == bom["serialNumber"]


def test_committed_sbom_parses_and_covers_runtime_deps():
    sbom_path = ROOT / "docs" / "audit" / "deep" / "sbom.cyclonedx.json"
    assert sbom_path.exists()
    doc = json.loads(sbom_path.read_text())
    assert doc["specVersion"] == "1.5"
    names = {c["name"].lower() for c in doc["components"]}
    assert {"numpy", "pyyaml", "pyln-client"} <= names


# --------------------------------------------------------------------------
# version-floor helpers (non-fatal probes)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plugin_mod():
    return load_plugin_module()


def test_parse_version_tuple(plugin_mod):
    f = plugin_mod._parse_version_tuple
    assert f("v24.11.1") == (24, 11, 1)
    assert f("2.11.0-beta") == (2, 11, 0)
    assert f("boltzcli version v2.12.0-9bf31be") == (2, 12, 0)
    assert f("24.11.1gl") == (24, 11, 1)
    assert f("") == ()
    assert f(None) == ()
    assert f("nonsense") == ()


def test_version_below_floor(plugin_mod):
    f = plugin_mod._version_below_floor
    assert f("v24.08.1", "24.11.1") is True
    assert f("v24.11.1", "24.11.1") is False
    assert f("v26.4", "24.11.1") is False
    assert f("2.10.0", "2.11.0") is True
    assert f(None, "24.11.1") is None
    assert f("garbage", "24.11.1") is None


def test_floor_constants_match_compat_doc(plugin_mod):
    assert plugin_mod.CLN_VERSION_FLOOR == "24.11.1"
    assert plugin_mod.BOLTZCLI_MIN_VERSION == "2.11.0"
