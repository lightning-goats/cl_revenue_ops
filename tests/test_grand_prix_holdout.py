"""Holdout seeds are private, immutable, and verifiable by commitment."""

import importlib.util
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "grand_prix_holdout.py"


def _module():
    spec = importlib.util.spec_from_file_location("grand_prix_holdout", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_seal_prints_only_commitment_and_reveal_verifies(tmp_path):
    module = _module()
    secret = tmp_path / "holdout.json"
    public = module.seal(secret)
    assert set(public) == {"schema", "commitment"}
    assert module.COMMITMENT_RE.fullmatch(public["commitment"])
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    revealed = module.reveal(secret, public["commitment"])
    assert revealed["verified"] is True
    assert revealed["seed"] > 0
    assert len(revealed["salt"]) == 64


def test_seal_refuses_to_replace_secret(tmp_path):
    module = _module()
    secret = tmp_path / "holdout.json"
    module.seal(secret)
    with pytest.raises(module.HoldoutError, match="replace"):
        module.seal(secret)


def test_reveal_fails_on_wrong_commitment(tmp_path):
    module = _module()
    secret = tmp_path / "holdout.json"
    module.seal(secret)
    with pytest.raises(module.HoldoutError, match="does not match"):
        module.reveal(secret, "sha256:" + "0" * 64)


def test_reveal_rejects_permissive_secret_mode(tmp_path):
    module = _module()
    secret = tmp_path / "holdout.json"
    public = module.seal(secret)
    secret.chmod(0o644)
    with pytest.raises(module.HoldoutError, match="0600"):
        module.reveal(secret, public["commitment"])
