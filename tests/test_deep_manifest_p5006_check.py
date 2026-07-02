"""P5-006: --check must reconcile the manifest against the live source set.

Parsing the recorded manifest alone is blind to a source file added after the
last generate (never chunked, never audited) or a manifest file since removed.
--check enumerates live tracked sources and FAILs on either drift.
"""

import contextlib
import importlib.util
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TOOL = os.path.join(_REPO, "tools", "audit", "deep_manifest.py")


def _load_dm():
    spec = importlib.util.spec_from_file_location("deep_manifest_p5006", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dm = _load_dm()

_MANIFEST_HEADER = (
    "| chunk_id | file | line_start | line_end | tier | blob | owner | status |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def _sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(dm, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(dm, "MANIFEST_PATH", "manifest.md")


def _row(chunk_id, file, start, end, blob, tier=1):
    return f"| {chunk_id} | {file} | {start} | {end} | {tier} | {blob} |  | UNASSIGNED |\n"


def _run_check():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = dm.check_drift()
    return rc, buf.getvalue()


def test_check_flags_new_source_file(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "manifest.md").write_text(
        _MANIFEST_HEADER + _row("cl-revenue-ops.py#1", "cl-revenue-ops.py", 1, 400, "aaa"))
    monkeypatch.setattr(dm, "enumerate_source_files",
                        lambda: ["cl-revenue-ops.py", "modules/newly_added.py"])
    monkeypatch.setattr(dm, "blob_hashes", lambda paths: {"cl-revenue-ops.py": "aaa"})
    monkeypatch.setattr(dm, "current_blob_hash", lambda p: "aaa")

    rc, out = _run_check()
    assert rc == 1
    assert "NEW SOURCE" in out
    assert "modules/newly_added.py" in out


def test_check_flags_removed_source_file(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "manifest.md").write_text(
        _MANIFEST_HEADER
        + _row("cl-revenue-ops.py#1", "cl-revenue-ops.py", 1, 400, "aaa")
        + _row("modules/gone.py#1", "modules/gone.py", 1, 400, "bbb"))
    monkeypatch.setattr(dm, "enumerate_source_files", lambda: ["cl-revenue-ops.py"])
    monkeypatch.setattr(dm, "blob_hashes",
                        lambda paths: {"cl-revenue-ops.py": "aaa", "modules/gone.py": "bbb"})
    monkeypatch.setattr(dm, "current_blob_hash",
                        lambda p: {"cl-revenue-ops.py": "aaa"}.get(p, "bbb"))

    rc, out = _run_check()
    assert rc == 1
    assert "REMOVED SOURCE" in out
    assert "modules/gone.py" in out


def test_check_clean_when_manifest_matches_live_tree(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "manifest.md").write_text(
        _MANIFEST_HEADER + _row("cl-revenue-ops.py#1", "cl-revenue-ops.py", 1, 400, "aaa"))
    monkeypatch.setattr(dm, "enumerate_source_files", lambda: ["cl-revenue-ops.py"])
    monkeypatch.setattr(dm, "blob_hashes", lambda paths: {"cl-revenue-ops.py": "aaa"})
    monkeypatch.setattr(dm, "current_blob_hash", lambda p: "aaa")

    rc, out = _run_check()
    assert rc == 0
    assert "CLEAN" in out
