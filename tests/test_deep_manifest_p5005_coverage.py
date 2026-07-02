"""P5-005: coverage gate must not over-count incidental cross-references.

Only the pinned `file:line@blob` column is a real citation, and it must match
by full repo-relative path, not basename. A bare file:line in a description
cell, or a same-basename file in another directory, must not mark a chunk
COVERED.
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
    spec = importlib.util.spec_from_file_location("deep_manifest_p5005", _TOOL)
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
    monkeypatch.setattr(dm, "LEDGER_PATH", "ledger.md")
    monkeypatch.setattr(dm, "ATTESTATIONS_PATH", "attest.md")


def _row(chunk_id, file, start, end, blob, tier=1):
    return f"| {chunk_id} | {file} | {start} | {end} | {tier} | {blob} |  | UNASSIGNED |\n"


def _ledger(pin_and_desc_rows):
    head = (
        "| ID | severity | dimension | file:line@blob | description | status | fix | test |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    return head + "".join(pin_and_desc_rows)


def test_parse_only_reads_pin_column(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "ledger.md").write_text(_ledger([
        "| P-1 | Low | x | cl-revenue-ops.py:10@abc123 | see modules/foo.py:50 "
        "for context | OPEN | | |\n",
    ]))
    assert dm.parse_ledger_citations(dm.LEDGER_PATH) == [("cl-revenue-ops.py", 10)]


def test_symbolic_blob_pin_still_counts(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "ledger.md").write_text(_ledger([
        "| P-1 | Low | x | modules/flow_analysis.py:1675@map | pin | OPEN | | |\n",
    ]))
    assert dm.parse_ledger_citations(dm.LEDGER_PATH) == [("modules/flow_analysis.py", 1675)]


def test_incidental_reference_does_not_cover_unrelated_chunk(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "manifest.md").write_text(
        _MANIFEST_HEADER
        + _row("cl-revenue-ops.py#1", "cl-revenue-ops.py", 1, 400, "abc123")
        + _row("modules/foo.py#1", "modules/foo.py", 1, 400, "def456")
    )
    (tmp_path / "ledger.md").write_text(_ledger([
        "| P-1 | Low | x | cl-revenue-ops.py:10@abc123 | ref modules/foo.py:50 "
        "incidental | OPEN | | |\n",
    ]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = dm.coverage()
    out = buf.getvalue()
    assert "COVERED: 1/2" in out, out
    assert rc == 1


def test_full_path_match_not_basename(monkeypatch, tmp_path):
    _sandbox(monkeypatch, tmp_path)
    (tmp_path / "manifest.md").write_text(
        _MANIFEST_HEADER
        + _row("modules/database.py#1", "modules/database.py", 1, 400, "aaa111")
        + _row("tools/database.py#1", "tools/database.py", 1, 400, "bbb222")
    )
    (tmp_path / "ledger.md").write_text(_ledger([
        "| P-1 | Low | x | modules/database.py:5@aaa111 | real pin | OPEN | | |\n",
    ]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dm.coverage()
    out = buf.getvalue()
    # The pin to modules/database.py must not cover tools/database.py.
    assert "COVERED: 1/2" in out, out
