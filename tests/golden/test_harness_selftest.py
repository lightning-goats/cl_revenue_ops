"""Self-test for the golden harness (Phase 0 Task 6)."""
import pytest

from tests.golden.util import golden_check, jsonify


def test_jsonify_handles_domain_shapes():
    import dataclasses
    import enum

    class Color(enum.Enum):
        RED = 1

    @dataclasses.dataclass
    class Point:
        x: int
        tag: Color

    assert jsonify(Point(1, Color.RED)) == {"x": 1, "tag": "RED"}
    assert jsonify((1, 2)) == [1, 2]


def test_golden_check_records_then_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.golden.util.FIXTURE_DIR", tmp_path)
    monkeypatch.setenv("GOLDEN_UPDATE", "1")
    golden_check("selftest/example", {"b": 2, "a": (1,)})
    monkeypatch.delenv("GOLDEN_UPDATE")
    golden_check("selftest/example", {"a": (1,), "b": 2})  # order-insensitive
    with pytest.raises(AssertionError):
        golden_check("selftest/example", {"a": [1], "b": 3})


def test_missing_fixture_fails_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.golden.util.FIXTURE_DIR", tmp_path)
    with pytest.raises(AssertionError, match="GOLDEN_UPDATE=1"):
        golden_check("selftest/absent", {"x": 1})
