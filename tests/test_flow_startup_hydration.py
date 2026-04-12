import pytest

from tests.plugin_test_utils import load_plugin_module


NOW = 1_700_000_000


def _compute_start(last_forward_ts, flow_window_days):
    mod = load_plugin_module()
    return mod._compute_forward_hydration_start(last_forward_ts, flow_window_days, NOW)


@pytest.mark.parametrize(
    "flow_window_days, expected_days",
    [
        (7, 14),
        (21, 21),
    ],
)
def test_none_last_forward_ts_covers_window_or_fourteen_days(flow_window_days, expected_days):
    start = _compute_start(None, flow_window_days)

    assert start == NOW - (expected_days * 86400)


def test_non_empty_table_with_meaningful_gap_uses_bounded_overlap_start():
    # 30 days old with a 7-day flow window should be capped by the 15-day floor.
    last_forward_ts = NOW - (30 * 86400)

    start = _compute_start(last_forward_ts, 7)

    assert start == NOW - (15 * 86400)


def test_very_recent_last_forward_ts_returns_none():
    last_forward_ts = NOW - 30 * 60

    assert _compute_start(last_forward_ts, 7) is None
