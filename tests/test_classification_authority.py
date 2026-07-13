"""Phase 3B: single classification authority.

Golden fixtures (role30d) plus the flow-analysis suite are the parity
oracle; these tests pin the extraction pattern itself."""
import inspect

from modules import classification
from modules.classification import ChannelRole, ChannelState


def test_authority_is_pure():
    """No I/O, clock, randomness, or plugin plumbing — scan real
    dependency patterns, not docstring prose."""
    source = inspect.getsource(classification)
    for forbidden in ("self.plugin", ".rpc.", "from .database",
                      "import time", "import random", "import json",
                      "import sqlite3", "Database("):
        assert forbidden not in source, forbidden
    import_lines = [line for line in source.splitlines()
                    if line.startswith(("import ", "from "))]
    allowed = {"from __future__ import annotations",
               "from enum import Enum",
               "from typing import Optional"}
    assert set(import_lines) <= allowed, import_lines


def test_enums_re_exported_by_identity():
    from modules.flow_analysis import ChannelState as FlowCS
    from modules.profitability_analyzer import ChannelRole as ProfCR
    assert FlowCS is ChannelState
    assert ProfCR is ChannelRole


def test_constants_aliased():
    from modules import flow_analysis
    assert flow_analysis.SINK_ENTER_OUTBOUND_RATIO == \
        classification.SINK_ENTER_OUTBOUND_RATIO
    assert flow_analysis.KALMAN_BALANCE_VETO_RATIO == \
        classification.KALMAN_BALANCE_VETO_RATIO


class TestFlowStateDecision:
    def test_threshold_stage(self):
        common = dict(source_threshold=0.05, sink_threshold=-0.05,
                      outbound_ratio=0.5, previous_state=None, turnover=0.02)
        assert classification.flow_state(
            kalman_ratio=0.10, **common) is ChannelState.SOURCE
        assert classification.flow_state(
            kalman_ratio=-0.10, **common) is ChannelState.SINK
        assert classification.flow_state(
            kalman_ratio=0.0, **common) is ChannelState.BALANCED_ACTIVE

    def test_hysteresis_bands(self):
        # A prior SINK stays SINK down to the exit band (0.72 < r < 0.78).
        assert classification.classify_balance_position(
            0.75, "sink", 0.0, 0.0) is ChannelState.SINK
        assert classification.classify_balance_position(
            0.75, None, 0.0, 0.0) is not ChannelState.SINK

    def test_direction_veto(self):
        # Draining (positive kalman beyond veto) forbids SINK label.
        assert classification.classify_balance_position(
            0.90, None, 0.10, 0.0) is not ChannelState.SINK
        # Filling forbids SOURCE label.
        assert classification.classify_balance_position(
            0.10, None, -0.10, 0.0) is not ChannelState.SOURCE

    def test_dormant_vs_balanced(self):
        assert classification.classify_balance_position(
            0.5, None, 0.0, 0.0) is ChannelState.DORMANT
        assert classification.classify_balance_position(
            0.5, None, 0.03, 0.0) is ChannelState.BALANCED


class TestRevenueRoleDecision:
    def test_matches_golden_scenarios(self):
        # Mirrors the Phase 0 role30d golden fixtures directly.
        assert classification.revenue_role_30d(
            window_30d_available=False, forward_count_30d=0,
            sourced_forward_count_30d=0,
            lifetime_role=ChannelRole.OUTBOUND_GATEWAY,
        ) is ChannelRole.OUTBOUND_GATEWAY
        assert classification.revenue_role_30d(
            window_30d_available=True, forward_count_30d=2,
            sourced_forward_count_30d=40,
            lifetime_role=ChannelRole.BALANCED,
        ) is ChannelRole.INBOUND_GATEWAY
        assert classification.revenue_role_30d(
            window_30d_available=True, forward_count_30d=3,
            sourced_forward_count_30d=4,
            lifetime_role=ChannelRole.BALANCED,
        ) is ChannelRole.DORMANT
        assert classification.revenue_role_30d(
            window_30d_available=True, forward_count_30d=20,
            sourced_forward_count_30d=22,
            lifetime_role=ChannelRole.BALANCED,
        ) is ChannelRole.BALANCED
