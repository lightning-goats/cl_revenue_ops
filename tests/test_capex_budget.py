"""Tests for unified capex budget engine."""

import os
import sys
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config, ConfigSnapshot


class TestCapexBudgetConfig:
    """Capex budget config fields exist with correct defaults."""

    def test_reinvestment_rate(self):
        assert Config().capex_reinvestment_rate == 0.50

    def test_bootstrap_bps(self):
        assert Config().capex_bootstrap_bps == 10

    def test_bootstrap_max_sats(self):
        assert Config().capex_bootstrap_max_sats == 200

    def test_grace_days(self):
        assert Config().capex_grace_days == 14

    def test_exploration_rate(self):
        assert Config().capex_exploration_rate == 0.10

    def test_tactical_rate(self):
        assert Config().capex_tactical_rate == 0.15

    def test_global_envelope(self):
        assert Config().capex_global_envelope_sats == 0

    def test_snapshot_includes_all_fields(self):
        snap = Config().snapshot()
        assert snap.capex_reinvestment_rate == 0.50
        assert snap.capex_bootstrap_bps == 10
        assert snap.capex_bootstrap_max_sats == 200
        assert snap.capex_grace_days == 14
        assert snap.capex_exploration_rate == 0.10
        assert snap.capex_tactical_rate == 0.15
        assert snap.capex_global_envelope_sats == 0
