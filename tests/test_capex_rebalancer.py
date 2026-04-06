"""Tests for capex-aware rebalancer."""

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


class TestCapexConfig:
    """New capex rebalancer config fields exist with correct defaults."""

    def test_reinvestment_rate_default(self):
        cfg = Config()
        assert cfg.rebalance_reinvestment_rate == 0.50

    def test_bootstrap_bps_default(self):
        cfg = Config()
        assert cfg.rebalance_bootstrap_bps == 10

    def test_bootstrap_max_sats_default(self):
        cfg = Config()
        assert cfg.rebalance_bootstrap_max_sats == 200

    def test_grace_days_default(self):
        cfg = Config()
        assert cfg.rebalance_grace_days == 14

    def test_snapshot_includes_capex_fields(self):
        cfg = Config()
        snap = cfg.snapshot()
        assert snap.rebalance_reinvestment_rate == 0.50
        assert snap.rebalance_bootstrap_bps == 10
        assert snap.rebalance_bootstrap_max_sats == 200
        assert snap.rebalance_grace_days == 14
