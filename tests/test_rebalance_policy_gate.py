"""Lazy-eval audit F0: the native rebalance pipeline never evaluated peer
policy — `revenue-policy set <peer> rebalance=disabled` and strategy=passive
(the canonical revenue-ignore) had NO effect on the highest-frequency spend
path. The engine now carries a policy gate: cheap eager filter at candidate
selection plus a mandatory lazy re-check at execution.

Direction semantics (a rebalance DRAINS the source and FILLS the dest):
- source peer: rebalance_mode in (disabled, sink_only) forbids draining
- dest peer:   rebalance_mode in (disabled, source_only) forbids filling
- strategy passive on either side forbids the pair entirely
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.policy_manager import FeeStrategy, RebalanceMode
from modules.rebalance_engine_v2 import RebalanceEngine


def _policy(strategy=FeeStrategy.DYNAMIC, mode=RebalanceMode.ENABLED):
    policy = MagicMock()
    policy.strategy = strategy
    policy.rebalance_mode = mode
    return policy


def _engine(policies=None):
    plugin = MagicMock()
    cfg = SimpleNamespace()
    database = MagicMock()
    manager = MagicMock()
    policies = policies or {}
    manager.get_policy.side_effect = (
        lambda peer: policies.get(peer, _policy()))
    return RebalanceEngine(plugin=plugin, config=cfg, database=database,
                           policy_manager=manager)


SRC = "02" + "a" * 64
DST = "02" + "b" * 64


def _pair():
    return SimpleNamespace(source_peer_id=SRC, dest_peer_id=DST,
                           source_channel_id="100x1x0",
                           dest_channel_id="200x1x0")


class TestPairPolicyGate:
    def test_enabled_both_sides_allows(self):
        ok, _ = _engine()._pair_policy_allowed(_pair())
        assert ok is True

    def test_source_drain_forbidden(self):
        for mode in (RebalanceMode.DISABLED, RebalanceMode.SINK_ONLY):
            engine = _engine({SRC: _policy(mode=mode)})
            ok, reason = engine._pair_policy_allowed(_pair())
            assert ok is False, mode
            assert "source" in reason

    def test_dest_fill_forbidden(self):
        for mode in (RebalanceMode.DISABLED, RebalanceMode.SOURCE_ONLY):
            engine = _engine({DST: _policy(mode=mode)})
            ok, reason = engine._pair_policy_allowed(_pair())
            assert ok is False, mode
            assert "dest" in reason

    def test_passive_strategy_forbids_either_side(self):
        for peer in (SRC, DST):
            engine = _engine({peer: _policy(strategy=FeeStrategy.PASSIVE)})
            ok, reason = engine._pair_policy_allowed(_pair())
            assert ok is False
            assert "passive" in reason

    def test_source_only_source_and_sink_only_dest_allowed(self):
        engine = _engine({SRC: _policy(mode=RebalanceMode.SOURCE_ONLY),
                          DST: _policy(mode=RebalanceMode.SINK_ONLY)})
        ok, _ = engine._pair_policy_allowed(_pair())
        assert ok is True

    def test_no_policy_manager_fails_closed(self):
        engine = RebalanceEngine(plugin=MagicMock(), config=SimpleNamespace(),
                                 database=MagicMock())
        engine._policy_manager = None  # explicit: simulate broken init
        ok, reason = engine._pair_policy_allowed(_pair())
        assert ok is False
        assert "fail closed" in reason

    def test_policy_error_fails_closed(self):
        manager = MagicMock()
        manager.get_policy.side_effect = RuntimeError("db gone")
        engine = RebalanceEngine(plugin=MagicMock(), config=SimpleNamespace(),
                                 database=MagicMock(), policy_manager=manager)
        ok, reason = engine._pair_policy_allowed(_pair())
        assert ok is False
        assert "fail closed" in reason
