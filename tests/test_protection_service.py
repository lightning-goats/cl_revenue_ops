"""Phase 3C: lifecycle/protection ownership service.

The close-protection goldens (via planner shims) are the parity oracle;
these tests exercise the service directly."""
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules import protection_service
from modules.classification import ChannelRole
from modules.protection_service import (
    ChannelLifecycle,
    close_protections,
    derive_lifecycle,
    lnplus_contract_protection,
    policy_close_block,
)

NOW = 1_752_400_000


def test_service_is_pure():
    source = inspect.getsource(protection_service)
    for forbidden in ("self.plugin", ".rpc.", "from .database",
                      "import time", "import random", "Database("):
        assert forbidden not in source, forbidden
    import_lines = [line for line in source.splitlines()
                    if line.startswith(("import ", "from "))]
    allowed = {"from __future__ import annotations",
               "from enum import Enum",
               "from typing import Any, Optional, Tuple",
               "from .classification import ChannelRole",
               "from .econ_snapshot import Protection",
               "from .econ_types import UnixTime"}
    assert set(import_lines) <= allowed, import_lines


def _prof(**over):
    base = dict(role_30d=ChannelRole.BALANCED,
                channel_role=ChannelRole.BALANCED,
                marginal_roi_percent=-90.0, window_30d_available=True,
                sourced_fee_30d_msat=0, days_open=100,
                revenue=SimpleNamespace(sourced_fee_contribution_sats=0))
    base.update(over)
    return SimpleNamespace(**base)


class TestPolicyGate:
    def _policy(self, strategy="dynamic", tags=()):
        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = strategy
        policy.has_tag.side_effect = lambda t: t in tags
        return policy

    def test_exact_legacy_reason_strings(self):
        assert policy_close_block(self._policy("static")) == \
            "Channel has static policy — close blocked"
        assert policy_close_block(self._policy("dynamic", ("no_close",))) \
            == "Channel tagged 'no_close' — close blocked"
        assert policy_close_block(self._policy("dynamic")) is None


class TestLnplusContract:
    def test_window(self):
        p = lnplus_contract_protection(NOW, 3, "24328")
        assert p is not None
        assert p.reason == "lnplus_contract" and p.owner == "lnplus"
        assert p.expires_at.value == NOW + 3 * 30 * 86400

    def test_garbage_returns_none(self):
        assert lnplus_contract_protection(None, 3, "x") is None
        assert lnplus_contract_protection(NOW, 0, "x") is None


class TestAggregation:
    def test_typed_protections_from_all_owners(self):
        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "dynamic"
        policy.has_tag.side_effect = lambda t: t == "no_close"
        result = close_protections(
            scid_display="111x222x0",
            prof=_prof(role_30d=ChannelRole.INBOUND_GATEWAY,
                       marginal_roi_percent=-10.0),
            flow_metrics=SimpleNamespace(confidence=0.9, forward_count=50),
            route_pair_channels=set(),
            policy=policy,
            lnplus_obligation={"opened_at": NOW, "duration_months": 3,
                               "swap_id": "24328"},
        )
        owners = [p.owner for p in result]
        assert owners == ["close_protection", "operator_policy", "lnplus"]
        assert result[0].reason == "INBOUND_GATEWAY"
        assert result[2].expires_at is not None

    def test_unprotected_channel_is_empty(self):
        result = close_protections(
            scid_display="111x222x0", prof=_prof(),
            flow_metrics=SimpleNamespace(confidence=0.9, forward_count=50),
            route_pair_channels=set())
        assert result == ()


class TestLifecycle:
    def test_precedence(self):
        assert derive_lifecycle(staged_for_close=True) is \
            ChannelLifecycle.RECYCLING
        assert derive_lifecycle(opening=True) is ChannelLifecycle.OPENING
        from modules.econ_snapshot import Protection
        prot = (Protection("x", "y", None),)
        assert derive_lifecycle(protections=prot) is \
            ChannelLifecycle.PROTECTED
        assert derive_lifecycle(underperforming=True) is \
            ChannelLifecycle.UNDERPERFORMING
        assert derive_lifecycle() is ChannelLifecycle.PRODUCTIVE
        # closing-track outranks protection
        assert derive_lifecycle(staged_for_close=True, protections=prot) \
            is ChannelLifecycle.RECYCLING

    def test_lifecycle_values_match_snapshot_schema(self):
        from modules.econ_snapshot import LIFECYCLES
        assert {m.name for m in ChannelLifecycle} == LIFECYCLES
