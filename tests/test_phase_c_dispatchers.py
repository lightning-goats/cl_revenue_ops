"""Phase C (operator-surface reduction 2026-08-01): dispatcher RPCs.

Pins the additive half of Phase C:

1. The new dispatchers (revenue-cycle /
   revenue-budget) route each subcommand
   onto the SAME shared helper the old standalone method calls — same
   underlying calls, same argument names, same result.
2. Removed peer-ban policy actions are rejected without touching
   PolicyManager.
3. Old names keep working and their dict responses gain ONLY an additive
   `deprecation` field; dispatcher responses never carry it.
4. Unknown subcommands return an error dict that lists the valid ones.

See docs/audits/OPERATOR_SURFACE_REDUCTION_2026-08-01.md (§1, §4) and the
2026-08-01 announcement in
docs/refactor/phase0/contract-compatibility-policy.md.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "a" * 64


def _strip(result):
    """Old-name result minus the (asserted-present) deprecation field."""
    assert isinstance(result, dict)
    assert "deprecation" in result, f"old name lost its deprecation field: {result}"
    out = dict(result)
    out.pop("deprecation")
    return out


def _assert_no_deprecation(result):
    assert isinstance(result, dict)
    assert "deprecation" not in result, (
        f"dispatcher response must not carry a deprecation field: {result}"
    )


class _OpenGate:
    """fee_authority_gate stand-in whose execution lease always grants."""

    def execution_lease(self, _operation):
        @contextmanager
        def cm():
            yield None
        return cm()


@pytest.fixture(scope="module")
def mod():
    return load_plugin_module()


@pytest.fixture()
def plugin(mod):
    return mod.plugin



class TestCycleDispatcher:
    def test_fees(self, mod, plugin, monkeypatch):
        monkeypatch.setattr(mod, "fee_authority_gate", _OpenGate())
        monkeypatch.setattr(mod, "run_fee_adjustment", lambda: [1, 2, 3])
        monkeypatch.setattr(mod, "revenue_fee_debug",
                            lambda plugin: {"summary": {}})

        new = mod.revenue_cycle(plugin, "fees")
        old = mod.revenue_fee_cycle(plugin)

        _assert_no_deprecation(new)
        assert new == _strip(old)
        assert new["ok"] is True and new["adjusted_channels"] == 3

    def test_rebalance(self, mod, plugin, monkeypatch):
        rebalancer = MagicMock()
        rebalancer.get_last_decision_summary.return_value = {"action": "hold"}
        rebalancer.rebalance_engine_v2.get_last_cycle_debug.return_value = {
            "candidates": []}
        monkeypatch.setattr(mod, "rebalancer", rebalancer)
        monkeypatch.setattr(mod, "run_rebalance_check", lambda: None)

        new = mod.revenue_cycle(plugin, "rebalance", max_candidates=7)
        old = mod.revenue_rebalance_cycle(plugin, max_candidates=7)

        _assert_no_deprecation(new)
        assert new == _strip(old)
        assert new["status"] == "success"
        for call in (rebalancer.rebalance_engine_v2
                     .get_last_cycle_debug.call_args_list):
            assert call.kwargs == {"max_candidates": 7}

    def test_flow(self, mod, plugin, monkeypatch):
        monkeypatch.setattr(mod, "flow_analyzer", MagicMock())
        monkeypatch.setattr(mod, "run_flow_analysis", lambda: None)

        new = mod.revenue_cycle(plugin, "flow")
        old = mod.revenue_analyze(plugin)

        _assert_no_deprecation(new)
        assert new == _strip(old) == {"status": "Flow analysis triggered"}

    def test_all(self, mod, plugin, monkeypatch):
        monkeypatch.setattr(mod, "fee_authority_gate", _OpenGate())
        controller = MagicMock()
        controller.wake_all_sleeping_channels.return_value = 4
        monkeypatch.setattr(mod, "fee_controller", controller)

        new = mod.revenue_cycle(plugin, "all")
        old = mod.revenue_wake_all(plugin)

        _assert_no_deprecation(new)
        assert new == _strip(old)
        assert new["channels_woken"] == 4

    def test_unknown_subsystem_lists_valid(self, mod, plugin):
        result = mod.revenue_cycle(plugin, "everything")
        assert "error" in result
        assert set(result["valid_subsystems"]) == {
            "fees", "rebalance", "flow", "all"}


# ---------------------------------------------------------------------------
# revenue-planner <view>
# ---------------------------------------------------------------------------

# revenue-budget
# ---------------------------------------------------------------------------

def _wire_budget_sections(mod, monkeypatch):
    monkeypatch.setattr(mod, "_total_cost_budget_status",
                        lambda window_hours=None: {"spent_sats": 10,
                                                   "window_hours": window_hours})
    capex_engine = MagicMock()
    capex_engine.compute_allocations.return_value = SimpleNamespace(
        channel_budgets={},
        priority_class="growth",
        global_envelope_sats=100,
        total_fleet_contribution_sats=0,
        allocated_by_priority_sats={},
    )
    monkeypatch.setattr(mod, "capex_engine", capex_engine)
    monkeypatch.setattr(mod, "data_service", None)


def _drop_clock_fields(capex):
    out = dict(capex)
    out.pop("timestamp", None)
    out.pop("generated_at", None)
    return out


class TestBudgetDispatcher:
    def test_combined_sections_match_old_methods(self, mod, plugin, monkeypatch):
        _wire_budget_sections(mod, monkeypatch)

        new = mod.revenue_budget(plugin)
        _assert_no_deprecation(new)
        assert set(new) == {"total_cost", "capex"}
        for section in new.values():
            _assert_no_deprecation(section)

        old_total = mod.revenue_total_cost_budget(plugin)
        old_capex = mod.revenue_capex_status(plugin)

        assert new["total_cost"] == _strip(old_total)
        assert (_drop_clock_fields(new["capex"])
                == _drop_clock_fields(_strip(old_capex)))

    def test_combined_window_hours_passthrough(self, mod, plugin, monkeypatch):
        _wire_budget_sections(mod, monkeypatch)
        new = mod.revenue_budget(plugin, window_hours=48)
        assert new["total_cost"]["window_hours"] == 48

    def test_ledger_forwards_filters(self, mod, plugin, monkeypatch):
        database = MagicMock()
        database.get_spend_ledger_summary.return_value = {"events": [],
                                                          "count": 0}
        monkeypatch.setattr(mod, "database", database)

        new = mod.revenue_budget(plugin, "ledger", window_hours=48,
                                 include_reservations=True)
        old = mod.revenue_spend_ledger(plugin, window_hours=48,
                                       include_reservations=True)

        _assert_no_deprecation(new)
        assert new == _strip(old)
        for call in database.get_spend_ledger_summary.call_args_list:
            assert call.kwargs == {"window_hours": 48,
                                   "include_reservations": True,
                                   "reservation_limit": 50}

    def test_unknown_section_lists_valid(self, mod, plugin):
        result = mod.revenue_budget(plugin, "capital")
        assert "error" in result
        assert result["valid_sections"] == ["ledger"]


# ---------------------------------------------------------------------------
# Removed peer-ban dispatcher regression
# ---------------------------------------------------------------------------

class TestRemovedPolicyBanActions:
    @pytest.mark.parametrize(
        ("action_input", "normalized_action"),
        [("ban", "ban"), ("BAN", "ban"), (" unban ", "unban"),
         ("list-banned", "list-banned")],
    )
    def test_removed_action_is_unknown_and_does_not_touch_policy_manager(
        self, mod, plugin, monkeypatch, action_input, normalized_action
    ):
        pm = MagicMock()
        monkeypatch.setattr(mod, "policy_manager", pm)

        result = mod.revenue_policy(plugin, action=action_input, peer_id=PEER)

        assert result["error"].startswith(f"Unknown action: {normalized_action}.")
        assert "'ban'" not in result["error"]
        assert "'unban'" not in result["error"]
        assert "'list-banned'" not in result["error"]
        assert pm.mock_calls == []


# ---------------------------------------------------------------------------
# Deprecation-notice text
# ---------------------------------------------------------------------------

class TestDeprecationNotices:

    def test_family_notices_point_at_their_dispatcher(self, mod, plugin,
                                                      monkeypatch):
        monkeypatch.setattr(mod, "fee_authority_gate", _OpenGate())
        monkeypatch.setattr(mod, "run_fee_adjustment", lambda: [])
        monkeypatch.setattr(mod, "revenue_fee_debug", lambda plugin: {})
        assert "revenue-cycle fees" in mod.revenue_fee_cycle(
            plugin)["deprecation"]


        monkeypatch.setattr(mod, "_total_cost_budget_status",
                            lambda window_hours=None: {})
        assert "revenue-budget" in mod.revenue_total_cost_budget(
            plugin)["deprecation"]

        database = MagicMock()
        database.get_spend_ledger_summary.return_value = {}
        monkeypatch.setattr(mod, "database", database)
        assert "revenue-budget ledger" in mod.revenue_spend_ledger(
            plugin)["deprecation"]

    def test_ignore_trio_removal_notice(self, mod, plugin, monkeypatch):
        pm = MagicMock()
        pm.get_all_policies.return_value = []
        monkeypatch.setattr(mod, "policy_manager", pm)

        listed = mod.revenue_list_ignored(plugin)
        assert "2026-09-05" in listed["deprecation"]
        assert "no replacement" in listed["deprecation"]

        # The locked write paths (error responses) carry the notice too.
        ignored = mod.revenue_ignore(plugin, PEER)
        assert "error" in ignored
        assert "no replacement" in ignored["deprecation"]
        unignored = mod.revenue_unignore(plugin, PEER)
        assert "error" in unignored
        assert "no replacement" in unignored["deprecation"]
