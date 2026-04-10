"""Tests for v3-specific audit module additions."""


def test_valid_skip_reasons_constant_exists():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS
    assert isinstance(VALID_SKIP_REASONS, frozenset)


def test_valid_skip_reasons_includes_v2_existing():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS
    existing = {
        "inside_band",
        "not_valuable",
        "no_partner",
        "cooldown",
        "no_budget",
        "max_pairs_reached",
        "outcompeted",
        "no_route",
        "route_over_budget",
    }
    assert existing.issubset(VALID_SKIP_REASONS)


def test_valid_skip_reasons_includes_v3_new():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS
    new = {
        "unknown_source_node",
        "unknown_dest_node",
        "unknown_layer",
        "askrene_child_died",
        "path_loops_through_us",
    }
    assert new.issubset(VALID_SKIP_REASONS)


def test_valid_skip_reasons_rejects_random_string():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS
    assert "definitely_not_a_real_reason" not in VALID_SKIP_REASONS


def test_format_pick_includes_router_field():
    from modules.rebalance_audit_v2 import RebalanceAudit
    line = RebalanceAudit.format_pick(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        amount_sats=1000,
        route_cost_sats=5,
        value_score=0.7,
        router="v3",
    )
    assert line.startswith("REBAL_PICK")
    assert "router=v3" in line


def test_format_skip_includes_router_field():
    from modules.rebalance_audit_v2 import RebalanceAudit
    line = RebalanceAudit.format_skip(
        channel_id="100x1x0",
        reason="no_route",
        value_class="hive",
        router="v3",
    )
    assert line.startswith("REBAL_SKIP")
    assert "router=v3" in line


def test_format_pick_defaults_router_to_v2():
    from modules.rebalance_audit_v2 import RebalanceAudit
    line = RebalanceAudit.format_pick(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        amount_sats=1000,
        route_cost_sats=5,
        value_score=0.7,
    )
    assert "router=v2" in line


def test_format_skip_defaults_router_to_v2():
    from modules.rebalance_audit_v2 import RebalanceAudit
    line = RebalanceAudit.format_skip(
        channel_id="100x1x0",
        reason="no_route",
        value_class="hive",
    )
    assert "router=v2" in line


def test_log_pick_passes_router_to_plugin_log():
    from unittest.mock import MagicMock
    from modules.rebalance_audit_v2 import RebalanceAudit

    plugin = MagicMock()
    audit = RebalanceAudit(plugin)
    audit.log_pick(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        amount_sats=1000,
        route_cost_sats=5,
        value_score=0.7,
        router="v3",
    )
    plugin.log.assert_called_once()
    msg = plugin.log.call_args.args[0]
    assert "router=v3" in msg


def test_log_skip_passes_router_to_plugin_log():
    from unittest.mock import MagicMock
    from modules.rebalance_audit_v2 import RebalanceAudit

    plugin = MagicMock()
    audit = RebalanceAudit(plugin)
    audit.log_skip(
        channel_id="100x1x0",
        reason="no_route",
        value_class="hive",
        router="v3",
    )
    plugin.log.assert_called_once()
    msg = plugin.log.call_args.args[0]
    assert "router=v3" in msg
