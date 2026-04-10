from unittest.mock import MagicMock

from modules.rebalance_audit_v2 import RebalanceAudit


# ------------------------------------------------------------------
# Pure formatter tests
# ------------------------------------------------------------------

def test_format_pick_includes_all_fields():
    msg = RebalanceAudit.format_pick(
        source_channel_id="111x1x0",
        dest_channel_id="222x2x0",
        amount_sats=50_000,
        route_cost_sats=120,
        value_score=0.87,
    )
    assert "REBAL_PICK" in msg
    assert "source=111x1x0" in msg
    assert "dest=222x2x0" in msg
    assert "amount=50000" in msg
    assert "route_cost_sats=120" in msg
    assert "value_score=0.87" in msg


def test_format_skip_includes_reason_and_budget():
    msg = RebalanceAudit.format_skip(
        channel_id="333x3x0",
        reason="skipped_no_budget",
        value_class="hive",
        remaining_budget_sats=0,
    )
    assert "REBAL_SKIP" in msg
    assert "channel=333x3x0" in msg
    assert "reason=skipped_no_budget" in msg
    assert "value_class=hive" in msg
    assert "budget=0" in msg


def test_format_skip_omits_route_cost_when_zero():
    msg = RebalanceAudit.format_skip(
        channel_id="444x4x0",
        reason="skipped_cooldown",
        value_class="active",
        remaining_budget_sats=1000,
        route_cost_sats=0,
    )
    assert "route_cost=" not in msg


def test_format_skip_with_detail():
    msg = RebalanceAudit.format_skip(
        channel_id="555x5x0",
        reason="skipped_policy",
        value_class="profitable",
        remaining_budget_sats=500,
        route_cost_sats=200,
        detail="peer_policy=passive",
    )
    assert "route_cost=200" in msg
    assert "detail=peer_policy=passive" in msg


# ------------------------------------------------------------------
# Plugin logging integration tests
# ------------------------------------------------------------------

def test_log_pick_calls_plugin_log():
    plugin = MagicMock()
    audit = RebalanceAudit(plugin)

    audit.log_pick(
        source_channel_id="111x1x0",
        dest_channel_id="222x2x0",
        amount_sats=50_000,
        route_cost_sats=120,
        value_score=0.87,
    )

    plugin.log.assert_called_once()
    logged_msg = plugin.log.call_args[0][0]
    assert "REBAL_PICK" in logged_msg
    assert "source=111x1x0" in logged_msg
    assert plugin.log.call_args[1]["level"] == "info"


def test_log_skip_calls_plugin_log():
    plugin = MagicMock()
    audit = RebalanceAudit(plugin)

    audit.log_skip(
        channel_id="333x3x0",
        reason="skipped_no_route",
        value_class="hive",
        remaining_budget_sats=2000,
    )

    plugin.log.assert_called_once()
    logged_msg = plugin.log.call_args[0][0]
    assert "REBAL_SKIP" in logged_msg
    assert "reason=skipped_no_route" in logged_msg
    assert plugin.log.call_args[1]["level"] == "info"


def test_log_cycle_summary_format():
    plugin = MagicMock()
    audit = RebalanceAudit(plugin)

    audit.log_cycle_summary(
        selected_count=3,
        skipped_count=7,
        total_valuable=10,
        total_channels=25,
        total_budget_sats=15_000,
    )

    plugin.log.assert_called_once()
    logged_msg = plugin.log.call_args[0][0]
    assert "REBAL_CYCLE" in logged_msg
    assert "selected=3" in logged_msg
    assert "skipped=7" in logged_msg
    assert "valuable=10/25" in logged_msg
    assert "budget_remaining=15000" in logged_msg
    assert plugin.log.call_args[1]["level"] == "info"
