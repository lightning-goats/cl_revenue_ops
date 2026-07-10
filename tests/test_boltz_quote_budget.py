"""Quote-subprocess budget for _build_boltz_balance_plan.

Every trigger-passing channel used to pay a boltzcli quote subprocess even
though only max_candidates recommendations survive the final slice. The plan
builder now pre-ranks trigger-passing candidates by (severity,
daily_contrib_est) and quotes only the top QUOTE_CANDIDATE_MULTIPLIER (3) x
max_candidates; the rest are skipped with reason 'quote_budget_exceeded'.
The multiplier margin exists because the profit guard needs the quote, so
some quoted candidates fail post-quote and a 1x budget could starve the
final slice.
"""

from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module

PEER = "02" + "c" * 64


def _make_module(channels):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()

    fee_controller = MagicMock()
    fee_controller._get_channels_info.return_value = channels
    fee_controller.get_dts_summary.return_value = {}
    mod.fee_controller = fee_controller

    database = MagicMock()
    database.get_top_route_pairs.return_value = []
    database.get_all_channel_states.return_value = []
    database.get_channel_rebalance_success_rate.return_value = {
        "total": 0,
        "success_rate": 1.0,
    }
    mod.database = database

    mod.profitability_analyzer = None
    mod.rebalancer = None

    bm = MagicMock()
    bm.budget.return_value = {}
    bm.quote.return_value = {"estimated_total_fee_sats": 100}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    mod._boltz_direction_allowed_by_policy = MagicMock(return_value=(True, None))
    mod._boltz_dynamic_channel_tuning = MagicMock(return_value={})
    return mod


def _forty_loop_out_channels():
    """40 channels above the 80% loop-out trigger with distinct severities.

    Channel 100xNx0 has local_pct = 80.4 + 0.4*N, so higher N = higher
    severity. All are large enough to clear the min-amount gate.
    """
    channels = {}
    capacity = 20_000_000
    for n in range(1, 41):
        local_pct = 80.4 + 0.4 * n  # 80.8 .. 96.4
        local_msat = int(capacity * 1000 * local_pct / 100.0)
        channels[f"100x{n}x0"] = {
            "peer_id": PEER,
            "capacity": capacity,
            "spendable_msat": local_msat,
            "receivable_msat": capacity * 1000 - local_msat,
        }
    return channels


class TestQuoteBudget:
    def test_forty_triggered_channels_pay_at_most_fifteen_quotes(self):
        mod = _make_module(_forty_loop_out_channels())
        bm = mod._require_boltz_manager.return_value

        plan = mod._build_boltz_balance_plan(
            require_profitable=False, max_candidates=5
        )

        assert "error" not in plan
        assert bm.quote.call_count <= 15
        # Quoted candidates all survived; the rest were budget-skipped.
        assert plan["total_candidates"] == 15
        budget_skips = [
            s
            for s in plan["skipped_examples"]
            if s.get("reason") == "quote_budget_exceeded"
        ]
        assert budget_skips, "expected quote_budget_exceeded skips"
        assert plan["skipped_count"] == 40 - 15

    def test_ranking_respected_highest_severity_quoted_first(self):
        mod = _make_module(_forty_loop_out_channels())

        plan = mod._build_boltz_balance_plan(
            require_profitable=False, max_candidates=5
        )

        # Channels 26..40 have the highest local_pct => highest severity.
        expected_quoted = {f"100x{n}x0" for n in range(26, 41)}
        quoted = {
            c["channel_id"]
            for c in plan["recommendations"]
        }
        assert quoted <= expected_quoted
        # Everything that hit the budget skip is from the low-severity tail.
        budget_skipped = {
            s["channel_id"]
            for s in plan["skipped_examples"]
            if s.get("reason") == "quote_budget_exceeded"
        }
        assert budget_skipped, "expected quote_budget_exceeded skips"
        assert budget_skipped.isdisjoint(expected_quoted)

    def test_small_candidate_sets_are_unaffected(self):
        # 2 triggered channels with max_candidates=5: budget (15) never binds.
        channels = dict(list(_forty_loop_out_channels().items())[:2])
        mod = _make_module(channels)
        bm = mod._require_boltz_manager.return_value

        plan = mod._build_boltz_balance_plan(
            require_profitable=False, max_candidates=5
        )

        assert "error" not in plan
        assert bm.quote.call_count == 2
        assert plan["total_candidates"] == 2
        assert all(
            s.get("reason") != "quote_budget_exceeded"
            for s in plan["skipped_examples"]
        )
