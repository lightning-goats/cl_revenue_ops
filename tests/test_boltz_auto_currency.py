"""'auto' currency must never reach a boltzcli quote subprocess.

The auto-cycle passes loop_in_currency='auto' / loop_out_currency='auto' into
_build_boltz_balance_plan. BoltzCliManager._norm_currency passes unknown
strings through uppercased, so an unresolved 'auto' would land in the CLI
args as `--to AUTO` / `--from AUTO`. The plan builder therefore resolves
'auto' to a concrete currency (BTC/LBTC, via _select_boltz_currency) at most
once per direction per plan build, before any candidate quote is issued.
"""

from unittest.mock import MagicMock

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
from tests.plugin_test_utils import load_plugin_module

PEER = "02" + "a" * 64


def _channel(peer_id, capacity, local_pct):
    local_msat = int(capacity * 1000 * local_pct / 100.0)
    return {
        "peer_id": peer_id,
        "capacity": capacity,
        "spendable_msat": local_msat,
        "receivable_msat": capacity * 1000 - local_msat,
    }


def _make_module(channels, real_bm=None):
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
    hive_hints = MagicMock()
    hive_hints.get_rebalance_bias.return_value = 1.0
    mod.hive_hints = hive_hints
    mod.hive_router = None
    mod.rebalancer = None

    if real_bm is None:
        real_bm = MagicMock()
        real_bm.budget.return_value = {}
        real_bm.quote.return_value = {"estimated_total_fee_sats": 100}
    mod._require_boltz_manager = MagicMock(return_value=real_bm)
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    mod._boltz_direction_allowed_by_policy = MagicMock(return_value=(True, None))
    mod._boltz_dynamic_channel_tuning = MagicMock(return_value={})
    return mod


def _make_real_bm(tmp_path, cli_calls):
    bm = BoltzCliManager(
        MagicMock(), MagicMock(), BoltzCliConfig(enabled=True, datadir=str(tmp_path))
    )

    def fake_run_json(args, timeout=None):
        cli_calls.append([str(a) for a in args])
        if args and args[0] == "quote":
            return {"boltzFee": 50, "networkFee": 50}
        if args and args[0] == "listswaps":
            return {"swaps": []}
        return {}

    def fake_run(args, timeout=None):
        cli_calls.append([str(a) for a in args])
        return "{}"

    bm._run_json = fake_run_json
    bm._run = fake_run
    return bm


class TestAutoNeverReachesSubprocess:
    def test_plan_quote_cli_args_never_contain_auto(self, tmp_path):
        # One loop-in candidate (10% local) and one loop-out candidate (90%).
        channels = {
            "100x1x0": _channel(PEER, 10_000_000, 10.0),
            "100x2x0": _channel(PEER, 10_000_000, 90.0),
        }
        cli_calls = []
        bm = _make_real_bm(tmp_path, cli_calls)
        mod = _make_module(channels, real_bm=bm)

        plan = mod._build_boltz_balance_plan(
            require_profitable=False,
            loop_in_currency="auto",
            loop_out_currency="auto",
        )

        assert "error" not in plan
        assert plan["total_candidates"] == 2
        quote_calls = [c for c in cli_calls if c and c[0] == "quote"]
        assert quote_calls, "expected at least one quote subprocess call"
        for call in cli_calls:
            for arg in call:
                assert arg.lower() != "auto", f"'auto' leaked into CLI args: {call}"
        # Every quote currency flag is a concrete currency.
        for call in quote_calls:
            for flag in ("--to", "--from"):
                if flag in call:
                    assert call[call.index(flag) + 1] in ("BTC", "LBTC"), call

    def test_auto_resolved_at_most_once_per_direction(self):
        # Two loop-out candidates: resolution (2 quotes) happens once, then
        # one quote per candidate => 4 quote calls total, none with 'auto'.
        channels = {
            "100x1x0": _channel(PEER, 10_000_000, 90.0),
            "100x2x0": _channel(PEER, 10_000_000, 92.0),
        }
        mod = _make_module(channels)
        bm = mod._require_boltz_manager.return_value

        plan = mod._build_boltz_balance_plan(
            require_profitable=False,
            loop_in_currency="auto",
            loop_out_currency="auto",
        )

        assert "error" not in plan
        assert plan["total_candidates"] == 2
        currencies = [c.kwargs.get("currency") for c in bm.quote.call_args_list]
        assert all(str(c).lower() != "auto" for c in currencies), currencies
        assert bm.quote.call_count == 4
        # Plan is annotated with the resolved currency per direction.
        thresholds = plan["thresholds"]
        assert thresholds["loop_out_currency"] == "AUTO"
        assert thresholds["resolved_loop_out_currency"] in ("BTC", "LBTC")
        # No loop-in candidate triggered: 'auto' was never resolved for it.
        assert thresholds["resolved_loop_in_currency"] is None

    def test_concrete_currency_is_passed_through_unresolved(self):
        channels = {"100x1x0": _channel(PEER, 10_000_000, 90.0)}
        mod = _make_module(channels)
        bm = mod._require_boltz_manager.return_value
        mod._select_boltz_currency = MagicMock()

        plan = mod._build_boltz_balance_plan(
            require_profitable=False,
            loop_in_currency="LBTC",
            loop_out_currency="BTC",
        )

        assert "error" not in plan
        mod._select_boltz_currency.assert_not_called()
        assert plan["thresholds"]["resolved_loop_out_currency"] == "BTC"
        assert plan["thresholds"]["resolved_loop_in_currency"] == "LBTC"
        assert bm.quote.call_args.kwargs["currency"] == "BTC"

    def test_resolution_failure_falls_back_to_btc(self):
        channels = {"100x1x0": _channel(PEER, 10_000_000, 90.0)}
        mod = _make_module(channels)
        bm = mod._require_boltz_manager.return_value
        mod._select_boltz_currency = MagicMock(side_effect=Exception("boom"))

        plan = mod._build_boltz_balance_plan(
            require_profitable=False,
            loop_out_currency="auto",
        )

        assert "error" not in plan
        assert plan["total_candidates"] == 1
        assert bm.quote.call_args.kwargs["currency"] == "BTC"
        assert plan["thresholds"]["resolved_loop_out_currency"] == "BTC"
