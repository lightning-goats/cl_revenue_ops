"""E-4.4 (2026-07 econ audit): reverse-swap plan-time cost.

Two defects:
1. The reverse (loop-out) quote omitted the LN routing fee paid to Boltz —
   plan-time totals and every budget gate understated the swap's cost.
2. The recursive fee-summing fallback in _estimate_swap_fee_sats summed a
   parent total AND its nested per-component breakdown (double-count),
   inflating spend accounting / over-throttling the budget gate.
"""

from unittest.mock import MagicMock

from modules.boltz_manager import BoltzCliManager, BoltzCliConfig


def _make_manager(**overrides):
    cfg_kwargs = {
        "enabled": True,
        "cli_path": "/usr/local/bin/boltzcli",
        "datadir": "/tmp/test_boltz",
        "daily_budget_sats": 3000,
        "enforce_budget": True,
    }
    cfg_kwargs.update(overrides)
    cfg = BoltzCliConfig(**cfg_kwargs)
    plugin = MagicMock()
    plugin.log = MagicMock()
    rpc = MagicMock()
    return BoltzCliManager(plugin, rpc, cfg)


class TestReverseQuoteIncludesRoutingFee:

    def test_reverse_quote_adds_routing_component_once(self):
        mgr = _make_manager(routing_fee_limit_ppm=1000)
        mgr._run_json = MagicMock(return_value={"boltzFee": 100, "networkFee": 50})

        q = mgr.quote(amount_sats=1_000_000, swap_type="reverse", currency="BTC")

        assert q["estimated_routing_fee_sats"] == 1_000  # 1M * 1000ppm
        # service+network (150) + routing (1000), counted exactly once.
        assert q["estimated_total_fee_sats"] == 1_150

    def test_per_call_limit_overrides_config(self):
        mgr = _make_manager(routing_fee_limit_ppm=1000)
        mgr._run_json = MagicMock(return_value={"boltzFee": 100})

        q = mgr.quote(amount_sats=1_000_000, swap_type="reverse",
                      currency="BTC", routing_fee_limit_ppm=250)

        assert q["estimated_routing_fee_sats"] == 250

    def test_unset_limit_contributes_zero(self):
        mgr = _make_manager(routing_fee_limit_ppm=0)
        mgr._run_json = MagicMock(return_value={"boltzFee": 100})

        q = mgr.quote(amount_sats=1_000_000, swap_type="reverse", currency="BTC")

        assert q["estimated_routing_fee_sats"] == 0
        assert q["estimated_total_fee_sats"] == 100

    def test_submarine_quote_has_no_routing_component(self):
        """Loop-in pays on-chain; we RECEIVE the LN leg — no routing cost."""
        mgr = _make_manager(routing_fee_limit_ppm=1000)
        mgr._run_json = MagicMock(return_value={"boltzFee": 100})

        q = mgr.quote(amount_sats=1_000_000, swap_type="submarine", currency="LBTC")

        assert q["estimated_routing_fee_sats"] == 0
        assert q["estimated_total_fee_sats"] == 100

    def test_budget_gate_counts_routing_component(self):
        mgr = _make_manager(routing_fee_limit_ppm=1000, daily_budget_sats=1_000)
        mgr.get_budget_status = MagicMock(
            return_value={"remaining_24h_sats_estimate": 1_000}
        )
        # Raw quote fees alone (150) fit the budget; with the 1000-sat
        # routing leg the swap must be rejected.
        check = mgr._enforce_budget_for_quote(
            {"boltzFee": 100, "networkFee": 50}, extra_fee_sats=1_000
        )
        assert check["allowed"] is False
        assert check["estimated_fee_sats"] == 1_150


class TestRecursiveFeeSumGuard:

    def test_parent_total_not_double_counted_with_breakdown(self):
        mgr = _make_manager()
        swap = {
            "fees": {
                "totalFees": 100,
                "breakdown": {"minerFee": 60, "serviceFee": 40},
            }
        }
        # Old fallback: 100 + 60 + 40 = 200 (double-count). Guarded: the
        # level with numeric fee fields is authoritative.
        assert mgr._estimate_swap_fee_sats(swap) == 100

    def test_breakdown_only_payload_still_sums_once(self):
        mgr = _make_manager()
        swap = {"details": {"minerFee": 60, "serviceFee": 40}}
        assert mgr._estimate_swap_fee_sats(swap) == 100

    def test_named_top_level_fields_take_precedence(self):
        mgr = _make_manager()
        swap = {"boltzFee": 10, "networkFee": 5, "nested": {"minerFee": 999}}
        assert mgr._estimate_swap_fee_sats(swap) == 15

    def test_percent_and_rate_fields_ignored(self):
        mgr = _make_manager()
        swap = {"quote": {"feePercent": 50, "feeRate": 12, "minerFee": 30}}
        assert mgr._estimate_swap_fee_sats(swap) == 30
