"""DD2 / P1-001, P1-004: hard ceilings bind even under force.

force may bypass soft gates (deadband, cooldown, per-cycle limits) but NEVER the
absolute rails:
  * revenue-set-fee: the [min_fee_ppm, max_fee_ppm] rail + absolute ceiling.
  * revenue-rebalance: the hard max rebalance amount, and rate-limiting must
    apply to force=false the same as force=true.
"""

from unittest.mock import MagicMock

from modules.config import Config
from tests.plugin_test_utils import load_plugin_module


# --------------------------------------------------------------------------
# set-fee hard rail
# --------------------------------------------------------------------------
def _fee_mod():
    mod = load_plugin_module()
    mod.fee_controller = MagicMock()
    mod.fee_controller.set_channel_fee.side_effect = lambda cid, fee, **kw: {
        "success": True, "channel_id": cid, "fee_ppm": fee,
    }
    mod.config = Config(min_fee_ppm=50, max_fee_ppm=5000)
    mod.force_rate_limiter = MagicMock()
    mod.force_rate_limiter.check_rate_limit.return_value = (True, "")
    return mod


def test_force_set_fee_above_max_clamped_to_max():
    mod = _fee_mod()
    result = mod.revenue_set_fee(mod.plugin, "123x456x0", 999999, force=True)
    assert result["status"] == "success"
    # set_channel_fee must have received the clamped max, not the raw value.
    called_fee = mod.fee_controller.set_channel_fee.call_args[0][1]
    assert called_fee == 5000
    assert result["new_fee_ppm"] == 5000


def test_force_set_fee_below_min_raised_to_min():
    mod = _fee_mod()
    result = mod.revenue_set_fee(mod.plugin, "123x456x0", 0, force=True)
    assert result["status"] == "success"
    called_fee = mod.fee_controller.set_channel_fee.call_args[0][1]
    assert called_fee == 50  # not settable to 0 below min_fee_ppm
    assert result["new_fee_ppm"] == 50


def test_force_set_fee_within_rail_unchanged():
    mod = _fee_mod()
    result = mod.revenue_set_fee(mod.plugin, "123x456x0", 1200, force=True)
    assert result["status"] == "success"
    called_fee = mod.fee_controller.set_channel_fee.call_args[0][1]
    assert called_fee == 1200
    assert "clamped_to_rail" not in result


def test_force_set_fee_respects_absolute_ceiling():
    mod = load_plugin_module()
    mod.fee_controller = MagicMock()
    mod.fee_controller.set_channel_fee.side_effect = lambda cid, fee, **kw: {
        "success": True, "channel_id": cid, "fee_ppm": fee,
    }
    # max_fee_ppm absurdly high; the absolute ABS_MAX_FEE_PPM must still bind.
    abs_max = mod.FeeController.ABS_MAX_FEE_PPM
    mod.config = Config(min_fee_ppm=50, max_fee_ppm=abs_max + 500_000)
    mod.force_rate_limiter = MagicMock()
    mod.force_rate_limiter.check_rate_limit.return_value = (True, "")

    result = mod.revenue_set_fee(mod.plugin, "123x456x0", abs_max + 400_000, force=True)
    called_fee = mod.fee_controller.set_channel_fee.call_args[0][1]
    assert called_fee == abs_max


# --------------------------------------------------------------------------
# rebalance rate-limit symmetry + hard amount cap
# --------------------------------------------------------------------------
def _reb_mod():
    mod = load_plugin_module()
    mod.rebalancer = MagicMock()
    mod.rebalancer.manual_rebalance.return_value = {"success": True, "message": "completed"}
    mod.config = Config(rebalance_min_amount=50000, rebalance_max_amount=5_000_000)
    mod.force_rate_limiter = MagicMock()
    mod.force_rate_limiter.check_rate_limit.return_value = (True, "")
    return mod


def test_rebalance_force_false_is_rate_limited():
    mod = _reb_mod()
    mod.force_rate_limiter.check_rate_limit.return_value = (False, "rate limited")
    result = mod.revenue_rebalance(mod.plugin, "1x1x1", "2x2x2", 100000, force=False)
    assert result["status"] == "error"
    assert "rate limited" in result["error"]
    mod.force_rate_limiter.check_rate_limit.assert_called_once_with("revenue-rebalance")


def test_rebalance_over_cap_rejected_force_false():
    mod = _reb_mod()
    result = mod.revenue_rebalance(mod.plugin, "1x1x1", "2x2x2", 10_000_000, force=False)
    assert result["status"] == "error"
    assert result["max_amount_sats"] == 5_000_000
    mod.rebalancer.manual_rebalance.assert_not_called()


def test_rebalance_over_cap_rejected_force_true():
    mod = _reb_mod()
    result = mod.revenue_rebalance(mod.plugin, "1x1x1", "2x2x2", 10_000_000, force=True)
    assert result["status"] == "error"
    assert result["max_amount_sats"] == 5_000_000
    mod.rebalancer.manual_rebalance.assert_not_called()


def test_rebalance_under_cap_passes():
    mod = _reb_mod()
    result = mod.revenue_rebalance(mod.plugin, "1x1x1", "2x2x2", 1_000_000, force=False)
    assert result["status"] == "success"
    mod.rebalancer.manual_rebalance.assert_called_once()
