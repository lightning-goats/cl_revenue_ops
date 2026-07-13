"""Workstream H (Boltz): recommendation-list batch arbitration."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


def _rec(channel="100x1x0", direction="loop_out", amount=250_000, fee=100):
    return {"channel_id": channel, "direction": direction,
            "amount_sats": amount,
            "economics": {"passes_profit_guard": True,
                          "estimated_swap_fee_sats": fee}}


def _mod(enabled=True):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.config = MagicMock()
    mod.config.snapshot.return_value = SimpleNamespace(
        econ_cycle_boltz_enabled=enabled)
    mod.econ_shadow = None
    return mod


def test_flag_off_untouched():
    mod = _mod(enabled=False)
    recs = [_rec(), _rec("200x1x0")]
    assert mod._arbitrate_boltz_recommendations(recs) is recs


def test_legacy_order_preserved():
    mod = _mod()
    recs = [_rec("900x1x0"), _rec("100x1x0")]  # plan-ranked, not sorted
    survivors = mod._arbitrate_boltz_recommendations(list(recs))
    assert [r["channel_id"] for r in survivors] == ["900x1x0", "100x1x0"]


def test_duplicates_deduped():
    mod = _mod()
    recs = [_rec("100x1x0"), _rec("100x1x0")]
    survivors = mod._arbitrate_boltz_recommendations(recs)
    assert len(survivors) == 1


def test_directions_map_to_intent_types():
    """SWAP_IN vs SWAP_OUT on the same channel are DIFFERENT intents —
    both survive (no false dedup)."""
    mod = _mod()
    recs = [_rec("100x1x0", direction="loop_out"),
            _rec("100x1x0", direction="loop_in")]
    survivors = mod._arbitrate_boltz_recommendations(recs)
    assert len(survivors) == 2


def test_fail_open_on_error(monkeypatch):
    mod = _mod()
    recs = [_rec()]
    monkeypatch.setattr("modules.econ_arbiter.arbitrate",
                        MagicMock(side_effect=RuntimeError("boom")))
    assert mod._arbitrate_boltz_recommendations(recs) is recs


def test_seam_is_wired():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "cl-revenue-ops.py").read_text()
    seam = source.find(
        "recommendations = _arbitrate_boltz_recommendations(recommendations)")
    loop = source.find("for rec in recommendations:", seam)
    assert 0 < seam < loop
