"""Phase 3D: rebalance-mode descriptors (Workstream F4)."""
import pathlib

from modules.rebalance_modes import MODES, engine_kwargs


def test_all_five_spec_modes_present():
    assert set(MODES) == {"normal", "hot_protection", "structural_drain",
                          "manual", "diagnostic"}


def test_kwargs_parity_with_legacy_call_sites():
    """The mode table must imply EXACTLY the boolean combinations the
    call sites used before 3D (behavior-identical routing)."""
    assert engine_kwargs("manual") == {
        "reserve_budget": False, "account_costs": False}
    assert engine_kwargs("diagnostic") == {
        "reserve_budget": True, "account_costs": True}
    assert engine_kwargs("normal") == {
        "reserve_budget": True, "account_costs": True}


def test_priority_ladder_matches_spec_table():
    assert MODES["manual"].priority > MODES["hot_protection"].priority \
        > MODES["normal"].priority > MODES["structural_drain"].priority \
        > 0 <= MODES["diagnostic"].priority


def test_accounting_ownership():
    # P4-020: manual callers own their accounting; everything else is
    # engine-owned on the unified rail.
    assert MODES["manual"].accounting_owner == "caller"
    assert not MODES["manual"].reserve_on_rail
    for name in ("normal", "hot_protection", "structural_drain",
                 "diagnostic"):
        assert MODES[name].accounting_owner == "engine", name
        assert MODES[name].reserve_on_rail, name


def test_call_sites_route_through_the_table():
    """Structural pin: no direct boolean mode flags remain at the
    rebalancer's engine call sites."""
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "modules" / "rebalancer.py").read_text()
    assert source.count('engine_kwargs("manual")') == 2
    assert source.count('engine_kwargs("diagnostic")') == 1
    assert "reserve_budget=True,\n                    account_costs=True" \
        not in source
