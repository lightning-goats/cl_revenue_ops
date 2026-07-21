"""P8-002: stall markup must be applied AFTER the risk-premium max().

_calculate_floor documents the formula:

    floor_ppm = max(base_floor, risk_premium) * stall_multiplier

but the original code applied the 20% stall markup to the base floor BEFORE
taking max() with the congestion risk premium. When the risk premium
dominates, the stall multiplier is silently dropped from the floor — an
under-charge. The stall markup must be applied after the risk-premium max()
so it multiplies whichever term wins.
"""

from unittest.mock import MagicMock


from modules.fee_authority import FeeAuthorityGate

def _fee_controller(mock_plugin, mock_database):
    from modules.fee_controller import FeeController

    config = MagicMock()
    return FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())


def test_stall_multiplier_applies_after_risk_premium(mock_plugin, mock_database):
    fc = _fee_controller(mock_plugin, mock_database)

    peer_id = "02" + "c" * 64
    # High stall risk -> 20% markup engaged.
    mock_database.get_peer_latency_stats.return_value = {"avg": 20.0, "std": 0.0}

    # Tiny open/close costs keep the base floor ~21 ppm, while a high
    # sat_per_vbyte makes the congestion risk premium dominate:
    #   risk_premium_ppm = sat_per_vbyte * 150 * 0.001 / 50_000 * 1e6
    #                    = sat_per_vbyte * 3
    sat_per_vbyte = 10_000.0
    chain_costs = {
        "open_cost_sats": 1_000,
        "close_cost_sats": 1_000,
        "sat_per_vbyte": sat_per_vbyte,
    }

    floor = fc._calculate_floor(
        5_000_000, chain_costs=chain_costs, peer_id=peer_id, opener="local"
    )

    risk_premium_ppm = int(sat_per_vbyte * 3)  # 30_000
    # Documented formula: the winning term (the risk premium) is multiplied by
    # the stall multiplier.
    expected = int(risk_premium_ppm * 1.2)  # 36_000
    assert floor == expected
    # And strictly above the un-multiplied risk premium — the under-charge the
    # buggy ordering (max(base*stall, risk)) produced.
    assert floor > risk_premium_ppm


def test_no_stall_no_extra_markup(mock_plugin, mock_database):
    """Without stall risk the floor equals the plain risk-premium max()."""
    fc = _fee_controller(mock_plugin, mock_database)

    peer_id = "02" + "d" * 64
    mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0}

    sat_per_vbyte = 10_000.0
    chain_costs = {
        "open_cost_sats": 1_000,
        "close_cost_sats": 1_000,
        "sat_per_vbyte": sat_per_vbyte,
    }

    floor = fc._calculate_floor(
        5_000_000, chain_costs=chain_costs, peer_id=peer_id, opener="local"
    )

    assert floor == int(sat_per_vbyte * 3)  # 30_000, no stall multiplier
