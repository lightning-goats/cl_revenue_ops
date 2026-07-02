from modules.fee_controller import compute_node_receivable_ratio, node_drain_pressure

def _ch(to_us_msat, total_msat, state="CHANNELD_NORMAL"):
    return {"to_us_msat": to_us_msat, "total_msat": total_msat, "state": state}

def test_receivable_ratio_source_heavy():
    # 90% local across two channels -> receivable ratio 0.10
    chans = [_ch(900_000_000, 1_000_000_000), _ch(900_000_000, 1_000_000_000)]
    assert abs(compute_node_receivable_ratio(chans) - 0.10) < 1e-6

def test_receivable_ratio_balanced():
    chans = [_ch(500_000_000, 1_000_000_000)]
    assert abs(compute_node_receivable_ratio(chans) - 0.50) < 1e-6

def test_receivable_ratio_skips_non_normal_and_bad_entries():
    chans = [_ch(900_000_000, 1_000_000_000), _ch(0, 1_000_000_000, state="CHANNELD_AWAITING_LOCKIN"), "garbage", {}]
    # only the first (normal) channel counts -> 0.10
    assert abs(compute_node_receivable_ratio(chans) - 0.10) < 1e-6

def test_receivable_ratio_zero_capacity_safe():
    assert compute_node_receivable_ratio([]) == 1.0

def test_drain_pressure_ramp():
    # target 0.30, floor 0.20
    assert node_drain_pressure(0.35, 0.30, 0.20) == 0.0     # healthy
    assert node_drain_pressure(0.30, 0.30, 0.20) == 0.0     # at target
    assert node_drain_pressure(0.20, 0.30, 0.20) == 1.0     # at floor -> full
    assert node_drain_pressure(0.10, 0.30, 0.20) == 1.0     # below floor -> clamped 1
    assert abs(node_drain_pressure(0.25, 0.30, 0.20) - 0.5) < 1e-6  # midpoint

def test_drain_pressure_degenerate_target_le_floor():
    assert node_drain_pressure(0.15, 0.20, 0.20) == 1.0
    assert node_drain_pressure(0.25, 0.20, 0.20) == 0.0
