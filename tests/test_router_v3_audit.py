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
