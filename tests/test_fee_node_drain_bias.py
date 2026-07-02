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


# ---------------------------------------------------------------------------
# Task 2: config knobs wired end-to-end + runtime-settable
# ---------------------------------------------------------------------------

def test_node_drain_bias_defaults():
    from modules.config import Config
    c = Config()
    assert c.node_drain_bias_enabled is False
    assert c.node_drain_bias_max == 0.3


def test_node_drain_bias_runtime_settable():
    from modules.config import PUBLIC_RUNTIME_KEYS
    assert 'node_drain_bias_enabled' in PUBLIC_RUNTIME_KEYS
    assert 'node_drain_bias_max' in PUBLIC_RUNTIME_KEYS


def test_node_drain_bias_field_types_and_ranges():
    from modules.config import CONFIG_FIELD_TYPES, CONFIG_FIELD_RANGES
    assert CONFIG_FIELD_TYPES['node_drain_bias_enabled'] is bool
    assert CONFIG_FIELD_TYPES['node_drain_bias_max'] is float
    assert CONFIG_FIELD_RANGES['node_drain_bias_max'] == (0.0, 0.5)
    assert 'node_drain_bias_enabled' not in CONFIG_FIELD_RANGES


def test_node_drain_bias_snapshot_mirrors_config():
    from modules.config import Config
    c = Config()
    snap = c.snapshot()
    assert snap.node_drain_bias_enabled == c.node_drain_bias_enabled
    assert snap.node_drain_bias_max == c.node_drain_bias_max


class TestNodeDrainBiasOptionsRegistered:
    """P6-002 regression guard: a config knob added to the Config dataclass
    without a matching plugin.add_option(...) registration is a silent
    no-op at startup (the operator-supplied value is never read). Verify
    both new node-drain-bias knobs are (a) present in CONFIG_FIELD_TYPES
    and (b) actually registered as a plugin option with a default that
    matches the Config dataclass default, using the same AST-based
    extraction as TestUpstreamPatternOptionsRegistered in
    tests/test_rebalancer_module.py (avoids importing the plugin module,
    which would register the whole plugin surface)."""

    FIELD_TO_OPTION = {
        "node_drain_bias_enabled": "revenue-ops-node-drain-bias-enabled",
        "node_drain_bias_max": "revenue-ops-node-drain-bias-max",
    }

    def _plugin_option_default(self, name):
        import ast
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree = ast.parse(open(os.path.join(root, "cl-revenue-ops.py")).read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_option"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            name_node = kw.get("name")
            if isinstance(name_node, ast.Constant) and name_node.value == name:
                default_node = kw.get("default")
                if isinstance(default_node, ast.Constant):
                    return default_node.value
        raise AssertionError(f"option {name} not found")

    def test_both_knobs_in_config_field_types(self):
        from modules.config import CONFIG_FIELD_TYPES
        for field in self.FIELD_TO_OPTION:
            assert field in CONFIG_FIELD_TYPES, f"{field} missing from CONFIG_FIELD_TYPES"

    def test_both_knobs_registered_as_plugin_options_with_matching_defaults(self):
        from modules.config import Config
        cfg = Config()
        for field, option_name in self.FIELD_TO_OPTION.items():
            option_default = self._plugin_option_default(option_name)
            field_default = getattr(cfg, field)
            if isinstance(field_default, bool):
                parsed_bool = str(option_default).lower() in ("true", "1", "yes")
                assert parsed_bool is field_default, (
                    f"{option_name} default {option_default!r} does not match "
                    f"{field}={field_default!r}"
                )
            else:
                assert type(field_default)(option_default) == field_default, (
                    f"{option_name} default {option_default!r} does not match "
                    f"{field}={field_default!r}"
                )
