"""Config tests for the v3 rebalance router opt-in keys."""


def test_config_has_rebalance_router_default_v2():
    from modules.config import Config
    cfg = Config()
    assert cfg.rebalance_router == "v2"


def test_config_has_askrene_layers_default_hive_fleet():
    from modules.config import Config
    cfg = Config()
    assert cfg.askrene_layers == "hive-fleet"


def test_rebalance_router_in_field_type_map():
    from modules.config import CONFIG_FIELD_TYPES
    assert CONFIG_FIELD_TYPES.get("rebalance_router") is str
    assert CONFIG_FIELD_TYPES.get("askrene_layers") is str


def test_rebalance_router_enum_values_v2_v3():
    from modules.config import STRING_ENUM_VALID_VALUES
    valid = STRING_ENUM_VALID_VALUES.get("rebalance_router")
    assert valid is not None
    assert "v2" in valid
    assert "v3" in valid
    assert "v4" not in valid


def test_snapshot_copies_new_fields():
    from modules.config import Config
    cfg = Config()
    cfg.rebalance_router = "v3"
    cfg.askrene_layers = "hive-fleet,hive-reputation"
    snap = cfg.snapshot()
    assert snap.rebalance_router == "v3"
    assert snap.askrene_layers == "hive-fleet,hive-reputation"
