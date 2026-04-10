"""Tests for exposing capex_probability_budget_bonus as a public runtime control."""


def test_capex_probability_budget_bonus_is_public_runtime_key():
    from modules.config import PUBLIC_RUNTIME_KEYS
    assert "capex_probability_budget_bonus" in PUBLIC_RUNTIME_KEYS


def test_is_public_runtime_key_returns_true_for_probability_bonus():
    from modules.config import Config
    assert Config.is_public_runtime_key("capex_probability_budget_bonus") is True


def test_public_runtime_keys_method_includes_probability_bonus():
    from modules.config import Config
    cfg = Config()
    assert "capex_probability_budget_bonus" in cfg.public_runtime_keys()


def test_public_runtime_dict_reflects_probability_bonus():
    from modules.config import Config
    cfg = Config()
    cfg.capex_probability_budget_bonus = 0.25
    d = cfg.public_runtime_dict()
    assert d["capex_probability_budget_bonus"] == 0.25


def test_classify_runtime_key_returns_public_for_probability_bonus():
    from modules.config import Config
    assert Config.classify_runtime_key("capex_probability_budget_bonus") == "public"
