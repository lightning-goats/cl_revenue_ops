"""Clean-room competitor response models are deterministic and fail closed."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "equivalent_competitor_controller.py"
FIXTURE = (
    ROOT / "tools" / "grand-prix"
    / "equivalent-controllers.v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("equivalent_competitor_controller", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_frozen_models_are_valid_and_claims_are_limited():
    result = _module().validate_models(_fixture())
    assert result["valid"] is True
    assert result["models"] == ["ln_operator", "torq"]
    assert result["catalog_digest"].startswith("sha256:")


def test_model_catalog_pins_the_current_research_catalog():
    model_catalog = _fixture()
    research = (
        ROOT / "tests" / "fixtures" / "competitive_improvement"
        / "competitor-research.v1.json"
    ).read_bytes()
    assert model_catalog["research_catalog_digest"] == (
        "sha256:" + hashlib.sha256(research).hexdigest()
    )


def test_ln_operator_sigmoid_reproduces_published_default_samples():
    module = _module()
    model = _fixture()["models"]["ln_operator"]
    expected = {0.05: 731, 0.20: 690, 0.50: 388, 0.80: 85, 0.95: 44}
    for ratio, fee in expected.items():
        assert module.target_fee_ppm(model, ratio) == pytest.approx(fee, abs=1)


def test_torq_workflow_is_aggressive_but_bounded_and_monotone():
    module = _module()
    model = _fixture()["models"]["torq"]
    fees = [module.target_fee_ppm(model, ratio) for ratio in (0, 0.2, 0.5, 0.8, 1)]
    assert fees == sorted(fees, reverse=True)
    assert 25 <= fees[-1] < fees[0] <= 2000


def test_policy_intents_skip_malformed_offline_and_deadband_rows():
    module = _module()
    model = _fixture()["models"]["ln_operator"]
    rows = [
        {"peer_connected": False, "peer_id": "offline"},
        {"peer_connected": True, "peer_id": "malformed", "total_msat": "bad"},
        {
            "peer_connected": True, "peer_id": "change",
            "fee_proportional_millionths": 100,
            "total_msat": "1000000msat", "to_us_msat": {"msat": 100000},
        },
        {
            "peer_connected": True, "peer_id": "deadband",
            "fee_proportional_millionths": 388,
            "total_msat": 1000000, "to_us_msat": 500000,
        },
    ]
    assert module.policy_intents(model, rows) == [{
        "peer_id": "change", "fee_base_msat": 0, "fee_ppm": 722,
        "previous_fee_ppm": 100,
    }]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["models"].pop("torq"), "LN Operator and Torq"),
        (
            lambda value: value["models"]["torq"].update(
                comparison_class="direct_runtime"
            ),
            "comparison class",
        ),
        (
            lambda value: value["models"]["ln_operator"].update(
                claim_scope="Direct product test"
            ),
            "claim scope",
        ),
        (
            lambda value: value["models"]["torq"]["formula"].update(
                maximum_ppm=6000
            ),
            "fee rails",
        ),
    ],
)
def test_unsafe_or_overclaiming_model_catalog_is_rejected(mutate, message):
    value = copy.deepcopy(_fixture())
    mutate(value)
    module = _module()
    with pytest.raises(module.EquivalentControllerError, match=message):
        module.validate_models(value)


def test_tool_is_pure_and_contains_no_dispatch_surface():
    source = TOOL.read_text(encoding="utf-8")
    for forbidden in (
        "import subprocess", "docker exec", "lightning-cli", "lncli ",
        "from pyln", "import pyln",
    ):
        assert forbidden not in source.casefold()
