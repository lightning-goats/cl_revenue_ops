"""Grand Prix protocol validation remains local-only and fail closed."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "competitive_improvement_protocol.py"
FIXTURE = ROOT / "tests" / "fixtures" / "competitive_improvement" / "grand-prix.v1.json"
RESEARCH_FIXTURE = (
    ROOT / "tests" / "fixtures" / "competitive_improvement"
    / "competitor-research.v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("competitive_improvement_protocol", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_valid_and_production_actions_remain_forbidden():
    result = _module().validate_protocol(_fixture())

    assert result["valid"] is True
    assert result["initial_controllers"] == ["revenue_ops", "clboss"]
    assert result["production_actions_authorized"] is False
    assert result["algorithm_arms"] == [
        "competitor_equivalent", "revenue_enhanced", "revenue_incumbent",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["scope"].update(production_actions_authorized=True), "production_actions_authorized"),
        (lambda value: value["scope"].update(xrebalance_series_1_unchanged=False), "xrebalance_series_1_unchanged"),
        (lambda value: value["calibration"]["allowed_aggregate_fields"].pop(), "complete privacy allowlist"),
        (lambda value: value["topology"].update(minimum_nodes=8), "24-32 node"),
        (lambda value: value["traffic"].update(sealed_holdout_seed_commitment="seed=7"), "complete SHA-256 commitment"),
        (lambda value: value["tournament"].update(minimum_volume_retention_ratio=0.9), "volume_retention"),
        (lambda value: value["improvement_loop"].update(promotion_requires_beating_incumbent_and_baseline=False), "promotion_requires"),
    ],
)
def test_unsafe_or_weakened_contract_is_rejected(mutate, message):
    value = copy.deepcopy(_fixture())
    mutate(value)
    module = _module()
    with pytest.raises(module.ProtocolError, match=message):
        module.validate_protocol(value)


def test_cli_is_pure_and_validates_fixture():
    result = subprocess.run([sys.executable, str(TOOL), str(FIXTURE)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["production_actions_authorized"] is False
    source = TOOL.read_text(encoding="utf-8")
    assert "pyln" not in source
    assert "import docker" not in source
    assert "subprocess" not in source
    assert "sendpay" not in source


def test_research_catalog_distinguishes_direct_and_equivalent_comparisons():
    catalog = json.loads(RESEARCH_FIXTURE.read_text(encoding="utf-8"))

    result = _module().validate_research_catalog(catalog)

    assert result["valid"] is True
    assert result["comparison_classes"] == {
        "clboss": "direct_runtime",
        "ln_operator": "algorithm_equivalent",
        "torq": "workflow_equivalent",
    }
    assert result["direct_runtime_statuses"] == {
        "clboss": "admitted", "ln_operator": "blocked", "torq": "blocked",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["cards"][1].update(
                comparison_class="direct_runtime"
            ),
            "only when directly admitted",
        ),
        (
            lambda value: value["cards"][2]["baseline_arm"].update(
                status="executed"
            ),
            "blocked direct runtime cannot be executed",
        ),
        (
            lambda value: value["cards"][0]["source_and_license"].update(
                revision="0" * 40
            ),
            "source revision does not match",
        ),
        (
            lambda value: value["cards"][1].update(safety_invariants=[]),
            "safety_invariants must be a non-empty string list",
        ),
    ],
)
def test_research_catalog_rejects_overclaiming_or_incomplete_cards(mutate, message):
    value = json.loads(RESEARCH_FIXTURE.read_text(encoding="utf-8"))
    mutate(value)
    module = _module()
    with pytest.raises(module.ProtocolError, match=message):
        module.validate_research_catalog(value)


def test_cli_validates_research_catalog_without_dispatching_actions():
    result = subprocess.run(
        [sys.executable, str(TOOL), str(RESEARCH_FIXTURE)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cards"] == ["clboss", "ln_operator", "torq"]
