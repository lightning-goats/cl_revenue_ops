"""Production-shaped Grand Prix manifests are private, matched, and pure."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "grand_prix_manifest.py"
FIXTURE = ROOT / "tests" / "fixtures" / "competitive_improvement" / "calibration.v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("grand_prix_manifest", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_calibration_is_private_and_reconciled():
    result = _module().validate_calibration(_fixture())
    assert result["valid"] is True
    assert result["channel_count"] == 10


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(peer_id="secret"), "forbidden production field"),
        (lambda value: value["privacy"].update(raw_identifiers_exported=True), "privacy proof"),
        (lambda value: value["channels"]["capacity_histogram_sats"].update(lt_2m=1), "reconcile"),
        (lambda value: value["traffic_30d"]["interarrival_distribution_seconds"].update(lt_1=1), "interarrival"),
    ],
)
def test_bad_calibration_fails_closed(mutate, message):
    module = _module()
    value = copy.deepcopy(_fixture())
    mutate(value)
    with pytest.raises(module.ManifestError, match=message):
        module.validate_calibration(value)


def test_built_topology_is_crossed_matched_and_has_alternatives():
    module = _module()
    topology = module.build_topology(_fixture(), public_seed=20260901)
    result = module.validate_topology(topology)
    assert result == {
        "schema": "polar-grand-prix-topology-v1",
        "valid": True,
        "nodes": 24,
        "channels": 57,
        "traffic_payments": 240,
        "matched_contender_channels": 16,
    }
    assert topology["holdout"]["seed_present"] is False
    directions = {row["direction"] for row in topology["traffic"]}
    assert directions == {"forward", "reverse"}
    reverse_share = sum(row["direction"] == "reverse" for row in topology["traffic"]) / 240
    assert 0.25 < reverse_share < 0.45
    endpoint_edges = [
        row for row in topology["channels"]
        if row["contender_lane"] is not None
        and ({row["source"], row["destination"]} & {
            "cln-payer", "lnd-payer", "cln-sink", "lnd-sink"
        })
    ]
    assert len(endpoint_edges) == 8
    assert min(row["capacity_sats"] for row in endpoint_edges) >= 8_000_000
    implementations = {row["name"]: row["implementation"] for row in topology["nodes"]}
    contender_funded_to_lnd = [
        row for row in topology["channels"]
        if row["source"] in {"identity-a", "identity-b"}
        and implementations[row["destination"]] == "lnd"
    ]
    assert max(row["capacity_sats"] for row in contender_funded_to_lnd) <= 15_000_000


def test_unmatched_contender_portfolio_is_rejected():
    module = _module()
    topology = module.build_topology(_fixture(), public_seed=1)
    edge = next(row for row in topology["channels"] if row["contender_lane"] == "identity-b")
    edge["capacity_sats"] += 1
    with pytest.raises(module.ManifestError, match="not matched"):
        module.validate_topology(topology)


def test_cli_builds_and_validates_without_runtime_dependencies(tmp_path):
    output = tmp_path / "topology.json"
    built = subprocess.run(
        [sys.executable, str(TOOL), "build-topology", str(FIXTURE),
         "--public-seed", "20260901", "--output", str(output)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert built.returncode == 0, built.stderr
    checked = subprocess.run(
        [sys.executable, str(TOOL), "validate-topology", str(output)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert checked.returncode == 0, checked.stderr
    source = TOOL.read_text(encoding="utf-8")
    assert "pyln" not in source
    assert "sendpay" not in source
    assert "ssh" not in source


def test_tournament_sources_have_no_archon_or_did_integration():
    """Tournament identities stay disposable local node-role labels only."""
    paths = [
        ROOT / "docs" / "optimization" / "plans" /
        "2026-09-01-competitive-improvement-program.md",
        ROOT / "tools" / "competitive_improvement_protocol.py",
        ROOT / "tools" / "docker_grand_prix_lab.py",
        ROOT / "tools" / "cl-revenue-ops-lab-wrapper",
        *sorted((ROOT / "tools").glob("grand_prix_*.py")),
        *sorted((ROOT / "tests" / "fixtures" / "competitive_improvement").glob("*.json")),
    ]
    forbidden = ("archon", "did:", "did_key", "did_web", "primary_did")
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in source, f"{path.relative_to(ROOT)} contains {token!r}"
