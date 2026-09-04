"""Docker-only Grand Prix orchestration is scoped, resumable, and tidy."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "docker_grand_prix_lab.py"


def _module():
    spec = importlib.util.spec_from_file_location("docker_grand_prix_lab_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_images_are_content_addressed():
    module = _module()
    for image in (module.BITCOIND_IMAGE, module.CLN_IMAGE, module.LND_IMAGE):
        assert "@sha256:" in image
        assert len(image.rsplit("@sha256:", 1)[1]) == 64


def _completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_create_is_metadata_only_and_preserves_exact_runtime_mix(tmp_path, monkeypatch):
    docker = _module()
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return _completed(command, returncode=1, stderr="not found")

    monkeypatch.setattr(docker, "_run", fake_run)
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    result = lab.call("create_network", {
        "name": "docker-equivalence",
        "description": "test",
        "nodes": [
            {"implementation": "bitcoind", "count": 1},
            {"implementation": "c-lightning", "count": 15},
            {"implementation": "LND", "count": 7},
        ],
    })

    lightning = result["network"]["nodes"]["lightning"]
    assert len(lightning) == 22
    assert sum(row["implementation"] == "c-lightning" for row in lightning) == 15
    assert sum(row["implementation"] == "LND" for row in lightning) == 7
    assert result["network"]["status"] == "Stopped"
    assert calls[0][:3] == ["docker", "network", "inspect"]
    assert lab.metadata_path.exists()


def test_reported_started_state_is_derived_from_all_expected_containers(tmp_path, monkeypatch):
    docker = _module()
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    metadata = {
        "id": 1_788_000_001,
        "name": "test",
        "status": "Started",
        "nodes": [{"name": "cln-payer", "implementation": "c-lightning"}],
    }
    running = {"revenue-gp-n1788000001-backend1"}
    monkeypatch.setattr(lab, "_container_running", lambda name: name in running)
    assert lab._network_payload(metadata)["status"] == "Stopped"
    running.add("revenue-gp-n1788000001-cln-payer")
    assert lab._network_payload(metadata)["status"] == "Started"


def test_lnd_readiness_uses_short_retryable_probe(tmp_path, monkeypatch):
    docker = _module()
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    calls = []

    def fake_lnd(network_id, node, *arguments, timeout):
        calls.append((network_id, node, arguments, timeout))
        return {"identity_pubkey": "02node"}

    monkeypatch.setattr(lab, "_lnd", fake_lnd)
    result = lab._wait_node(
        1_788_000_001, {"name": "lnd-sink", "implementation": "LND"}
    )
    assert result == {"identity_pubkey": "02node"}
    assert calls == [
        (1_788_000_001, "lnd-sink", ("getinfo",), 10)
    ]


def test_cleanup_removes_only_resolved_run_label_resources(tmp_path, monkeypatch):
    docker = _module()
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    network_id = 1_788_000_001
    lab._write({
        "schema": "docker-grand-prix-lab-v1",
        "id": network_id,
        "name": "test",
        "status": "Started",
        "nodes": [],
    })
    resolved = {
        "container": [f"revenue-gp-n{network_id}-backend1"],
        "network": [f"revenue-gp-n{network_id}"],
        "volume": [f"revenue-gp-n{network_id}-backend1-data"],
    }
    commands = []
    monkeypatch.setattr(lab, "_labeled_names", lambda kind, _network_id: resolved[kind])
    monkeypatch.setattr(lab, "_container_running", lambda _name: False)
    monkeypatch.setattr(
        docker,
        "_run",
        lambda command, **_kwargs: (
            commands.append(command) or _completed(command)
        ),
    )

    result = lab.call("stop_network", {"networkId": network_id})

    assert commands == [
        ["docker", "rm", "-f", resolved["container"][0]],
        ["docker", "network", "rm", resolved["network"][0]],
        ["docker", "volume", "rm", resolved["volume"][0]],
    ]
    assert result["network"]["status"] == "Stopped"
    stored = json.loads(lab.metadata_path.read_text(encoding="utf-8"))
    assert stored["removed_resources"] == {
        "containers": resolved["container"],
        "networks": resolved["network"],
        "volumes": resolved["volume"],
    }


def test_cleanup_fails_closed_on_any_unscoped_resource(tmp_path, monkeypatch):
    docker = _module()
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    monkeypatch.setattr(
        docker,
        "_run",
        lambda command, **_kwargs: _completed(command, stdout="unrelated-volume\n"),
    )
    with pytest.raises(docker.DockerLabError, match="refusing cleanup"):
        lab._labeled_names("volume", 1_788_000_001)


def test_unsupported_operation_and_malformed_inputs_are_non_mutating(tmp_path):
    docker = _module()
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    with pytest.raises(docker.DockerLabError, match="unsupported"):
        lab.call("delete_everything", {})
    with pytest.raises(docker.DockerLabError, match="positive integer"):
        lab.container_name(True, "backend1")
    assert not lab.metadata_path.exists()


@pytest.mark.parametrize("inventory,creates", [("[]", True), ('["existing"]', False)])
def test_bitcoin_wallet_creation_is_idempotent(tmp_path, monkeypatch, inventory, creates):
    docker = _module()
    lab = docker.DockerGrandPrixLab(tmp_path / "state.json")
    calls = []

    def fake_bitcoin(_network_id, *arguments):
        calls.append(arguments)
        return _completed(arguments, stdout=inventory if arguments == ("listwallets",) else "{}")

    monkeypatch.setattr(lab, "_bitcoin", fake_bitcoin)
    lab._ensure_bitcoin_wallet(1)
    assert (("createwallet", "grand-prix") in calls) is creates


def test_cln_channel_open_uses_unambiguous_keyed_arguments():
    source = TOOL.read_text(encoding="utf-8")
    assert '"-k", "fundchannel"' in source
    assert 'f"id={destination_pubkey}"' in source
    assert 'f"amount={capacity}"' in source
    assert 'connect_argument: list[str]' in source
    assert 'if destination["implementation"] == "c-lightning"' in source
    assert '"--numgraphsyncpeers=0"' in source
