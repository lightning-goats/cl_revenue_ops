import json
from pathlib import Path

from tools import revenue_validation_common as mod


def test_load_config_reads_nodes_and_schedule(tmp_path: Path) -> None:
    cfg = tmp_path / "revenue_validation.yaml"
    cfg.write_text(
        """
schedule:
  timezone: America/Denver
  run_time: "06:00"
nodes:
  lnnode:
    t0: "2026-04-23T00:00:00Z"
    transport: ["ssh", "lnnode"]
""".strip(),
        encoding="utf-8",
    )

    data = mod.load_config(cfg)

    assert data["schedule"]["run_time"] == "06:00"
    assert data["nodes"]["lnnode"]["transport"] == ["ssh", "lnnode"]


def test_build_node_command_wraps_ssh_transport() -> None:
    node = {"transport": ["ssh", "lnnode"]}

    cmd = mod.build_node_command(node, "lightning-cli getinfo")

    assert cmd == ["ssh", "lnnode", "lightning-cli getinfo"]


def test_build_node_command_wraps_docker_exec_transport() -> None:
    node = {
        "transport": ["docker", "exec", "cl-hive-node-hive-nexus-02", "sh", "-lc"]
    }

    cmd = mod.build_node_command(node, "lightning-cli getinfo")

    assert cmd == [
        "docker",
        "exec",
        "cl-hive-node-hive-nexus-02",
        "sh",
        "-lc",
        "lightning-cli getinfo",
    ]


def test_node_day_dir_uses_dated_results_path() -> None:
    cfg = {"paths": {"results_root": "results/revenue-validation"}}

    path = mod.node_day_dir(cfg, "2026-04-23", "lnnode")

    assert path == Path("results/revenue-validation/2026-04-23/lnnode")


def test_write_json_file_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "snapshot.json"

    mod.write_json_file(out, {"status": "ok"})

    assert json.loads(out.read_text(encoding="utf-8")) == {"status": "ok"}


def test_run_result_records_command_status() -> None:
    result = mod.RunResult(
        command=["ssh", "lnnode", "lightning-cli getinfo"],
        ok=False,
        stdout="",
        stderr="boom",
        returncode=1,
    )

    assert result.command[0] == "ssh"
    assert result.ok is False
    assert result.returncode == 1


def test_evaluation_identity_is_explicit_and_versioned() -> None:
    node = {
        "evaluation": {
            "id": "optimization-phase0-measurement-preflight-v1",
            "version": 1,
            "state": "preflight",
            "formal_window_active": False,
            "t0": "2026-08-13T00:00:00Z",
        }
    }

    identity = mod.evaluation_identity(node)

    assert identity["id"] == "optimization-phase0-measurement-preflight-v1"
    assert identity["version"] == 1
    assert identity["t0"] == "2026-08-13T00:00:00Z"
    assert identity["formal_window_active"] is False


def test_evaluation_identity_rejects_naive_or_unversioned_new_identity() -> None:
    for evaluation in (
        {"id": "new", "state": "preflight", "formal_window_active": False, "t0": "2026-08-13T00:00:00Z"},
        {"id": "new", "version": 1, "state": "preflight", "formal_window_active": False, "t0": "2026-08-13T00:00:00"},
    ):
        try:
            mod.evaluation_identity({"evaluation": evaluation})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid evaluation identity must fail closed")


def test_evaluation_identity_rejects_noncanonical_ids() -> None:
    for evaluation_id in ("alpha/beta", "alpha?beta", " alpha"):
        try:
            mod.evaluation_identity({
                "evaluation": {
                    "id": evaluation_id,
                    "version": 1,
                    "state": "preflight",
                    "formal_window_active": False,
                    "t0": "2026-08-13T00:00:00Z",
                }
            })
        except ValueError as exc:
            assert "filename-safe" in str(exc)
        else:
            raise AssertionError("noncanonical evaluation id must fail closed")


def test_repository_validation_config_uses_current_nonformal_preflight_identity() -> None:
    config = mod.load_config("config/revenue_validation.yaml")

    identity = mod.evaluation_identity(config["nodes"]["lnnode"])

    assert identity == {
        "id": "optimization-phase0-measurement-preflight-v1",
        "version": 1,
        "state": "preflight",
        "formal_window_active": False,
        "t0": "2026-08-13T00:00:00Z",
    }
    assert identity["t0"] != "2026-04-23T16:31:01Z"
