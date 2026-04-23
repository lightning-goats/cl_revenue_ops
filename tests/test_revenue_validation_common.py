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
