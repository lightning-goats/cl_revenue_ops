import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "polar_plugin_deploy.py"
SPEC = importlib.util.spec_from_file_location("polar_plugin_deploy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


@pytest.mark.parametrize(
    "name",
    [
        "polar-n1-revenue-node",
        "polar-n42-revenue-node",
    ],
)
def test_validate_container_accepts_only_polar_revenue_role(name):
    assert deploy.validate_container_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "polar-n0-revenue-node",
        "polar-n3-cln-competitor",
        "revenue-node",
        "polar-n3-revenue-node;touch /tmp/bad",
    ],
)
def test_validate_container_rejects_unsafe_or_wrong_role(name):
    with pytest.raises(deploy.DeployError):
        deploy.validate_container_name(name)


def test_plan_is_non_live_and_starts_with_all_safety_rails():
    plan = deploy.build_plan("polar-n7-revenue-node", "abc123")

    assert plan["fresh_root_required"] is True
    assert plan["startup"] == {
        "dry_run": True,
        "daily_budget_sats": 0,
        "persist_paused": True,
    }
    start = plan["plugin_start_command"]
    assert "revenue-ops-dry-run=true" in start
    assert "revenue-ops-daily-budget-sats=0" in start
    assert plan["pause_command"][-2:] == ["paused", "true"]


def test_committed_wrapper_is_copied_from_archive_layout():
    assert deploy.ARCHIVE_WRAPPER.endswith(
        "/tools/cl-revenue-ops-polar-wrapper"
    )
    assert deploy.PLUGIN_WRAPPER.endswith("/cl-revenue-ops-polar-wrapper")
    assert deploy.ARCHIVE_WRAPPER != deploy.PLUGIN_WRAPPER


def test_failed_post_deploy_safety_check_stops_plugin_and_removes_root(
    monkeypatch, tmp_path
):
    commands = []

    def fake_run(command, **kwargs):
        command = list(command)
        commands.append(command)
        if command[:2] == ["git", "archive"]:
            output = next(part.split("=", 1)[1] for part in command if part.startswith("--output="))
            Path(output).write_bytes(b"archive")
        return SimpleNamespace(returncode=0, stdout="")

    def fake_capture(command, **kwargs):
        command = list(command)
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        if command[:2] == ["docker", "inspect"]:
            return "true"
        if command[-1] == "revenue-status":
            return '{"status":"running","operator_controls":{"values":{"paused":false,"daily_budget_sats":0}}}'
        if command[-1] == "revenue-ops-dry-run":
            return '{"configs":{"revenue-ops-dry-run":{"value_str":"true"}}}'
        if "sha256sum" in command:
            return "hash  file"
        if command[-1] == "--version":
            return "Python 3.11.0"
        if command[-1] == "freeze":
            return "pyln-client==24.11"
        raise AssertionError(f"unexpected capture: {command}")

    monkeypatch.setattr(deploy, "_run", fake_run)
    monkeypatch.setattr(deploy, "_capture", fake_capture)

    with pytest.raises(deploy.DeployError, match="safety verification"):
        deploy.deploy("polar-n7-revenue-node", "HEAD", tmp_path)

    assert any("subcommand=stop" in command for command in commands)
    assert ["docker", "exec", "polar-n7-revenue-node", "rm", "-rf", "--", deploy.PLUGIN_ROOT] in commands
