#!/usr/bin/env python3
"""Install an exact cl_revenue_ops revision into a fresh Polar CLN node.

The default mode prints the mutation plan. ``--apply`` is required to touch
Docker.  The target name is deliberately restricted to the revenue-node role
created by ``polar_mixed_client_lab.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


PLUGIN_ROOT = "/opt/cl_revenue_ops"
PLUGIN_WRAPPER = f"{PLUGIN_ROOT}/cl-revenue-ops-polar-wrapper"
ARCHIVE_WRAPPER = f"{PLUGIN_ROOT}/tools/cl-revenue-ops-polar-wrapper"
CONTAINER_RE = re.compile(r"^polar-n[1-9][0-9]*-revenue-node$")


class DeployError(RuntimeError):
    """The requested deployment is unsafe or failed verification."""


def validate_container_name(container: str) -> str:
    """Return a safe Polar revenue-node name or reject the target."""
    if not CONTAINER_RE.fullmatch(container):
        raise DeployError(
            "container must match polar-n<positive-id>-revenue-node"
        )
    return container


def plugin_start_command(container: str) -> list[str]:
    """Build the safety-first plugin start command."""
    validate_container_name(container)
    return [
        "docker",
        "exec",
        "-u",
        "clightning",
        container,
        "lightning-cli",
        "--network=regtest",
        "-k",
        "plugin",
        "subcommand=start",
        f"plugin={PLUGIN_WRAPPER}",
        "revenue-ops-dry-run=true",
        "revenue-ops-daily-budget-sats=0",
    ]


def pause_command(container: str) -> list[str]:
    """Build the command that persists the operator pause rail."""
    validate_container_name(container)
    return [
        "docker",
        "exec",
        "-u",
        "clightning",
        container,
        "lightning-cli",
        "--network=regtest",
        "revenue-config",
        "set",
        "paused",
        "true",
    ]


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        text=True,
        capture_output=capture,
        cwd=cwd,
    )


def _capture(command: Sequence[str], *, cwd: Path | None = None) -> str:
    return _run(command, capture=True, cwd=cwd).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(container: str, revision: str) -> dict[str, Any]:
    """Return the ordered, credential-free deployment plan."""
    validate_container_name(container)
    return {
        "container": container,
        "revision": revision,
        "plugin_root": PLUGIN_ROOT,
        "fresh_root_required": True,
        "system_packages": ["python3", "python3-venv", "ca-certificates"],
        "runtime_requirements": "requirements.txt",
        "startup": {
            "dry_run": True,
            "daily_budget_sats": 0,
            "persist_paused": True,
        },
        "plugin_start_command": plugin_start_command(container),
        "pause_command": pause_command(container),
    }


def deploy(container: str, revision: str, repo_root: Path) -> dict[str, Any]:
    """Deploy and verify one exact committed revision into a fresh container."""
    validate_container_name(container)
    commit = _capture(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo_root,
    )
    if not commit:
        raise DeployError(f"could not resolve revision {revision!r}")

    running = _capture(
        ["docker", "inspect", "--format", "{{.State.Running}}", container]
    )
    if running != "true":
        raise DeployError(f"container {container!r} is not running")

    root_probe = _run(
        ["docker", "exec", container, "test", "!", "-e", PLUGIN_ROOT],
        check=False,
    )
    if root_probe.returncode != 0:
        raise DeployError(
            f"{PLUGIN_ROOT} already exists in {container}; fresh-only deploy refused"
        )

    root_created = False
    plugin_started = False
    try:
        with tempfile.TemporaryDirectory(prefix="cl-revenue-ops-polar-") as tmp:
            archive = Path(tmp) / f"cl-revenue-ops-{commit[:12]}.tar"
            _run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    f"--output={archive}",
                    commit,
                ],
                cwd=repo_root,
            )
            archive_hash = _sha256(archive)

            _run(["docker", "exec", container, "apt-get", "update"])
            _run(
                [
                    "docker",
                    "exec",
                    container,
                    "apt-get",
                    "install",
                    "-y",
                    "--no-install-recommends",
                    "python3",
                    "python3-venv",
                    "ca-certificates",
                ]
            )
            _run(["docker", "exec", container, "mkdir", "-p", PLUGIN_ROOT])
            root_created = True
            _run(
                [
                    "docker",
                    "cp",
                    str(archive),
                    f"{container}:{PLUGIN_ROOT}/source.tar",
                ]
            )
            _run(
                [
                    "docker",
                    "exec",
                    container,
                    "tar",
                    "-xf",
                    f"{PLUGIN_ROOT}/source.tar",
                    "-C",
                    PLUGIN_ROOT,
                ]
            )
            _run(
                [
                    "docker",
                    "exec",
                    container,
                    "cp",
                    ARCHIVE_WRAPPER,
                    PLUGIN_WRAPPER,
                ]
            )
            _run(
                ["docker", "exec", container, "chmod", "755", PLUGIN_WRAPPER]
            )
            _run(
                [
                    "docker",
                    "exec",
                    container,
                    "python3",
                    "-m",
                    "venv",
                    f"{PLUGIN_ROOT}/.venv",
                ]
            )
            _run(
                [
                    "docker",
                    "exec",
                    container,
                    f"{PLUGIN_ROOT}/.venv/bin/pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    f"{PLUGIN_ROOT}/requirements.txt",
                ]
            )

        _run(plugin_start_command(container))
        plugin_started = True
        _run(pause_command(container))

        # Keep every post-install readback inside the transaction. A malformed
        # status response or failed safety assertion must not leave a plugin
        # tree behind that looks successfully deployed.
        status_raw = _capture(
            [
                "docker",
                "exec",
                "-u",
                "clightning",
                container,
                "lightning-cli",
                "--network=regtest",
                "revenue-status",
            ]
        )
        dry_run_raw = _capture(
            [
                "docker",
                "exec",
                "-u",
                "clightning",
                container,
                "lightning-cli",
                "--network=regtest",
                "listconfigs",
                "revenue-ops-dry-run",
            ]
        )
        hashes = _capture(
            [
                "docker",
                "exec",
                container,
                "sha256sum",
                f"{PLUGIN_ROOT}/cl-revenue-ops.py",
                f"{PLUGIN_ROOT}/modules/fee_controller.py",
                PLUGIN_WRAPPER,
            ]
        )
        python_version = _capture(
            [
                "docker",
                "exec",
                container,
                f"{PLUGIN_ROOT}/.venv/bin/python",
                "--version",
            ]
        )
        packages = _capture(
            [
                "docker",
                "exec",
                container,
                f"{PLUGIN_ROOT}/.venv/bin/pip",
                "freeze",
            ]
        ).splitlines()

        status = json.loads(status_raw)
        dry_run = json.loads(dry_run_raw)
        controls = status.get("operator_controls", {}).get("values", {})
        config_value = (
            dry_run.get("configs", {})
            .get("revenue-ops-dry-run", {})
            .get("value_str")
        )
        if (
            status.get("status") != "running"
            or controls.get("paused") is not True
            or controls.get("daily_budget_sats") != 0
            or config_value != "true"
        ):
            raise DeployError("post-deploy safety verification failed")
    except Exception:
        if plugin_started:
            _run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "clightning",
                    container,
                    "lightning-cli",
                    "--network=regtest",
                    "-k",
                    "plugin",
                    "subcommand=stop",
                    f"plugin={PLUGIN_WRAPPER}",
                ],
                check=False,
            )
        if root_created:
            _run(
                ["docker", "exec", container, "rm", "-rf", "--", PLUGIN_ROOT],
                check=False,
            )
        raise

    return {
        "status": "deployed",
        "container": container,
        "commit": commit,
        "archive_sha256": archive_hash,
        "python": python_version,
        "packages": packages,
        "file_hashes": hashes.splitlines(),
        "safety": {
            "plugin_status": status.get("status"),
            "paused": controls.get("paused"),
            "daily_budget_sats": controls.get("daily_budget_sats"),
            "dry_run": config_value,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        plan = build_plan(args.container, args.revision)
        result = (
            deploy(
                args.container,
                args.revision,
                Path(__file__).resolve().parents[1],
            )
            if args.apply
            else {"status": "plan", **plan}
        )
    except (DeployError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(f"polar plugin deploy failed: {exc}") from exc

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
