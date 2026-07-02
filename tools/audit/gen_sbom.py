#!/usr/bin/env python3
"""Generate a CycloneDX 1.5 SBOM for the cl-revenue-ops environment (Phase 3C).

Enumerates every distribution installed in the interpreter that runs this
script and emits a CycloneDX JSON document. Run it with the SAME interpreter
the plugin uses (e.g. the project ``.venv``) so the SBOM reflects the real
runtime supply chain:

    .venv/bin/python tools/audit/gen_sbom.py -o docs/audit/deep/sbom.cyclonedx.json

Stdlib-only and deterministic: components are sorted by canonical name, and the
``serialNumber`` is a uuid5 over the component purls, so re-running on an
unchanged environment yields a byte-identical body (only ``metadata.timestamp``
tracks wall-clock, and it honors ``SOURCE_DATE_EPOCH`` for reproducible builds).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import uuid

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

# Fixed namespace so serialNumber is a deterministic function of the component set.
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/cl_revenue_ops/sbom")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _plugin_version(repo_root: str) -> str:
    path = os.path.join(repo_root, "cl-revenue-ops.py")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r'^PLUGIN_VERSION\s*=\s*[\'"]([^\'"]+)[\'"]', line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return "unknown"


def collect_components():
    seen = {}
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:
            continue
        if not name or not version:
            continue
        key = _canonical(name)
        # De-dup (multiple metadata dirs); keep first deterministic.
        if key in seen:
            continue
        purl = f"pkg:pypi/{key}@{version}"
        licenses = []
        lic = None
        try:
            lic = dist.metadata.get("License")
        except Exception:
            lic = None
        if lic and lic.strip() and lic.strip().lower() != "unknown":
            licenses = [{"license": {"name": lic.strip()}}]
        else:
            # Fall back to License-style Trove classifiers.
            try:
                classifiers = dist.metadata.get_all("Classifier") or []
            except Exception:
                classifiers = []
            for c in classifiers:
                if c.startswith("License :: "):
                    licenses = [{"license": {"name": c.split("::")[-1].strip()}}]
                    break
        comp = {
            "type": "library",
            "bom-ref": purl,
            "name": name,
            "version": version,
            "purl": purl,
        }
        if licenses:
            comp["licenses"] = licenses
        seen[key] = comp
    return [seen[k] for k in sorted(seen)]


def build_bom(repo_root: str):
    components = collect_components()

    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.isdigit():
        ts = _dt.datetime.fromtimestamp(int(epoch), tz=_dt.timezone.utc)
    else:
        ts = _dt.datetime.now(tz=_dt.timezone.utc)
    timestamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    serial_seed = "\n".join(c["purl"] for c in components)
    serial = f"urn:uuid:{uuid.uuid5(_NAMESPACE, serial_seed)}"

    plugin_ver = _plugin_version(repo_root)
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial,
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "cl_revenue_ops",
                    "name": "gen_sbom.py",
                    "version": "1.0.0",
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": f"pkg:pypi/cl-revenue-ops@{plugin_ver}",
                "name": "cl-revenue-ops",
                "version": plugin_ver,
                "description": (
                    "Core Lightning revenue-operations plugin. SBOM captures the "
                    "Python distributions installed in the plugin runtime "
                    "interpreter."
                ),
            },
            "properties": [
                {"name": "python.version", "value": sys.version.split()[0]},
                {"name": "python.executable", "value": sys.executable},
            ],
        },
        "components": components,
    }
    return bom


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a CycloneDX SBOM for the current interpreter.")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(repo_root, "docs", "audit", "deep", "sbom.cyclonedx.json"),
        help="Output path (default: docs/audit/deep/sbom.cyclonedx.json)",
    )
    args = parser.parse_args(argv)

    bom = build_bom(repo_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(bom, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"gen_sbom: wrote {len(bom['components'])} components to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
