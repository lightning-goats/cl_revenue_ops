#!/usr/bin/env python3
"""Verify the hermes listforwards-window chain is lossless per node.

Read-only audit over /home/sat/cl-mycelium-hermes (or $CL_MYCELIUM_HERMES_ROOT).
Checks, per node:
  - chain contiguity: each window's start equals the previous window's next_start
  - truncated windows are retried (same start reappears) rather than skipped
  - duplicate coverage: overlapping windows are detected and quantified
  - totals: settled forward count and fee_msat sum after dedup, for
    reconciliation against revenue-history / revenue_dashboard

Exit code 0 when every chain is contiguous (truncation retries allowed),
1 when any gap is found.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CL_MYCELIUM_HERMES_ROOT", "/home/sat/cl-mycelium-hermes"))
NODES = ("hive-nexus-01", "hive-nexus-02")


def iter_windows(node_root: Path):
    for path in sorted(node_root.glob("*/*/commands/listforwards-window.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.loads(fh.read(), strict=False)
        except Exception as exc:
            yield path, None, f"unreadable: {exc}"
            continue
        meta = payload.get("_window") if isinstance(payload, dict) else None
        if not isinstance(meta, dict):
            yield path, None, "missing _window metadata"
            continue
        yield path, (meta, payload.get("forwards") or []), None


def audit_node(node: str) -> dict:
    node_root = ROOT / node
    findings = {
        "node": node,
        "windows": 0,
        "unreadable": [],
        "gaps": [],
        "overlaps": [],
        "truncated_windows": 0,
        "truncated_unretried": [],
        "forwards_total": 0,
        "forwards_deduped": 0,
        "settled_count": 0,
        "settled_fee_msat": 0,
        "index_min": None,
        "index_max": None,
    }
    prev_next_start = None
    prev_path = None
    pending_truncated_start = None
    seen_keys: set[tuple] = set()

    for path, item, err in iter_windows(node_root):
        if err:
            findings["unreadable"].append({"path": str(path), "error": err})
            continue
        meta, forwards = item
        findings["windows"] += 1
        start = meta.get("start")
        next_start = meta.get("next_start")

        if prev_next_start is not None and start != prev_next_start:
            if start is not None and prev_next_start is not None and start < prev_next_start:
                findings["overlaps"].append({
                    "path": str(path), "start": start, "expected": prev_next_start,
                })
            else:
                findings["gaps"].append({
                    "path": str(path), "start": start, "expected": prev_next_start,
                    "prev": str(prev_path),
                })
        if meta.get("truncated"):
            findings["truncated_windows"] += 1
            pending_truncated_start = start
        elif pending_truncated_start is not None:
            if start != pending_truncated_start:
                findings["truncated_unretried"].append({
                    "path": str(path), "truncated_start": pending_truncated_start, "start": start,
                })
            pending_truncated_start = None

        for fwd in forwards:
            findings["forwards_total"] += 1
            key = (
                fwd.get("created_index"),
                fwd.get("updated_index"),
                fwd.get("in_channel"),
                fwd.get("in_htlc_id"),
                fwd.get("status"),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            findings["forwards_deduped"] += 1
            upd = fwd.get("updated_index")
            if isinstance(upd, int):
                if findings["index_min"] is None or upd < findings["index_min"]:
                    findings["index_min"] = upd
                if findings["index_max"] is None or upd > findings["index_max"]:
                    findings["index_max"] = upd
            if fwd.get("status") == "settled":
                findings["settled_count"] += 1
                fee = fwd.get("fee_msat")
                if isinstance(fee, str) and fee.endswith("msat"):
                    fee = fee[:-4]
                try:
                    findings["settled_fee_msat"] += int(fee)
                except (TypeError, ValueError):
                    pass

        prev_next_start = next_start
        prev_path = path

    return findings


def main() -> int:
    ok = True
    for node in NODES:
        f = audit_node(node)
        status = "CONTIGUOUS" if not f["gaps"] and not f["unreadable"] else "BROKEN"
        if f["gaps"] or f["unreadable"]:
            ok = False
        print(f"== {node}: {status} ==")
        print(f"  windows={f['windows']} unreadable={len(f['unreadable'])} gaps={len(f['gaps'])} "
              f"overlaps={len(f['overlaps'])} truncated={f['truncated_windows']} "
              f"truncated_unretried={len(f['truncated_unretried'])}")
        print(f"  forwards: raw={f['forwards_total']} deduped={f['forwards_deduped']} "
              f"settled={f['settled_count']} settled_fees={f['settled_fee_msat']/1000:.3f} sat")
        print(f"  updated_index span: {f['index_min']}..{f['index_max']}")
        for gap in f["gaps"][:10]:
            print(f"  GAP at {gap['path']}: start={gap['start']} expected={gap['expected']}")
        for bad in f["unreadable"][:10]:
            print(f"  UNREADABLE {bad['path']}: {bad['error']}")
        for bad in f["truncated_unretried"][:10]:
            print(f"  TRUNCATED-UNRETRIED {bad['path']}: truncated_start={bad['truncated_start']} next start={bad['start']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
