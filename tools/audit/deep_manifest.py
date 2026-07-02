#!/usr/bin/env python3
"""Deep-audit coverage manifest / gate for cl_revenue_ops.

Read-only, deterministic, stdlib + `git` subprocess only.

This tool underpins P1 ("Accountable coverage") and Phase 8 (drift detection)
of the deep-audit campaign described in
docs/audit/deep/README.md. It divides every tracked *source* line of the
plugin into blob-pinned chunks and tracks whether each chunk is COVERED by a
finding or a structured attestation.

Subcommands (default = generate):
    generate     Write docs/audit/deep/coverage-manifest.md (default).
    --check      Recompute current git blob hashes and flag any chunk whose
                 file blob changed since the manifest was written (Phase 8
                 drift detection). Exit 1 if drift found.
    --coverage   Read the findings-ledger + attestations file and report the
                 percentage of chunks that are COVERED. Exit 1 if <100%.

Design notes
------------
* Enumeration is via `git ls-files` filtered to source paths; blob hashes are
  the object ids at HEAD (equivalent to `git rev-parse HEAD:<path>`), fetched
  in bulk via `git ls-tree -r HEAD`.
* Chunking is a pure function of (line count, tier chunk size), so the manifest
  is fully reproducible: re-running `generate` on an unchanged tree yields a
  byte-identical table.
* A chunk is COVERED iff (a) a findings-ledger row cites a line inside it, or
  (b) an attestation exists for its chunk_id. EXAMPLE-marked rows/blocks are
  ignored so the template ships at honest 0% coverage.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Repo layout / constants
# --------------------------------------------------------------------------- #

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEEP_DIR = os.path.join("docs", "audit", "deep")
MANIFEST_PATH = os.path.join(DEEP_DIR, "coverage-manifest.md")
LEDGER_PATH = os.path.join(DEEP_DIR, "findings-ledger.md")
ATTESTATIONS_PATH = os.path.join(DEEP_DIR, "attestations.md")

TIER1_CHUNK_LINES = 400
TIER23_CHUNK_LINES = 700

# Tier assignment is by module basename (see docs/audit/deep/README.md).
TIER1_BASENAMES = {
    "cl-revenue-ops.py",
    "fee_controller.py",
    "database.py",
    "boltz_manager.py",
    "rebalance_engine_v2.py",
    "rebalancer.py",
    "rebalance_native_executor_v2.py",
    "rebalance_execution.py",
    "rebalance_executor_v2.py",
    "policy_manager.py",
    "capex_budget.py",
}
TIER2_BASENAMES = {
    "capacity_planner.py",
    "profitability_analyzer.py",
    "hive_hints.py",
    "flow_analysis.py",
    "config.py",
    "hive_router.py",
    "rebalance_router_v3.py",
    "rebalance_hive_router.py",
    "rebalance_coordination_overlay.py",
    "rebalance_router_v2.py",
    "rebalance_planner_v2.py",
    "rebalance_state_v2.py",
    "rebalance_route_policy.py",
}
# Everything else (modules/*, tools/**, scripts/**) is Tier 3.

# Paths that participate in the deep source read. Exclusions below.
_EXCLUDE_DIR_PARTS = {
    "tests",
    "docs",
    ".worktrees",
    ".venv",
    "__pycache__",
    "config",
    "fixtures",
}


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def _is_source_path(path: str) -> bool:
    """Filter tracked paths down to the deep-audit SOURCE set."""
    parts = path.split("/")
    # Directory-based exclusions (any component).
    if any(p in _EXCLUDE_DIR_PARTS for p in parts):
        return False
    if path.endswith(".md"):
        return False
    # Include rules.
    if path == "cl-revenue-ops.py":
        return True
    if path.startswith("modules/") and path.endswith(".py"):
        return True
    if path.startswith("tools/") and path.endswith(".py"):
        return True
    if path.startswith("scripts/"):
        return True
    return False


def enumerate_source_files() -> List[str]:
    """Deterministically sorted list of tracked source paths."""
    tracked = _git("ls-files").splitlines()
    files = [p for p in tracked if _is_source_path(p)]
    return sorted(files)


def blob_hashes(paths: List[str]) -> Dict[str, str]:
    """Map path -> git blob hash at HEAD, in one `git ls-tree` call."""
    result: Dict[str, str] = {}
    if not paths:
        return result
    out = _git("ls-tree", "-r", "HEAD", "--", *paths)
    for line in out.splitlines():
        # "<mode> <type> <hash>\t<path>"
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) == 3 and fields[1] == "blob":
            result[path] = fields[2]
    return result


def current_blob_hash(path: str) -> Optional[str]:
    """Current HEAD blob hash for a single path (equiv. `git rev-parse HEAD:<path>`)."""
    try:
        out = _git("rev-parse", f"HEAD:{path}").strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


def line_count(path: str) -> int:
    abs_path = os.path.join(REPO_ROOT, path)
    with open(abs_path, "rb") as fh:
        data = fh.read()
    if not data:
        return 0
    n = data.count(b"\n")
    if not data.endswith(b"\n"):
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Tier + chunking
# --------------------------------------------------------------------------- #


def tier_for(path: str) -> int:
    base = os.path.basename(path)
    if base in TIER1_BASENAMES:
        return 1
    if base in TIER2_BASENAMES:
        return 2
    return 3


def chunk_size_for(tier: int) -> int:
    return TIER1_CHUNK_LINES if tier == 1 else TIER23_CHUNK_LINES


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file: str
    line_start: int
    line_end: int
    tier: int
    blob: str
    owner: str = ""
    status: str = "UNASSIGNED"

    def contains(self, line: int) -> bool:
        return self.line_start <= line <= self.line_end


def chunk_file(path: str, tier: int, blob: str, nlines: int) -> List[Chunk]:
    size = chunk_size_for(tier)
    chunks: List[Chunk] = []
    if nlines <= 0:
        # Represent empty/near-empty files as a single degenerate chunk so
        # coverage accounting still names them.
        chunks.append(
            Chunk(
                chunk_id=f"{path}#1",
                file=path,
                line_start=1,
                line_end=max(nlines, 1),
                tier=tier,
                blob=blob,
            )
        )
        return chunks
    n_chunks = math.ceil(nlines / size)
    for i in range(n_chunks):
        start = i * size + 1
        end = min((i + 1) * size, nlines)
        chunks.append(
            Chunk(
                chunk_id=f"{path}#{i + 1}",
                file=path,
                line_start=start,
                line_end=end,
                tier=tier,
                blob=blob,
            )
        )
    return chunks


def build_chunks() -> Tuple[List[Chunk], Dict[str, int]]:
    """Return (chunks, per-file line counts)."""
    files = enumerate_source_files()
    blobs = blob_hashes(files)
    chunks: List[Chunk] = []
    counts: Dict[str, int] = {}
    for path in files:
        blob = blobs.get(path) or current_blob_hash(path) or "MISSING"
        nlines = line_count(path)
        counts[path] = nlines
        tier = tier_for(path)
        chunks.extend(chunk_file(path, tier, blob, nlines))
    return chunks, counts


# --------------------------------------------------------------------------- #
# Manifest rendering
# --------------------------------------------------------------------------- #


def render_manifest(chunks: List[Chunk], counts: Dict[str, int]) -> str:
    total_lines = sum(counts.values())
    tier_lines = {1: 0, 2: 0, 3: 0}
    tier_chunks = {1: 0, 2: 0, 3: 0}
    for path, n in counts.items():
        tier_lines[tier_for(path)] += n
    for c in chunks:
        tier_chunks[c.tier] += 1

    lines: List[str] = []
    lines.append("# Deep-Audit Coverage Manifest")
    lines.append("")
    lines.append(
        "Generated by `tools/audit/deep_manifest.py` (read-only, deterministic)."
    )
    lines.append(
        "Do not hand-edit chunk rows; regenerate. Owner/status columns are the "
        "one place auditors write into (or track them in the findings-ledger / "
        "attestations files)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total tracked source files: **{len(counts)}**")
    lines.append(f"- Total tracked source lines: **{total_lines}**")
    lines.append(f"- Total chunks: **{len(chunks)}**")
    lines.append("")
    lines.append("| Tier | Chunk size | Files | Lines | Chunks |")
    lines.append("| --- | --- | --- | --- | --- |")
    for tier in (1, 2, 3):
        nfiles = sum(1 for p in counts if tier_for(p) == tier)
        lines.append(
            f"| {tier} | {chunk_size_for(tier)} | {nfiles} | "
            f"{tier_lines[tier]} | {tier_chunks[tier]} |"
        )
    lines.append("")
    lines.append("## Chunks")
    lines.append("")
    lines.append(
        "| chunk_id | file | line_start | line_end | tier | blob | owner | status |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in chunks:
        owner = c.owner or ""
        lines.append(
            f"| {c.chunk_id} | {c.file} | {c.line_start} | {c.line_end} | "
            f"{c.tier} | {c.blob} | {owner} | {c.status} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_manifest() -> Tuple[List[Chunk], Dict[str, int]]:
    chunks, counts = build_chunks()
    content = render_manifest(chunks, counts)
    os.makedirs(os.path.join(REPO_ROOT, DEEP_DIR), exist_ok=True)
    with open(os.path.join(REPO_ROOT, MANIFEST_PATH), "w") as fh:
        fh.write(content)
    return chunks, counts


# --------------------------------------------------------------------------- #
# Manifest parsing (for --check / --coverage against the recorded manifest)
# --------------------------------------------------------------------------- #

_ROW_RE = re.compile(
    r"^\|\s*(?P<chunk_id>[^|]+?)\s*\|\s*(?P<file>[^|]+?)\s*\|\s*"
    r"(?P<start>\d+)\s*\|\s*(?P<end>\d+)\s*\|\s*(?P<tier>\d+)\s*\|\s*"
    r"(?P<blob>[0-9a-fA-F]+|MISSING)\s*\|\s*(?P<owner>[^|]*?)\s*\|\s*"
    r"(?P<status>[^|]+?)\s*\|\s*$"
)


def parse_manifest(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    abs_path = os.path.join(REPO_ROOT, path)
    with open(abs_path) as fh:
        for line in fh:
            m = _ROW_RE.match(line.rstrip("\n"))
            if not m:
                continue
            chunks.append(
                Chunk(
                    chunk_id=m.group("chunk_id"),
                    file=m.group("file"),
                    line_start=int(m.group("start")),
                    line_end=int(m.group("end")),
                    tier=int(m.group("tier")),
                    blob=m.group("blob"),
                    owner=m.group("owner"),
                    status=m.group("status"),
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Drift detection (--check)
# --------------------------------------------------------------------------- #


def check_drift() -> int:
    manifest_abs = os.path.join(REPO_ROOT, MANIFEST_PATH)
    if not os.path.exists(manifest_abs):
        print(f"ERROR: manifest not found at {MANIFEST_PATH}; run generate first.")
        return 2
    chunks = parse_manifest(MANIFEST_PATH)
    # Compute current blob per distinct file once.
    files = sorted({c.file for c in chunks})
    current = blob_hashes(files)
    for f in files:
        if f not in current:
            current[f] = current_blob_hash(f) or "MISSING"

    drifted = [c for c in chunks if current.get(c.file) != c.blob]
    if not drifted:
        print(f"CLEAN: {len(chunks)} chunks across {len(files)} files; "
              "no blob drift since manifest was written.")
        return 0

    drifted_files = sorted({c.file for c in drifted})
    print(f"DRIFT: {len(drifted)} chunk(s) across {len(drifted_files)} file(s) "
          "need re-attestation (file blob changed since manifest):")
    for c in drifted:
        print(f"  {c.chunk_id}  manifest={c.blob[:12]}  "
              f"current={(current.get(c.file) or 'MISSING')[:12]}")
    return 1


# --------------------------------------------------------------------------- #
# Coverage (--coverage)
# --------------------------------------------------------------------------- #

# findings-ledger row: ID | severity | dimension | file:line@blob | ...
_LEDGER_ROW_RE = re.compile(r"^\|(?P<cells>.+)\|\s*$")
# The canonical citation is the PINNED column: `file:line@<blob>`. Only a
# citation carrying the `@<blob>` suffix is a real pin; a bare `file:line`
# in a description cell is an incidental cross-reference and must NOT count
# toward coverage (P5-005). blob may be a hex object id or a symbolic tag
# such as `map` / `campaign`, so accept any alnum run after `@`.
_PIN_CITATION_RE = re.compile(r"(?P<file>[^\s|@]+):(?P<line>\d+)@[0-9A-Za-z]+")
# Column index of the pinned `file:line@blob` cell in a findings-ledger row
# (ID | severity | dimension | file:line@blob | description | status | ...).
_PIN_COLUMN_INDEX = 3

# attestation heading: "### <chunk_id>" (EXAMPLE-prefixed headings are skipped).
_ATTEST_HEADING_RE = re.compile(r"^#{2,4}\s+(?P<rest>.+?)\s*$")
_ATTEST_FIELD_RE = re.compile(r"^\s*[-*]?\s*chunk_id\s*:\s*(?P<id>\S+)")


def parse_ledger_citations(path: str) -> List[Tuple[str, int]]:
    """Return (file, line) pins from real (non-EXAMPLE) ledger rows.

    Only the canonical pinned column (`file:line@blob`, column index
    ``_PIN_COLUMN_INDEX``) is scanned, and only citations carrying an
    ``@<blob>`` suffix count. An incidental ``file:line`` mentioned in a
    description cell is deliberately ignored so it cannot mark an unrelated
    chunk COVERED (P5-005).
    """
    citations: List[Tuple[str, int]] = []
    abs_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(abs_path):
        return citations
    with open(abs_path) as fh:
        for raw in fh:
            m = _LEDGER_ROW_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            cells = [c.strip() for c in m.group("cells").split("|")]
            if not cells:
                continue
            header_like = cells[0].lower() in {"id", ""} or set(cells[0]) <= {"-"}
            if header_like:
                continue
            if "EXAMPLE" in raw.upper():
                continue
            if len(cells) <= _PIN_COLUMN_INDEX:
                continue
            # Scan ONLY the pinned file:line@blob column.
            pin_cell = cells[_PIN_COLUMN_INDEX]
            for fm in _PIN_CITATION_RE.finditer(pin_cell):
                citations.append((fm.group("file"), int(fm.group("line"))))
    return citations


def parse_attested_chunk_ids(path: str) -> List[str]:
    """Return chunk_ids from real (non-EXAMPLE) attestation blocks."""
    ids: List[str] = []
    abs_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(abs_path):
        return ids
    with open(abs_path) as fh:
        lines = fh.readlines()

    # Split into blocks by markdown headings; a block is EXAMPLE if its heading
    # or any line before the next heading is EXAMPLE-marked.
    current_block: List[str] = []
    blocks: List[List[str]] = []
    for raw in lines:
        if _ATTEST_HEADING_RE.match(raw) and raw.lstrip().startswith("#"):
            # Only treat level 2-4 headings as block starts.
            if current_block:
                blocks.append(current_block)
            current_block = [raw]
        else:
            current_block.append(raw)
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        text = "".join(block)
        if "EXAMPLE" in text.upper():
            continue
        # Prefer an explicit `chunk_id:` field; else the heading token.
        found: Optional[str] = None
        for raw in block:
            fm = _ATTEST_FIELD_RE.match(raw)
            if fm:
                found = fm.group("id")
                break
        if found is None:
            hm = _ATTEST_HEADING_RE.match(block[0])
            if hm:
                token = hm.group("rest").split()[0] if hm.group("rest").split() else ""
                # Heading token must look like a chunk_id (contains '#').
                if "#" in token:
                    found = token
        if found:
            ids.append(found.strip())
    return ids


def coverage() -> int:
    manifest_abs = os.path.join(REPO_ROOT, MANIFEST_PATH)
    if not os.path.exists(manifest_abs):
        print(f"ERROR: manifest not found at {MANIFEST_PATH}; run generate first.")
        return 2
    chunks = parse_manifest(MANIFEST_PATH)
    by_id = {c.chunk_id: c for c in chunks}

    covered: Dict[str, str] = {}  # chunk_id -> reason

    # (b) attestations
    for cid in parse_attested_chunk_ids(ATTESTATIONS_PATH):
        if cid in by_id and cid not in covered:
            covered[cid] = "attestation"

    # (a) finding citations. Match by FULL repo-relative path, not basename,
    # so a pin to modules/foo.py cannot cover a same-named tools/foo.py chunk
    # (P5-005).
    for f, line in parse_ledger_citations(LEDGER_PATH):
        for c in chunks:
            if c.chunk_id in covered:
                continue
            if c.file == f and c.contains(line):
                covered[c.chunk_id] = "finding"

    total = len(chunks)
    ncov = len(covered)
    pct = (100.0 * ncov / total) if total else 100.0

    # Per-tier breakdown.
    tier_total = {1: 0, 2: 0, 3: 0}
    tier_cov = {1: 0, 2: 0, 3: 0}
    for c in chunks:
        tier_total[c.tier] += 1
        if c.chunk_id in covered:
            tier_cov[c.tier] += 1

    print("Deep-audit coverage")
    print(f"  COVERED: {ncov}/{total} chunks = {pct:.2f}%")
    for tier in (1, 2, 3):
        tt = tier_total[tier]
        tc = tier_cov[tier]
        tp = (100.0 * tc / tt) if tt else 100.0
        print(f"    Tier {tier}: {tc}/{tt} = {tp:.2f}%")
    n_att = sum(1 for r in covered.values() if r == "attestation")
    n_find = sum(1 for r in covered.values() if r == "finding")
    print(f"  by attestation: {n_att}   by finding: {n_find}")
    return 0 if ncov == total else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Recompute blob hashes and flag chunks needing re-attestation.",
    )
    group.add_argument(
        "--coverage",
        action="store_true",
        help="Report %% of chunks COVERED by findings + attestations.",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check_drift()
    if args.coverage:
        return coverage()

    # default: generate
    chunks, counts = write_manifest()
    total_lines = sum(counts.values())
    tier_chunks = {1: 0, 2: 0, 3: 0}
    for c in chunks:
        tier_chunks[c.tier] += 1
    print(f"Wrote {MANIFEST_PATH}")
    print(f"  files: {len(counts)}   total source lines: {total_lines}")
    print(f"  chunks: {len(chunks)} "
          f"(T1={tier_chunks[1]} T2={tier_chunks[2]} T3={tier_chunks[3]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
