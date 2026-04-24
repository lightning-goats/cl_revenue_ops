#!/usr/bin/env python3
"""Client-side compatibility launcher for the deleted cl-hive MCP server."""

from __future__ import annotations

import ast
import json
import os
import re
import runpy
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Iterable


DEFAULT_CL_HIVE_PATH = Path("/home/sat/bin/cl-hive")
DEFAULT_REVENUE_OPS_PATH = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path("/tmp/hive-mcp-compat")
DEFAULT_SOURCE_REV = "99a14ea225833036f8d5bab0b10f65dd99e8630f"
SOURCE_FILES = {
    "tools/mcp-hive-server.py": "mcp-hive-server.py",
    "tools/advisor_db.py": "advisor_db.py",
}
PLUGIN_METHOD_PATTERN = re.compile(
    r'@plugin\.(?:async_method|method)\(\s*["\']([^"\']+)["\']'
)
OBSERVABILITY_EXTRA_METHODS = frozenset(
    {
        "getinfo",
        "hive-health",
        "listforwards",
        "listpeerchannels",
        "listpeers",
    }
)
LEGACY_RPC_RENAMES = {
    "hive-getinfo": "getinfo",
    "hive-listforwards": "listforwards",
    "hive-listpeerchannels": "listpeerchannels",
    "hive-listpeers": "listpeers",
}
OBSERVABILITY_FUNCTION_REPLACEMENTS = {
    "health_check": """
async def health_check(self, timeout: float = 5.0) -> Dict[str, Any]:
    \"\"\"Quick health check on all nodes using current Hive and CLN endpoints.\"\"\"

    async def check_node(name: str, node: NodeConnection) -> tuple:
        try:
            start = asyncio.get_running_loop().time()
            health_result, info_result = await asyncio.wait_for(
                asyncio.gather(
                    node.call("hive-health"),
                    node.call("getinfo"),
                    return_exceptions=True,
                ),
                timeout=timeout,
            )
            latency = asyncio.get_running_loop().time() - start

            payload: Dict[str, Any] = {
                "status": "healthy",
                "latency_ms": round(latency * 1000),
            }
            warnings: List[str] = []

            if isinstance(info_result, Exception):
                warnings.append(str(info_result) or type(info_result).__name__)
            elif isinstance(info_result, dict) and "error" in info_result:
                warnings.append(str(info_result["error"]))
            elif isinstance(info_result, dict):
                payload["alias"] = info_result.get("alias", "unknown")
                payload["blockheight"] = info_result.get("blockheight", 0)
                payload["id"] = info_result.get("id")
                payload["network"] = info_result.get("network")

            if isinstance(health_result, Exception):
                warnings.append(str(health_result) or type(health_result).__name__)
            elif isinstance(health_result, dict) and "error" in health_result:
                warnings.append(str(health_result["error"]))
            elif isinstance(health_result, dict):
                payload["hive"] = health_result

            if "alias" not in payload and "hive" not in payload:
                return (
                    name,
                    {"status": "error", "error": "; ".join(warnings) or "No health data"},
                )
            if warnings:
                payload["warnings"] = warnings
            return (name, payload)
        except asyncio.TimeoutError:
            return (name, {"status": "timeout", "error": f"No response in {timeout}s"})
        except Exception as e:
            return (name, {"status": "error", "error": str(e) or type(e).__name__})

    tasks = [check_node(name, node) for name, node in self.nodes.items()]
    results_list = await asyncio.gather(*tasks)
    return dict(results_list)
""",
    "handle_revenue_hive_status": """
async def handle_revenue_hive_status(args: Dict) -> Dict:
    \"\"\"Get current Hive and revenue integration status.\"\"\"
    node_name = args.get("node")

    node = fleet.get_node(node_name)
    if not node:
        return {"error": f"Unknown node: {node_name}"}

    hive_status, revenue_status = await asyncio.gather(
        node.call("hive-status"),
        node.call("revenue-status"),
        return_exceptions=True,
    )

    result: Dict[str, Any] = {"node": node_name}
    if isinstance(hive_status, Exception):
        result["hive_status"] = {"error": str(hive_status)}
    else:
        result["hive_status"] = hive_status

    if isinstance(revenue_status, Exception):
        result["revenue_status"] = {"error": str(revenue_status)}
    else:
        result["revenue_status"] = revenue_status

    hive_ok = isinstance(result["hive_status"], dict) and "error" not in result["hive_status"]
    revenue_ok = (
        isinstance(result["revenue_status"], dict)
        and "error" not in result["revenue_status"]
    )
    if hive_ok and revenue_ok:
        result["status"] = "ok"
    elif hive_ok or revenue_ok:
        result["status"] = "degraded"
    else:
        result["status"] = "error"
    return result
""",
    "handle_hive_node_diagnostic": """
async def handle_hive_node_diagnostic(args: Dict) -> Dict:
    \"\"\"Comprehensive single-node diagnostic using current RPC surfaces.\"\"\"
    node_name = args.get("node")

    node = fleet.get_node(node_name)
    if not node:
        return {"error": f"Unknown node: {node_name}"}

    now = int(time.time())
    since_24h = now - 86400
    result: Dict[str, Any] = {"node": node_name}

    info_result, channels_result, forwards_result, revenue_status = await asyncio.gather(
        node.call("getinfo"),
        node.call("listpeerchannels"),
        node.call("listforwards", {"status": "settled"}),
        node.call("revenue-status"),
        return_exceptions=True,
    )

    if isinstance(info_result, Exception):
        result["info"] = {"error": str(info_result)}
    elif isinstance(info_result, dict) and "error" in info_result:
        result["info"] = info_result
    else:
        result["info"] = {
            "alias": info_result.get("alias", "unknown"),
            "id": info_result.get("id"),
            "blockheight": info_result.get("blockheight", 0),
            "network": info_result.get("network"),
            "num_peers": info_result.get("num_peers", 0),
        }

    if isinstance(channels_result, Exception):
        result["channels"] = {"error": str(channels_result)}
    elif isinstance(channels_result, dict) and "error" in channels_result:
        result["channels"] = channels_result
    else:
        channels = channels_result.get("channels", [])
        total_capacity_msat = 0
        total_local_msat = 0
        channel_count = 0
        zero_balance_channels = []
        for ch in channels:
            state = ch.get("state", "")
            if "CHANNELD_NORMAL" not in state:
                continue
            channel_count += 1
            totals = _channel_totals(ch)
            total_capacity_msat += totals["total_msat"]
            total_local_msat += totals["local_msat"]
            if totals["total_msat"] == 0:
                zero_balance_channels.append(ch.get("short_channel_id", "unknown"))
        result["channels"] = {
            "count": channel_count,
            "total_capacity_sats": total_capacity_msat // 1000,
            "total_local_sats": total_local_msat // 1000,
            "total_remote_sats": (total_capacity_msat - total_local_msat) // 1000,
            "avg_balance_ratio": round(total_local_msat / total_capacity_msat, 3)
            if total_capacity_msat
            else 0,
            "zero_balance_channels": zero_balance_channels,
        }

    if isinstance(forwards_result, Exception):
        result["forwards_24h"] = {"error": str(forwards_result)}
    elif isinstance(forwards_result, dict) and "error" in forwards_result:
        result["forwards_24h"] = forwards_result
    else:
        result["forwards_24h"] = _forward_stats(
            forwards_result.get("forwards", []),
            since_24h,
            now,
        )

    if isinstance(revenue_status, Exception):
        result["revenue_status"] = {"error": str(revenue_status)}
        result["plugins"] = ["cl-hive"]
        result["rebalance_status"] = {
            "status": "unavailable",
            "reason": "revenue-status unavailable",
        }
    elif isinstance(revenue_status, dict) and "error" in revenue_status:
        result["revenue_status"] = revenue_status
        result["plugins"] = ["cl-hive"]
        result["rebalance_status"] = {
            "status": "unavailable",
            "reason": str(revenue_status["error"]),
        }
    else:
        result["revenue_status"] = revenue_status
        result["plugins"] = ["cl-hive", "cl-revenue-ops"]
        result["plugin_inventory_note"] = (
            "Legacy hive-plugin-list RPC was removed; only currently reachable plugin surfaces are reported."
        )
        result["rebalance_status"] = {
            "status": "unavailable",
            "reason": "inspect revenue_status.rebalance_decision instead.",
            "rebalance_decision": revenue_status.get("rebalance_decision"),
        }

    return result
""",
    "handle_revenue_ops_health": """
async def handle_revenue_ops_health(args: Dict) -> Dict:
    \"\"\"Validate cl-revenue-ops operational surfaces using current endpoints.\"\"\"
    node_name = args.get("node")

    node = fleet.get_node(node_name)
    if not node:
        return {"error": f"Unknown node: {node_name}"}

    checks: Dict[str, Dict[str, Any]] = {}
    dashboard, status, spend, boltz = await asyncio.gather(
        node.call("revenue-dashboard", {"window_days": 7}),
        node.call("revenue-status"),
        node.call("revenue-spend-ledger", {"window_hours": 24, "include_reservations": True}),
        node.call("revenue-boltz-wallet"),
        return_exceptions=True,
    )

    if isinstance(dashboard, Exception):
        checks["dashboard"] = {"status": "error", "detail": str(dashboard)}
    elif isinstance(dashboard, dict) and "error" in dashboard:
        checks["dashboard"] = {"status": "error", "detail": dashboard["error"]}
    else:
        financial_health = dashboard.get("financial_health")
        period = dashboard.get("period")
        if isinstance(financial_health, dict) and isinstance(period, dict):
            checks["dashboard"] = {
                "status": "pass",
                "gross_revenue_sats": period.get("gross_revenue_sats", 0),
                "forward_count": period.get("forward_count", 0),
                "bleeder_count": dashboard.get("bleeder_count", 0),
            }
        else:
            checks["dashboard"] = {
                "status": "warn",
                "detail": "Dashboard returned but missing current financial_health/period fields",
            }

    if isinstance(status, Exception):
        checks["status"] = {"status": "error", "detail": str(status)}
    elif isinstance(status, dict) and "error" in status:
        checks["status"] = {"status": "error", "detail": status["error"]}
    else:
        required_keys = {"operator_controls", "fee_decision", "rebalance_decision"}
        if required_keys <= set(status):
            checks["status"] = {
                "status": "pass",
                "version": status.get("version"),
                "recent_fee_changes": len(status.get("recent_fee_changes", [])),
                "recent_rebalances": len(status.get("recent_rebalances", [])),
            }
        else:
            checks["status"] = {
                "status": "warn",
                "detail": "Revenue status returned but is missing current decision metadata",
            }

    if isinstance(spend, Exception):
        checks["spend_ledger"] = {"status": "error", "detail": str(spend)}
    elif isinstance(spend, dict) and "error" in spend:
        checks["spend_ledger"] = {"status": "error", "detail": spend["error"]}
    else:
        if "spent_24h_sats" in spend or "spent_by_category" in spend:
            checks["spend_ledger"] = {
                "status": "pass",
                "spent_24h_sats": spend.get("spent_24h_sats"),
                "active_reservations": len(spend.get("active_reservations", [])),
            }
        else:
            checks["spend_ledger"] = {
                "status": "warn",
                "detail": "Spend ledger returned but missing spend summary fields",
            }

    if isinstance(boltz, Exception):
        checks["boltz_wallet"] = {"status": "error", "detail": str(boltz)}
    elif isinstance(boltz, dict) and "error" in boltz:
        checks["boltz_wallet"] = {"status": "error", "detail": boltz["error"]}
    else:
        checks["boltz_wallet"] = {
            "status": "pass",
            "keys": sorted(boltz.keys())[:10],
        }

    statuses = [check["status"] for check in checks.values()]
    if statuses and all(status == "pass" for status in statuses):
        overall = "healthy"
    elif statuses and all(status == "error" for status in statuses):
        overall = "unhealthy"
    elif "error" in statuses:
        overall = "degraded"
    else:
        overall = "warning"

    return {
        "node": node_name,
        "overall_health": overall,
        "checks": checks,
        "note": (
            "Heavy diagnostics such as revenue-profitability and revenue-rebalance-debug are excluded from the heartbeat to avoid timeout-driven false negatives."
        ),
    }
""",
    "handle_advisor_get_context_brief": """
async def handle_advisor_get_context_brief(args: Dict) -> Dict:
    \"\"\"Get pre-run context summary for AI advisor.\"\"\"
    db = ensure_advisor_db()
    days = args.get("days", 7)

    brief = db.get_context_brief(days)
    snapshots = db.get_recent_snapshots(limit=1)
    latest_snapshot_ts = snapshots[0].get("timestamp") if snapshots else None
    latest_snapshot_age_seconds = (
        int(time.time()) - int(latest_snapshot_ts)
        if latest_snapshot_ts is not None
        else None
    )
    stale_after_seconds = 86400
    is_stale = (
        latest_snapshot_age_seconds is None
        or latest_snapshot_age_seconds > stale_after_seconds
    )

    result = {
        "period_days": brief.period_days,
        "total_capacity_sats": brief.total_capacity_sats,
        "capacity_change_pct": brief.capacity_change_pct,
        "total_channels": brief.total_channels,
        "channel_count_change": brief.channel_count_change,
        "period_revenue_sats": brief.period_revenue_sats,
        "revenue_change_pct": brief.revenue_change_pct,
        "channels_depleting": brief.channels_depleting,
        "channels_filling": brief.channels_filling,
        "critical_velocity_channels": brief.critical_velocity_channels,
        "unresolved_alerts": brief.unresolved_alerts,
        "recent_decisions_count": brief.recent_decisions_count,
        "decisions_by_type": brief.decisions_by_type,
        "summary_text": brief.summary_text,
        "freshness": {
            "latest_snapshot_timestamp": latest_snapshot_ts,
            "latest_snapshot_age_seconds": latest_snapshot_age_seconds,
            "stale_after_seconds": stale_after_seconds,
            "is_stale": is_stale,
        },
    }
    if is_stale:
        result["staleness_reason"] = (
            "Advisor context is older than the freshness threshold; treat it as historical context, not live operations state."
        )
    return result
""",
}


def _read_git_file(cl_hive_path: Path, source_rev: str, rel_path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cl_hive_path), "show", f"{source_rev}:{rel_path}"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"failed to restore {rel_path} from {cl_hive_path} at {source_rev}: {stderr}"
        ) from exc

    return completed.stdout


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return

    path.write_text(content, encoding="utf-8")
    if path.suffix == ".py":
        path.chmod(0o755)


def _extract_plugin_methods(entrypoint: Path) -> set[str]:
    try:
        text = entrypoint.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing RPC entrypoint: {entrypoint}") from exc

    methods = set(PLUGIN_METHOD_PATTERN.findall(text))
    if not methods:
        raise RuntimeError(f"no RPC methods found in {entrypoint}")
    return methods


def _read_allowlist(path: Path) -> set[str]:
    try:
        methods = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing RPC allowlist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid RPC allowlist JSON: {path}") from exc
    if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
        raise RuntimeError(f"invalid RPC allowlist payload: {path}")
    return set(methods)


def write_rpc_allowlist(
    *,
    cache_dir: Path,
    cl_hive_path: Path = DEFAULT_CL_HIVE_PATH,
    revenue_ops_path: Path = DEFAULT_REVENUE_OPS_PATH,
    existing_allowlist_path: Path | None = None,
) -> Path:
    methods = sorted(
        OBSERVABILITY_EXTRA_METHODS
        | _extract_plugin_methods(cl_hive_path / "cl-hive.py")
        | _extract_plugin_methods(revenue_ops_path / "cl-revenue-ops.py")
    )
    if existing_allowlist_path is not None:
        methods = sorted(set(methods) & _read_allowlist(existing_allowlist_path))
    allowlist_path = cache_dir / "allowed-methods.json"
    _write_if_changed(
        allowlist_path,
        json.dumps(methods, indent=2) + "\n",
    )
    return allowlist_path


def _tool_name_from_call(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


def _tool_name_from_block(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg != "name":
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _function_defs(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in defs:
            defs[node.name] = node
    return defs


def _indent_block(source: str, indent: int) -> str:
    prefix = " " * indent
    return "".join(
        f"{prefix}{line}" if line.strip() else line
        for line in source.splitlines(keepends=True)
    )


def patch_observability_source(server_source: str) -> str:
    patched_source = server_source
    for legacy_method, current_method in LEGACY_RPC_RENAMES.items():
        patched_source = patched_source.replace(f'"{legacy_method}"', f'"{current_method}"')

    tree = ast.parse(patched_source)
    function_defs = _function_defs(tree)
    source_lines = patched_source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []

    for name, replacement in OBSERVABILITY_FUNCTION_REPLACEMENTS.items():
        node = function_defs.get(name)
        if node is None:
            continue
        replacements.append(
            (
                node.lineno,
                node.end_lineno,
                _indent_block(dedent(replacement).lstrip("\n"), node.col_offset),
            )
        )

    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        source_lines[start - 1 : end] = replacement.splitlines(keepends=True)

    return "".join(source_lines)


def _tool_handlers(tree: ast.AST) -> list[tuple[str, str]]:
    for node in getattr(tree, "body", []):
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "TOOL_HANDLERS":
                value = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL_HANDLERS":
                    value = node.value
                    break
        if not isinstance(value, ast.Dict):
            continue
        handlers: list[tuple[str, str]] = []
        for key, handler in zip(value.keys, value.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if not isinstance(handler, ast.Name):
                continue
            handlers.append((key.value, handler.id))
        return handlers
    raise RuntimeError("TOOL_HANDLERS registry not found in compatibility server")


def _list_tool_blocks(
    tree: ast.AST,
    source_lines: list[str],
) -> tuple[int, int, list[tuple[str, str]]]:
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name != "list_tools":
            continue
        if not node.body:
            break
        for stmt in node.body:
            if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.List):
                continue
            blocks: list[tuple[str, str]] = []
            for elt in stmt.value.elts:
                if not isinstance(elt, ast.Call):
                    continue
                tool_name = _tool_name_from_block(elt)
                if not tool_name:
                    continue
                blocks.append(
                    (
                        tool_name,
                        "".join(source_lines[elt.lineno - 1 : elt.end_lineno]),
                    )
                )
            if not blocks:
                raise RuntimeError("No Tool() blocks found in list_tools")
            return stmt.value.elts[0].lineno, stmt.value.elts[-1].end_lineno, blocks
    raise RuntimeError("list_tools() function not found in compatibility server")


def _tool_handlers_span(tree: ast.AST) -> tuple[int, int]:
    for node in getattr(tree, "body", []):
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "TOOL_HANDLERS":
                value = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL_HANDLERS":
                    value = node.value
                    break
        if not isinstance(value, ast.Dict) or not value.keys or not value.values:
            continue
        return value.keys[0].lineno, value.values[-1].end_lineno
    raise RuntimeError("TOOL_HANDLERS registry span not found in compatibility server")


def _top_level_function_spans(tree: ast.AST) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans[node.name] = (node.lineno, node.end_lineno)
    return spans


def _function_dependency_graph(
    tree: ast.AST,
) -> tuple[
    dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    defs = _function_defs(tree)
    direct_rpc_calls = {name: set() for name in defs}
    direct_tool_calls = {name: set() for name in defs}
    helper_calls = {name: set() for name in defs}

    for name, node in defs.items():
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name) and child.func.id in defs and child.func.id != name:
                helper_calls[name].add(child.func.id)
            method_name = _tool_name_from_call(child)
            if not method_name:
                continue
            if not isinstance(child.func, ast.Attribute):
                continue
            if child.func.attr not in {"call", "call_all"}:
                continue
            if "-" in method_name:
                direct_rpc_calls[name].add(method_name)
            else:
                direct_tool_calls[name].add(method_name)

    return defs, direct_rpc_calls, direct_tool_calls, helper_calls


def _resolve_function_dependencies(
    name: str,
    direct_rpc_calls: dict[str, set[str]],
    direct_tool_calls: dict[str, set[str]],
    helper_calls: dict[str, set[str]],
    memo: dict[str, tuple[set[str], set[str]]],
    visiting: set[str],
) -> tuple[set[str], set[str]]:
    if name in memo:
        return memo[name]
    if name in visiting:
        return set(), set()

    visiting.add(name)
    rpc_calls = set(direct_rpc_calls.get(name, ()))
    tool_calls = set(direct_tool_calls.get(name, ()))
    for helper_name in helper_calls.get(name, ()):
        helper_rpc, helper_tools = _resolve_function_dependencies(
            helper_name,
            direct_rpc_calls,
            direct_tool_calls,
            helper_calls,
            memo,
            visiting,
        )
        rpc_calls |= helper_rpc
        tool_calls |= helper_tools
    visiting.remove(name)
    memo[name] = (rpc_calls, tool_calls)
    return memo[name]


def _reachable_top_level_functions(
    roots: set[str],
    helper_calls: dict[str, set[str]],
    top_level_names: set[str],
) -> set[str]:
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in reachable or name not in top_level_names:
            continue
        reachable.add(name)
        pending.extend(helper for helper in helper_calls.get(name, ()) if helper in top_level_names)
    return reachable


def _supported_tool_names(tree: ast.AST, allowed_methods: set[str]) -> set[str]:
    defs, direct_rpc_calls, direct_tool_calls, helper_calls = _function_dependency_graph(
        tree
    )
    handlers = _tool_handlers(tree)
    tool_names = {tool_name for tool_name, _ in handlers}

    dependency_cache: dict[str, tuple[set[str], set[str]]] = {}
    candidate_tools: set[str] = set()
    for tool_name, handler_name in handlers:
        if handler_name not in defs:
            continue
        rpc_calls, _tool_calls = _resolve_function_dependencies(
            handler_name,
            direct_rpc_calls,
            direct_tool_calls,
            helper_calls,
            dependency_cache,
            set(),
        )
        if rpc_calls <= allowed_methods:
            candidate_tools.add(tool_name)

    changed = True
    while changed:
        changed = False
        for tool_name, handler_name in handlers:
            if tool_name not in candidate_tools or handler_name not in defs:
                continue
            _rpc_calls, tool_calls = _resolve_function_dependencies(
                handler_name,
                direct_rpc_calls,
                direct_tool_calls,
                helper_calls,
                dependency_cache,
                set(),
            )
            internal_tool_calls = {
                dep_tool for dep_tool in tool_calls if dep_tool in tool_names
            }
            if not internal_tool_calls <= candidate_tools:
                candidate_tools.remove(tool_name)
                changed = True

    return candidate_tools


def prune_compat_server_source(
    server_source: str,
    *,
    allowed_methods: Iterable[str],
) -> str:
    tree = ast.parse(server_source)
    source_lines = server_source.splitlines(keepends=True)
    supported_tools = _supported_tool_names(tree, set(allowed_methods))
    handler_entries = _tool_handlers(tree)
    function_spans = _top_level_function_spans(tree)
    _defs, direct_rpc_calls, direct_tool_calls, helper_calls = _function_dependency_graph(tree)
    top_level_names = set(function_spans)
    supported_handler_names = {
        handler_name
        for tool_name, handler_name in handler_entries
        if tool_name in supported_tools
    }
    unsupported_handler_names = {
        handler_name
        for tool_name, handler_name in handler_entries
        if tool_name not in supported_tools
    }
    reachable_top_level = _reachable_top_level_functions(
        supported_handler_names,
        helper_calls,
        top_level_names,
    )
    dependency_cache: dict[str, tuple[set[str], set[str]]] = {}
    unsupported_helper_names = {
        name
        for name in top_level_names
        if name.startswith("_")
        and name not in reachable_top_level
        and _resolve_function_dependencies(
            name,
            direct_rpc_calls,
            direct_tool_calls,
            helper_calls,
            dependency_cache,
            set(),
        )[0]
        - set(allowed_methods)
    }

    list_start, list_end, tool_blocks = _list_tool_blocks(tree, source_lines)
    handlers_start, handlers_end = _tool_handlers_span(tree)

    replacements = [
        (
            handlers_start,
            handlers_end,
            "".join(
                f'    "{tool_name}": {handler_name},\n'
                for tool_name, handler_name in handler_entries
                if tool_name in supported_tools
            ),
        ),
        (
            list_start,
            list_end,
            "".join(
                block
                for tool_name, block in tool_blocks
                if tool_name in supported_tools
            ),
        ),
    ]
    replacements.extend(
        (
            function_spans[name][0],
            function_spans[name][1],
            "",
        )
        for name in sorted(unsupported_handler_names | unsupported_helper_names)
        if name in function_spans
    )

    for start, end, replacement in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        source_lines[start - 1 : end] = replacement.splitlines(keepends=True)

    return "".join(source_lines)


def prune_compat_server(server_path: Path, *, allowed_methods: Iterable[str]) -> None:
    source = server_path.read_text(encoding="utf-8")
    pruned_source = prune_compat_server_source(
        source,
        allowed_methods=allowed_methods,
    )
    _write_if_changed(server_path, pruned_source)


def patch_observability_server(server_path: Path) -> None:
    source = server_path.read_text(encoding="utf-8")
    patched_source = patch_observability_source(source)
    _write_if_changed(server_path, patched_source)



def ensure_compat_sources(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cl_hive_path: Path = DEFAULT_CL_HIVE_PATH,
    source_rev: str = DEFAULT_SOURCE_REV,
) -> Path:
    for rel_path, filename in SOURCE_FILES.items():
        content = _read_git_file(cl_hive_path, source_rev, rel_path)
        _write_if_changed(cache_dir / filename, content)

    return cache_dir / "mcp-hive-server.py"


def main() -> None:
    cl_hive_path = Path(
        os.environ.get("HIVE_MCP_COMPAT_CL_HIVE_PATH", str(DEFAULT_CL_HIVE_PATH))
    )
    revenue_ops_path = Path(
        os.environ.get(
            "HIVE_MCP_COMPAT_REVENUE_OPS_PATH",
            str(DEFAULT_REVENUE_OPS_PATH),
        )
    )
    cache_dir = Path(
        os.environ.get("HIVE_MCP_COMPAT_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    )
    source_rev = os.environ.get("HIVE_MCP_COMPAT_SOURCE_REV", DEFAULT_SOURCE_REV)
    existing_allowlist = os.environ.get("HIVE_ALLOWED_METHODS")

    server_path = ensure_compat_sources(
        cache_dir=cache_dir,
        cl_hive_path=cl_hive_path,
        source_rev=source_rev,
    )
    patch_observability_server(server_path)
    allowlist_path = write_rpc_allowlist(
        cache_dir=cache_dir,
        cl_hive_path=cl_hive_path,
        revenue_ops_path=revenue_ops_path,
        existing_allowlist_path=Path(existing_allowlist) if existing_allowlist else None,
    )
    prune_compat_server(
        server_path,
        allowed_methods=_read_allowlist(allowlist_path),
    )
    os.environ["HIVE_ALLOWED_METHODS"] = str(allowlist_path)
    runpy.run_path(str(server_path), run_name="__main__")


if __name__ == "__main__":
    main()
