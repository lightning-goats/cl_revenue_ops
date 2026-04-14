import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_tree() -> ast.Module:
    source = (ROOT / "cl-revenue-ops.py").read_text()
    return ast.parse(source, filename="cl-revenue-ops.py")


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in cl-revenue-ops.py")


def _contains_refresh_call(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == "refresh_hive_runtime":
            return True
    return False


def test_rebalance_check_loop_refreshes_hive_runtime_before_running():
    tree = _load_tree()
    rebalance_loop = _find_function(tree, "rebalance_check_loop")
    assert _contains_refresh_call(rebalance_loop), (
        "rebalance_check_loop must refresh hive runtime so post-restart "
        "rebalance cycles do not run with empty hive hints"
    )
