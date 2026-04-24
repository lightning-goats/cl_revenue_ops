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


def _calls_function(func: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == name:
            return True
    return False


def _first_call_index(func: ast.FunctionDef, name: str) -> int:
    for index, node in enumerate(ast.walk(func)):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == name:
            return index
    raise AssertionError(f"{func.name} does not call {name}")


def test_rebalance_check_loop_refreshes_hive_runtime_before_running():
    tree = _load_tree()
    rebalance_loop = _find_function(tree, "rebalance_check_loop")
    assert _contains_refresh_call(rebalance_loop), (
        "rebalance_check_loop must refresh hive runtime so post-restart "
        "rebalance cycles do not run with empty hive hints"
    )


def test_manual_rebalance_cycle_refreshes_hive_runtime_before_running():
    tree = _load_tree()
    rebalance_cycle = _find_function(tree, "revenue_rebalance_cycle")
    refresh_index = _first_call_index(rebalance_cycle, "refresh_hive_runtime")
    run_index = _first_call_index(rebalance_cycle, "run_rebalance_check")
    assert refresh_index < run_index, (
        "manual revenue-rebalance-cycle must refresh hive hints before planning"
    )


def test_manual_fee_cycle_refreshes_hive_inputs_before_adjusting():
    tree = _load_tree()
    fee_cycle = _find_function(tree, "revenue_fee_cycle")
    refresh_index = _first_call_index(fee_cycle, "_refresh_fee_cycle_hive_inputs")
    adjust_index = _first_call_index(fee_cycle, "run_fee_adjustment")
    assert refresh_index < adjust_index, (
        "manual revenue-fee-cycle must refresh hive hints before adjusting fees"
    )


def test_scheduled_fee_loop_uses_same_hive_refresh_helper():
    tree = _load_tree()
    fee_loop = _find_function(tree, "fee_adjustment_loop")
    assert _calls_function(fee_loop, "_refresh_fee_cycle_hive_inputs"), (
        "scheduled and manual fee cycles should share the same hive refresh path"
    )
