"""Cross-repo metabolic Level 2c integration tests.

These tests build cl-mycelium metabolic_influence/v1 payloads from the local
cl-hive checkout and feed them to cl_revenue_ops consumers. They intentionally
stay in zero-budget / dry-run style unit surfaces and do not call action RPCs.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_controller import FeeController
from modules.rebalance_engine_v2 import RebalanceEngine
from modules.rebalance_types_v2 import PairCandidate
from tests.plugin_test_utils import load_plugin_module


CL_MYCELIUM_ROOT = Path("/home/sat/bin/cl-hive")
DIRECT = "02" + "a" * 64
FLEET = "02" + "b" * 64
OUT_OF_SCOPE = "02" + "c" * 64


def test_revenue_status_reports_no_spend_or_execution_in_canary_fixture():
    module = load_plugin_module()
    module.database = MagicMock()
    module.database.get_all_channel_states.return_value = []
    module.database.get_recent_fee_changes.return_value = []
    module.database.get_recent_rebalances.return_value = []
    module.config = MagicMock()
    module.config.public_runtime_keys.return_value = ["daily_budget_sats", "dry_run"]
    module.config.public_runtime_dict.return_value = {"daily_budget_sats": 0, "dry_run": True}
    module.fee_controller = MagicMock()
    module.fee_controller.get_last_decision_summary.return_value = {
        "action": "hold",
        "reason": "dry_run_canary",
        "dominant_input": "fee_controller",
        "safety_block": False,
    }
    module.rebalancer = MagicMock()
    module.rebalancer.get_last_decision_summary.return_value = {
        "action": "hold",
        "reason": "zero_budget_canary",
        "dominant_input": "rebalancer",
        "safety_block": True,
        "budget_blocked": True,
    }

    status = module.revenue_status(MagicMock())

    assert status["operator_controls"]["values"]["daily_budget_sats"] == 0
    assert status["operator_controls"]["values"]["dry_run"] is True
    assert status["recent_fee_changes"] == []
    assert status["recent_rebalances"] == []
    assert status["fee_decision"]["action"] == "hold"
    assert status["rebalance_decision"]["action"] == "hold"
    assert status["rebalance_decision"]["budget_blocked"] is True
