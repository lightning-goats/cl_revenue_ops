"""Compatibility aliases for the v2 rebalance execution contract.

The historical external executor was removed. New code should import
``ExecutionResult`` from ``modules.rebalance_execution`` and use
``NativeRouteExecutor`` directly.
"""

from __future__ import annotations

from .rebalance_execution import ExecutionResult
from .rebalance_native_executor_v2 import NativeRouteExecutor as RebalanceExecutor

__all__ = ["ExecutionResult", "RebalanceExecutor"]
