"""
cl-revenue-ops modules package

This package contains the core modules for the Revenue Operations plugin, grouped by
responsibility. Only a subset is re-exported below for convenience (`from .module import X`);
most modules are imported directly by cl-revenue-ops.py. Module families:

- Fee pricing: fee_controller (DTS+PID dynamic fee optimization), flow_analysis
  (sink/source detection and flow metrics)
- Rebalancing: rebalancer, rebalance_engine_v2/executor_v2/native_executor_v2/planner_v2,
  rebalance_router_v2/v3, rebalance_route_policy, rebalance_state_v2,
  rebalance_types_v2, rebalance_flow_facts, rebalance_audit_v2, rebalance_coordination_overlay,
  rebalance_execution
- Capital allocation: capacity_planner (channel opens/closes/defibrillation),
  capex_budget (budget/reservation engine), capital_efficiency (dead-capital analysis)
- On-/off-chain swaps: boltz_manager (submarine swap integration),
  lnplus_swaps (lightningnetwork.plus liquidity-swap automation)
- Analysis & reporting: profitability_analyzer, demand_flow, segment_observations
- Infrastructure: config (configuration and constants), database (SQLite storage layer),
  data_service, policy_manager (peer-level policy management, v1.4), utils
"""

from .flow_analysis import FlowAnalyzer, ChannelState, FlowMetrics
from .fee_controller import FeeController
from .rebalancer import EVRebalancer, RebalanceCandidate
from .config import Config
from .database import Database
from .data_service import DataService
from .policy_manager import PolicyManager, FeeStrategy, RebalanceMode, PeerPolicy

__all__ = [
    'FlowAnalyzer',
    'ChannelState',
    'FlowMetrics',
    'FeeController',
    'EVRebalancer',
    'RebalanceCandidate',
    'Config',
    'Database',
    'DataService',
    'PolicyManager',
    'FeeStrategy',
    'RebalanceMode',
    'PeerPolicy'
]
