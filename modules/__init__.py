"""
cl-revenue-ops modules package

This package contains the core modules for the Revenue Operations plugin:
- flow_analysis: Sink/Source detection and flow metrics
- fee_controller: DTS+PID dynamic fee optimization
- rebalancer: EV-based profit-aware rebalancing
- config: Configuration and constants
- database: SQLite storage layer
- policy_manager: Peer-level policy management (v1.4)
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
