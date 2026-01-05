"""
cl-revenue-ops modules package

This package contains the core modules for the Revenue Operations plugin:
- flow_analysis: Sink/Source detection and flow metrics
- fee_controller: Hill Climbing (Perturb & Observe) dynamic fee optimization
- rebalancer: EV-based profit-aware rebalancing
- clboss_manager: Interface for clboss unmanage commands
- config: Configuration and constants
- database: SQLite storage layer
- policy_manager: Peer-level policy management (v1.4)
"""

from .flow_analysis import FlowAnalyzer, ChannelState, FlowMetrics
from .fee_controller import PIDFeeController
from .rebalancer import EVRebalancer, RebalanceCandidate
from .clboss_manager import ClbossManager
from .config import Config
from .database import Database
from .policy_manager import PolicyManager, FeeStrategy, RebalanceMode, PeerPolicy

__all__ = [
    'FlowAnalyzer',
    'ChannelState', 
    'FlowMetrics',
    'PIDFeeController',
    'EVRebalancer',
    'RebalanceCandidate',
    'ClbossManager',
    'Config',
    'Database',
    'PolicyManager',
    'FeeStrategy',
    'RebalanceMode',
    'PeerPolicy'
]
