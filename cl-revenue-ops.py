#!/usr/bin/env python3
"""
cl-revenue-ops: A Revenue Operations Plugin for Core Lightning

This plugin acts as a "Revenue Operations" layer that sits on top of the clboss 
automated manager. While clboss handles channel creation and node reliability,
this plugin overrides clboss for fee setting and rebalancing decisions to 
maximize profitability based on economic principles rather than heuristics.

MANAGER-OVERRIDE PATTERN:
-------------------------
Before changing any channel state, this plugin checks if the peer is managed 
by clboss. If it is, we issue the `clboss-unmanage` command for that specific 
peer and tag (e.g., lnfee) to prevent clboss from reverting our changes.

This allows us to:
1. Let clboss handle what it's good at (channel creation, peer selection)
2. Take over the economic decisions (fee setting, rebalancing) where we can
   apply more sophisticated algorithms

Dependencies:
- pyln-client: Core Lightning plugin framework
- bookkeeper plugin (built-in): Historical routing data
- External rebalancer (sling): Executes rebalance payments

Author: Lightning Goats Team
License: MIT
"""

import os
import sys
import time
import json
import random
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path

from pyln.client import Plugin, RpcError

# Import our modules
from modules.flow_analysis import FlowAnalyzer, ChannelState
from modules.fee_controller import PIDFeeController
from modules.rebalancer import EVRebalancer
from modules.clboss_manager import ClbossManager
from modules.config import Config
from modules.database import Database
from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
from modules.capacity_planner import CapacityPlanner
from modules.metrics import PrometheusExporter, MetricNames, METRIC_HELP

# Initialize the plugin
plugin = Plugin()

# =============================================================================
# THREAD-SAFE RPC WRAPPER (Phase 5.5: High-Uptime Stability)
# =============================================================================
# pyln-client's RPC is not inherently thread-safe for concurrent calls.
# This lock serializes all RPC calls to prevent race conditions when
# multiple background loops (Fee, Flow, Rebalance) fire simultaneously.

RPC_LOCK = threading.Lock()


class ThreadSafeRpcProxy:
    """
    A thread-safe proxy for the plugin's RPC interface.
    
    This wrapper ensures that all RPC calls are serialized through a lock,
    preventing race conditions when multiple background threads make
    concurrent calls to lightningd.
    
    Usage:
        # Instead of: plugin.rpc.listpeers()
        # Modules use: self.plugin.rpc.listpeers()  (unchanged syntax)
        # But the plugin they receive has this proxy as its .rpc attribute
    """
    
    def __init__(self, rpc):
        """Wrap the original RPC object."""
        self._rpc = rpc
    
    def __getattr__(self, name):
        """
        Intercept attribute access to wrap RPC method calls.
        
        Returns a thread-safe wrapper function that acquires the lock
        before calling the actual RPC method.
        """
        original_method = getattr(self._rpc, name)
        
        if callable(original_method):
            def thread_safe_method(*args, **kwargs):
                with RPC_LOCK:
                    return original_method(*args, **kwargs)
            return thread_safe_method
        else:
            # For non-callable attributes, return directly
            return original_method
    
    def call(self, method_name, payload=None):
        """
        Thread-safe wrapper for the generic RPC call method.
        
        This is used for plugin-specific calls like sling-job, sling-stats, etc.
        """
        with RPC_LOCK:
            if payload:
                return self._rpc.call(method_name, payload)
            return self._rpc.call(method_name)


class ThreadSafePluginProxy:
    """
    A proxy for the Plugin object that provides thread-safe RPC access.
    
    This allows modules to use the same interface (self.plugin.rpc.method())
    while ensuring all RPC calls are serialized through the lock.
    """
    
    def __init__(self, plugin):
        """Wrap the original plugin with a thread-safe RPC proxy."""
        self._plugin = plugin
        self.rpc = ThreadSafeRpcProxy(plugin.rpc)
    
    def log(self, message, level='info'):
        """Delegate logging to the original plugin."""
        self._plugin.log(message, level=level)
    
    def __getattr__(self, name):
        """Delegate all other attribute access to the original plugin."""
        return getattr(self._plugin, name)


# Global instances (initialized in init)
flow_analyzer: Optional[FlowAnalyzer] = None
fee_controller: Optional[PIDFeeController] = None
rebalancer: Optional[EVRebalancer] = None
clboss_manager: Optional[ClbossManager] = None
database: Optional[Database] = None
config: Optional[Config] = None
profitability_analyzer: Optional[ChannelProfitabilityAnalyzer] = None
capacity_planner: Optional[CapacityPlanner] = None
metrics_exporter: Optional[PrometheusExporter] = None
safe_plugin: Optional['ThreadSafePluginProxy'] = None  # Thread-safe plugin wrapper

# SCID to Peer ID cache for reputation tracking
# Maps short_channel_id -> peer_id for quick lookups
_scid_to_peer_cache: Dict[str, str] = {}


# =============================================================================
# PLUGIN OPTIONS
# =============================================================================

plugin.add_option(
    name='revenue-ops-db-path',
    default='~/.lightning/revenue_ops.db',
    description='Path to the SQLite database for storing state'
)

plugin.add_option(
    name='revenue-ops-flow-interval',
    default='3600',
    description='Interval in seconds for flow analysis (default: 1 hour)'
)

plugin.add_option(
    name='revenue-ops-fee-interval',
    default='1800',
    description='Interval in seconds for fee adjustments (default: 30 min)'
)

plugin.add_option(
    name='revenue-ops-rebalance-interval',
    default='900',
    description='Interval in seconds for rebalance checks (default: 15 min)'
)

plugin.add_option(
    name='revenue-ops-target-flow',
    default='100000',
    description='Target daily flow in sats per channel (default: 100,000)'
)

plugin.add_option(
    name='revenue-ops-min-fee-ppm',
    default='10',
    description='Minimum fee floor in PPM (default: 10)'
)

plugin.add_option(
    name='revenue-ops-max-fee-ppm',
    default='5000',
    description='Maximum fee ceiling in PPM (default: 5000)'
)

plugin.add_option(
    name='revenue-ops-rebalance-min-profit',
    default='10',
    description='Minimum profit in sats to trigger rebalance (default: 10)'
)

plugin.add_option(
    name='revenue-ops-pid-kp',
    default='0.5',
    description='PID Proportional gain (default: 0.5)'
)

plugin.add_option(
    name='revenue-ops-pid-ki',
    default='0.1',
    description='PID Integral gain (default: 0.1)'
)

plugin.add_option(
    name='revenue-ops-pid-kd',
    default='0.05',
    description='PID Derivative gain (default: 0.05)'
)

plugin.add_option(
    name='revenue-ops-flow-window-days',
    default='7',
    description='Number of days to analyze for flow calculation (default: 7)'
)

plugin.add_option(
    name='revenue-ops-clboss-enabled',
    default='true',
    description='Whether to interact with clboss for unmanage commands (default: true)'
)

plugin.add_option(
    name='revenue-ops-rebalancer',
    default='sling',
    description='Rebalancer plugin to use (default: sling)'
)

plugin.add_option(
    name='revenue-ops-daily-budget-sats',
    default='5000',
    description='Max rebalancing fees to spend in 24 hours - acts as floor when proportional budget enabled (default: 5000)'
)

plugin.add_option(
    name='revenue-ops-min-wallet-reserve',
    default='1000000',
    description='Minimum total funds (on-chain + off-chain) to keep in reserve (default: 1,000,000)'
)

plugin.add_option(
    name='revenue-ops-proportional-budget',
    default='false',
    description='If true, scale daily budget based on 24h revenue (default: false)'
)

plugin.add_option(
    name='revenue-ops-proportional-budget-pct',
    default='0.05',
    description='Percentage of 24h revenue to use as budget when proportional budget enabled (default: 0.05 = 5%)'
)

plugin.add_option(
    name='revenue-ops-dry-run',
    default='false',
    description='If true, log actions but do not execute (default: false)'
)

plugin.add_option(
    name='revenue-ops-htlc-congestion-threshold',
    default='0.8',
    description='HTLC slot utilization threshold (0.0-1.0) above which channel is considered congested (default: 0.8)'
)

plugin.add_option(
    name='revenue-ops-enable-reputation',
    default='true',
    description='If true, weight volume by peer reputation (success rate) in fee decisions (default: true)'
)

plugin.add_option(
    name='revenue-ops-reputation-decay',
    default='0.98',
    description='Reputation decay factor applied per flow-interval (default: 0.98). 0.98^24 ≈ 0.61 daily decay.'
)

plugin.add_option(
    name='revenue-ops-enable-prometheus',
    default='false',
    description='If true, start Prometheus metrics exporter HTTP server (default: false)'
)

plugin.add_option(
    name='revenue-ops-prometheus-port',
    default='9800',
    description='Port for Prometheus HTTP metrics server (default: 9800)'
)

plugin.add_option(
    name='revenue-ops-enable-kelly',
    default='false',
    description='If true, scale rebalance budget using Kelly Criterion based on peer reputation (default: false)'
)

plugin.add_option(
    name='revenue-ops-kelly-fraction',
    default='0.5',
    description='Multiplier for Kelly fraction (default: 0.5 = Half Kelly). Full Kelly (1.0) maximizes growth but has high volatility.'
)


# =============================================================================
# INITIALIZATION
# =============================================================================

@plugin.init()
def init(options: Dict[str, Any], configuration: Dict[str, Any], plugin: Plugin, **kwargs):
    """
    Initialize the Revenue Operations plugin.
    
    This is called once when the plugin starts. We:
    1. Parse and validate options
    2. Initialize the database
    3. Create instances of our analysis modules
    4. Set up timers for periodic execution
    5. Start Prometheus metrics exporter (if enabled)
    """
    global flow_analyzer, fee_controller, rebalancer, clboss_manager, database, config, profitability_analyzer, capacity_planner, metrics_exporter, safe_plugin
    
    plugin.log("Initializing cl-revenue-ops plugin...")
    
    # Build configuration from options
    config = Config(
        db_path=os.path.expanduser(options['revenue-ops-db-path']),
        flow_interval=int(options['revenue-ops-flow-interval']),
        fee_interval=int(options['revenue-ops-fee-interval']),
        rebalance_interval=int(options['revenue-ops-rebalance-interval']),
        target_flow=int(options['revenue-ops-target-flow']),
        min_fee_ppm=int(options['revenue-ops-min-fee-ppm']),
        max_fee_ppm=int(options['revenue-ops-max-fee-ppm']),
        rebalance_min_profit=int(options['revenue-ops-rebalance-min-profit']),
        pid_kp=float(options['revenue-ops-pid-kp']),
        pid_ki=float(options['revenue-ops-pid-ki']),
        pid_kd=float(options['revenue-ops-pid-kd']),
        flow_window_days=int(options['revenue-ops-flow-window-days']),
        clboss_enabled=options['revenue-ops-clboss-enabled'].lower() == 'true',
        rebalancer_plugin=options['revenue-ops-rebalancer'],
        daily_budget_sats=int(options['revenue-ops-daily-budget-sats']),
        min_wallet_reserve=int(options['revenue-ops-min-wallet-reserve']),
        enable_proportional_budget=options['revenue-ops-proportional-budget'].lower() == 'true',
        proportional_budget_pct=float(options['revenue-ops-proportional-budget-pct']),
        dry_run=options['revenue-ops-dry-run'].lower() == 'true',
        htlc_congestion_threshold=float(options['revenue-ops-htlc-congestion-threshold']),
        enable_reputation=options['revenue-ops-enable-reputation'].lower() == 'true',
        reputation_decay=float(options['revenue-ops-reputation-decay']),
        enable_prometheus=options['revenue-ops-enable-prometheus'].lower() == 'true',
        prometheus_port=int(options['revenue-ops-prometheus-port']),
        enable_kelly=options['revenue-ops-enable-kelly'].lower() == 'true',
        kelly_fraction=float(options['revenue-ops-kelly-fraction'])
    )
    
    plugin.log(f"Configuration loaded: target_flow={config.target_flow}, "
               f"fee_range=[{config.min_fee_ppm}, {config.max_fee_ppm}], "
               f"dry_run={config.dry_run}")
    
    # =========================================================================
    # STARTUP DEPENDENCY CHECKS (Phase 4: Stability & Scaling)
    # Verify external plugins are available before initializing dependent modules
    # =========================================================================
    try:
        # Try modern 'plugin list' command first, fallback to 'listplugins' for older nodes
        try:
            # Modern CLN (v23.08+)
            plugins_result = plugin.rpc.plugin("list")
        except RpcError:
            # Fallback for older CLN versions
            plugins_result = plugin.rpc.listplugins()
            
        active_plugins = [p.get("name", "").lower() for p in plugins_result.get("plugins", [])]
        
        # Check for sling plugin
        sling_found = any("sling" in name for name in active_plugins)
        if not sling_found:
            plugin.log(
                "Dependency 'sling' not found. Rebalancing module disabled. "
                "Install cln-sling to enable rebalancing.",
                level='warn'
            )
            config.sling_available = False
        else:
            plugin.log("Dependency check: sling plugin detected")
            config.sling_available = True
        
        # Check for bookkeeper plugin
        bookkeeper_found = any("bookkeeper" in name for name in active_plugins)
        if not bookkeeper_found:
            plugin.log(
                "Dependency 'bookkeeper' not found. Using 'listforwards' fallback for flow analysis. "
                "Enable bookkeeper for accurate cost tracking.",
                level='info'
            )
        else:
            plugin.log("Dependency check: bookkeeper plugin detected")
            
    except Exception as e:
        plugin.log(f"Error checking plugin dependencies: {e}", level='warn')
        # Assume plugins are available if check fails
        config.sling_available = True
    
    # Create thread-safe RPC proxy (Phase 5.5: High-Uptime Stability)
    # All background threads share a single RPC connection - serialize access
    # to prevent corruption from concurrent calls to lightningd
    safe_plugin = ThreadSafePluginProxy(plugin)
    plugin.log("Thread-safe RPC proxy initialized")
    
    # Initialize database
    database = Database(config.db_path, safe_plugin)
    database.initialize()
    
    
    # Snapshot currently connected peers for baseline state on restart
    # This establishes a known state for uptime tracking after plugin restarts
    try:
        peers = safe_plugin.rpc.listpeers()
        total_peers = len(peers.get("peers", []))
        connected_peers = 0
        snapshot_count = 0
        
        plugin.log(f"Checking {total_peers} peers for connection snapshot...")
        
        for peer in peers.get("peers", []):
            if peer.get("connected", False):
                connected_peers += 1
                peer_id = peer["id"]
                # Only insert snapshot if no recent history exists (within 1 hour)
                has_recent = database.has_recent_connection_history(peer_id, 3600)
                plugin.log(f"Peer {peer_id[:12]}... is connected, has_recent_history={has_recent}", level='debug')
                if not has_recent:
                    database.record_connection_event(peer_id, "snapshot")
                    snapshot_count += 1
        
        plugin.log(f"Connection baseline: {connected_peers} connected peers, snapshotted {snapshot_count} new peers")
    except Exception as e:
        plugin.log(f"Error snapshotting peer connections: {e}", level='warn')
        import traceback
        plugin.log(f"Traceback: {traceback.format_exc()}", level='warn')
    
    # Initialize Prometheus metrics exporter (Phase 2: Observability)
    if config.enable_prometheus:
        metrics_exporter = PrometheusExporter(port=config.prometheus_port, plugin=safe_plugin)
        if not metrics_exporter.start_server():
            plugin.log("Prometheus metrics disabled due to server startup failure", level='warn')
            metrics_exporter = None
    else:
        metrics_exporter = None
        plugin.log("Prometheus metrics exporter disabled by configuration")
    
    # Initialize clboss manager (handles unmanage commands)
    clboss_manager = ClbossManager(safe_plugin, config)
    
    # Initialize profitability analyzer (with metrics exporter)
    profitability_analyzer = ChannelProfitabilityAnalyzer(safe_plugin, config, database, metrics_exporter)
    
    # Initialize analysis modules with profitability analyzer and metrics exporter
    flow_analyzer = FlowAnalyzer(safe_plugin, config, database)
    capacity_planner = CapacityPlanner(safe_plugin, config, profitability_analyzer, flow_analyzer)
    fee_controller = PIDFeeController(safe_plugin, config, database, clboss_manager, profitability_analyzer, metrics_exporter)
    rebalancer = EVRebalancer(safe_plugin, config, database, clboss_manager, metrics_exporter)
    rebalancer.set_profitability_analyzer(profitability_analyzer)
    
    # Set up periodic background tasks using threading
    # Note: plugin.log() is safe to call from threads in pyln-client
    # We use daemon threads so they don't block shutdown
    
    def flow_analysis_loop():
        """Background loop for flow analysis."""
        # Initial delay to let lightningd fully start
        time.sleep(10)
        while True:
            try:
                plugin.log("Running scheduled flow analysis...")
                run_flow_analysis()
                
                # Run cleanup on each iteration (it's a fast DELETE query)
                # Keeps history tables from growing unbounded over months
                # Use flow_window_days + 1 day buffer, minimum 8 days
                if database:
                    days_to_keep = max(8, config.flow_window_days + 1)
                    database.cleanup_old_data(days_to_keep=days_to_keep)
                
                # Export peer reputation metrics (Phase 2: Observability)
                if metrics_exporter and database:
                    update_peer_reputation_metrics()
                
                # Update last run timestamp for health monitoring
                if metrics_exporter:
                    metrics_exporter.set_gauge(
                        MetricNames.SYSTEM_LAST_RUN_TIMESTAMP,
                        int(time.time()),
                        {"task": "flow"},
                        METRIC_HELP.get(MetricNames.SYSTEM_LAST_RUN_TIMESTAMP, "")
                    )
                    
            except Exception as e:
                plugin.log(f"Error in flow analysis: {e}", level='error')
            # Calculate +/- 20% jitter
            jitter_seconds = int(config.flow_interval * 0.2)
            sleep_time = config.flow_interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Flow analysis sleeping for {sleep_time}s")
            time.sleep(sleep_time)
    
    def fee_adjustment_loop():
        """Background loop for fee adjustment."""
        # Initial delay to let flow analysis run first
        time.sleep(60)
        while True:
            try:
                plugin.log("Running scheduled fee adjustment...")
                run_fee_adjustment()
                
                # Update last run timestamp for health monitoring
                if metrics_exporter:
                    metrics_exporter.set_gauge(
                        MetricNames.SYSTEM_LAST_RUN_TIMESTAMP,
                        int(time.time()),
                        {"task": "fee"},
                        METRIC_HELP.get(MetricNames.SYSTEM_LAST_RUN_TIMESTAMP, "")
                    )
            except Exception as e:
                plugin.log(f"Error in fee adjustment: {e}", level='error')
            # Calculate +/- 20% jitter
            jitter_seconds = int(config.fee_interval * 0.2)
            sleep_time = config.fee_interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Fee adjustment sleeping for {sleep_time}s")
            time.sleep(sleep_time)
    
    def rebalance_check_loop():
        """Background loop for rebalance checks."""
        # Skip rebalancing entirely if sling is not available
        if not config.sling_available:
            plugin.log("Rebalance loop disabled: sling plugin not found")
            return
        
        # Initial delay to let other analyses run first
        time.sleep(120)
        while True:
            try:
                plugin.log("Running scheduled rebalance check...")
                run_rebalance_check()
                
                # Update last run timestamp for health monitoring
                if metrics_exporter:
                    metrics_exporter.set_gauge(
                        MetricNames.SYSTEM_LAST_RUN_TIMESTAMP,
                        int(time.time()),
                        {"task": "rebalance"},
                        METRIC_HELP.get(MetricNames.SYSTEM_LAST_RUN_TIMESTAMP, "")
                    )
            except Exception as e:
                plugin.log(f"Error in rebalance check: {e}", level='error')
            # Calculate +/- 20% jitter
            jitter_seconds = int(config.rebalance_interval * 0.2)
            sleep_time = config.rebalance_interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Rebalance check sleeping for {sleep_time}s")
            time.sleep(sleep_time)
    
    def snapshot_peers_delayed():
        """
        One-time delayed snapshot of connected peers.
        
        Sleeps to allow lightningd to establish connections, then records
        a snapshot for all currently connected peers. Exits after completion.
        """
        delay_seconds = 60
        plugin.log(f"Startup snapshot: waiting {delay_seconds}s for network connections...")
        time.sleep(delay_seconds)
        
        try:
            peers = safe_plugin.rpc.listpeers()
            connected_count = 0
            snapshot_count = 0
            
            for peer in peers.get("peers", []):
                if peer.get("connected", False):
                    connected_count += 1
                    peer_id = peer["id"]
                    # Only snapshot if no recent history exists
                    if not database.has_recent_connection_history(peer_id, 3600):
                        database.record_connection_event(peer_id, "snapshot")
                        snapshot_count += 1
            
            plugin.log(f"Startup snapshot: Recorded {snapshot_count} of {connected_count} connected peers")
        except Exception as e:
            plugin.log(f"Error in delayed snapshot: {e}", level='warn')
            import traceback
            plugin.log(f"Traceback: {traceback.format_exc()}", level='warn')
    
    # Start background threads (daemon=True so they don't block shutdown)
    threading.Thread(target=flow_analysis_loop, daemon=True, name="flow-analysis").start()
    threading.Thread(target=fee_adjustment_loop, daemon=True, name="fee-adjustment").start()
    threading.Thread(target=rebalance_check_loop, daemon=True, name="rebalance-check").start()
    threading.Thread(target=snapshot_peers_delayed, daemon=True, name="startup-snapshot").start()
    
    plugin.log("cl-revenue-ops plugin initialized successfully!")
    return None


# =============================================================================
# CORE LOGIC FUNCTIONS
# =============================================================================

def run_flow_analysis():
    """
    Module 1: Flow Analysis & Sink/Source Detection
    
    Query bookkeeper to calculate the "Net Flow" of every channel over 
    the last N days. Calculate FlowRatio and mark channels as Source/Sink/Balanced.
    
    Also applies reputation decay to ensure recent peer behavior matters more
    than ancient history.
    """
    if flow_analyzer is None:
        plugin.log("Flow analyzer not initialized", level='error')
        return
    
    try:
        results = flow_analyzer.analyze_all_channels()
        plugin.log(f"Flow analysis complete: {len(results)} channels analyzed")
        
        # Log summary
        sources = sum(1 for r in results.values() if r.state == ChannelState.SOURCE)
        sinks = sum(1 for r in results.values() if r.state == ChannelState.SINK)
        balanced = sum(1 for r in results.values() if r.state == ChannelState.BALANCED)
        plugin.log(f"Channel states: {sources} sources, {sinks} sinks, {balanced} balanced")
        
        # Apply reputation decay (Phase 3: Time-windowing)
        # This ensures recent peer behavior matters more than ancient history
        if database and config and config.enable_reputation:
            database.decay_reputation(config.reputation_decay)
            plugin.log(f"Applied reputation decay (factor={config.reputation_decay})")
        
    except Exception as e:
        plugin.log(f"Flow analysis failed: {e}", level='error')
        raise


def run_fee_adjustment():
    """
    Module 2: PID Fee Controller (Dynamic Pricing)
    
    Adjust channel fees based on the Flow Analysis using a PID controller.
    Before setting fees, unmanage from clboss to prevent conflicts.
    """
    if fee_controller is None:
        plugin.log("Fee controller not initialized", level='error')
        return
    
    try:
        adjustments = fee_controller.adjust_all_fees()
        plugin.log(f"Fee adjustment complete: {len(adjustments)} channels adjusted")
        
    except Exception as e:
        plugin.log(f"Fee adjustment failed: {e}", level='error')
        raise


def run_rebalance_check():
    """
    Module 3: EV-Based Rebalancing (Profit-Aware)
    
    Identify rebalance candidates based on expected value calculation.
    Only trigger rebalances when the EV is positive and significant.
    """
    if rebalancer is None:
        plugin.log("Rebalancer not initialized", level='error')
        return
    
    try:
        candidates = rebalancer.find_rebalance_candidates()
        plugin.log(f"Rebalance check complete: {len(candidates)} profitable candidates found")
        
        for candidate in candidates:
            rebalancer.execute_rebalance(candidate)
            
    except Exception as e:
        plugin.log(f"Rebalance check failed: {e}", level='error')
        raise


def update_peer_reputation_metrics():
    """
    Export peer reputation data to Prometheus metrics.
    
    Phase 2: Observability - Track peer reliability scores.
    
    Called periodically from flow_analysis_loop to update:
    - cl_revenue_peer_reputation_score: Success rate (0.0 to 1.0)
    - cl_revenue_peer_success_count: Total successful forwards
    - cl_revenue_peer_failure_count: Total failed forwards
    """
    if database is None or metrics_exporter is None:
        return
    
    try:
        reputations = database.get_all_peer_reputations()
        
        for rep in reputations:
            peer_id = rep.get('peer_id', '')
            if not peer_id:
                continue
            
            labels = {"peer_id": peer_id}
            
            # Gauge: Reputation score (success rate 0.0 to 1.0)
            metrics_exporter.set_gauge(
                MetricNames.PEER_REPUTATION_SCORE,
                rep.get('score', 1.0),
                labels,
                METRIC_HELP.get(MetricNames.PEER_REPUTATION_SCORE, "")
            )
            
            # Gauge: Success count (using gauge so we can see current state)
            metrics_exporter.set_gauge(
                MetricNames.PEER_SUCCESS_COUNT,
                rep.get('successes', 0),
                labels,
                METRIC_HELP.get(MetricNames.PEER_SUCCESS_COUNT, "")
            )
            
            # Gauge: Failure count
            metrics_exporter.set_gauge(
                MetricNames.PEER_FAILURE_COUNT,
                rep.get('failures', 0),
                labels,
                METRIC_HELP.get(MetricNames.PEER_FAILURE_COUNT, "")
            )
        
        plugin.log(f"Updated Prometheus metrics for {len(reputations)} peer reputations", level='debug')
        
    except Exception as e:
        plugin.log(f"Error updating peer reputation metrics: {e}", level='warn')


# =============================================================================
# RPC METHODS - Exposed to lightning-cli
# =============================================================================

@plugin.method("revenue-status")
def revenue_status(plugin: Plugin) -> Dict[str, Any]:
    """
    Get the current status of the revenue operations plugin.
    
    Usage: lightning-cli revenue-status
    """
    if database is None:
        return {"error": "Plugin not fully initialized"}
    
    channel_states = database.get_all_channel_states()
    fee_history = database.get_recent_fee_changes(limit=10)
    rebalance_history = database.get_recent_rebalances(limit=10)
    
    return {
        "status": "running",
        "config": {
            "target_flow_sats": config.target_flow,
            "fee_range_ppm": [config.min_fee_ppm, config.max_fee_ppm],
            "rebalance_min_profit_sats": config.rebalance_min_profit,
            "dry_run": config.dry_run
        },
        "channel_states": channel_states,
        "recent_fee_changes": fee_history,
        "recent_rebalances": rebalance_history
    }


@plugin.method("revenue-analyze")
def revenue_analyze(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run flow analysis on demand (optionally for a specific channel).
    
    Usage: lightning-cli revenue-analyze [channel_id]
    """
    if flow_analyzer is None:
        return {"error": "Plugin not fully initialized"}
    
    if channel_id:
        result = flow_analyzer.analyze_channel(channel_id)
        return {"channel": channel_id, "analysis": result.to_dict() if result else None}
    else:
        run_flow_analysis()
        return {"status": "Flow analysis triggered"}



@plugin.method("revenue-capacity-report")
def revenue_capacity_report(plugin: Plugin, **kwargs):
    """
    Generate a strategic capital redeployment report.
    
    Identifies "Winner" channels for capital injection (Splice-In)
    and "Loser" channels for capital extraction (Splice-Out/Close).
    """
    if capacity_planner is None:
        raise RpcError("revenue-capacity-report", {}, "Capacity planner not initialized")
        
    return capacity_planner.generate_report()


@plugin.method("revenue-set-fee")
def revenue_set_fee(plugin: Plugin, channel_id: str, fee_ppm: int) -> Dict[str, Any]:
    """
    Manually set fee for a channel (with clboss unmanage).
    
    Usage: lightning-cli revenue-set-fee channel_id fee_ppm
    """
    if fee_controller is None:
        return {"error": "Plugin not fully initialized"}
    
    try:
        result = fee_controller.set_channel_fee(channel_id, fee_ppm, manual=True)
        return {"status": "success", "channel": channel_id, "new_fee_ppm": fee_ppm, **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-rebalance")
def revenue_rebalance(plugin: Plugin, 
                      from_channel: str, 
                      to_channel: str, 
                      amount_sats: int,
                      max_fee_sats: Optional[int] = None) -> Dict[str, Any]:
    """
    Manually trigger a rebalance with profit constraints.
    
    Usage: lightning-cli revenue-rebalance from_channel to_channel amount_sats [max_fee_sats]
    """
    if rebalancer is None:
        return {"error": "Plugin not fully initialized"}
    
    if config and not config.sling_available:
        return {"error": "Rebalancing disabled: sling plugin not found. Install cln-sling to enable."}
    
    try:
        result = rebalancer.manual_rebalance(from_channel, to_channel, amount_sats, max_fee_sats)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-clboss-status")
def revenue_clboss_status(plugin: Plugin) -> Dict[str, Any]:
    """
    Check which channels are currently unmanaged from clboss.
    
    Usage: lightning-cli revenue-clboss-status
    """
    if clboss_manager is None:
        return {"error": "Plugin not fully initialized"}
    
    return clboss_manager.get_unmanaged_status()


@plugin.method("revenue-profitability")
def revenue_profitability(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get channel profitability analysis.
    
    Shows each channel's:
    - Total costs (opening + rebalancing)
    - Total revenue (routing fees)
    - Net profit/loss
    - ROI percentage
    - Profitability classification (profitable, break_even, underwater, zombie)
    
    Usage: lightning-cli revenue-profitability [channel_id]
    """
    if profitability_analyzer is None:
        return {"error": "Plugin not fully initialized"}
    
    try:
        if channel_id:
            # Analyze single channel
            result = profitability_analyzer.analyze_channel(channel_id)
            if result:
                return {
                    "channel_id": channel_id,
                    "profitability": {
                        "total_costs_sats": result.costs.total_cost_sats,
                        "total_revenue_sats": result.revenue.fees_earned_sats,
                        "net_profit_sats": result.net_profit_sats,
                        "roi_percentage": round(result.roi_percent, 2),
                        "profitability_class": result.classification.value,
                        "days_active": result.days_open,
                        "volume_routed_sats": result.revenue.volume_routed_sats,
                        "forward_count": result.revenue.forward_count,
                        "fee_multiplier": profitability_analyzer.get_fee_multiplier(channel_id)
                    }
                }
            else:
                return {"channel_id": channel_id, "error": "No data available"}
        else:
            # Analyze all channels
            all_results = profitability_analyzer.analyze_all_channels()
            
            # Group by profitability class
            summary = {
                "profitable": [],
                "break_even": [],
                "underwater": [],
                "zombie": []
            }
            total_profit = 0
            total_revenue = 0
            total_costs = 0
            
            for ch_id, result in all_results.items():
                channel_summary = {
                    "channel_id": ch_id,
                    "net_profit_sats": result.net_profit_sats,
                    "roi_percentage": round(result.roi_percent, 2),
                    "days_active": result.days_open
                }
                summary[result.classification.value].append(channel_summary)
                total_profit += result.net_profit_sats
                total_revenue += result.revenue.fees_earned_sats
                total_costs += result.costs.total_cost_sats
            
            return {
                "summary": {
                    "total_channels": len(all_results),
                    "profitable_count": len(summary["profitable"]),
                    "break_even_count": len(summary["break_even"]),
                    "underwater_count": len(summary["underwater"]),
                    "zombie_count": len(summary["zombie"]),
                    "total_profit_sats": total_profit,
                    "total_revenue_sats": total_revenue,
                    "total_costs_sats": total_costs,
                    "overall_roi_pct": round((total_profit / total_costs * 100) if total_costs > 0 else 0, 2)
                },
                "channels_by_class": summary
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-history")
def revenue_history(plugin: Plugin) -> Dict[str, Any]:
    """
    Get lifetime financial history including closed channels.
    
    Reports aggregate financial performance since the plugin was installed,
    including data from channels that have since been closed. This provides
    a true "Lifetime P&L" view.
    
    Returns:
        - Lifetime Revenue (total routing fees earned)
        - Lifetime Costs (opening fees + rebalancing fees)
        - Lifetime Net Profit (revenue - costs)
        - Lifetime ROI percentage
        - Total number of forwards processed
    
    Usage: lightning-cli revenue-history
    """
    if profitability_analyzer is None:
        return {"error": "Plugin not initialized"}
    
    try:
        return profitability_analyzer.get_lifetime_report()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-remanage")
def revenue_remanage(plugin: Plugin, peer_id: str, tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Re-enable clboss management for a peer (release our override).
    
    Usage: lightning-cli revenue-remanage peer_id [tag]
    """
    if clboss_manager is None:
        return {"error": "Plugin not fully initialized"}
    
    try:
        result = clboss_manager.remanage(peer_id, tag)
        return {"status": "success", "peer_id": peer_id, **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# HOOKS - React to Lightning events
# =============================================================================

@plugin.hook("htlc_accepted")
def on_htlc_accepted(onion: Dict, htlc: Dict, plugin: Plugin, **kwargs) -> Dict[str, str]:
    """
    Hook called when an HTLC is accepted.
    
    We can use this to track live routing activity and update our flow metrics
    in real-time rather than waiting for periodic analysis.
    
    For now, we just let it pass through - the periodic analysis from bookkeeper
    is sufficient for initial implementation.
    """
    # Just continue - we don't want to interfere with routing
    return {"result": "continue"}


def _resolve_scid_to_peer(scid: str) -> Optional[str]:
    """
    Resolve a short_channel_id to its peer_id.
    
    Uses a cache to avoid repeated RPC calls. Cache is refreshed if the
    SCID is not found (channel might be new).
    
    Args:
        scid: Short channel ID (e.g., "123x456x0")
        
    Returns:
        peer_id (node pubkey) or None if not found
    """
    global _scid_to_peer_cache
    
    # Check cache first
    if scid in _scid_to_peer_cache:
        return _scid_to_peer_cache[scid]
    
    # Cache miss - refresh cache from listpeerchannels
    # Use safe_plugin for thread-safe RPC access
    try:
        result = safe_plugin.rpc.listpeerchannels()
        for channel in result.get("channels", []):
            channel_scid = channel.get("short_channel_id") or channel.get("channel_id")
            peer_id = channel.get("peer_id")
            if channel_scid and peer_id:
                _scid_to_peer_cache[channel_scid] = peer_id
        
        # Try again after refresh
        return _scid_to_peer_cache.get(scid)
    except RpcError as e:
        plugin.log(f"Error resolving SCID {scid} to peer: {e}", level='warn')
        return None
        

def _parse_msat(msat_val: Any) -> int:
    """
    Safely convert msat values to integers.
    Handles '1000msat' strings, raw integers, Millisatoshi objects, and plain numeric strings.
    """
    if msat_val is None:
        return 0
    if hasattr(msat_val, 'millisatoshis'):
        return int(msat_val.millisatoshis)
    if isinstance(msat_val, int):
        return msat_val
    if isinstance(msat_val, str):
        # Strip suffix if present
        if msat_val.endswith('msat'):
            clean_val = msat_val[:-4]
        else:
            clean_val = msat_val
            
        try:
            return int(clean_val)
        except ValueError:
            return 0
    return 0


@plugin.subscribe("forward_event")
def on_forward_event(forward_event: Dict, plugin: Plugin, **kwargs):
    """
    Notification when a forward completes (success or failure).
    
    We use this for:
    1. Real-time flow tracking (settled forwards)
    2. Peer reputation tracking (success/failure rates)
    
    Reputation tracking helps identify unreliable peers for traffic intelligence.
    """
    if database is None:
        return
    
    status = forward_event.get("status")
    in_channel = forward_event.get("in_channel")
    
    # Normalize SCID: replace colons with 'x' for consistency
    if in_channel:
        in_channel = in_channel.replace(':', 'x')
    
    # Track peer reputation for all forward outcomes
    if in_channel:
        peer_id = _resolve_scid_to_peer(in_channel)
        if peer_id:
            if status == "settled":
                # Success - increment success count
                database.update_peer_reputation(peer_id, is_success=True)
            elif status in ("failed", "local_failed"):
                # Failure - increment failure count
                database.update_peer_reputation(peer_id, is_success=False)
            
            # Real-time metrics update (Phase 2: Observability)
            # Update Prometheus metrics immediately after DB update
            if metrics_exporter:
                rep = database.get_peer_reputation(peer_id)
                labels = {"peer_id": peer_id}
                
                metrics_exporter.set_gauge(
                    MetricNames.PEER_REPUTATION_SCORE,
                    rep.get('score', 1.0),
                    labels
                )
                metrics_exporter.set_gauge(
                    MetricNames.PEER_SUCCESS_COUNT,
                    rep.get('successes', 0),
                    labels
                )
                metrics_exporter.set_gauge(
                    MetricNames.PEER_FAILURE_COUNT,
                    rep.get('failures', 0),
                    labels
                )
    
    # Record successful forwards for flow metrics
    if status == "settled":
        out_channel = forward_event.get("out_channel")
        if out_channel:
            out_channel = out_channel.replace(':', 'x')
            
        in_msat = _parse_msat(forward_event.get("in_msatoshi", 0))
        out_msat = _parse_msat(forward_event.get("out_msatoshi", 0))
        fee_msat = _parse_msat(forward_event.get("fee_msatoshi", 0))
        
        # Calculate resolution duration (Risk Premium tracking)
        # durations in CLN are usually in seconds (float)
        received_time = forward_event.get("received_time", 0)
        resolved_time = forward_event.get("resolved_time", 0)
        resolution_duration = resolved_time - received_time if resolved_time > 0 else 0
        
        database.record_forward(in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_duration)


@plugin.subscribe("connect")
def on_peer_connect(plugin: Plugin, **kwargs):
    """
    Notification when a peer connects.
    
    Records the connection event for uptime tracking.
    """
    if database is None:
        return
    
    # Log full structure for debugging
    plugin.log(f"Connect notification: {kwargs}", level='debug')
    
    # Try multiple extraction methods for compatibility
    peer_id = None
    
    # Method 1: Nested under 'connect' key
    if 'connect' in kwargs and isinstance(kwargs['connect'], dict):
        peer_id = kwargs['connect'].get('id')
    
    # Method 2: Direct 'id' key
    if not peer_id and 'id' in kwargs:
        peer_id = kwargs['id']
    
    # Method 3: Check for nested peer_id
    if not peer_id and 'connect' in kwargs and isinstance(kwargs['connect'], dict):
        peer_id = kwargs['connect'].get('peer_id')
    
    if peer_id:
        database.record_connection_event(peer_id, "connected")
        plugin.log(f"Peer connected: {peer_id[:12]}...", level='info')
    else:
        plugin.log(f"Connect event - could not extract peer_id from: {kwargs}", level='warn')


@plugin.subscribe("disconnect")
def on_peer_disconnect(plugin: Plugin, **kwargs):
    """
    Notification when a peer disconnects.
    
    Records the disconnection event for uptime tracking.
    """
    if database is None:
        return
    
    # Log full structure for debugging
    plugin.log(f"Disconnect notification: {kwargs}", level='debug')
    
    # Try multiple extraction methods for compatibility
    peer_id = None
    
    # Method 1: Nested under 'disconnect' key
    if 'disconnect' in kwargs and isinstance(kwargs['disconnect'], dict):
        peer_id = kwargs['disconnect'].get('id')
    
    # Method 2: Direct 'id' key
    if not peer_id and 'id' in kwargs:
        peer_id = kwargs['id']
    
    # Method 3: Check for nested peer_id
    if not peer_id and 'disconnect' in kwargs and isinstance(kwargs['disconnect'], dict):
        peer_id = kwargs['disconnect'].get('peer_id')
    
    if peer_id:
        database.record_connection_event(peer_id, "disconnected")
        plugin.log(f"Peer disconnected: {peer_id[:12]}...", level='info')
    else:
        plugin.log(f"Disconnect event - could not extract peer_id from: {kwargs}", level='warn')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plugin.run()
