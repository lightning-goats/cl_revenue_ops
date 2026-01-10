"""
Flow Analysis module for cl-revenue-ops

MODULE 1: Flow Analysis & Sink/Source Detection

This module analyzes routing flow through each channel to classify them as:
- SOURCE: Channels that are draining (sats flowing out)
- SINK: Channels that are filling up (sats flowing in)
- BALANCED: Channels with roughly equal in/out flow

The classification drives fee and rebalancing decisions:
- Sources need higher fees (scarce outbound liquidity)
- Sinks need lower fees (encourage outflow)
- Balanced channels are at target state

Data Sources:
1. bookkeeper plugin (preferred) - provides accounting-grade data
2. listforwards RPC - fallback if bookkeeper unavailable
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from pyln.client import Plugin, RpcError


class ChannelState(Enum):
    """
    Classification of channel flow state.
    
    SOURCE: Net outflow - channel is draining
    SINK: Net inflow - channel is filling
    BALANCED: Roughly equal flow - ideal state
    UNKNOWN: Not enough data to classify
    CONGESTED: HTLC slots near exhaustion (>80% used)
    """
    SOURCE = "source"
    SINK = "sink"
    BALANCED = "balanced"
    UNKNOWN = "unknown"
    CONGESTED = "congested"


@dataclass
class FlowMetrics:
    """
    Flow metrics for a single channel.
    
    Attributes:
        channel_id: Short channel ID
        peer_id: Node ID of the peer
        sats_in: Total sats routed into this channel (from peer)
        sats_out: Total sats routed out of this channel (to peer)
        capacity: Channel capacity in sats
        flow_ratio: (sats_out - sats_in) / capacity
        state: Classified state (SOURCE/SINK/BALANCED/CONGESTED)
        daily_volume: Average daily routing volume
        analysis_window_days: Number of days analyzed
        htlc_min: Minimum HTLC amount (msat)
        htlc_max: Maximum HTLC amount (msat)
        active_htlcs: Number of currently active HTLCs
        max_htlcs: Maximum allowed HTLCs on the channel
        is_congested: True if HTLC slots are >80% utilized
        our_balance: Current outbound balance in sats
    """
    channel_id: str
    peer_id: str
    sats_in: int
    sats_out: int
    capacity: int
    flow_ratio: float
    state: ChannelState
    daily_volume: int
    analysis_window_days: int
    htlc_min: int = 0
    htlc_max: int = 0
    active_htlcs: int = 0
    max_htlcs: int = 483  # Default per BOLT spec
    is_congested: bool = False
    our_balance: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "sats_in": self.sats_in,
            "sats_out": self.sats_out,
            "capacity": self.capacity,
            "flow_ratio": round(self.flow_ratio, 4),
            "state": self.state.value,
            "daily_volume": self.daily_volume,
            "analysis_window_days": self.analysis_window_days,
            "htlc_min": self.htlc_min,
            "htlc_max": self.htlc_max,
            "active_htlcs": self.active_htlcs,
            "max_htlcs": self.max_htlcs,
            "is_congested": self.is_congested,
            "our_balance": self.our_balance
        }


class FlowAnalyzer:
    """
    Analyzes routing flow to classify channels as Source/Sink/Balanced.
    
    Flow Analysis Logic:
    1. Query bookkeeper or listforwards for historical routing data
    2. Calculate net flow for each channel using Exponential Moving Average (EMA)
    3. Compute FlowRatio = (EMA_Out - EMA_In) / Capacity
    4. Classify based on thresholds:
       - FlowRatio > 0.5: SOURCE (draining)
       - FlowRatio < -0.5: SINK (filling)
       - Otherwise: BALANCED
    """
    
    def __init__(self, plugin: Plugin, config, database):
        """
        Initialize the flow analyzer.
        
        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance for persistence
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        self._bookkeeper_available: Optional[bool] = None
    
    def is_bookkeeper_available(self) -> bool:
        """
        Check if the bookkeeper plugin is available.
        
        Returns:
            True if bookkeeper commands are available
        """
        if self._bookkeeper_available is not None:
            return self._bookkeeper_available
        
        try:
            # Try a bookkeeper command
            self.plugin.rpc.call("bkpr-listbalances")
            self._bookkeeper_available = True
            self.plugin.log("bookkeeper plugin detected and available")
            return True
        except RpcError as e:
            if "Unknown command" in str(e):
                self._bookkeeper_available = False
                self.plugin.log("bookkeeper not available, using listforwards fallback")
                return False
            # Other errors might be temporary
            return True
        except Exception as e:
            self.plugin.log(f"Error checking bookkeeper: {e}", level='warn')
            self._bookkeeper_available = False
            return False
    
    def analyze_all_channels(self) -> Dict[str, FlowMetrics]:
        """
        Analyze flow for all channels.
        
        This is the main entry point, called periodically by the timer.
        
        Returns:
            Dict mapping channel_id to FlowMetrics
        """
        results = {}
        
        # Get list of all channels
        channels = self._get_channels()
        
        if not channels:
            self.plugin.log("No channels found to analyze")
            return results
        
        self.plugin.log(f"Analyzing flow for {len(channels)} channels")
        
        # Get flow data from listforwards (most reliable source with correct channel IDs)
        # UPDATED: Now returns daily buckets for EMA calculation
        flow_data_daily = self._get_daily_flow_from_listforwards()
        
        # Analyze each channel
        for channel in channels:
            channel_id = channel.get("short_channel_id") or channel.get("channel_id")
            if not channel_id:
                continue
            
            peer_id = channel.get("peer_id", "")
            
            # Calculate capacity - may be null in some CLN versions
            # Always fetch spendable/receivable first for balance calculation
            spendable_msat = channel.get("spendable_msat", 0) or 0
            receivable_msat = channel.get("receivable_msat", 0) or 0

            capacity_msat = channel.get("capacity_msat")
            if capacity_msat is None or capacity_msat == 0:
                # Calculate from spendable + receivable (approximate)
                capacity = (spendable_msat + receivable_msat) // 1000
            else:
                capacity = capacity_msat // 1000

            if capacity == 0:
                capacity = channel.get("capacity", 0)

            # Get current balance for fallback inference
            our_balance = spendable_msat // 1000
            
            # Get daily buckets for this channel
            channel_daily = flow_data_daily.get(channel_id, [])
            
            # Calculate EMA flow and total volume
            ema_in, ema_out, total_in, total_out = self._calculate_ema_flow(channel_daily)
            
            # Extract HTLC information for congestion detection
            htlc_min = channel.get("htlc_min_msat", 0)
            htlc_max = channel.get("htlc_max_msat", 0)
            active_htlcs = channel.get("active_htlcs", 0)
            max_htlcs = channel.get("max_htlcs", 483)
            
            # Calculate metrics (with balance fallback for zero-flow channels)
            metrics = self._calculate_metrics(
                channel_id=channel_id,
                peer_id=peer_id,
                sats_in=total_in,
                sats_out=total_out,
                ema_in=ema_in,    # Pass EMA for classification
                ema_out=ema_out,  # Pass EMA for classification
                capacity=capacity,
                our_balance=our_balance,
                htlc_min=htlc_min,
                htlc_max=htlc_max,
                active_htlcs=active_htlcs,
                max_htlcs=max_htlcs
            )
            
            results[channel_id] = metrics
            
            # Store in database
            self.database.update_channel_state(
                channel_id=channel_id,
                peer_id=peer_id,
                state=metrics.state.value,
                flow_ratio=metrics.flow_ratio,
                sats_in=total_in,
                sats_out=total_out,
                capacity=capacity
            )
        
        return results
    
    def analyze_channel(self, channel_id: str) -> Optional[FlowMetrics]:
        """
        Analyze flow for a specific channel.
        
        Args:
            channel_id: The channel to analyze
            
        Returns:
            FlowMetrics for the channel, or None if not found
        """
        # Get channel info
        channel = self._get_channel(channel_id)
        if not channel:
            return None
        
        peer_id = channel.get("peer_id", "")
        
        # Calculate capacity
        capacity_msat = channel.get("capacity_msat")
        spendable_msat = channel.get("spendable_msat", 0) or 0
        receivable_msat = channel.get("receivable_msat", 0) or 0
        
        if capacity_msat is None or capacity_msat == 0:
            capacity = (spendable_msat + receivable_msat) // 1000
        else:
            capacity = capacity_msat // 1000
        
        if capacity == 0:
            capacity = channel.get("capacity", 0)
        
        our_balance = spendable_msat // 1000
        
        # Get daily flow data
        flow_data_daily = self._get_daily_flow_from_listforwards(channel_id)
        channel_daily = flow_data_daily.get(channel_id, [])
        
        ema_in, ema_out, total_in, total_out = self._calculate_ema_flow(channel_daily)
        
        # Extract HTLC information
        htlc_min = channel.get("htlc_min_msat", 0)
        htlc_max = channel.get("htlc_max_msat", 0)
        active_htlcs = channel.get("active_htlcs", 0)
        max_htlcs = channel.get("max_htlcs", 483)
        
        return self._calculate_metrics(
            channel_id=channel_id,
            peer_id=peer_id,
            sats_in=total_in,
            sats_out=total_out,
            ema_in=ema_in,
            ema_out=ema_out,
            capacity=capacity,
            our_balance=our_balance,
            htlc_min=htlc_min,
            htlc_max=htlc_max,
            active_htlcs=active_htlcs,
            max_htlcs=max_htlcs
        )
    
    def _calculate_metrics(self, channel_id: str, peer_id: str,
                          sats_in: int, sats_out: int, capacity: int,
                          ema_in: float = 0.0, ema_out: float = 0.0,
                          our_balance: int = 0,
                          htlc_min: int = 0, htlc_max: int = 0,
                          active_htlcs: int = 0, max_htlcs: int = 483) -> FlowMetrics:
        """
        Calculate flow metrics and classify a channel using EMA.
        
        The FlowRatio formula (EMA-based):
        FlowRatio = (EMA_Out - EMA_In) / Capacity
        
        This makes the classification responsive to recent trend reversals.
        
        Interpretation:
        - Positive ratio: Net outflow trend (SOURCE)
        - Negative ratio: Net inflow trend (SINK)
        - Near zero: Balanced flow
        """
        has_flow_data = sats_in > 0 or sats_out > 0
        
        # Calculate flow ratio from EMA data
        if capacity > 0:
            flow_ratio = (ema_out - ema_in) / capacity
        else:
            flow_ratio = 0.0
        
        # Check HTLC slot congestion FIRST
        htlc_utilization = active_htlcs / max_htlcs if max_htlcs > 0 else 0.0
        is_congested = htlc_utilization > self.config.htlc_congestion_threshold
        
        if is_congested:
            state = ChannelState.CONGESTED
            self.plugin.log(
                f"Channel {channel_id} is CONGESTED: {active_htlcs}/{max_htlcs} "
                f"HTLC slots used ({htlc_utilization:.1%})"
            )
        elif has_flow_data:
            # Use EMA flow data for classification
            if flow_ratio > self.config.source_threshold:
                state = ChannelState.SOURCE
            elif flow_ratio < self.config.sink_threshold:
                state = ChannelState.SINK
            else:
                state = ChannelState.BALANCED
        else:
            # FALLBACK: Infer from current balance
            outbound_ratio = our_balance / capacity if capacity > 0 else 0.5
            
            if outbound_ratio < 0.30:
                state = ChannelState.SOURCE
                flow_ratio = 0.6
            elif outbound_ratio > 0.70:
                state = ChannelState.SINK
                flow_ratio = -0.6
            else:
                state = ChannelState.BALANCED
                flow_ratio = 0.0
        
        # Calculate daily volume (simple average for display/stats)
        total_volume = sats_in + sats_out
        daily_volume = total_volume // max(self.config.flow_window_days, 1)
        
        return FlowMetrics(
            channel_id=channel_id,
            peer_id=peer_id,
            sats_in=sats_in,
            sats_out=sats_out,
            capacity=capacity,
            flow_ratio=flow_ratio,
            state=state,
            daily_volume=daily_volume,
            analysis_window_days=self.config.flow_window_days,
            htlc_min=htlc_min,
            htlc_max=htlc_max,
            active_htlcs=active_htlcs,
            max_htlcs=max_htlcs,
            is_congested=is_congested,
            our_balance=our_balance
        )
    
    def _get_daily_flow_from_listforwards(self, channel_id: Optional[str] = None) -> Dict[str, List[Dict[str, int]]]:
        """
        Get daily flow buckets from local database.
        
        TODO #19: "Double-Dip" Fix - Uses local SQLite instead of listforwards RPC.
        The forwards table is populated by the forward_event hook in real-time,
        and hydrated on startup to fill any gaps while the plugin was offline.
        
        This eliminates the heaviest RPC call in the plugin, reducing CPU usage
        by ~90% during flow analysis cycles on high-volume nodes.
        
        Instead of summing everything, this buckets data by day (0 = today, 1 = yesterday, etc.)
        to support EMA calculation.
        
        Returns:
            Dict mapping channel_id to a list of daily buckets:
            {'scid': [{'in': 100, 'out': 50}, {'in': 200, 'out': 80}, ...]}
        """
        window_days = self.config.flow_window_days
        
        try:
            # Use local database aggregation instead of RPC
            flow_data = self.database.get_daily_flow_buckets(
                window_days=window_days,
                channel_id=channel_id
            )
            return flow_data
            
        except Exception as e:
            self.plugin.log(f"Error querying flow from database: {e}", level='error')
            return {}
    
    def _calculate_ema_flow(self, daily_buckets: List[Dict[str, int]]) -> Tuple[float, float, int, int]:
        """
        Calculate Exponential Moving Average (EMA) for flow.
        
        Weights recent days significantly higher to reduce lag.
        Formula: 
           Weight = decay_factor ^ age
           EMA = Sum(Value * Weight) / Sum(Weight)
        
        Using decay_factor = 0.8:
           Day 0 (Today): 1.0
           Day 1: 0.8
           Day 2: 0.64
           ...
           
        Returns:
            (ema_in, ema_out, total_in, total_out)
        """
        if not daily_buckets:
            return 0.0, 0.0, 0, 0
            
        ema_in = 0.0
        ema_out = 0.0
        total_weight = 0.0
        total_in = 0
        total_out = 0
        
        decay_factor = 0.8
        
        for age, bucket in enumerate(daily_buckets):
            weight = decay_factor ** age
            
            ema_in += bucket['in'] * weight
            ema_out += bucket['out'] * weight
            
            total_in += bucket['in']
            total_out += bucket['out']
            
            total_weight += weight
            
        if total_weight <= 0:
            return 0.0, 0.0, 0, 0
            
        ema_in /= total_weight
        ema_out /= total_weight
            
        return ema_in, ema_out, total_in, total_out

    def _get_channels(self) -> List[Dict[str, Any]]:
        """
        Get list of all channels from lightningd with HTLC information.
        
        Extracts HTLC slot limits and current usage for congestion detection:
        - htlc_minimum_msat: Minimum HTLC amount
        - htlc_maximum_msat: Maximum HTLC amount
        - max_accepted_htlcs: Maximum number of HTLCs allowed
        - htlcs: List of currently active HTLCs
        """
        try:
            result = self.plugin.rpc.listpeerchannels()
            channels = []
            
            # listpeerchannels returns channels grouped by peer
            for channel_info in result.get("channels", []):
                if channel_info.get("state") == "CHANNELD_NORMAL":
                    # Extract HTLC limits and current usage
                    # htlc_minimum_msat and htlc_maximum_msat are our advertised limits
                    channel_info["htlc_min_msat"] = channel_info.get("htlc_minimum_msat", 0)
                    channel_info["htlc_max_msat"] = channel_info.get("htlc_maximum_msat", 0)
                    
                    # max_accepted_htlcs is the limit on concurrent HTLCs
                    # Default is 483 per BOLT #2
                    channel_info["max_htlcs"] = channel_info.get("max_accepted_htlcs", 483)
                    
                    # Count active HTLCs from the htlcs array
                    htlcs = channel_info.get("htlcs", [])
                    channel_info["active_htlcs"] = len(htlcs) if htlcs else 0
                    
                    channels.append(channel_info)
            
            return channels
        except RpcError as e:
            self.plugin.log(f"Error getting channels: {e}", level='error')
            return []
    
    def _get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific channel."""
        channels = self._get_channels()
        for channel in channels:
            scid = channel.get("short_channel_id") or channel.get("channel_id")
            if scid == channel_id:
                return channel
        return None
    
    def get_channel_state(self, channel_id: str) -> ChannelState:
        """
        Get the cached state of a channel.
        
        Args:
            channel_id: The channel to check
            
        Returns:
            The channel's current state classification
        """
        state_data = self.database.get_channel_state(channel_id)
        if state_data:
            return ChannelState(state_data.get("state", "unknown"))
        return ChannelState.UNKNOWN
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get all channels classified as SOURCE (draining)."""
        return self.database.get_channels_by_state("source")
    
    def get_sinks(self) -> List[Dict[str, Any]]:
        """Get all channels classified as SINK (filling)."""
        return self.database.get_channels_by_state("sink")
    
    def get_balanced(self) -> List[Dict[str, Any]]:
        """Get all channels classified as BALANCED."""
        return self.database.get_channels_by_state("balanced")