"""
Channel Profitability Analyzer Module

This module tracks costs and revenue for each channel to determine profitability.
It provides actionable data for fee setting and rebalancing decisions.

Key metrics tracked:
- Channel open cost (on-chain fees)
- Rebalance costs (fees paid to acquire liquidity)
- Routing revenue (fees earned from forwarding)
- Volume routed (total sats forwarded)

Classifications:
- PROFITABLE: ROI > 0, earning more than costs
- BREAK_EVEN: ROI ~0, covering costs but not much profit
- UNDERWATER: ROI < 0, losing money
- ZOMBIE: Underwater + low volume, should consider closing
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import time

from pyln.client import Plugin

if TYPE_CHECKING:
    from .hive_bridge import HiveFeeIntelligenceBridge


class ProfitabilityClass(Enum):
    """Channel profitability classification."""
    PROFITABLE = "profitable"      # ROI > 10%
    BREAK_EVEN = "break_even"      # ROI between -10% and 10%
    UNDERWATER = "underwater"      # ROI < -10%
    STAGNANT_CANDIDATE = "stagnant_candidate"  # 0 forwards in last 7 days
    ZOMBIE = "zombie"              # Underwater + failed diagnostic recovery


class ChannelRole(Enum):
    """
    Channel flow role classification based on directional activity.

    Helps identify what purpose a channel serves in the routing topology:
    - INBOUND_GATEWAY: Primarily sources volume from the network (>70% inbound)
    - OUTBOUND_GATEWAY: Primarily exits payments to the network (>70% outbound)
    - BALANCED: Roughly equal flow in both directions (within 70/30)
    - DORMANT: Little to no flow in either direction
    """
    INBOUND_GATEWAY = "inbound_gateway"    # >70% of activity is sourcing inbound
    OUTBOUND_GATEWAY = "outbound_gateway"  # >70% of activity is exit outbound
    BALANCED = "balanced"                   # Flow in both directions
    DORMANT = "dormant"                     # No significant activity


@dataclass
class ChannelCosts:
    """
    Cost tracking for a channel.
    
    Attributes:
        channel_id: Channel short ID
        peer_id: Peer node ID
        open_cost_sats: On-chain fees paid to open
        rebalance_cost_sats: Total fees paid for rebalancing
        total_cost_sats: Sum of all costs
    """
    channel_id: str
    peer_id: str
    open_cost_sats: int
    rebalance_cost_sats: int
    
    @property
    def total_cost_sats(self) -> int:
        return self.open_cost_sats + self.rebalance_cost_sats


@dataclass
class ChannelRevenue:
    """
    Revenue tracking for a channel.

    Attributes:
        channel_id: Channel short ID
        fees_earned_sats: Total routing fees earned (as exit channel)
        volume_routed_sats: Total sats forwarded through channel (as exit)
        forward_count: Number of successful forwards (as exit)
        sourced_volume_sats: Volume that entered through this channel (as entry)
        sourced_fee_contribution_sats: Fees earned on exits for forwards sourced here
        sourced_forward_count: Number of forwards where this was entry channel
    """
    channel_id: str
    fees_earned_sats: int
    volume_routed_sats: int
    forward_count: int
    # Inbound contribution metrics (channel as entry point)
    sourced_volume_sats: int = 0
    sourced_fee_contribution_sats: int = 0
    sourced_forward_count: int = 0

    @property
    def total_contribution_sats(self) -> int:
        """Total value: direct fees + fees enabled by sourcing volume."""
        return self.fees_earned_sats + self.sourced_fee_contribution_sats

    @property
    def total_forward_count(self) -> int:
        """Total forwards: as exit + as entry."""
        return self.forward_count + self.sourced_forward_count


@dataclass 
class ChannelProfitability:
    """
    Complete profitability analysis for a channel.
    
    Attributes:
        channel_id: Channel short ID
        peer_id: Peer node ID
        capacity_sats: Channel capacity
        costs: Cost breakdown
        revenue: Revenue breakdown
        net_profit_sats: Revenue - Costs (Total/Accounting view)
        roi_percent: Return on investment percentage (Total ROI - includes open cost)
        classification: Profitability class
        cost_per_sat_routed: Average cost per sat of volume
        fee_per_sat_routed: Average fee earned per sat of volume
        days_open: How long the channel has been open
        last_routed: Timestamp of last routing activity
    
    Important Distinction:
        - roi_percent (Total ROI): Accounting view including sunk costs (open_cost_sats)
        - marginal_roi: Operational view - only considers ongoing costs (rebalance_costs)
        
    The marginal_roi is what matters for operational decisions:
    A channel that is covering its rebalancing costs is operationally profitable,
    even if it hasn't "paid back" the initial opening cost (sunk cost fallacy).
    """
    channel_id: str
    peer_id: str
    capacity_sats: int
    costs: ChannelCosts
    revenue: ChannelRevenue
    net_profit_sats: int
    roi_percent: float
    classification: ProfitabilityClass
    cost_per_sat_routed: float
    fee_per_sat_routed: float
    days_open: int
    last_routed: Optional[int]
    
    @property
    def marginal_roi(self) -> float:
        """
        Calculate Marginal ROI (Operational profitability).

        This metric EXCLUDES open_cost_sats (sunk cost) and focuses only on
        operational profitability: are we covering our rebalancing costs?

        Uses total_contribution_sats which includes:
        - Direct fees earned (as exit channel)
        - Sourced fee contribution (fees enabled by sourcing inbound volume)

        Formula: (total_contribution - rebalance_costs) / max(1, rebalance_costs)

        Returns:
            Marginal ROI as a decimal (e.g., 0.5 = 50% marginal return)
            Returns 1.0 if no rebalance costs and earning/contributing
            Returns 0.0 if no rebalance costs and no contribution
        """
        # Use total contribution (direct fees + sourced fee contribution)
        total_contribution = self.revenue.total_contribution_sats
        rebalance_costs = self.costs.rebalance_cost_sats

        # If no rebalancing has occurred, check if channel is contributing value
        if rebalance_costs == 0:
            # No operational costs - if contributing anything, it's pure profit
            return 1.0 if total_contribution > 0 else 0.0

        # Marginal profit = total contribution minus rebalancing costs (NO open cost!)
        marginal_profit = total_contribution - rebalance_costs

        # ROI relative to rebalancing investment
        return marginal_profit / max(1, rebalance_costs)
    
    @property
    def marginal_roi_percent(self) -> float:
        """Marginal ROI as a percentage."""
        return self.marginal_roi * 100
    
    @property
    def is_operationally_profitable(self) -> bool:
        """
        Check if channel is operationally profitable (covering rebalance costs).

        This is the key metric for fee decisions - we should NOT penalize channels
        just because they haven't paid back their opening cost.
        """
        return self.marginal_roi >= 0

    @property
    def channel_role(self) -> ChannelRole:
        """
        Classify channel's primary flow role based on directional activity.

        Uses forward counts (not volume) to classify:
        - >70% as entry channel → INBOUND_GATEWAY
        - >70% as exit channel → OUTBOUND_GATEWAY
        - Otherwise → BALANCED or DORMANT

        Returns:
            ChannelRole enum value
        """
        total_forwards = self.revenue.total_forward_count

        # Dormant if less than 10 forwards total
        if total_forwards < 10:
            return ChannelRole.DORMANT

        # Calculate ratios
        inbound_ratio = self.revenue.sourced_forward_count / total_forwards
        outbound_ratio = self.revenue.forward_count / total_forwards

        # >70% in one direction = gateway
        if inbound_ratio > 0.70:
            return ChannelRole.INBOUND_GATEWAY
        elif outbound_ratio > 0.70:
            return ChannelRole.OUTBOUND_GATEWAY
        else:
            return ChannelRole.BALANCED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "capacity_sats": self.capacity_sats,
            "open_cost_sats": self.costs.open_cost_sats,
            "rebalance_cost_sats": self.costs.rebalance_cost_sats,
            "total_cost_sats": self.costs.total_cost_sats,
            "fees_earned_sats": self.revenue.fees_earned_sats,
            "volume_routed_sats": self.revenue.volume_routed_sats,
            "forward_count": self.revenue.forward_count,
            "net_profit_sats": self.net_profit_sats,
            "roi_percent": round(self.roi_percent, 2),
            "marginal_roi_percent": round(self.marginal_roi_percent, 2),
            "is_operationally_profitable": self.is_operationally_profitable,
            "classification": self.classification.value,
            "cost_per_sat_routed": round(self.cost_per_sat_routed, 6),
            "fee_per_sat_routed": round(self.fee_per_sat_routed, 6),
            "days_open": self.days_open,
            "last_routed": self.last_routed,
            # Issue #21: Inbound vs outbound revenue attribution
            "channel_role": self.channel_role.value,
            "inbound_flow": {
                "payment_count": self.revenue.sourced_forward_count,
                "volume_sats": self.revenue.sourced_volume_sats,
                "contribution_to_other_channels_sats": self.revenue.sourced_fee_contribution_sats
            },
            "outbound_flow": {
                "payment_count": self.revenue.forward_count,
                "volume_sats": self.revenue.volume_routed_sats,
                "revenue_earned_sats": self.revenue.fees_earned_sats
            }
        }


class ChannelProfitabilityAnalyzer:
    """
    Analyzes channel profitability for informed fee and rebalancing decisions.
    
    This module:
    1. Tracks costs (open fees, rebalance fees)
    2. Tracks revenue (routing fees earned)
    3. Calculates ROI and classifies channels
    4. Provides multipliers for fee and rebalance decisions
    """
    
    # Classification thresholds
    PROFITABLE_ROI_THRESHOLD = 0.10    # > 10% ROI
    UNDERWATER_ROI_THRESHOLD = -0.10   # < -10% ROI
    ZOMBIE_DAYS_INACTIVE = 30          # No routing for 30 days
    ZOMBIE_MIN_LOSS_SATS = 1000        # Minimum loss to be zombie
    
    def __init__(self, plugin: Plugin, config, database,
                 hive_bridge: Optional["HiveFeeIntelligenceBridge"] = None):
        """
        Initialize the profitability analyzer.

        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance for persistence
            hive_bridge: Optional bridge to cl-hive for NNLB health reporting
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        self.hive_bridge = hive_bridge

        # Cache for profitability data (refreshed periodically)
        self._profitability_cache: Dict[str, ChannelProfitability] = {}
        self._cache_timestamp: int = 0
        self._cache_ttl: int = 300  # 5 minutes

        # Track last health report to avoid spam
        self._last_health_report: int = 0
        self._health_report_interval: int = 300  # Report every 5 minutes max
        
    def _parse_msat(self, msat_val: Any) -> int:
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
    
    def analyze_all_channels(self) -> Dict[str, ChannelProfitability]:
        """
        Analyze profitability for all channels.
        
        This method is optimized to batch fetch revenue data with a single
        RPC call to listforwards, avoiding N+1 query overhead.
        
        Returns:
            Dict mapping channel_id to ChannelProfitability
        """
        results = {}
        
        try:
            # Get all channels
            channels = self._get_all_channels()
            
            # Batch fetch all revenue data with a single RPC call
            all_revenue_data = self._get_all_revenue_data()
            
            for channel_id, channel_info in channels.items():
                # Pass precalculated revenue to avoid per-channel RPC calls
                precalculated_revenue = all_revenue_data.get(channel_id)
                profitability = self.analyze_channel(
                    channel_id, channel_info, precalculated_revenue=precalculated_revenue
                )
                if profitability:
                    results[channel_id] = profitability
            
            # Update cache
            self._profitability_cache = results
            self._cache_timestamp = int(time.time())
            
            # Log summary
            classifications = {}
            for p in results.values():
                cls = p.classification.value
                classifications[cls] = classifications.get(cls, 0) + 1
            
            self.plugin.log(
                f"Profitability analysis complete: {len(results)} channels - "
                f"{classifications}"
            )

            # Report health and liquidity to cl-hive for fleet coordination
            # INFORMATION ONLY - no fund transfers between nodes
            self._report_health_to_hive()
            self._report_liquidity_state_to_hive()

        except Exception as e:
            self.plugin.log(f"Error in profitability analysis: {e}", level='error')

        return results
    
    def analyze_channel(self, channel_id: str, 
                       channel_info: Optional[Dict] = None,
                       precalculated_revenue: Optional[ChannelRevenue] = None) -> Optional[ChannelProfitability]:
        """
        Analyze profitability for a single channel.
        
        Args:
            channel_id: Channel to analyze
            channel_info: Optional channel info (fetched if not provided)
            precalculated_revenue: Optional pre-fetched revenue data to avoid
                                   per-channel RPC calls when batch processing
            
        Returns:
            ChannelProfitability object or None if analysis fails
        """
        try:
            # Get channel info if not provided
            if channel_info is None:
                channels = self._get_all_channels()
                channel_info = channels.get(channel_id)
                if not channel_info:
                    return None
            
            peer_id = channel_info.get("peer_id", "")
            capacity = channel_info.get("capacity", 0)
            funding_txid = channel_info.get("funding_txid", "")
            opener = channel_info.get("opener", "local")
            
            # Get costs from database
            # Pass opener to correctly handle remote vs local channel costs
            costs = self._get_channel_costs(channel_id, peer_id, funding_txid, capacity, opener)
            
            # Get revenue from routing history
            # Use precalculated data if provided, otherwise fall back to single RPC call
            if precalculated_revenue is not None:
                revenue = precalculated_revenue
            else:
                revenue = self._get_channel_revenue(channel_id)

            # Calculate metrics using total contribution (direct + sourced)
            # This properly values channels that source inbound volume
            total_contribution = revenue.total_contribution_sats
            net_profit = total_contribution - costs.total_cost_sats

            # ROI based on total investment (costs)
            if costs.total_cost_sats > 0:
                roi = net_profit / costs.total_cost_sats
            else:
                # No costs recorded (e.g. remote open, no rebalancing)
                # Infinite ROI if contributing value, 0 otherwise
                roi = 1.0 if total_contribution > 0 else 0.0

            # Cost/fee per sat routed (using total volume: exit + sourced)
            total_volume = revenue.volume_routed_sats + revenue.sourced_volume_sats
            if total_volume > 0:
                cost_per_sat = costs.total_cost_sats / total_volume
                fee_per_sat = total_contribution / total_volume
            else:
                cost_per_sat = 0.0
                fee_per_sat = 0.0
            
            # Days open
            open_timestamp = channel_info.get("open_timestamp", int(time.time()))
            days_open = (int(time.time()) - open_timestamp) // 86400
            
            # Last routing activity
            last_routed = self._get_last_routing_time(channel_id)
            
            # Classify
            classification = self._classify_channel(
                roi, net_profit, last_routed, days_open,
                channel_id=channel_id
            )
            
            profitability = ChannelProfitability(
                channel_id=channel_id,
                peer_id=peer_id,
                capacity_sats=capacity,
                costs=costs,
                revenue=revenue,
                net_profit_sats=net_profit,
                roi_percent=roi * 100,
                classification=classification,
                cost_per_sat_routed=cost_per_sat,
                fee_per_sat_routed=fee_per_sat,
                days_open=days_open,
                last_routed=last_routed
            )
            
            return profitability
            
        except Exception as e:
            self.plugin.log(
                f"Error analyzing channel {channel_id}: {e}", 
                level='warn'
            )
            return None
    
    def get_profitability(self, channel_id: str) -> Optional[ChannelProfitability]:
        """
        Get profitability data for a channel (uses cache if fresh).
        
        Args:
            channel_id: Channel to look up
            
        Returns:
            ChannelProfitability or None
        """
        # Check cache freshness
        if (int(time.time()) - self._cache_timestamp) > self._cache_ttl:
            self.analyze_all_channels()
        
        return self._profitability_cache.get(channel_id)
    
    def get_fee_multiplier(self, channel_id: str) -> float:
        """
        Get fee multiplier based on channel's MARGINAL (operational) profitability.
        
        CRITICAL: Uses marginal_roi, NOT total ROI.
        
        This avoids the SUNK COST FALLACY:
        - A channel should NOT be penalized with high fees just because
          it had a high opening cost that hasn't been recovered yet.
        - What matters operationally is: Is this channel covering its
          ONGOING costs (rebalancing) with its fee revenue?
        
        If a channel is operationally profitable (marginal_roi >= 0),
        it's working well and should keep competitive fees to maintain volume.
        
        Args:
            channel_id: Channel to get multiplier for
            
        Returns:
            Fee multiplier (1.0 = no change)
        """
        profitability = self.get_profitability(channel_id)
        
        if not profitability:
            return 1.0  # No data, no adjustment
        
        # Use MARGINAL ROI for operational decisions
        # This ignores sunk costs (open_cost_sats)
        marginal_roi = profitability.marginal_roi
        
        # Fee multipliers based on operational profitability
        if marginal_roi > 0.20:  # > 20% marginal return
            # Highly profitable operationally - keep fees competitive
            return 0.95
        elif marginal_roi >= 0:  # Breaking even or better on operations
            # Covering costs - no change needed
            return 1.0
        elif marginal_roi >= -0.20:  # -20% to 0 marginal return
            # Slight operational loss - modest fee increase
            return 1.05
        elif marginal_roi >= -0.50:  # -50% to -20% marginal return
            # Significant operational loss - larger fee increase
            return 1.10
        else:  # < -50% marginal return
            # Severe operational loss - check if zombie
            if profitability.classification == ProfitabilityClass.ZOMBIE:
                return 1.0  # Don't bother adjusting zombies, flag for closure
            return 1.15  # Try to recover operational costs
    
    def get_marginal_roi(self, channel_id: str) -> Optional[float]:
        """
        Get the marginal ROI for a channel.
        
        This is the operational profitability metric that excludes sunk costs.
        
        Args:
            channel_id: Channel to get marginal ROI for
            
        Returns:
            Marginal ROI as decimal, or None if no data
        """
        profitability = self.get_profitability(channel_id)
        if not profitability:
            return None
        return profitability.marginal_roi
    
    def get_rebalance_priority(self, channel_id: str) -> float:
        """
        Get rebalance priority multiplier based on profitability.
        
        Higher priority = more worth rebalancing.
        
        Args:
            channel_id: Channel to get priority for
            
        Returns:
            Priority multiplier (1.0 = normal, >1 = higher, <1 = lower)
        """
        profitability = self.get_profitability(channel_id)
        
        if not profitability:
            return 1.0  # No data, normal priority
        
        # Priority based on classification
        priorities = {
            ProfitabilityClass.PROFITABLE: 1.5,          # High priority - proven earner
            ProfitabilityClass.BREAK_EVEN: 1.0,          # Normal priority
            ProfitabilityClass.UNDERWATER: 0.5,          # Low priority - not worth much
            ProfitabilityClass.STAGNANT_CANDIDATE: 0.3,  # Very low - needs diagnostic first
            ProfitabilityClass.ZOMBIE: 0.0,              # Skip entirely - don't waste fees
        }

        return priorities.get(profitability.classification, 1.0)
    
    def get_max_rebalance_fee_multiplier(self, channel_id: str) -> float:
        """
        Get multiplier for maximum rebalance fee budget.
        
        Profitable channels are worth paying more to keep full.
        
        Args:
            channel_id: Channel to get multiplier for
            
        Returns:
            Budget multiplier (1.0 = normal, >1 = pay more, <1 = pay less)
        """
        profitability = self.get_profitability(channel_id)
        
        if not profitability:
            return 1.0
        
        # Budget based on ROI - pay up to what the channel has proven to earn
        if profitability.classification == ProfitabilityClass.PROFITABLE:
            # Pay more for proven earners - up to 1.5x normal budget
            return min(1.5, 1.0 + (profitability.roi_percent / 100))
        elif profitability.classification == ProfitabilityClass.BREAK_EVEN:
            return 1.0
        elif profitability.classification == ProfitabilityClass.UNDERWATER:
            return 0.5  # Half budget - already losing money
        elif profitability.classification == ProfitabilityClass.STAGNANT_CANDIDATE:
            # BUG FIX: Stagnant channels need some budget for diagnostic rebalances
            return 0.3  # Reduced budget - needs diagnostic testing first
        else:  # ZOMBIE
            return 0.0  # No budget - don't rebalance
    
    def should_rebalance(self, channel_id: str) -> Tuple[bool, str]:
        """
        Determine if a channel should be rebalanced based on profitability.
        
        Args:
            channel_id: Channel to check
            
        Returns:
            Tuple of (should_rebalance, reason)
        """
        profitability = self.get_profitability(channel_id)
        
        if not profitability:
            return True, "no_profitability_data"
        
        if profitability.classification == ProfitabilityClass.ZOMBIE:
            return False, f"zombie_channel (ROI={profitability.roi_percent:.1f}%, inactive {self._days_since_routed(profitability)}+ days)"
        
        if profitability.classification == ProfitabilityClass.UNDERWATER:
            # Allow rebalancing but log warning
            return True, f"underwater_channel (ROI={profitability.roi_percent:.1f}%) - consider if worth it"
        
        return True, f"{profitability.classification.value} (ROI={profitability.roi_percent:.1f}%)"
    
    def record_rebalance_cost(self, channel_id: str, peer_id: str, 
                              cost_sats: int, amount_sats: int):
        """
        Record a rebalance cost for a channel.
        
        Called after a successful rebalance to track costs.
        
        Args:
            channel_id: Channel that was rebalanced into
            peer_id: Peer node ID
            cost_sats: Fee paid for the rebalance
            amount_sats: Amount rebalanced
        """
        self.database.record_rebalance_cost(
            channel_id=channel_id,
            peer_id=peer_id,
            cost_sats=cost_sats,
            amount_sats=amount_sats,
            timestamp=int(time.time())
        )
        
        # Invalidate cache for this channel
        if channel_id in self._profitability_cache:
            del self._profitability_cache[channel_id]
    
    def record_channel_open_cost(self, channel_id: str, peer_id: str,
                                  open_cost_sats: int, capacity_sats: int):
        """
        Record the cost to open a channel.
        
        Args:
            channel_id: New channel ID
            peer_id: Peer node ID
            open_cost_sats: On-chain fees paid
            capacity_sats: Channel capacity
        """
        self.database.record_channel_open_cost(
            channel_id=channel_id,
            peer_id=peer_id,
            open_cost_sats=open_cost_sats,
            capacity_sats=capacity_sats,
            timestamp=int(time.time())
        )
    
    def get_zombie_channels(self, validate_exists: bool = True) -> List[ChannelProfitability]:
        """
        Get list of zombie channels that should be considered for closure.

        Issue #29: Now validates that channels still exist before reporting them
        as zombies. This prevents false positives from closed channels lingering
        in the profitability cache.

        Args:
            validate_exists: If True (default), cross-reference with active
                           channel list to filter out closed channels.

        Returns:
            List of ChannelProfitability for zombie channels
        """
        if (int(time.time()) - self._cache_timestamp) > self._cache_ttl:
            self.analyze_all_channels()

        zombies = [
            p for p in self._profitability_cache.values()
            if p.classification == ProfitabilityClass.ZOMBIE
        ]

        if validate_exists:
            # Get current active channel IDs
            active_channel_ids = set(self._get_all_channels().keys())
            # Filter to only include channels that still exist
            zombies = [z for z in zombies if z.channel_id in active_channel_ids]

        return zombies

    def prune_closed_channels(self) -> int:
        """
        Remove closed channels from profitability cache (Issue #29).

        This prevents stale data from closed channels from accumulating
        in the cache and causing false positive zombie reports.

        Returns:
            Number of entries removed from cache
        """
        # Get current active channel IDs
        active_channel_ids = set(self._get_all_channels().keys())

        # Find closed channels in cache
        cached_ids = set(self._profitability_cache.keys())
        closed_ids = cached_ids - active_channel_ids

        # Remove closed channels from cache
        for channel_id in closed_ids:
            del self._profitability_cache[channel_id]

        if closed_ids:
            self.plugin.log(
                f"Pruned {len(closed_ids)} closed channels from profitability cache: "
                f"{list(closed_ids)[:5]}{'...' if len(closed_ids) > 5 else ''}",
                level='debug'
            )

        return len(closed_ids)
    
    def get_profitable_channels(self) -> List[ChannelProfitability]:
        """
        Get list of profitable channels (for prioritization).
        
        Returns:
            List of ChannelProfitability for profitable channels, sorted by ROI
        """
        if (int(time.time()) - self._cache_timestamp) > self._cache_ttl:
            self.analyze_all_channels()
        
        profitable = [
            p for p in self._profitability_cache.values()
            if p.classification == ProfitabilityClass.PROFITABLE
        ]
        
        return sorted(profitable, key=lambda p: p.roi_percent, reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all channels.

        Returns:
            Summary dict with totals and breakdowns including flow role distribution
        """
        if (int(time.time()) - self._cache_timestamp) > self._cache_ttl:
            self.analyze_all_channels()

        total_costs = 0
        total_revenue = 0
        total_volume = 0
        total_sourced_volume = 0
        total_sourced_contribution = 0
        classifications = {}
        role_distribution = {}

        for p in self._profitability_cache.values():
            total_costs += p.costs.total_cost_sats
            total_revenue += p.revenue.fees_earned_sats
            total_volume += p.revenue.volume_routed_sats
            total_sourced_volume += p.revenue.sourced_volume_sats
            total_sourced_contribution += p.revenue.sourced_fee_contribution_sats

            cls = p.classification.value
            classifications[cls] = classifications.get(cls, 0) + 1

            # Track role distribution (Issue #21)
            role = p.channel_role.value
            role_distribution[role] = role_distribution.get(role, 0) + 1

        net_profit = total_revenue - total_costs
        overall_roi = (net_profit / total_costs * 100) if total_costs > 0 else 0

        return {
            "total_channels": len(self._profitability_cache),
            "total_cost_sats": total_costs,
            "total_revenue_sats": total_revenue,
            "net_profit_sats": net_profit,
            "overall_roi_percent": round(overall_roi, 2),
            "total_volume_routed_sats": total_volume,
            "classifications": classifications,
            "zombie_channels": len(self.get_zombie_channels()),
            "cache_age_seconds": int(time.time()) - self._cache_timestamp,
            # Issue #21: Flow direction metrics
            "total_sourced_volume_sats": total_sourced_volume,
            "total_sourced_contribution_sats": total_sourced_contribution,
            "role_distribution": role_distribution
        }

    def _report_health_to_hive(self) -> bool:
        """
        Report our health status to cl-hive for NNLB coordination.

        This shares INFORMATION only - no sats move between nodes.
        The health data is used by cl-hive to calculate our NNLB health tier,
        which affects how aggressively we rebalance our own channels.

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.hive_bridge:
            return False

        # Rate limit health reports
        now = int(time.time())
        if (now - self._last_health_report) < self._health_report_interval:
            return False

        # Get current summary
        summary = self.get_summary()
        classifications = summary.get("classifications", {})
        total_channels = summary.get("total_channels", 0)

        # Extract classification counts
        profitable = classifications.get("profitable", 0)
        underwater = classifications.get("underwater", 0)
        stagnant = classifications.get("stagnant_candidate", 0)
        zombie = classifications.get("zombie", 0)

        # Determine revenue trend from ROI
        roi = summary.get("overall_roi_percent", 0)
        if roi > 5:
            revenue_trend = "improving"
        elif roi < -5:
            revenue_trend = "declining"
        else:
            revenue_trend = "stable"

        # Calculate liquidity score from channel balance distribution
        liquidity_score = self._calculate_liquidity_score()

        # Report to hive
        success = self.hive_bridge.report_health_update(
            profitable_channels=profitable,
            underwater_channels=underwater + zombie,  # Combine underwater + zombie
            stagnant_channels=stagnant,
            total_channels=total_channels,
            revenue_trend=revenue_trend,
            liquidity_score=liquidity_score
        )

        if success:
            self._last_health_report = now
            self.plugin.log(
                f"NNLB: Reported health to hive - profitable={profitable}, "
                f"underwater={underwater + zombie}, stagnant={stagnant}, "
                f"trend={revenue_trend}",
                level='debug'
            )

        return success

    def _report_liquidity_state_to_hive(self) -> bool:
        """
        Report our liquidity state to cl-hive for fleet coordination.

        This shares INFORMATION only - no sats move between nodes.
        The liquidity state helps other hive members make coordinated
        decisions about fees and rebalancing.

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.hive_bridge:
            return False

        # Rate limit liquidity reports (use same interval as health)
        now = int(time.time())
        if (now - self._last_health_report) < self._health_report_interval:
            return True  # Report health and liquidity together

        depleted_channels = []
        saturated_channels = []

        for channel_id, prof in self._profitability_cache.items():
            # Get channel state for balance info
            state = self.database.get_channel_state(channel_id)
            if not state:
                continue

            local = state.get("local_balance_sats", 0)
            capacity = state.get("capacity_sats", 0)
            peer_id = state.get("peer_id", "")

            if capacity <= 0 or not peer_id:
                continue

            local_pct = local / capacity

            if local_pct < 0.20:
                # Depleted channel - we need outbound
                depleted_channels.append({
                    "peer_id": peer_id,
                    "local_pct": round(local_pct, 3),
                    "capacity_sats": capacity
                })
            elif local_pct > 0.80:
                # Saturated channel - we need inbound
                saturated_channels.append({
                    "peer_id": peer_id,
                    "local_pct": round(local_pct, 3),
                    "capacity_sats": capacity
                })

        # Report to hive
        success = self.hive_bridge.report_liquidity_state(
            depleted_channels=depleted_channels,
            saturated_channels=saturated_channels,
            rebalancing_active=False,  # Will be updated by rebalancer when active
            rebalancing_peers=[]
        )

        if success:
            self.plugin.log(
                f"LIQUIDITY: Reported to hive - depleted={len(depleted_channels)}, "
                f"saturated={len(saturated_channels)}",
                level='debug'
            )

        return success

    def _calculate_liquidity_score(self) -> int:
        """
        Calculate liquidity balance score from channel data.

        A well-balanced node has channels near 50% local balance.
        Depleted (<20%) or saturated (>80%) channels hurt the score.

        Returns:
            Liquidity score (0-100, higher is better)
        """
        if not self._profitability_cache:
            return 50  # Default to neutral

        total_penalty = 0
        count = 0

        for channel_id, prof in self._profitability_cache.items():
            # Get channel state for balance info
            state = self.database.get_channel_state(channel_id)
            if not state:
                continue

            local = state.get("local_balance_sats", 0)
            capacity = state.get("capacity_sats", 0)
            if capacity <= 0:
                continue

            local_pct = local / capacity

            # Calculate distance from ideal (50%)
            distance = abs(local_pct - 0.5)

            # Penalty increases with distance from 50%
            # 0% or 100% local = 50 penalty points (worst)
            # 50% local = 0 penalty points (ideal)
            penalty = distance * 100
            total_penalty += penalty
            count += 1

        if count == 0:
            return 50

        # Average penalty across channels
        avg_penalty = total_penalty / count

        # Convert to score (0-100, higher is better)
        score = int(max(0, min(100, 100 - avg_penalty)))
        return score

    def get_channels_by_role(self, role: ChannelRole) -> List[ChannelProfitability]:
        """
        Get all channels with a specific flow role.

        Args:
            role: ChannelRole to filter by

        Returns:
            List of ChannelProfitability objects with the specified role
        """
        if (int(time.time()) - self._cache_timestamp) > self._cache_ttl:
            self.analyze_all_channels()

        return [
            p for p in self._profitability_cache.values()
            if p.channel_role == role
        ]

    def get_inbound_gateways(self) -> List[ChannelProfitability]:
        """
        Get channels that primarily source inbound volume.

        These channels are valuable because they bring payments into the node,
        enabling routing revenue on outbound channels.

        Returns:
            List sorted by sourced fee contribution (highest first)
        """
        gateways = self.get_channels_by_role(ChannelRole.INBOUND_GATEWAY)
        return sorted(
            gateways,
            key=lambda p: p.revenue.sourced_fee_contribution_sats,
            reverse=True
        )

    def get_outbound_gateways(self) -> List[ChannelProfitability]:
        """
        Get channels that primarily exit payments to the network.

        These channels earn direct routing fees.

        Returns:
            List sorted by fees earned (highest first)
        """
        gateways = self.get_channels_by_role(ChannelRole.OUTBOUND_GATEWAY)
        return sorted(
            gateways,
            key=lambda p: p.revenue.fees_earned_sats,
            reverse=True
        )
    
    def get_lifetime_report(self) -> Dict[str, Any]:
        """
        Get lifetime financial history report including closed channels.

        Unlike get_summary() which only considers active channels,
        this method queries the database directly for ALL historical
        data to provide a true "Lifetime P&L" view.

        Returns:
            Dictionary with lifetime financial metrics:
                - lifetime_revenue_sats: Total routing fees earned
                - lifetime_opening_costs_sats: Total channel opening fees
                - lifetime_closure_costs_sats: Total channel closure fees (Accounting v2.0)
                - lifetime_splice_costs_sats: Total splice fees (Accounting v2.0)
                - lifetime_rebalance_costs_sats: Total rebalancing fees paid
                - lifetime_total_costs_sats: Opening + Closure + Splice + Rebalance costs
                - lifetime_net_profit_sats: Revenue - Total Costs
                - lifetime_roi_percent: ROI percentage
                - lifetime_forward_count: Total number of forwards
                - closed_channels_summary: Summary of closed channel P&L
        """
        # Get aggregate stats from database (includes closed channels)
        stats = self.database.get_lifetime_stats()

        # Convert revenue from msat to sats
        lifetime_revenue_sats = stats["total_revenue_msat"] // 1000

        # Get costs (including closure and splice costs - Accounting v2.0)
        lifetime_opening_costs_sats = stats["total_opening_cost_sats"]
        lifetime_closure_costs_sats = stats.get("total_closure_cost_sats", 0)
        lifetime_splice_costs_sats = stats.get("total_splice_cost_sats", 0)
        lifetime_rebalance_costs_sats = stats["total_rebalance_cost_sats"]

        # Calculate totals (now includes closure and splice costs)
        lifetime_total_costs_sats = (
            lifetime_opening_costs_sats +
            lifetime_closure_costs_sats +
            lifetime_splice_costs_sats +
            lifetime_rebalance_costs_sats
        )
        lifetime_net_profit_sats = lifetime_revenue_sats - lifetime_total_costs_sats

        # Calculate ROI (avoid division by zero)
        if lifetime_total_costs_sats > 0:
            lifetime_roi_percent = round(
                (lifetime_net_profit_sats / lifetime_total_costs_sats) * 100, 2
            )
        else:
            # No costs incurred - infinite ROI if any revenue, 0 otherwise
            lifetime_roi_percent = 100.0 if lifetime_revenue_sats > 0 else 0.0

        # Get closed channels summary (Accounting v2.0)
        closed_summary = self.database.get_closed_channels_summary()

        return {
            "lifetime_revenue_sats": lifetime_revenue_sats,
            "lifetime_opening_costs_sats": lifetime_opening_costs_sats,
            "lifetime_closure_costs_sats": lifetime_closure_costs_sats,
            "lifetime_splice_costs_sats": lifetime_splice_costs_sats,
            "lifetime_rebalance_costs_sats": lifetime_rebalance_costs_sats,
            "lifetime_total_costs_sats": lifetime_total_costs_sats,
            "lifetime_net_profit_sats": lifetime_net_profit_sats,
            "lifetime_roi_percent": lifetime_roi_percent,
            "lifetime_forward_count": stats["total_forwards"],
            "closed_channels_summary": closed_summary
        }

    # =========================================================================
    # Phase 8: P&L Dashboard Methods
    # =========================================================================

    def get_pnl_summary(self, window_days: int = 30) -> Dict[str, Any]:
        """
        Get P&L summary for a given time window.

        Calculates key financial metrics for the Sovereign Dashboard:
        - Gross Revenue: Total routing fees earned
        - Operating Expense (OpEx): Total costs (rebalance + closure + splice)
        - Net Profit: Revenue - OpEx
        - Operating Margin: (Net Profit / Gross Revenue) * 100

        Args:
            window_days: Time window for calculations (default 30 days, minimum 1)

        Returns:
            Dict with revenue, opex breakdown, net_profit, margin
        """
        # BUG FIX: Validate window_days to prevent empty/confusing results
        if window_days < 1:
            window_days = 1

        since_timestamp = int(time.time()) - (window_days * 86400)

        # Get revenue (routing fees earned)
        gross_revenue_sats = self.database.get_total_routing_revenue(since_timestamp)

        # Get OpEx components (Accounting v2.0: includes closure and splice costs)
        rebalance_cost_sats = self.database.get_total_rebalance_fees(since_timestamp)
        closure_cost_sats = self.database.get_closure_costs_since(since_timestamp)
        splice_cost_sats = self.database.get_splice_costs_since(since_timestamp)

        # Total OpEx
        opex_sats = rebalance_cost_sats + closure_cost_sats + splice_cost_sats

        # Calculate net profit
        net_profit_sats = gross_revenue_sats - opex_sats

        # Calculate operating margin (avoid division by zero)
        if gross_revenue_sats > 0:
            operating_margin_pct = round((net_profit_sats / gross_revenue_sats) * 100, 2)
        else:
            # No revenue - margin is undefined, use 0 if no costs, -100 if costs
            operating_margin_pct = 0.0 if opex_sats == 0 else -100.0

        return {
            'window_days': window_days,
            'gross_revenue_sats': gross_revenue_sats,
            'opex_sats': opex_sats,
            'rebalance_cost_sats': rebalance_cost_sats,
            'closure_cost_sats': closure_cost_sats,
            'splice_cost_sats': splice_cost_sats,
            'net_profit_sats': net_profit_sats,
            'operating_margin_pct': operating_margin_pct
        }

    def identify_bleeders(self, window_days: int = 30) -> List[Dict[str, Any]]:
        """
        Identify "Bleeder" channels that are losing money.

        A Bleeder is a channel where:
        - Net P&L < 0 (rebalance costs exceed total contribution value)
        - Has activity (either as exit or entry channel)

        Total contribution includes:
        - Direct fees (earned as exit channel)
        - Sourced fee contribution (fees enabled by sourcing inbound volume)

        This prevents misclassifying channels that source valuable inbound
        volume but don't earn direct exit fees.

        Args:
            window_days: Time window for analysis (default 30 days, minimum 1)

        Returns:
            List of bleeder channel dicts with P&L breakdown, sorted by loss
        """
        # BUG FIX: Validate window_days to prevent empty/confusing results
        if window_days < 1:
            window_days = 1

        bleeders = []

        try:
            # Get all active channels
            channels = self._get_all_channels()

            for channel_id, info in channels.items():
                # Get FULL P&L including inbound contribution
                pnl = self.database.get_channel_full_pnl(channel_id, window_days)

                # Total activity = exit forwards + sourced forwards
                total_activity = pnl['direct_forward_count'] + pnl['sourced_forward_count']

                # Check for bleeder condition: net < 0 AND has activity
                # Now uses total_contribution which includes sourced fee value
                if pnl['net_pnl_sats'] < 0 and total_activity > 0:
                    bleeders.append({
                        'channel_id': channel_id,
                        'peer_id': info.get('peer_id', ''),
                        'capacity_sats': info.get('capacity', 0),
                        # Direct revenue (as exit channel)
                        'direct_revenue_sats': pnl['direct_revenue_sats'],
                        # Inbound contribution (fees enabled by sourcing volume)
                        'sourced_fee_contribution_sats': pnl['sourced_fee_contribution_sats'],
                        'sourced_volume_sats': pnl['sourced_volume_sats'],
                        # Combined metrics
                        'total_contribution_sats': pnl['total_contribution_sats'],
                        'rebalance_cost_sats': pnl['rebalance_cost_sats'],
                        'net_pnl_sats': pnl['net_pnl_sats'],
                        'direct_forward_count': pnl['direct_forward_count'],
                        'sourced_forward_count': pnl['sourced_forward_count'],
                        'total_forward_count': total_activity,
                        'loss_per_forward': abs(pnl['net_pnl_sats']) // max(total_activity, 1),
                        # Legacy fields for backward compatibility
                        'revenue_sats': pnl['direct_revenue_sats'],
                        'forward_count': pnl['direct_forward_count']
                    })

            # Sort by loss (most negative first)
            bleeders.sort(key=lambda x: x['net_pnl_sats'])

        except Exception as e:
            self.plugin.log(f"Error identifying bleeders: {e}", level='error')

        return bleeders

    def calculate_roc(self, window_days: int = 30) -> Dict[str, Any]:
        """
        Calculate Return on Capacity (ROC).

        ROC measures the yield on deployed capital, normalized to annual percentage:
        ROC = (Net_Profit_window / Total_Capacity) * (365 / window_days)

        This tells operators their annualized return on the BTC locked in channels.

        Args:
            window_days: Time window for net profit calculation (default 30 days, minimum 1)

        Returns:
            Dict with total_capacity, net_profit, roc_pct, and annualized_roc_pct
        """
        # BUG FIX: Validate window_days to prevent division by zero in annualization
        if window_days < 1:
            window_days = 1

        # Get P&L for the window
        pnl = self.get_pnl_summary(window_days)

        # Get total channel capacity
        total_capacity_sats = 0
        try:
            channels = self._get_all_channels()
            for info in channels.values():
                total_capacity_sats += info.get('capacity', 0)
        except Exception as e:
            self.plugin.log(f"Error getting capacity for ROC: {e}", level='error')

        # Calculate ROC (avoid division by zero)
        if total_capacity_sats > 0:
            # ROC for the window period
            roc_pct = (pnl['net_profit_sats'] / total_capacity_sats) * 100

            # Annualized ROC
            annualized_roc_pct = roc_pct * (365 / window_days)
        else:
            roc_pct = 0.0
            annualized_roc_pct = 0.0

        return {
            'window_days': window_days,
            'total_capacity_sats': total_capacity_sats,
            'net_profit_sats': pnl['net_profit_sats'],
            'roc_pct': round(roc_pct, 4),
            'annualized_roc_pct': round(annualized_roc_pct, 2)
        }

    def get_tlv(self) -> Dict[str, int]:
        """
        Calculate Total Liquidating Value (TLV).

        TLV represents the node's "Net Worth" if all channels were
        cooperatively closed today:
        TLV = On-chain Balance + Sum(Channel Local Balances)

        Returns:
            Dict with onchain_sats, local_balance_sats, tlv_sats
        """
        onchain_sats = 0
        local_balance_sats = 0
        remote_balance_sats = 0
        channel_count = 0

        try:
            # Get on-chain balance from listfunds
            listfunds = self.plugin.rpc.listfunds()

            for output in listfunds.get("outputs", []):
                if output.get("status") == "confirmed":
                    amount_msat = self._parse_msat(output.get("amount_msat", 0))
                    onchain_sats += amount_msat // 1000

            # Get channel balances
            for channel in listfunds.get("channels", []):
                if channel.get("state") != "CHANNELD_NORMAL":
                    continue

                our_amount_msat = self._parse_msat(channel.get("our_amount_msat", 0))
                amount_msat = self._parse_msat(channel.get("amount_msat", 0))

                local_balance_sats += our_amount_msat // 1000
                remote_balance_sats += (amount_msat - our_amount_msat) // 1000
                channel_count += 1

        except Exception as e:
            self.plugin.log(f"Error calculating TLV: {e}", level='error')

        return {
            'onchain_sats': onchain_sats,
            'local_balance_sats': local_balance_sats,
            'remote_balance_sats': remote_balance_sats,
            'tlv_sats': onchain_sats + local_balance_sats,
            'channel_count': channel_count
        }

    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _get_all_channels(self) -> Dict[str, Dict[str, Any]]:
        """Get all channels with their info."""
        channels = {}
        
        try:
            result = self.plugin.rpc.listpeerchannels()
            
            for channel in result.get("channels", []):
                state = channel.get("state", "")
                if state != "CHANNELD_NORMAL":
                    continue
                
                channel_id = channel.get("short_channel_id")
                if not channel_id:
                    continue
                
                # Calculate capacity
                capacity_msat = self._parse_msat(channel.get("total_msat", 0))
                capacity = capacity_msat // 1000  # Convert to sats
                
                # If capacity is 0, try spendable + receivable
                if capacity == 0:
                    spendable_msat = self._parse_msat(channel.get("spendable_msat", 0))
                    receivable_msat = self._parse_msat(channel.get("receivable_msat", 0))
                    capacity = (spendable_msat + receivable_msat) // 1000
                
                funding_txid = channel.get("funding_txid", "")
                
                # Get open timestamp from bookkeeper or estimate from SCID
                open_timestamp = self._get_channel_open_timestamp(
                    channel_id, funding_txid
                )
                
                channels[channel_id] = {
                    "peer_id": channel.get("peer_id", ""),
                    "capacity": capacity,
                    "funding_txid": funding_txid,
                    "open_timestamp": open_timestamp,
                    "opener": channel.get("opener", "local")  # Extract opener (local/remote)
                }
                
        except Exception as e:
            self.plugin.log(f"Error getting channels: {e}", level='error')
        
        return channels
    
    def _get_channel_open_timestamp(self, channel_id: str, funding_txid: str) -> int:
        """
        Get the timestamp when a channel was opened.
        
        Methods (in priority order):
        1. Bookkeeper channel_open event - has exact timestamp
        2. Estimate from SCID block height - approximate but reliable
        3. Fallback to 30 days ago
        
        Args:
            channel_id: Short channel ID (e.g., "902205x123x0")
            funding_txid: Funding transaction ID
            
        Returns:
            Unix timestamp of channel open
        """
        # Method 1: Query bookkeeper for channel_open event
        if funding_txid:
            bkpr_timestamp = self._get_open_timestamp_from_bookkeeper(funding_txid)
            if bkpr_timestamp:
                return bkpr_timestamp
        
        # Method 2: Estimate from SCID block height
        # SCID format is "blockheight x txindex x output"
        if channel_id and 'x' in channel_id:
            try:
                block_height = int(channel_id.split('x')[0])
                # Estimate: ~10 minutes per block, blocks since genesis
                # Bitcoin mainnet started ~Jan 3, 2009
                # Block 0 = 1231006505
                genesis_timestamp = 1231006505
                seconds_per_block = 600  # 10 minutes average
                estimated_timestamp = genesis_timestamp + (block_height * seconds_per_block)
                
                # Sanity check - should be in the past
                now = int(time.time())
                if estimated_timestamp < now:
                    return estimated_timestamp
                    
            except (ValueError, IndexError):
                pass
        
        # Method 3: Fallback to 30 days ago
        return int(time.time()) - (86400 * 30)
    
    def _get_open_timestamp_from_bookkeeper(self, funding_txid: str) -> Optional[int]:
        """
        Get channel open timestamp from bookkeeper.
        
        Bookkeeper records channel_open events with the exact timestamp.
        
        Args:
            funding_txid: The funding transaction ID
            
        Returns:
            Unix timestamp, or None if not found
        """
        try:
            # Bookkeeper account names use reversed txid bytes
            reversed_txid = self._reverse_txid(funding_txid)
            
            # Query bookkeeper for this account's events
            result = self.plugin.rpc.call(
                "bkpr-listaccountevents",
                {"account": reversed_txid}
            )
            
            events = result.get("events", [])
            
            # Look for channel_open event
            for event in events:
                if (event.get("type") == "chain" and 
                    event.get("tag") == "channel_open"):
                    timestamp = event.get("timestamp")
                    if timestamp:
                        return int(timestamp)
                        
        except Exception as e:
            self.plugin.log(
                f"Error getting open timestamp from bookkeeper: {e}",
                level='debug'
            )
        
        return None
    
    def _get_channel_costs(self, channel_id: str, peer_id: str, 
                          funding_txid: str, capacity_sats: int = 0,
                          opener: str = "local") -> ChannelCosts:
        """
        Get costs for a channel from bookkeeper and database.
        
        Cost sources (in priority order):
        1. If opener == 'remote': Cost is 0 (we didn't pay to open it).
        2. Bookkeeper onchain_fee events for the funding tx (most accurate).
        3. Database cached value.
        4. Config estimated_open_cost_sats (fallback).
        
        Args:
            channel_id: Short channel ID
            peer_id: Peer node ID
            funding_txid: Funding transaction ID
            capacity_sats: Channel capacity
            opener: Who opened the channel ('local' or 'remote')
        """
        # Get rebalance costs - combine database records with bookkeeper data
        db_rebalance_costs = self.database.get_channel_rebalance_costs(channel_id)
        bkpr_rebalance_costs = self._get_rebalance_costs_from_bookkeeper(channel_id, funding_txid)
        
        # Use the higher value (bookkeeper may have more complete history)
        rebalance_costs = max(db_rebalance_costs, bkpr_rebalance_costs)
        
        # Determine open cost
        open_cost = None
        db_open_cost = self.database.get_channel_open_cost(channel_id)

        if opener == 'remote':
            # Remote opener pays the fees -> Cost to us is 0
            open_cost = 0
            
            # SELF-HEALING: If we previously recorded a cost for a remote channel (e.g. fallback), fix it.
            if db_open_cost is not None and db_open_cost > 0:
                self.plugin.log(
                    f"Self-healing: Fixed open cost for remote channel {channel_id} "
                    f"(was {db_open_cost}, now 0)",
                    level='info'
                )
                self.database.record_channel_open_cost(
                    channel_id, peer_id, 0, capacity_sats
                )
        else:
            # Local opener -> We paid fees. Proceed with lookup logic.
            
            # Use cached value if available
            open_cost = db_open_cost
            
            # RETROACTIVE FIX: Re-query channels stored with fallback value
            if db_open_cost is not None and db_open_cost == self.config.estimated_open_cost_sats:
                if funding_txid:
                    self.plugin.log(
                        f"Stored cost for {channel_id} is fallback value "
                        f"({db_open_cost} sats). Attempting re-query with summation logic...",
                        level='debug'
                    )
                    requeried_cost = self._get_open_cost_from_bookkeeper(funding_txid, capacity_sats)
                    
                    if requeried_cost is not None:
                        self.plugin.log(
                            f"Retroactive fix for {channel_id}: updated open_cost from "
                            f"{db_open_cost} sats (fallback) to {requeried_cost} sats (actual)",
                            level='info'
                        )
                        self.database.record_channel_open_cost(
                            channel_id, peer_id, requeried_cost, capacity_sats
                        )
                        open_cost = requeried_cost
            
            # SANITY CHECK: Detect invalid open_cost (capital mistaken as expense)
            if open_cost is not None and capacity_sats > 0:
                open_cost = self._sanity_check_open_cost(
                    channel_id, peer_id, funding_txid, open_cost, capacity_sats
                )
            
            # Query bookkeeper if not found
            if open_cost is None and funding_txid:
                open_cost = self._get_open_cost_from_bookkeeper(funding_txid, capacity_sats)
                if open_cost is not None:
                    self.database.record_channel_open_cost(
                        channel_id, peer_id, open_cost, capacity_sats
                    )
            
            # Final fallback
            if open_cost is None:
                open_cost = self.config.estimated_open_cost_sats
                self.plugin.log(
                    f"Using estimated open cost ({open_cost} sats) for {channel_id} - "
                    f"bookkeeper data not available",
                    level='debug'
                )
        
        return ChannelCosts(
            channel_id=channel_id,
            peer_id=peer_id,
            open_cost_sats=open_cost,
            rebalance_cost_sats=rebalance_costs
        )
    
    def _sanity_check_open_cost(self, channel_id: str, peer_id: str,
                                 funding_txid: str, open_cost: int,
                                 capacity_sats: int) -> int:
        """
        Sanity check and self-heal invalid open_cost values.
        
        This detects cases where:
        1. The channel funding amount (principal capital) was incorrectly recorded
           as an expense (open cost), causing healthy channels to appear as losses.
        2. A batch transaction fee was attributed to a single channel, resulting
           in abnormally high "open cost" values.
        
        Uses _is_valid_fee_amount for consistent validation across all code paths.
        
        Args:
            channel_id: Channel short ID
            peer_id: Peer node ID
            funding_txid: Funding transaction ID
            open_cost: The open cost value to validate
            capacity_sats: Channel capacity in sats
            
        Returns:
            Corrected open_cost value (either validated original, re-queried, or fallback)
        """
        # Use the centralized validation helper for consistent logic
        if self._is_valid_fee_amount(open_cost, capacity_sats, funding_txid):
            return open_cost  # Value looks reasonable
        
        # DETECTED INVALID OPEN COST - trigger self-healing
        self.plugin.log(
            f"Sanity check for {channel_id}: open_cost {open_cost} sats failed validation "
            f"(capacity: {capacity_sats} sats). Triggering recalculation.",
            level='debug'
        )
        
        # Step 1: Force re-query from bookkeeper
        corrected_cost = None
        if funding_txid:
            corrected_cost = self._get_open_cost_from_bookkeeper(funding_txid, capacity_sats)
        
        # Step 2: Validate the re-queried value using centralized helper
        if corrected_cost is not None and not self._is_valid_fee_amount(corrected_cost, capacity_sats, funding_txid):
            self.plugin.log(
                f"Re-query for {channel_id} still returned invalid value "
                f"({corrected_cost} sats). Using fallback.",
                level='debug'
            )
            corrected_cost = None
        
        # Step 3: Use fallback if re-query failed or still invalid
        if corrected_cost is None:
            corrected_cost = self.config.estimated_open_cost_sats
            self.plugin.log(
                f"Fallback triggered for {channel_id}: using estimated_open_cost_sats "
                f"({corrected_cost} sats)",
                level='info'
            )
        else:
            self.plugin.log(
                f"Corrected open_cost for {channel_id}: {corrected_cost} sats "
                f"(was: {open_cost} sats)",
                level='debug'
            )
        
        # Step 4: Update database with corrected value (self-healing)
        self.database.record_channel_open_cost(
            channel_id, peer_id, corrected_cost, capacity_sats
        )
        self.plugin.log(
            f"Database updated with corrected open_cost for {channel_id}",
            level='debug'
        )
        
        return corrected_cost
    
    def _get_open_cost_from_bookkeeper(self, funding_txid: str, 
                                        capacity_sats: int = 0) -> Optional[int]:
        """
        Query bookkeeper for actual on-chain fee paid for channel open.
        
        Bookkeeper tracks onchain_fee events per txid as a SERIES of adjustments.
        The total fee is the Sum of Credits minus Sum of Debits for a given TXID.
        
        This is critical for batch transactions where bookkeeper may:
        1. Credit the total batch fee to one account
        2. Issue debits to redistribute the fee across all accounts in the batch
        
        For example, a batch open might show:
        - credit_msat: 833,443,000 (initial attribution)
        - debit_msat: 416,721,000 (redistribution to another channel)
        - debit_msat: 416,656,000 (redistribution to another channel)
        - Net: 66,000 msat = 66 sats (actual fee for this channel)
        
        Args:
            funding_txid: The funding transaction ID
            capacity_sats: Channel capacity in sats (for validation)
            
        Returns:
            On-chain fee in sats, or None if not found or invalid
        """
        try:
            # Bookkeeper account names use reversed txid bytes
            # e.g., txid 9e14b256... becomes account 940fec8a...
            reversed_txid = self._reverse_txid(funding_txid)
            
            # Query bookkeeper for this account's events
            result = self.plugin.rpc.call(
                "bkpr-listaccountevents",
                {"account": reversed_txid}
            )
            
            events = result.get("events", [])
            
            # Sum ALL onchain_fee events for this txid (credits - debits)
            total_credit_msat = 0
            total_debit_msat = 0
            found_events = False
            
            for event in events:
                if (event.get("type") == "onchain_fee" and 
                    event.get("txid") == funding_txid):
                    found_events = True
                    total_credit_msat += event.get("credit_msat", 0)
                    total_debit_msat += event.get("debit_msat", 0)
            
            if found_events:
                # Net fee = credits - debits
                net_fee_msat = total_credit_msat - total_debit_msat
                fee_sats = net_fee_msat // 1000
                
                self.plugin.log(
                    f"Bookkeeper fee calculation for {funding_txid}: "
                    f"credits={total_credit_msat}msat, debits={total_debit_msat}msat, "
                    f"net={fee_sats}sats",
                    level='debug'
                )
                
                # Sanity check: fee should be positive and reasonable
                if fee_sats < 0:
                    self.plugin.log(
                        f"Negative fee calculated ({fee_sats} sats) for {funding_txid} - "
                        f"returning None",
                        level='warn'
                    )
                    return None
                
                # Final validation: ensure it's not the funding principal
                if not self._is_valid_fee_amount(fee_sats, capacity_sats, funding_txid):
                    self.plugin.log(
                        f"Net fee {fee_sats} sats failed validation for {funding_txid}",
                        level='debug'
                    )
                    return None
                
                return fee_sats
            
            # Alternative: check wallet account for the same txid
            # This catches cases where we opened the channel
            wallet_result = self.plugin.rpc.call(
                "bkpr-listaccountevents",
                {"account": "wallet"}
            )
            
            wallet_events = wallet_result.get("events", [])
            wallet_credit_msat = 0
            wallet_debit_msat = 0
            wallet_found = False
            
            for event in wallet_events:
                if (event.get("type") == "onchain_fee" and
                    event.get("txid") == funding_txid):
                    wallet_found = True
                    wallet_credit_msat += event.get("credit_msat", 0)
                    wallet_debit_msat += event.get("debit_msat", 0)
            
            if wallet_found:
                # For wallet, the fee we paid is typically debits - credits
                # (opposite of channel account perspective)
                net_fee_msat = wallet_debit_msat - wallet_credit_msat
                fee_sats = net_fee_msat // 1000
                
                self.plugin.log(
                    f"Wallet fee calculation for {funding_txid}: "
                    f"debits={wallet_debit_msat}msat, credits={wallet_credit_msat}msat, "
                    f"net={fee_sats}sats",
                    level='debug'
                )
                
                # Sanity check: fee should be positive and reasonable
                if fee_sats < 0:
                    self.plugin.log(
                        f"Negative wallet fee calculated ({fee_sats} sats) for "
                        f"{funding_txid} - returning None",
                        level='warn'
                    )
                    return None
                
                # Final validation: ensure it's not the funding principal
                if not self._is_valid_fee_amount(fee_sats, capacity_sats, funding_txid):
                    self.plugin.log(
                        f"Wallet net fee {fee_sats} sats failed validation for {funding_txid}",
                        level='debug'
                    )
                    return None
                
                return fee_sats
                        
        except Exception as e:
            self.plugin.log(
                f"Error querying bookkeeper for {funding_txid}: {e}",
                level='debug'
            )
        
        return None
    
    def _is_valid_fee_amount(self, fee_sats: int, capacity_sats: int, 
                             funding_txid: str) -> bool:
        """
        Validate that a fee amount is actually a mining fee, not an invalid value.
        
        This is a final failsafe to catch cases where the funding principal
        (channel capacity) is incorrectly returned instead of the actual
        on-chain mining fee.
        
        NOTE: The previous "Batch Fee Check" has been removed because the
        summation logic in _get_open_cost_from_bookkeeper now correctly nets
        credits and debits, naturally handling batch transactions without
        needing heuristic rejection.
        
        Validation rules (returns False if ANY match):
        - fee_sats > 50,000 (Hard Cap - no mining fee should ever be this high)
        - fee_sats >= 90% of capacity (Principal Check - funding amount)
        - fee_sats > capacity (clearly invalid - fee can't exceed capacity)
        
        Args:
            fee_sats: The fee amount to validate
            capacity_sats: Channel capacity in sats
            funding_txid: Transaction ID (for logging)
            
        Returns:
            True if fee appears valid, False if it looks like funding amount
        """
        # Hard Cap: Reject any fee above 50,000 sats
        # No legitimate channel opening fee should ever be this high.
        # This catches data artifacts like change outputs or batch fee totals.
        if fee_sats > 50000:
            self.plugin.log(
                f"Rejected absurd fee: {fee_sats} > 50,000 hard cap for {funding_txid}",
                level='debug'
            )
            return False
        
        # If we don't have capacity info, accept any reasonable value
        if capacity_sats <= 0:
            return True
        
        # Principal Check: Reject if fee >= 90% of capacity
        # This is clearly the funding amount, not a mining fee
        if fee_sats >= capacity_sats * 0.90:
            self.plugin.log(
                f"Rejected principal: {fee_sats} sats >= 90% of capacity "
                f"({capacity_sats} sats) for {funding_txid}",
                level='debug'
            )
            return False
        
        # Sanity check: fee should never exceed capacity
        if fee_sats > capacity_sats:
            self.plugin.log(
                f"Rejected invalid fee: {fee_sats} sats > capacity "
                f"({capacity_sats} sats) for {funding_txid}",
                level='debug'
            )
            return False
        
        return True
    
    def _get_rebalance_costs_from_bookkeeper(self, channel_id: str, funding_txid: Optional[str] = None) -> int:
        """
        Query bookkeeper for rebalance costs (self-payment rebalance fees).
        
        Rebalance self-payments show up in bookkeeper as:
        - 'invoice' events on the destination channel (we paid ourselves)
        - The fees_msat field shows what we paid in routing fees
        
        We look for invoice events where we paid to ourselves
        by checking if there's a matching credit on another of our channels.
        
        Args:
            channel_id: The channel to get rebalance costs for
            funding_txid: The funding transaction ID (optional, will look up if not provided)
            
        Returns:
            Total rebalance costs in sats
        """
        total_fees_sats = 0
        
        try:
            # Use provided funding_txid or look it up
            if not funding_txid:
                channels = self._get_all_channels()
                if channel_id not in channels:
                    return 0
                funding_txid = channels[channel_id].get("funding_txid", "")
            
            if not funding_txid:
                return 0
            
            reversed_txid = self._reverse_txid(funding_txid)
            
            # Query bookkeeper for this channel's events
            result = self.plugin.rpc.call(
                "bkpr-listaccountevents",
                {"account": reversed_txid}
            )
            
            events = result.get("events", [])
            
            # Look for invoice events with fees (payments we made)
            for event in events:
                if event.get("type") == "channel" and event.get("tag") == "invoice":
                    # Debit on invoice = we paid out (could be rebalance)
                    fees_msat = event.get("fees_msat", 0)
                    if fees_msat and fees_msat > 0:
                        # This is a fee we paid - likely a rebalance self-payment
                        total_fees_sats += fees_msat // 1000
            
            if total_fees_sats > 0:
                self.plugin.log(
                    f"Found {total_fees_sats} sats in rebalance costs from bookkeeper for {channel_id}",
                    level='debug'
                )
                        
        except Exception as e:
            self.plugin.log(
                f"Error getting rebalance costs from bookkeeper for {channel_id}: {e}",
                level='debug'
            )
        
        return total_fees_sats
    
    def _reverse_txid(self, txid: str) -> str:
        """
        Reverse a transaction ID (byte-swap).
        
        Bitcoin txids are displayed in reverse byte order.
        Bookkeeper uses the reversed form as account names.
        
        Args:
            txid: Transaction ID in standard display format
            
        Returns:
            Reversed txid (bytes swapped)
        """
        # Convert hex string to bytes, reverse, convert back
        try:
            txid_bytes = bytes.fromhex(txid)
            reversed_bytes = txid_bytes[::-1]
            return reversed_bytes.hex()
        except (ValueError, AttributeError):
            return txid
    
    def _get_all_revenue_data(self) -> Dict[str, ChannelRevenue]:
        """
        Batch fetch revenue data for all channels with a single RPC call.

        This method calls listforwards(status="settled") once and aggregates:
        - Exit metrics: fees_earned and volume_routed by out_channel
        - Entry metrics: sourced_volume and sourced_fee_contribution by in_channel

        This provides a complete picture of channel value, including channels
        that primarily source inbound volume rather than earn exit fees.

        Returns:
            Dict mapping channel_id to ChannelRevenue with both exit and entry metrics
        """
        revenue_map: Dict[str, Dict[str, int]] = {}

        def init_channel(channel_id: str) -> None:
            """Initialize channel entry with all metrics."""
            if channel_id not in revenue_map:
                revenue_map[channel_id] = {
                    "fees_earned": 0,
                    "volume_routed": 0,
                    "forward_count": 0,
                    "sourced_volume": 0,
                    "sourced_fee_contribution": 0,
                    "sourced_forward_count": 0
                }

        try:
            # Single RPC call to fetch ALL settled forwards
            result = self.plugin.rpc.listforwards(status="settled")

            for forward in result.get("forwards", []):
                out_channel = forward.get("out_channel")
                in_channel = forward.get("in_channel")
                fee_msat = self._parse_msat(forward.get("fee_msat", 0))
                out_msat = self._parse_msat(forward.get("out_msat", 0))
                in_msat = self._parse_msat(forward.get("in_msat", 0))

                # Track EXIT metrics (out_channel earns the fee)
                if out_channel:
                    init_channel(out_channel)
                    revenue_map[out_channel]["fees_earned"] += fee_msat // 1000
                    revenue_map[out_channel]["volume_routed"] += out_msat // 1000
                    revenue_map[out_channel]["forward_count"] += 1

                # Track ENTRY metrics (in_channel sourced the volume)
                # The fee was enabled by the in_channel providing the route
                if in_channel:
                    init_channel(in_channel)
                    revenue_map[in_channel]["sourced_volume"] += in_msat // 1000
                    revenue_map[in_channel]["sourced_fee_contribution"] += fee_msat // 1000
                    revenue_map[in_channel]["sourced_forward_count"] += 1

        except Exception as e:
            self.plugin.log(
                f"Error batch fetching revenue data: {e}",
                level='warn'
            )

        # Convert to ChannelRevenue objects
        result_map: Dict[str, ChannelRevenue] = {}
        for channel_id, data in revenue_map.items():
            result_map[channel_id] = ChannelRevenue(
                channel_id=channel_id,
                fees_earned_sats=data["fees_earned"],
                volume_routed_sats=data["volume_routed"],
                forward_count=data["forward_count"],
                sourced_volume_sats=data["sourced_volume"],
                sourced_fee_contribution_sats=data["sourced_fee_contribution"],
                sourced_forward_count=data["sourced_forward_count"]
            )

        return result_map
    
    def _get_channel_revenue(self, channel_id: str) -> ChannelRevenue:
        """
        Get revenue for a single channel from routing history.

        This fetches both:
        - Exit metrics: fees earned when channel is out_channel
        - Entry metrics: volume sourced when channel is in_channel

        Note: For batch operations, use _get_all_revenue_data() instead to
        avoid N+1 query overhead. This method is retained for single-channel
        lookups where batch fetching would be wasteful.
        """
        # Exit metrics (channel as out_channel)
        fees_earned = 0
        volume_routed = 0
        forward_count = 0
        # Entry metrics (channel as in_channel)
        sourced_volume = 0
        sourced_fee_contribution = 0
        sourced_forward_count = 0

        try:
            # Get forwards where we earned fees (out_channel = this channel)
            result = self.plugin.rpc.listforwards(
                out_channel=channel_id,
                status="settled"
            )

            for forward in result.get("forwards", []):
                fee_msat = self._parse_msat(forward.get("fee_msat", 0))
                fees_earned += fee_msat // 1000

                out_msat = self._parse_msat(forward.get("out_msat", 0))
                volume_routed += out_msat // 1000

                forward_count += 1

        except Exception as e:
            self.plugin.log(
                f"Error getting exit revenue for {channel_id}: {e}",
                level='warn'
            )

        try:
            # Get forwards where we sourced volume (in_channel = this channel)
            result = self.plugin.rpc.listforwards(
                in_channel=channel_id,
                status="settled"
            )

            for forward in result.get("forwards", []):
                in_msat = self._parse_msat(forward.get("in_msat", 0))
                sourced_volume += in_msat // 1000

                # The fee was enabled by this channel sourcing the forward
                fee_msat = self._parse_msat(forward.get("fee_msat", 0))
                sourced_fee_contribution += fee_msat // 1000

                sourced_forward_count += 1

        except Exception as e:
            self.plugin.log(
                f"Error getting inbound contribution for {channel_id}: {e}",
                level='warn'
            )

        return ChannelRevenue(
            channel_id=channel_id,
            fees_earned_sats=fees_earned,
            volume_routed_sats=volume_routed,
            forward_count=forward_count,
            sourced_volume_sats=sourced_volume,
            sourced_fee_contribution_sats=sourced_fee_contribution,
            sourced_forward_count=sourced_forward_count
        )
    
    def _get_last_routing_time(self, channel_id: str) -> Optional[int]:
        """
        Get timestamp of last routing activity on a channel.

        Checks BOTH directions:
        - As out_channel (exit): when channel forwarded outbound
        - As in_channel (entry): when channel sourced inbound volume

        A channel is "active" if it's routing in either direction.
        """
        latest_time: Optional[int] = None

        try:
            # Check outbound activity (as exit channel)
            result = self.plugin.rpc.listforwards(
                out_channel=channel_id,
                status="settled"
            )
            forwards = result.get("forwards", [])
            if forwards:
                out_latest = max(forwards, key=lambda f: f.get("received_time", 0))
                latest_time = int(out_latest.get("received_time", 0))
        except Exception:
            pass

        try:
            # Check inbound activity (as entry channel)
            result = self.plugin.rpc.listforwards(
                in_channel=channel_id,
                status="settled"
            )
            forwards = result.get("forwards", [])
            if forwards:
                in_latest = max(forwards, key=lambda f: f.get("received_time", 0))
                in_time = int(in_latest.get("received_time", 0))
                if latest_time is None or in_time > latest_time:
                    latest_time = in_time
        except Exception:
            pass

        return latest_time
    
    def _classify_channel(self, roi: float, net_profit: int,
                         last_routed: Optional[int], days_open: int,
                         channel_id: Optional[str] = None) -> ProfitabilityClass:
        """Classify a channel based on profitability metrics."""
        
        # Check for inactivity
        if last_routed:
            days_inactive = (int(time.time()) - last_routed) // 86400
        else:
            days_inactive = days_open  # Never routed
        
        # 1. Check for ZOMBIE (refinement with Defibrillator logic)
        # Requirement: Underwater AND inactive AND (at least 2 diagnostic attempts in 14 days)
        # If there are no diagnostic attempts, it's just UNDERWATER or STAGNANT_CANDIDATE
        if channel_id and roi < self.UNDERWATER_ROI_THRESHOLD:
            diag_stats = self.database.get_diagnostic_rebalance_stats(channel_id, days=14)
            
            # --- FIX FOR NoneType CRASH STARTS HERE ---
            attempt_count = diag_stats.get("attempt_count", 0)
            last_success_time = diag_stats.get("last_success_time") or 0 # Explicitly use 0 if None
            
            if attempt_count >= 2:
                if last_success_time > 0:
                    hours_since_diag_success = (int(time.time()) - last_success_time) // 3600
                    
                    # If it succeeded but still didn't route in 48h, it's a zombie
                    if hours_since_diag_success > 48 and (not last_routed or last_routed < last_success_time):
                        return ProfitabilityClass.ZOMBIE
                elif last_success_time == 0:
                    # No success in 2+ attempts (and last_success_time is guaranteed int 0 here)
                    return ProfitabilityClass.ZOMBIE
            # --- FIX FOR NoneType CRASH ENDS HERE ---
                    
        # 2. Check for STAGNANT_CANDIDATE (0 forwards in last 7 days + unprofitable)
        # Bug C Fix: Ensure low-volume profitable channels are BREAK_EVEN/BALANCED, not STAGNANT.
        # STAGNANT_CANDIDATE only if ROI < -10% and inactive for 7+ days.
        if days_inactive >= 7 and roi < -0.10:
            return ProfitabilityClass.STAGNANT_CANDIDATE
        
        # 3. Standard ROI Classifications
        if roi > self.PROFITABLE_ROI_THRESHOLD:
            return ProfitabilityClass.PROFITABLE
        elif roi < self.UNDERWATER_ROI_THRESHOLD:
            return ProfitabilityClass.UNDERWATER
        else:
            return ProfitabilityClass.BREAK_EVEN
    
    def _days_since_routed(self, profitability: ChannelProfitability) -> int:
        """Calculate days since last routing activity."""
        if profitability.last_routed:
            return (int(time.time()) - profitability.last_routed) // 86400
        return profitability.days_open