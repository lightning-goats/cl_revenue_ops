"""
Hill Climbing Fee Controller module for cl-revenue-ops

MODULE 2: Revenue-Maximizing Fee Controller (Dynamic Pricing)

This module implements a Hill Climbing (Perturb & Observe) algorithm
for dynamically adjusting channel fees to maximize revenue.

Why Hill Climbing Instead of PID?
- PID targets a static flow rate, ignoring price elasticity
- Hill Climbing actively seeks the revenue-maximizing fee point
- It adapts to changing market conditions and peer behavior

Hill Climbing Algorithm:
1. Perturb: Make a small fee change in a direction
2. Observe: Measure the resulting revenue change
3. Decide:
   - If Revenue Increased: Keep going in the same direction
   - If Revenue Decreased: Reverse direction (we went too far)
4. Repeat: Continuously seek the optimal fee point

Revenue Calculation:
- Revenue = Volume * Fee
- We track revenue over time windows to measure impact of changes

Constraints:
- Never drop below floor (economic minimum based on chain costs)
- Never exceed ceiling (prevent absurd fees)
- Use liquidity bucket multipliers as secondary weighting
- Unmanage from clboss before setting fees

The Hill Climber provides adaptive, revenue-seeking fee adjustments that
find the optimal price point where volume * fee is maximized.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

from pyln.client import Plugin, RpcError

from .config import Config, ChainCostDefaults, LiquidityBuckets
from .database import Database
from .clboss_manager import ClbossManager, ClbossTags
from .metrics import PrometheusExporter, MetricNames, METRIC_HELP

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer


@dataclass
class HillClimbState:
    """
    State of the Hill Climbing fee optimizer for one channel.
    
    UPDATED: Uses rate-based feedback (revenue per hour) instead of
    absolute revenue to eliminate lag from using 7-day averages.
    
    Attributes:
        last_revenue_rate: Revenue rate in sats/hour observed since last fee change
        last_fee_ppm: Fee that was in effect during last period
        trend_direction: Current search direction (1 = increasing, -1 = decreasing)
        step_ppm: Current step size in PPM (subject to wiggle dampening)
        last_update: Timestamp of last update
        consecutive_same_direction: How many times we've moved in same direction
        is_sleeping: Deadband hysteresis - True if channel is in sleep mode
        sleep_until: Unix timestamp when to wake up from sleep mode
        stable_cycles: Number of consecutive stable cycles (for entering sleep)
        last_broadcast_fee_ppm: The last fee PPM broadcasted to the network
    """
    last_revenue_rate: float = 0.0  # Revenue rate in sats/hour
    last_fee_ppm: int = 0
    trend_direction: int = 1  # 1 = try increasing fee, -1 = try decreasing
    step_ppm: int = 50  # Current step size (decays on reversal)
    last_update: int = 0
    consecutive_same_direction: int = 0
    is_sleeping: bool = False  # Deadband hysteresis sleep state
    sleep_until: int = 0  # Unix timestamp when to wake up
    stable_cycles: int = 0  # Consecutive stable cycles counter
    last_broadcast_fee_ppm: int = 0  # Last fee PPM broadcasted to the network
    last_state: str = 'balanced'  # State during last broadcast


@dataclass
class FeeAdjustment:
    """
    Record of a fee adjustment.
    
    Attributes:
        channel_id: Channel that was adjusted
        peer_id: Peer node ID
        old_fee_ppm: Previous fee
        new_fee_ppm: New fee after adjustment
        reason: Explanation of the adjustment
        hill_climb_values: Hill Climbing algorithm internal values
    """
    channel_id: str
    peer_id: str
    old_fee_ppm: int
    new_fee_ppm: int
    reason: str
    hill_climb_values: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "old_fee_ppm": self.old_fee_ppm,
            "new_fee_ppm": self.new_fee_ppm,
            "reason": self.reason,
            "hill_climb_values": self.hill_climb_values
        }


class HillClimbingFeeController:
    """
    Hill Climbing (Perturb & Observe) fee controller for revenue maximization.
    
    The controller aims to find the revenue-maximizing fee by iteratively
    adjusting fees and observing the revenue impact.
    
    Key Principles:
    1. Revenue Focus: Maximize Volume * Fee, not just volume
    2. Adaptive: Learns from revenue changes to find optimal fees
    3. Bounded: Respects floor/ceiling constraints
    4. Liquidity-aware: Uses bucket multipliers as weights
    5. clboss override: Unmanages from clboss before setting fees
    
    Hill Climbing Parameters:
    - step_ppm: Base fee change per iteration (default 50 ppm)
    - step_percent: Alternative step as percentage (default 5%)
    - min_observation_window: Minimum time between changes (default 6 hours)
    """
    
    # Hill Climbing parameters
    STEP_PPM = 50           # Initial step size in PPM
    STEP_PERCENT = 0.05     # Percentage step size (5%)
    MIN_STEP_PPM = 10       # Minimum step size (floor for dampening)
    MAX_STEP_PPM = 200      # Maximum step size
    MAX_CONSECUTIVE = 5     # Max consecutive moves in same direction before reducing step
    DAMPENING_FACTOR = 0.5  # Step size decay factor on direction reversal (halve the step)
    MIN_OBSERVATION_HOURS = 1.0  # Minimum hours between fee changes for valid signal
    VOLATILITY_THRESHOLD = 0.50  # 50% change in revenue rate triggers volatility reset
    
    # Deadband Hysteresis parameters (Phase 4: Stability & Scaling)
    # These reduce gossip noise by suppressing fee updates when the market is stable
    STABILITY_THRESHOLD = 0.01   # 1% change - consider market stable if below this
    WAKE_UP_THRESHOLD = 0.20     # 20% revenue spike triggers immediate wake-up
    SLEEP_CYCLES = 4             # Sleep for 4x the fee interval
    STABLE_CYCLES_REQUIRED = 3   # Number of flat cycles before entering sleep mode
    
    def __init__(self, plugin: Plugin, config: Config, database: Database, 
                 clboss_manager: ClbossManager,
                 profitability_analyzer: Optional["ChannelProfitabilityAnalyzer"] = None,
                 metrics_exporter: Optional[PrometheusExporter] = None):
        """
        Initialize the fee controller.
        
        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance
            clboss_manager: ClbossManager for handling overrides
            profitability_analyzer: Optional profitability analyzer for ROI-based adjustments
            metrics_exporter: Optional Prometheus metrics exporter for observability
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        self.clboss = clboss_manager
        self.profitability = profitability_analyzer
        self.metrics = metrics_exporter
        
        # In-memory cache of Hill Climbing states (also persisted to DB)
        self._hill_climb_states: Dict[str, HillClimbState] = {}
    
    def adjust_all_fees(self) -> List[FeeAdjustment]:
        """
        Adjust fees for all channels using Hill Climbing optimization.
        
        This is the main entry point, called periodically by the timer.
        
        Returns:
            List of FeeAdjustment records for channels that were adjusted
        """
        adjustments = []
        
        # Get all channel states from flow analysis
        channel_states = self.database.get_all_channel_states()
        
        if not channel_states:
            self.plugin.log("No channel state data for fee adjustment")
            return adjustments
        
        # Get current channel info for capacity and balance
        channels = self._get_channels_info()
        
        # OPTIMIZATION: Hoist feerates RPC call outside the loop
        # This reduces N RPC calls to 1 per adjust_all_fees cycle
        chain_costs = self._get_dynamic_chain_costs()
        
        for state in channel_states:
            channel_id = state.get("channel_id")
            peer_id = state.get("peer_id")
            
            if not channel_id or not peer_id:
                continue
            
            # Get channel info
            channel_info = channels.get(channel_id)
            if not channel_info:
                continue
            
            try:
                adjustment = self._adjust_channel_fee(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=state,
                    channel_info=channel_info,
                    chain_costs=chain_costs
                )
                
                if adjustment:
                    adjustments.append(adjustment)
                    
            except Exception as e:
                self.plugin.log(f"Error adjusting fee for {channel_id}: {e}", level='error')
        
        return adjustments
    
    def _adjust_channel_fee(self, channel_id: str, peer_id: str,
                           state: Dict[str, Any], 
                           channel_info: Dict[str, Any],
                           chain_costs: Optional[Dict[str, int]] = None) -> Optional[FeeAdjustment]:
        """
        Adjust fee for a single channel using Hill Climbing optimization.
        
        UPDATED: Rate-Based Feedback with Wiggle Dampening
        
        Key Changes from Previous Version:
        1. Rate-Based Feedback: Uses volume since last fee change (not 7-day average)
           to measure revenue per hour, eliminating lag in the feedback loop.
        2. Wiggle Dampening: When the algorithm reverses direction (overshot peak),
           the step size is decayed by DAMPENING_FACTOR to converge on the optimum.
        
        Hill Climbing (Perturb & Observe) Algorithm:
        1. Get volume since last fee change via get_volume_since()
        2. Calculate revenue RATE (sats/hour) = (volume * fee) / hours_elapsed
        3. Compare current revenue rate to last period's rate
        4. If rate increased: continue in same direction
        5. If rate decreased: reverse direction AND reduce step (dampening)
        6. Apply step change in calculated direction
        
        Args:
            channel_id: Channel to adjust
            peer_id: Peer node ID
            state: Channel state from flow analysis
            channel_info: Current channel info (capacity, balance, etc.)
            chain_costs: Pre-fetched chain costs from feerates RPC (optimization)
            
        Returns:
            FeeAdjustment if fee was changed, None otherwise
        """
        # Detect critical state (Phase 5.5)
        is_congested = (state and state.get("state") == "congested")
        
        # Get current fee
        raw_chain_fee = channel_info.get("fee_proportional_millionths", 0)
        current_fee_ppm = raw_chain_fee
        if current_fee_ppm == 0:
            current_fee_ppm = self.config.min_fee_ppm  # Initialize if not set
        
        # Load Hill Climbing state
        hc_state = self._get_hill_climb_state(channel_id)
        
        # =====================================================================
        # ZERO-FEE PROBE: Defibrillator Override (Phase 8.1)
        # =====================================================================
        probe_flag = self.database.get_channel_probe(channel_id)
        is_under_probe = (probe_flag is not None)
        
        now = int(time.time())
        
        # Decision for target fee (The Alpha Sequence)
        is_fire_sale = False
        if self.profitability:
            from .profitability_analyzer import ProfitabilityClass
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data and prof_data.days_open > 90:
                if prof_data.classification == ProfitabilityClass.ZOMBIE:
                    is_fire_sale = True
                elif prof_data.classification == ProfitabilityClass.UNDERWATER:
                    if prof_data.roi_percent < -50.0:
                        is_fire_sale = True
            
            # MOMENTUM GUARD: Protect recovering channels from Fire Sale (Phase 5.5)
            # Channels with positive operational ROI are paying back their debt -
            # don't kill them just because they had high opening costs.
            if is_fire_sale and prof_data:
                marginal_roi = prof_data.marginal_roi
                if marginal_roi > 0.05 and prof_data.days_open < 180:
                    self.plugin.log(
                        f"MOMENTUM GUARD: Channel {channel_id[:12]}... is recovering "
                        f"(Marginal ROI {marginal_roi:.1%}). Suspending Fire Sale to allow price discovery.",
                        level='info'
                    )
                    is_fire_sale = False
        
        # =====================================================================
        # DEADBAND HYSTERESIS: Sleep Status Check (Phase 4: Stability & Scaling)
        # Reduces gossip noise by suppressing fee updates when the market is stable
        # =====================================================================
        if hc_state.is_sleeping:
            # Check if it's time to wake up (sleep timer expired)
            if now > hc_state.sleep_until:
                # Timer expired - wake up
                hc_state.is_sleeping = False
                hc_state.sleep_until = 0
                hc_state.stable_cycles = 0
                self._save_hill_climb_state(channel_id, hc_state)
                self.plugin.log(
                    f"HYSTERESIS: Channel {channel_id[:12]}... waking up (sleep timer expired)",
                    level='info'
                )
            else:
                # Still within sleep period - check for revenue spike that should wake us
                # Calculate current revenue rate to detect significant changes
                if self.config.enable_reputation:
                    volume_since_sats = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
                else:
                    volume_since_sats = self.database.get_volume_since(channel_id, hc_state.last_update)
                
                hours_elapsed = (now - hc_state.last_update) / 3600.0 if hc_state.last_update > 0 else 1.0
                hours_elapsed = max(hours_elapsed, 0.1)  # Prevent division by zero
                
                revenue_sats = (volume_since_sats * current_fee_ppm) // 1_000_000
                current_revenue_rate = revenue_sats / hours_elapsed
                
                # Calculate percent change from last known rate
                last_rate = max(1.0, hc_state.last_revenue_rate)  # Avoid division by zero
                delta = abs(current_revenue_rate - hc_state.last_revenue_rate)
                percent_change = delta / last_rate
                
                if percent_change > self.WAKE_UP_THRESHOLD:
                    # Significant revenue spike detected - wake up immediately!
                    hc_state.is_sleeping = False
                    hc_state.sleep_until = 0
                    hc_state.stable_cycles = 0
                    self._save_hill_climb_state(channel_id, hc_state)
                    self.plugin.log(
                        f"HYSTERESIS: Channel {channel_id[:12]}... waking up due to revenue spike "
                        f"({percent_change:.0%} change, threshold={self.WAKE_UP_THRESHOLD:.0%})",
                        level='info'
                    )
                else:
                    # No significant change - stay asleep, skip this adjustment cycle
                    self.plugin.log(
                        f"HYSTERESIS: Channel {channel_id[:12]}... sleeping "
                        f"(wake in {(hc_state.sleep_until - now) // 60} min)",
                        level='debug'
                    )
                    # Export sleep state metric (Observability)
                    if self.metrics:
                        self.metrics.set_gauge(
                            MetricNames.CHANNEL_IS_SLEEPING,
                            1,
                            {"channel_id": channel_id, "peer_id": peer_id},
                            METRIC_HELP.get(MetricNames.CHANNEL_IS_SLEEPING, "")
                        )
                    return None
        
        # PROFITABILITY SHIELD: Protect high-value peers from reputation penalties
        # If a channel is highly profitable (ROI > 10%), we ignore its "reputation" score.
        # This ensures we don't price-out "messy but rich" peers (high volume but occasional failures).
        is_shielded = False
        if self.profitability:
            from .profitability_analyzer import ProfitabilityClass
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data and prof_data.classification == ProfitabilityClass.PROFITABLE:
                is_shielded = True
                self.plugin.log(
                    f"PROFITABILITY SHIELD: Shielding profitable peer {peer_id[:12]}... "
                    f"(ROI={prof_data.roi_percent:.1f}%) - Reputation penalty ignored.",
                    level='info'
                )

        # RATE-BASED FEEDBACK: Get volume SINCE LAST FEE CHANGE (not 7-day average)
        # This eliminates the lag from averaging that made the controller blind
        #
        # REPUTATION-WEIGHTED VOLUME: If enabled, discount volume by peer success rate
        # This prevents spammy peers with high failure rates from influencing fees
        # Effective Volume = Raw Volume * Peer_Success_Rate
        #
        # EXCEPTION: If channel is SHIELDED, we always use raw volume.
        if self.config.enable_reputation and not is_shielded:
            volume_since_sats = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
        else:
            volume_since_sats = self.database.get_volume_since(channel_id, hc_state.last_update)
        
        # FLAP PROTECTION: Penalize flapping peers' volume for revenue signal
        # Peers with high disconnect rates have dampened revenue signals so we
        # don't optimize fees based on unreliable traffic patterns.
        # Formula: effective_volume = volume * (uptime_pct / 100)
        # 
        # NOTE: Shielded channels are NOT protected from Flap Protection.
        # Unstable connections are bad regardless of profitability.
        uptime_pct = self.database.get_peer_uptime_percent(peer_id, 86400)  # 24h window
        uptime_factor = uptime_pct / 100.0  # Convert 0-100 to 0-1
        if uptime_factor < 1.0:
            original_volume = volume_since_sats
            volume_since_sats = int(volume_since_sats * uptime_factor)
            self.plugin.log(
                f"FLAP PROTECTION: Dampening volume for {channel_id[:12]}... "
                f"({original_volume} -> {volume_since_sats} sats, uptime={uptime_pct:.1f}%)",
                level='debug'
            )
        
        # Calculate time elapsed since last update
        if hc_state.last_update > 0:
            hours_elapsed = (now - hc_state.last_update) / 3600.0
        else:
            hours_elapsed = 0.0
        
        # EDGE CASE: Protect against division by zero or very small time windows
        # If the user manually triggers analysis twice instantly, hours_elapsed could be tiny
        if hours_elapsed < self.MIN_OBSERVATION_HOURS:
            self.plugin.log(
                f"Skipping {channel_id[:12]}...: observation window too short "
                f"({hours_elapsed:.2f}h < {self.MIN_OBSERVATION_HOURS}h minimum)",
                level='debug'
            )
            # Still too early for valid signal - skip this channel for now
            if hc_state.last_update > 0:  # Only skip if we have prior state
                return None
            # First run - continue with initialization
            hours_elapsed = 1.0  # Use 1 hour as default for first run
        
        # Calculate REVENUE RATE (sats/hour) - this is our feedback signal
        # Revenue = Volume * Fee_PPM / 1_000_000
        revenue_sats = (volume_since_sats * current_fee_ppm) // 1_000_000
        current_revenue_rate = revenue_sats / hours_elapsed if hours_elapsed > 0 else 0.0
        
        # Get capacity and balance for liquidity adjustments
        capacity = channel_info.get("capacity", 1)
        spendable = channel_info.get("spendable_msat", 0) // 1000
        outbound_ratio = spendable / capacity if capacity > 0 else 0.5
        
        bucket = LiquidityBuckets.get_bucket(outbound_ratio)
        liquidity_multiplier = LiquidityBuckets.get_fee_multiplier(bucket)
        
        # Get flow state for bias
        flow_state = state.get("state", "balanced")
        flow_state_multiplier = 1.0
        if flow_state == "source":
            flow_state_multiplier = 1.25  # Sources are scarce - higher fees
        elif flow_state == "sink":
            flow_state_multiplier = 0.80  # Sinks fill for free - lower floor
        
        # Get profitability multiplier (uses marginal ROI now)
        profitability_multiplier = 1.0
        marginal_roi_info = "unknown"
        if self.profitability:
            profitability_multiplier = self.profitability.get_fee_multiplier(channel_id)
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data:
                marginal_roi_info = f"marginal_roi={prof_data.marginal_roi_percent:.1f}%"
        
        # Calculate Floor and Ceiling
        floor_ppm = self._calculate_floor(capacity, chain_costs=chain_costs, peer_id=peer_id)
        floor_ppm = max(floor_ppm, self.config.min_fee_ppm)
        # Apply flow state to floor (sinks can go lower)
        floor_ppm = int(floor_ppm * flow_state_multiplier)
        floor_ppm = max(floor_ppm, 1)  # Never go below 1 ppm
        
        ceiling_ppm = self.config.max_fee_ppm
        
        # PRIORITY OVERRIDE: Zero-Fee Probe takes precedence over Fire Sale
        # We must allow the diagnostic probe (0 PPM) to run to verify liveness
        # before resigning ourselves to liquidation pricing (1 PPM).
        if is_under_probe:
            is_fire_sale = False
        
        # Target Decision Block (The Alpha Sequence)
        new_fee_ppm = 0
        target_found = False
        
        # Priority 1: Congestion (Emergency High Fee)
        if is_congested:
            new_fee_ppm = ceiling_ppm
            decision_reason = "CONGESTION"
            new_direction = hc_state.trend_direction
            step_ppm = hc_state.step_ppm
            volatility_reset = False
            rate_change = 0.0
            previous_rate = hc_state.last_revenue_rate
            target_found = True
            
        # Priority 2: Fire Sale (Dumping Inventory)
        elif is_fire_sale:
            new_fee_ppm = 1
            decision_reason = "FIRE_SALE"
            new_direction = hc_state.trend_direction
            step_ppm = hc_state.step_ppm
            volatility_reset = False
            rate_change = 0.0
            previous_rate = hc_state.last_revenue_rate
            target_found = True
            
        # Priority 3: Zero-Fee Probe Logic (Jumpstarting)
        if not target_found and is_under_probe:
            # Calculate current revenue rate (reuse logic from rate calculation below)
            if self.config.enable_reputation and not is_shielded:
                v_since = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
            else:
                v_since = self.database.get_volume_since(channel_id, hc_state.last_update)
            
            h_elapsed = (now - hc_state.last_update) / 3600.0 if hc_state.last_update > 0 else 1.0
            rev_sats = (v_since * current_fee_ppm) // 1_000_000
            curr_rev_rate = rev_sats / h_elapsed if h_elapsed > 0 else 0.0
            
            if curr_rev_rate > 0.0:
                # WAKE UP: Success!
                self.database.clear_channel_probe(channel_id)
                self.plugin.log(
                    f"DEFIBRILLATOR SUCCESS: Channel {channel_id} routed under 0-fee probe. Resuming Hill Climber.",
                    level='info'
                )
                is_under_probe = False  # Continue to standard Hill Climbing this cycle
            else:
                # Still probing
                new_fee_ppm = 0  # Force 0 PPM
                decision_reason = "ZERO_FEE_PROBE"
                new_direction = hc_state.trend_direction
                step_ppm = hc_state.step_ppm
                volatility_reset = False
                rate_change = 0.0
                previous_rate = hc_state.last_revenue_rate
                target_found = True

        # Priority 4: Hill Climbing (Discovery)
        if not target_found:
            # HILL CLIMBING DECISION (Rate-Based)
            rate_change = current_revenue_rate - hc_state.last_revenue_rate
            last_direction = hc_state.trend_direction
            previous_rate = hc_state.last_revenue_rate
            
            step_ppm = hc_state.step_ppm
            if step_ppm <= 0:
                step_ppm = self.STEP_PPM
            
            # VOLATILITY RESET & DEADBAND HYSTERESIS
            volatility_reset = False
            rate_change_ratio = 0.0
            if hc_state.last_update > 0 and hc_state.last_revenue_rate > 0:
                delta_rate = abs(current_revenue_rate - hc_state.last_revenue_rate)
                rate_change_ratio = delta_rate / max(1.0, hc_state.last_revenue_rate)
                
                if rate_change_ratio > self.VOLATILITY_THRESHOLD:
                    step_ppm = self.STEP_PPM
                    volatility_reset = True
                    hc_state.stable_cycles = 0

            # DEADBAND HYSTERESIS: Enter Sleep Mode Check
            if hc_state.last_update > 0 and rate_change_ratio < self.STABILITY_THRESHOLD:
                hc_state.stable_cycles += 1
                if hc_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED:
                    sleep_duration_seconds = self.config.fee_interval * self.SLEEP_CYCLES
                    hc_state.is_sleeping = True
                    hc_state.sleep_until = now + sleep_duration_seconds
                    hc_state.last_revenue_rate = current_revenue_rate
                    hc_state.last_fee_ppm = current_fee_ppm
                    hc_state.last_update = now
                    self._save_hill_climb_state(channel_id, hc_state)
                    self.plugin.log(f"HYSTERESIS: Market Calm - Channel {channel_id[:12]}... entering sleep mode.", level='info')
                    return None
            else:
                if rate_change_ratio >= self.STABILITY_THRESHOLD:
                    hc_state.stable_cycles = 0

            # Direction Decision
            if hc_state.last_update == 0:
                new_direction = 1
                decision_reason = "initial"
            elif rate_change > 0:
                new_direction = last_direction
                decision_reason = "rate_up"
                hc_state.consecutive_same_direction += 1
                if rate_change_ratio > 0.20:
                    step_ppm = min(int(step_ppm * 2), self.MAX_STEP_PPM)
            elif rate_change < 0:
                new_direction = -last_direction
                decision_reason = "rate_down"
                hc_state.consecutive_same_direction = 0
                step_ppm = max(self.MIN_STEP_PPM, int(step_ppm * self.DAMPENING_FACTOR))
            else:
                new_direction = -last_direction
                decision_reason = "rate_flat"
                hc_state.consecutive_same_direction = 0
                step_ppm = max(self.MIN_STEP_PPM, int(step_ppm * self.DAMPENING_FACTOR))

            # Apply step constraints
            step_percent = max(current_fee_ppm * self.STEP_PERCENT, self.MIN_STEP_PPM)
            step_ppm = max(step_ppm, int(step_percent))
            step_ppm = min(step_ppm, self.MAX_STEP_PPM)
            if hc_state.consecutive_same_direction > self.MAX_CONSECUTIVE:
                step_ppm = max(self.MIN_STEP_PPM, step_ppm // 2)

            base_new_fee = current_fee_ppm + (new_direction * step_ppm)
            new_fee_ppm = int(base_new_fee * liquidity_multiplier * profitability_multiplier)
            new_fee_ppm = max(floor_ppm, min(ceiling_ppm, new_fee_ppm))


        # Check if fee changed meaningfully (Alpha Guard)
        fee_change = abs(new_fee_ppm - current_fee_ppm)
        if current_fee_ppm < 100:
            min_change = 1
        else:
            min_change = max(5, current_fee_ppm * 0.03)
            
        if fee_change < min_change and not (is_congested or is_fire_sale):
            return None
        
        # =====================================================================
        # GOSSIP HYSTERESIS: The 5% Gate (Phase 5.5)
        # Reduce network noise by only broadcasting significant changes.
        # =====================================================================
        delta_broadcast = abs(new_fee_ppm - hc_state.last_broadcast_fee_ppm)
        threshold = hc_state.last_broadcast_fee_ppm * 0.05
        
        # Override: Always broadcast if entering/exiting critical states
        # or if we have never broadcasted before
        significant_change = (delta_broadcast > threshold) or \
                             (hc_state.last_broadcast_fee_ppm <= 1) or \
                             (new_fee_ppm <= 1) or \
                             (target_found and hc_state.last_state != decision_reason) or \
                             (not target_found and hc_state.last_state in ("CONGESTION", "FIRE_SALE"))

        if not significant_change:
            # HYSTERESIS: Skip RPC, update internal target, but PAUSE observation window
            hc_state.last_fee_ppm = new_fee_ppm
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            # IMPORTANT: Do NOT update hc_state.last_update here (Observation Pause)
            self._save_hill_climb_state(channel_id, hc_state)
            
            self.plugin.log(
                f"HYSTERESIS: Target fee {new_fee_ppm} is <5% delta from broadcast {hc_state.last_broadcast_fee_ppm}. "
                f"Skipping gossip; pausing observation.",
                level='info'
            )
            return None

        # Build reason string (with rate info)
        volatility_note = " [VOLATILITY_RESET]" if volatility_reset else ""
        reason = (f"HillClimb: rate={current_revenue_rate:.2f}sats/hr ({decision_reason}){volatility_note}, "
                 f"direction={'up' if new_direction > 0 else 'down'}, "
                 f"step={step_ppm}ppm, state={flow_state}, "
                 f"liquidity={bucket} ({outbound_ratio:.0%}), "
                 f"{marginal_roi_info}")
        
        # IDEMPOTENCY GUARD: Skip RPC if target is physically set (Phase 5.5)
        if new_fee_ppm == raw_chain_fee:
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.last_fee_ppm = raw_chain_fee
            hc_state.last_broadcast_fee_ppm = new_fee_ppm
            hc_state.last_state = decision_reason
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            hc_state.last_update = now  # Reset observation timer
            self._save_hill_climb_state(channel_id, hc_state)
            return None
        
        # Apply the fee change (Significant change -> Broadcast)
        result = self.set_channel_fee(channel_id, new_fee_ppm, reason=reason)
        
        if result.get("success"):
            # Update state with new broadcast fee and refresh timer
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.last_fee_ppm = current_fee_ppm
            hc_state.last_broadcast_fee_ppm = new_fee_ppm
            hc_state.last_state = decision_reason
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            hc_state.last_update = now
            self._save_hill_climb_state(channel_id, hc_state)
            # Export metrics (Phase 2: Observability)
            if self.metrics:
                labels = {"channel_id": channel_id, "peer_id": peer_id}
                
                # Gauge: Current fee PPM
                self.metrics.set_gauge(
                    MetricNames.CHANNEL_FEE_PPM,
                    new_fee_ppm,
                    labels,
                    METRIC_HELP.get(MetricNames.CHANNEL_FEE_PPM, "")
                )
                
                # Gauge: Revenue rate (sats/hour)
                self.metrics.set_gauge(
                    MetricNames.CHANNEL_REVENUE_RATE_SATS_HR,
                    current_revenue_rate,
                    labels,
                    METRIC_HELP.get(MetricNames.CHANNEL_REVENUE_RATE_SATS_HR, "")
                )
                
                # Gauge: Sleep state (0 = awake, actively adjusting)
                self.metrics.set_gauge(
                    MetricNames.CHANNEL_IS_SLEEPING,
                    0,
                    labels,
                    METRIC_HELP.get(MetricNames.CHANNEL_IS_SLEEPING, "")
                )
            
            return FeeAdjustment(
                channel_id=channel_id,
                peer_id=peer_id,
                old_fee_ppm=current_fee_ppm,
                new_fee_ppm=new_fee_ppm,
                reason=reason,
                hill_climb_values={
                    "current_revenue_rate": current_revenue_rate,
                    "previous_revenue_rate": previous_rate,
                    "rate_change": rate_change,
                    "volume_since_sats": volume_since_sats,
                    "hours_elapsed": hours_elapsed,
                    "direction": new_direction,
                    "step_ppm": step_ppm,
                    "consecutive_same_direction": hc_state.consecutive_same_direction,
                    "volatility_reset": volatility_reset
                }
            )
        
        return None
    
    def set_channel_fee(self, channel_id: str, fee_ppm: int, 
                       reason: str = "manual", manual: bool = False) -> Dict[str, Any]:
        """
        Set the fee for a channel, handling clboss override.
        
        MANAGER-OVERRIDE PATTERN:
        1. Get peer ID for the channel
        2. Call clboss-unmanage to prevent conflicts
        3. Set the fee using setchannelfee
        4. Record the change
        
        Args:
            channel_id: Channel to update
            fee_ppm: New fee in parts per million
            reason: Explanation for the change
            manual: True if manually triggered (vs automatic)
            
        Returns:
            Result dict with success status and details
        """
        result = {
            "success": False,
            "channel_id": channel_id,
            "fee_ppm": fee_ppm,
            "message": ""
        }
        
        try:
            # Get channel info to find peer ID and current fee
            channels = self._get_channels_info()
            channel_info = channels.get(channel_id)
            
            if not channel_info:
                result["message"] = f"Channel {channel_id} not found"
                return result
            
            peer_id = channel_info.get("peer_id", "")
            old_fee_ppm = channel_info.get("fee_proportional_millionths", 0)
            
            # Step 1: Unmanage from clboss
            # This is critical - we MUST do this before setting fees
            if not self.clboss.ensure_unmanaged_for_channel(
                channel_id, peer_id, ClbossTags.FEE, self.database
            ):
                self.plugin.log(
                    f"Warning: Could not unmanage {peer_id} from clboss, "
                    "fee may be reverted", level='warn'
                )
            
            # Step 2: Set the fee
            if self.config.dry_run:
                self.plugin.log(f"[DRY RUN] Would set fee for {channel_id} to {fee_ppm} PPM")
                result["success"] = True
                result["message"] = "Dry run - no changes made"
                return result
            
            # Use setchannel command
            # setchannel id [feebase] [feeppm] [htlcmin] [htlcmax] [enforcedelay] [ignorefeelimits]
            self.plugin.rpc.setchannel(
                channel_id,                    # id
                self.config.base_fee_msat,     # feebase (msat)
                fee_ppm                        # feeppm
            )
            
            # Step 3: Record the change
            self.database.record_fee_change(
                channel_id=channel_id,
                peer_id=peer_id,
                old_fee_ppm=old_fee_ppm,
                new_fee_ppm=fee_ppm,
                reason=reason,
                manual=manual
            )
            
            result["success"] = True
            result["old_fee_ppm"] = old_fee_ppm
            result["message"] = f"Fee set to {fee_ppm} PPM"
            
            self.plugin.log(
                f"Set fee for {channel_id[:16]}...: {old_fee_ppm} -> {fee_ppm} PPM "
                f"({reason})"
            )
            
        except RpcError as e:
            result["message"] = f"RPC error: {str(e)}"
            self.plugin.log(f"Failed to set fee for {channel_id}: {e}", level='error')
        except Exception as e:
            result["message"] = f"Error: {str(e)}"
            self.plugin.log(f"Error setting fee: {e}", level='error')
        
        return result
    
    def _calculate_floor(self, capacity_sats: int, 
                         chain_costs: Optional[Dict[str, int]] = None,
                         peer_id: Optional[str] = None) -> int:
        """
        Calculate the economic floor fee for a channel.
        
        The floor ensures we never charge less than the channel costs us.
        Uses live mempool fee rates when available for accurate cost estimation.
        
        ALGORITHM:
        1. Base Floor: Amortized open/close costs over lifetime volume.
           (Phase 7: REPLACEMENT COST PRICING logic)
        2. Risk Premium: Additional fee needed to cover on-chain enforcement diff
           during high congestion for typical HTLC sizes.
        3. HTLC Hold Risk Premium: Markup for peers with high "Stall Risk"
           (peers that tie up capital for long durations).
           
        floor_ppm = max(base_floor, risk_premium) * stall_multiplier
        
        Args:
            capacity_sats: Channel capacity
            chain_costs: Pre-fetched chain costs from feerates RPC (optimization).
            peer_id: Optional peer ID to check for HTLC hold latency.
            
        Returns:
            Minimum fee in PPM
        """
        # Use provided chain_costs (hoisted from adjust_all_fees for efficiency)
        # Falls back to static defaults if chain_costs is None (RPC failed)
        dynamic_costs = chain_costs
        floor_ppm = ChainCostDefaults.calculate_floor_ppm(capacity_sats)
        
        if dynamic_costs:
            # 1. Calculate Base Floor (Cost Recovery) using REPLACEMENT COST
            # We ignore historical costs (what we paid) and look at what it costs
            # to replace the channel today.
            open_cost = dynamic_costs.get("open_cost_sats", ChainCostDefaults.CHANNEL_OPEN_COST_SATS)
            close_cost = dynamic_costs.get("close_cost_sats", ChainCostDefaults.CHANNEL_CLOSE_COST_SATS)
            
            total_chain_cost = open_cost + close_cost
            estimated_lifetime_volume = ChainCostDefaults.DAILY_VOLUME_SATS * ChainCostDefaults.CHANNEL_LIFETIME_DAYS
            
            if estimated_lifetime_volume > 0:
                base_floor = (total_chain_cost / estimated_lifetime_volume) * 1_000_000
                
                # Check if replacement cost is driving the floor up significantly
                if base_floor > floor_ppm:
                    self.plugin.log(
                        f"REPLACEMENT COST PRICING: Raising floor to {int(base_floor)} PPM "
                        f"based on current chain fees.", 
                        level='debug'
                    )
                
                floor_ppm = max(floor_ppm, int(base_floor))
        
        # 3. HTLC Hold Risk Premium (Stall Defense)
        if peer_id:
            latency = self.database.get_peer_latency_stats(peer_id, window_seconds=86400)
            avg_res = latency.get('avg', 0)
            std_res = latency.get('std', 0)
            
            if avg_res > 10.0 or std_res > 5.0:
                self.plugin.log(
                    f"HTLC HOLD DEFENSE: Peer {peer_id[:16]}... has high Stall Risk "
                    f"(avg={avg_res:.1f}s, std={std_res:.1f}s). Applying 20% markup to floor.",
                    level='info'
                )
                floor_ppm = int(floor_ppm * 1.2)
                
        # 2. Calculate Risk Premium (Congestion Defense)
            # When mempool is congested, force-closing becomes expensive.
            # We must charge enough to justify the risk of smaller HTLCs getting stuck/trimmed.
            sat_per_vbyte = dynamic_costs.get("sat_per_vbyte", 0.0)
            
            if sat_per_vbyte > 0:
                # Conservative estimate for a commitment tx weight (approx 150 vbytes)
                COMMITMENT_TX_VBYTES = 150
                # Reference HTLC size to evaluate risk against (50k sats = ~$50)
                # Smaller values mean we charge HIGHER fees to discourage dust
                AVG_HTLC_SIZE_SATS = 50_000
                
                # RISK PROBABILITY: The chance that any specific HTLC will force-close the channel.
                # We don't charge the full on-chain cost for every packet (that would be ~180,000 PPM).
                # We charge the Expectation of cost: Cost * Probability.
                # Assumes ~1 in 1000 HLTCs causes a force-close scenario.
                RISK_PROBABILITY = 0.001
                
                # Formula: (Cost to enforce * Probability) / Value protected * 1M
                # (sat_vbyte * 150 * 0.001) / 250k * 1M
                risk_premium_ppm = int((sat_per_vbyte * COMMITMENT_TX_VBYTES * RISK_PROBABILITY * 1_000_000) / AVG_HTLC_SIZE_SATS)
                
                # Apply Risk Premium if it exceeds the base floor
                if risk_premium_ppm > floor_ppm:
                    # Log warning only if congestion is significant (e.g. > 100 sat/vB)
                    if sat_per_vbyte > 100:
                        self.plugin.log(
                            f"CONGESTION DEFENSE: High fees ({sat_per_vbyte:.1f} sat/vB). "
                            f"Raising floor from {floor_ppm} to {risk_premium_ppm} PPM "
                            f"(Risk Premium).",
                            level='info'
                        )
                    floor_ppm = risk_premium_ppm
        
        return max(1, int(floor_ppm))
    
    def _get_dynamic_chain_costs(self) -> Optional[Dict[str, int]]:
        """
        Get dynamic chain cost estimates from feerates RPC.
        
        Uses current mempool fee rates to estimate:
        - Channel open cost (funding tx, ~140 vbytes typical)
        - Channel close cost (commitment tx, ~200 vbytes typical)
        
        Returns:
            Dict with open_cost_sats and close_cost_sats, or None if unavailable
        """
        try:
            # Query feerates - prefer 'perkb' style for calculations
            feerates = self.plugin.rpc.feerates(style="perkb")
            
            # Get a medium-term estimate (12 blocks ~2 hours)
            perkb = feerates.get("perkb", {})
            
            # Try different fee rate estimates in order of preference
            sat_per_kvb = (
                perkb.get("opening") or      # CLN's channel opening estimate
                perkb.get("mutual_close") or  # Mutual close estimate  
                perkb.get("unilateral_close") or  # Unilateral close estimate
                perkb.get("floor") or         # Minimum relay fee
                1000                          # Fallback 1 sat/vbyte
            )
            
            # Convert to sat/vbyte
            sat_per_vbyte = sat_per_kvb / 1000
            
            # Typical transaction sizes (conservative estimates)
            # Funding tx: ~140 vbytes (1 input, 2 outputs)
            # Mutual close: ~170 vbytes  
            # Unilateral close: ~200 vbytes (with anchor outputs)
            FUNDING_TX_VBYTES = 140
            CLOSE_TX_VBYTES = 200  # Use unilateral as worst case
            
            open_cost_sats = int(sat_per_vbyte * FUNDING_TX_VBYTES)
            close_cost_sats = int(sat_per_vbyte * CLOSE_TX_VBYTES)
            
            # Sanity bounds
            open_cost_sats = max(500, min(50000, open_cost_sats))
            close_cost_sats = max(300, min(50000, close_cost_sats))
            
            self.plugin.log(
                f"Dynamic chain costs: open={open_cost_sats} sats, close={close_cost_sats} sats "
                f"(at {sat_per_vbyte:.1f} sat/vB)",
                level='debug'
            )
            
            return {
                "open_cost_sats": open_cost_sats,
                "close_cost_sats": close_cost_sats,
                "sat_per_vbyte": sat_per_vbyte
            }
            
        except Exception as e:
            self.plugin.log(f"Error getting feerates: {e}", level='debug')
            return None
    
    def _get_hill_climb_state(self, channel_id: str) -> HillClimbState:
        """
        Get Hill Climbing state for a channel.
        
        Checks in-memory cache first, then database.
        Updated to use rate-based feedback (last_revenue_rate), step_ppm,
        and deadband hysteresis fields (is_sleeping, sleep_until, stable_cycles).
        """
        if channel_id in self._hill_climb_states:
            return self._hill_climb_states[channel_id]
        
        # Load from database (uses the fee_strategy_state table)
        db_state = self.database.get_fee_strategy_state(channel_id)
        
        hc_state = HillClimbState(
            last_revenue_rate=db_state.get("last_revenue_rate", 0.0),
            last_fee_ppm=db_state.get("last_fee_ppm", 0),
            trend_direction=db_state.get("trend_direction", 1),
            step_ppm=db_state.get("step_ppm", self.STEP_PPM),
            last_update=db_state.get("last_update", 0),
            consecutive_same_direction=db_state.get("consecutive_same_direction", 0),
            is_sleeping=bool(db_state.get("is_sleeping", 0)),
            sleep_until=db_state.get("sleep_until", 0),
            stable_cycles=db_state.get("stable_cycles", 0),
            last_broadcast_fee_ppm=db_state.get("last_broadcast_fee_ppm", 0),
            last_state=db_state.get("last_state", "balanced")
        )
        
        self._hill_climb_states[channel_id] = hc_state
        return hc_state
    
    def _save_hill_climb_state(self, channel_id: str, state: HillClimbState):
        """Save Hill Climbing state to cache and database (including hysteresis fields)."""
        self._hill_climb_states[channel_id] = state
        self.database.update_fee_strategy_state(
            channel_id=channel_id,
            last_revenue_rate=state.last_revenue_rate,
            last_fee_ppm=state.last_fee_ppm,
            trend_direction=state.trend_direction,
            step_ppm=state.step_ppm,
            consecutive_same_direction=state.consecutive_same_direction,
            last_broadcast_fee_ppm=state.last_broadcast_fee_ppm,
            last_state=state.last_state,
            is_sleeping=1 if state.is_sleeping else 0,
            sleep_until=state.sleep_until,
            stable_cycles=state.stable_cycles
        )
    
    def _get_channels_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current info for all channels.
        
        Returns:
            Dict mapping channel_id to channel info
        """
        channels = {}
        
        try:
            result = self.plugin.rpc.listpeerchannels()
            
            for channel in result.get("channels", []):
                if channel.get("state") != "CHANNELD_NORMAL":
                    continue
                
                channel_id = channel.get("short_channel_id") or channel.get("channel_id")
                if channel_id:
                    # Get balance info
                    spendable_msat = channel.get("spendable_msat", 0) or 0
                    receivable_msat = channel.get("receivable_msat", 0) or 0
                    
                    # Calculate capacity - may be null in some CLN versions
                    total_msat = channel.get("total_msat") or channel.get("capacity_msat")
                    if not total_msat:
                        total_msat = spendable_msat + receivable_msat
                    
                    # Get fee info - in newer CLN it's under updates.local
                    updates = channel.get("updates", {})
                    local_updates = updates.get("local", {})
                    
                    # Try updates.local first, fall back to top-level
                    fee_base = local_updates.get("fee_base_msat") or channel.get("fee_base_msat", 0)
                    fee_ppm = local_updates.get("fee_proportional_millionths") or channel.get("fee_proportional_millionths", 0)
                    
                    channels[channel_id] = {
                        "channel_id": channel_id,
                        "peer_id": channel.get("peer_id", ""),
                        "capacity": total_msat // 1000 if total_msat else 0,
                        "spendable_msat": spendable_msat,
                        "receivable_msat": receivable_msat,
                        "fee_base_msat": fee_base,
                        "fee_proportional_millionths": fee_ppm
                    }
                    
        except RpcError as e:
            self.plugin.log(f"Error getting channel info: {e}", level='error')
        
        return channels
    
    def reset_hill_climb_state(self, channel_id: str):
        """
        Reset Hill Climbing state for a channel.
        
        Use this when manually intervening or if the controller
        is behaving erratically.
        """
        hc_state = HillClimbState()
        self._save_hill_climb_state(channel_id, hc_state)
        self.plugin.log(f"Reset Hill Climbing state for {channel_id}")


# Keep alias for backward compatibility
PIDFeeController = HillClimbingFeeController