from collections import defaultdict, deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Deque, Dict, Optional


@dataclass
class SurgeSample:
    ts: float
    amount_msat: int
    incoming_peer_id: str
    incoming_channel_id: str
    outgoing_channel_id: str


@dataclass
class SurgeOverlayState:
    baseline_fee_ppm: Optional[int] = None
    active_fee_ppm: Optional[int] = None
    active: bool = False
    cooldown_until: float = 0.0
    last_trigger_reason: str = ""
    last_apply_result: str = "idle"
    last_attempt_ts: float = 0.0
    pending_action: Optional[str] = None
    pending_fee_ppm: Optional[int] = None
    pending_baseline_fee_ppm: Optional[int] = None


@dataclass
class SurgeOverlayAction:
    channel_id: str
    action: str
    fee_ppm: int
    baseline_fee_ppm: Optional[int] = None


class RealtimeSurgeDefense:
    def __init__(
        self,
        *,
        enabled: bool,
        surge_window_seconds: int,
        surge_trigger_pct: float,
        surge_multiplier_min: float,
        surge_multiplier_max: float,
        surge_cooldown_seconds: int,
        surge_setchannel_min_interval_seconds: int,
        channel_capacity_msat: Callable[[str], Optional[int]],
        current_fee_ppm: Callable[[str], Optional[int]],
        apply_fee_callback: Callable[[str, int], bool],
        time_fn: Callable[[], float] = time.time,
        peer_volume_share_threshold: float = 0.80,
        peer_htlc_share_threshold: float = 0.80,
        max_fee_ppm: int = 5_000,
    ):
        self.enabled = enabled
        self.surge_window_seconds = surge_window_seconds
        self.surge_trigger_pct = surge_trigger_pct
        self.surge_multiplier_min = surge_multiplier_min
        self.surge_multiplier_max = surge_multiplier_max
        self.surge_cooldown_seconds = surge_cooldown_seconds
        self.surge_setchannel_min_interval_seconds = surge_setchannel_min_interval_seconds
        self.channel_capacity_msat = channel_capacity_msat
        self.current_fee_ppm = current_fee_ppm
        self.apply_fee_callback = apply_fee_callback
        self.time_fn = time_fn
        self.peer_volume_share_threshold = peer_volume_share_threshold
        self.peer_htlc_share_threshold = peer_htlc_share_threshold
        self.max_fee_ppm = max_fee_ppm

        self._samples: Dict[str, Deque[SurgeSample]] = defaultdict(deque)
        self._states: Dict[str, SurgeOverlayState] = {}
        self._pending_actions: Deque[SurgeOverlayAction] = deque()
        self._trigger_history: Deque[float] = deque()
        self._lock = threading.RLock()

    def ingest_sample(self, sample: SurgeSample) -> Optional[Dict[str, object]]:
        with self._lock:
            if not self.enabled:
                return None
            if not sample.outgoing_channel_id or sample.amount_msat <= 0:
                return None

            channel_id = sample.outgoing_channel_id
            samples = self._samples[channel_id]
            samples.append(sample)
            self._prune_samples(channel_id, sample.ts)

            metrics = self._compute_window_metrics(channel_id)
            if not self._should_trigger(metrics):
                return None

            state = self._states.get(channel_id)
            if state is None:
                state = SurgeOverlayState()
                self._states[channel_id] = state

            state.cooldown_until = max(
                state.cooldown_until,
                sample.ts + self.surge_cooldown_seconds,
            )
            state.last_trigger_reason = self._build_trigger_reason(metrics)

            if state.pending_action == "revert":
                self._clear_pending_action(channel_id)

            baseline_fee_ppm = self._resolve_baseline_fee(channel_id, state)
            if baseline_fee_ppm is None:
                self._cleanup_idle_channel(channel_id)
                return None

            target_fee_ppm = self._compute_bounded_surge_fee(
                baseline_fee_ppm,
                metrics["moved_pct"],
            )

            if sample.ts - state.last_attempt_ts < self.surge_setchannel_min_interval_seconds:
                state.last_apply_result = "debounced"
                return {
                    "channel_id": channel_id,
                    "target_fee_ppm": target_fee_ppm,
                    "result": state.last_apply_result,
                }

            if state.active and state.active_fee_ppm is not None and target_fee_ppm <= state.active_fee_ppm:
                state.last_apply_result = "cooldown_extended"
                return {
                    "channel_id": channel_id,
                    "target_fee_ppm": state.active_fee_ppm,
                    "result": state.last_apply_result,
                }

            if (
                state.pending_action == "apply"
                and state.pending_fee_ppm is not None
                and target_fee_ppm <= state.pending_fee_ppm
            ):
                state.last_apply_result = "queued"
                return {
                    "channel_id": channel_id,
                    "target_fee_ppm": state.pending_fee_ppm,
                    "result": state.last_apply_result,
                }

            self._enqueue_action(
                SurgeOverlayAction(
                    channel_id=channel_id,
                    action="apply",
                    fee_ppm=target_fee_ppm,
                    baseline_fee_ppm=baseline_fee_ppm,
                )
            )
            self._record_trigger(sample.ts)
            state.last_apply_result = "queued"

            return {
                "channel_id": channel_id,
                "target_fee_ppm": target_fee_ppm,
                "result": state.last_apply_result,
            }

    def process_pending_actions(self) -> None:
        with self._lock:
            now = self.time_fn()
            for channel_id in list(self._states):
                self._prune_samples(channel_id, now)
                state = self._states.get(channel_id)
                if state is None:
                    continue

                if (
                    state.active
                    and state.pending_action is None
                    and state.baseline_fee_ppm is not None
                    and now >= state.cooldown_until
                    and not self._should_trigger(self._compute_window_metrics(channel_id))
                ):
                    self._enqueue_action(
                        SurgeOverlayAction(
                            channel_id=channel_id,
                            action="revert",
                            fee_ppm=state.baseline_fee_ppm,
                        )
                    )

                self._cleanup_idle_channel(channel_id)

            pending_count = len(self._pending_actions)
            for _ in range(pending_count):
                action = self._pending_actions.popleft()
                state = self._states.get(action.channel_id)
                if state is None:
                    continue

                if not self._is_current_pending_action(state, action):
                    continue

                if now - state.last_attempt_ts < self.surge_setchannel_min_interval_seconds:
                    self._pending_actions.append(action)
                    continue

                if action.action == "apply":
                    metrics = self._compute_window_metrics(action.channel_id)
                    if not self._should_trigger(metrics):
                        self._clear_pending_action(action.channel_id)
                        state.last_apply_result = "stale_dropped"
                        self._cleanup_idle_channel(action.channel_id)
                        continue

                state.last_attempt_ts = self.time_fn()
                try:
                    ok = bool(self.apply_fee_callback(action.channel_id, action.fee_ppm))
                except Exception:
                    ok = False

                if action.action == "apply":
                    if ok:
                        self._clear_pending_action(action.channel_id)
                        state.active = True
                        state.baseline_fee_ppm = action.baseline_fee_ppm
                        state.active_fee_ppm = action.fee_ppm
                        state.last_apply_result = "applied"
                    else:
                        state.last_apply_result = "apply_failed"
                        state.pending_action = action.action
                        state.pending_fee_ppm = action.fee_ppm
                        state.pending_baseline_fee_ppm = action.baseline_fee_ppm
                        self._pending_actions.append(action)
                        if not state.active:
                            state.baseline_fee_ppm = None
                            state.active_fee_ppm = None
                elif action.action == "revert":
                    if ok:
                        self._clear_pending_action(action.channel_id)
                        state.active = False
                        state.baseline_fee_ppm = None
                        state.active_fee_ppm = None
                        state.cooldown_until = 0.0
                        state.last_trigger_reason = ""
                        state.last_apply_result = "reverted"
                    else:
                        state.last_apply_result = "revert_failed"
                        self._pending_actions.append(action)

                self._cleanup_idle_channel(action.channel_id)

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            now = self.time_fn()
            self._prune_trigger_history(now)
            channel_ids = set(self._samples) | set(self._states)
            channels: Dict[str, Dict[str, object]] = {}

            for channel_id in sorted(channel_ids):
                self._prune_samples(channel_id, now)
                self._cleanup_idle_channel(channel_id)
                if channel_id not in self._samples and channel_id not in self._states:
                    continue

                metrics = self._compute_window_metrics(channel_id)
                state = self._states.get(channel_id)
                channels[channel_id] = {
                    "active": state.active if state else False,
                    "baseline_fee_ppm": state.baseline_fee_ppm if state else None,
                    "active_fee_ppm": state.active_fee_ppm if state else None,
                    "cooldown_until": state.cooldown_until if state else 0.0,
                    "cooldown_remaining_sec": (
                        max(0.0, state.cooldown_until - now) if state else 0.0
                    ),
                    "cooldown_remaining_seconds": (
                        max(0.0, state.cooldown_until - now) if state else 0.0
                    ),
                    "last_trigger_reason": state.last_trigger_reason if state else "",
                    "last_apply_result": state.last_apply_result if state else "idle",
                    "last_attempt_ts": state.last_attempt_ts if state else 0.0,
                    **metrics,
                }

            return {
                "enabled": self.enabled,
                "active_channel_count": sum(
                    1 for state in self._states.values() if state.active
                ),
                "trigger_count_1h": self._count_recent_triggers(now, 3600),
                "trigger_count_24h": self._count_recent_triggers(now, 86400),
                "channels": channels,
            }

    def is_overlay_active(self, channel_id: str) -> bool:
        """Report whether a channel is currently surge-protected or has a queued overlay write."""
        if not channel_id:
            return False

        with self._lock:
            state = self._states.get(channel_id)
            return bool(state and (state.active or state.pending_action is not None))

    def _resolve_baseline_fee(
        self,
        channel_id: str,
        state: SurgeOverlayState,
    ) -> Optional[int]:
        if state.baseline_fee_ppm is not None:
            return state.baseline_fee_ppm
        if state.pending_action == "apply" and state.pending_baseline_fee_ppm is not None:
            return state.pending_baseline_fee_ppm

        baseline_fee_ppm = self.current_fee_ppm(channel_id)
        if baseline_fee_ppm is None:
            return None
        return int(baseline_fee_ppm)

    def _enqueue_action(self, action: SurgeOverlayAction) -> None:
        self._pending_actions = deque(
            queued
            for queued in self._pending_actions
            if queued.channel_id != action.channel_id
        )
        self._pending_actions.append(action)

        state = self._states.setdefault(action.channel_id, SurgeOverlayState())
        state.pending_action = action.action
        state.pending_fee_ppm = action.fee_ppm
        state.pending_baseline_fee_ppm = action.baseline_fee_ppm

    def _clear_pending_action(self, channel_id: str) -> None:
        self._pending_actions = deque(
            queued
            for queued in self._pending_actions
            if queued.channel_id != channel_id
        )
        state = self._states.get(channel_id)
        if state is None:
            return
        state.pending_action = None
        state.pending_fee_ppm = None
        state.pending_baseline_fee_ppm = None

    @staticmethod
    def _is_current_pending_action(
        state: SurgeOverlayState,
        action: SurgeOverlayAction,
    ) -> bool:
        return (
            state.pending_action == action.action
            and state.pending_fee_ppm == action.fee_ppm
            and state.pending_baseline_fee_ppm == action.baseline_fee_ppm
        )

    def _cleanup_idle_channel(self, channel_id: str) -> None:
        state = self._states.get(channel_id)
        if state is None:
            if channel_id in self._samples and not self._samples[channel_id]:
                self._samples.pop(channel_id, None)
            return

        has_samples = bool(self._samples.get(channel_id))
        if state.active or state.pending_action is not None or has_samples:
            return

        self._states.pop(channel_id, None)
        self._samples.pop(channel_id, None)

    def _prune_samples(self, channel_id: str, now: float) -> None:
        samples = self._samples.get(channel_id)
        if samples is None:
            return

        while samples and now - samples[0].ts > self.surge_window_seconds:
            samples.popleft()

        if not samples:
            self._samples.pop(channel_id, None)

    def _record_trigger(self, ts: float) -> None:
        self._trigger_history.append(ts)
        self._prune_trigger_history(ts)

    def _prune_trigger_history(self, now: float) -> None:
        while self._trigger_history and now - self._trigger_history[0] > 86400:
            self._trigger_history.popleft()

    def _count_recent_triggers(self, now: float, window_seconds: int) -> int:
        return sum(1 for ts in self._trigger_history if now - ts <= window_seconds)

    def _compute_window_metrics(self, channel_id: str) -> Dict[str, float]:
        samples = self._samples.get(channel_id, ())
        total_amount_msat = 0
        htlc_count = 0
        volume_by_peer: Dict[str, int] = defaultdict(int)
        count_by_peer: Dict[str, int] = defaultdict(int)

        for sample in samples:
            total_amount_msat += sample.amount_msat
            htlc_count += 1
            volume_by_peer[sample.incoming_peer_id] += sample.amount_msat
            count_by_peer[sample.incoming_peer_id] += 1

        channel_capacity_msat = self.channel_capacity_msat(channel_id) or 0
        moved_pct = (
            total_amount_msat / channel_capacity_msat
            if channel_capacity_msat > 0
            else 0.0
        )
        top_volume_share = (
            max(volume_by_peer.values()) / total_amount_msat
            if total_amount_msat > 0
            else 0.0
        )
        top_htlc_share = (
            max(count_by_peer.values()) / htlc_count
            if htlc_count > 0
            else 0.0
        )

        return {
            "moved_pct": moved_pct,
            "htlc_count": htlc_count,
            "top_incoming_peer_volume_share": top_volume_share,
            "top_incoming_peer_htlc_share": top_htlc_share,
        }

    def _should_trigger(self, metrics: Dict[str, float]) -> bool:
        if metrics["moved_pct"] < self.surge_trigger_pct:
            return False

        return (
            metrics["top_incoming_peer_volume_share"] >= self.peer_volume_share_threshold
            or metrics["top_incoming_peer_htlc_share"] >= self.peer_htlc_share_threshold
        )

    def _compute_bounded_surge_fee(self, baseline_fee_ppm: int, moved_pct: float) -> int:
        if baseline_fee_ppm <= 0:
            return 0

        severity = 0.0
        if self.surge_trigger_pct > 0:
            severity = (moved_pct - self.surge_trigger_pct) / self.surge_trigger_pct
        severity = min(max(severity, 0.0), 1.0)

        multiplier = self.surge_multiplier_min + (
            severity * (self.surge_multiplier_max - self.surge_multiplier_min)
        )
        surge_fee_ppm = int(math.ceil(baseline_fee_ppm * multiplier))
        return min(surge_fee_ppm, self.max_fee_ppm)

    @staticmethod
    def _build_trigger_reason(metrics: Dict[str, float]) -> str:
        return (
            "moved_pct="
            f"{metrics['moved_pct']:.4f},"
            "top_volume_share="
            f"{metrics['top_incoming_peer_volume_share']:.4f},"
            "top_htlc_share="
            f"{metrics['top_incoming_peer_htlc_share']:.4f}"
        )
