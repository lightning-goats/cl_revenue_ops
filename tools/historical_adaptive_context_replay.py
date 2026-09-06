"""Offline delayed-feedback, fixed-share-inspired historical-count mixture.

Research only: no RPC, database writes, runtime admission or economic claim.
Frozen prediction probabilities travel with each pending outcome; never
recompute a past forecast after fitting that outcome. This asynchronous capped
variant does not claim the synchronous Fixed Share regret theorem.
"""

from collections import Counter, defaultdict
import math

from tools.historical_online_context_replay import _Model
from tools.historical_route_context_replay import (
    DAY, NS, MAX_ROWS, HALF_LIFE_DAYS, SHRINKAGE, AMOUNT_BOUNDARIES_MSAT,
    HistoryError, _validate_events, validate_bounds,
)


SHARE = 0.01
HISTORICAL_CAP = 0.5
CONTEXTS = ("pooled", "outgoing", "outgoing_amount")
ARMS = ("cold", "warm", "fixed_half", "adaptive", "adaptive_capped")


class _Gate:
    def __init__(self, cap=1.0):
        if type(cap) not in (int, float) or not math.isfinite(cap) or not 0.5 <= cap <= 1:
            raise HistoryError("invalid historical influence cap")
        self.cap = cap
        self.weight = 0.5
        self.updates = 0

    def observe(self, warm_probability, cold_probability):
        """Use the original forecast's likelihood, on outcome availability.

        Probabilities must be strictly positive on the common outcome space.
        Rescale before multiplication to avoid tiny-likelihood underflow.
        Cap the new weight itself, not just an invisible proposal.
        """
        for probability in (warm_probability, cold_probability):
            if (type(probability) not in (int, float) or not math.isfinite(probability)
                    or not 0 < probability <= 1):
                raise HistoryError("invalid frozen forecast probability")
        scale = max(warm_probability, cold_probability)
        warm = self.weight * (warm_probability / scale)
        cold = (1 - self.weight) * (cold_probability / scale)
        posterior = warm / (warm + cold)
        self.weight = min(self.cap, SHARE + (1 - 2 * SHARE) * posterior)
        self.updates += 1


def evaluate_adaptive(rows, start, split, end):
    """Same base experts/alphabet as the prior online-count ablation.

    Gate updates happen in (settlement time, created index) order using saved
    pre-update forecasts. Equal-time outcomes are withheld from all predictions
    at that instant. Prefix arrivals that settle late update base counts but
    not gate losses, because no pre-split gate forecast was issued for them.
    """
    validate_bounds(start, split, end)
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise HistoryError("bounded event list required")
    if any(not isinstance(row, dict) for row in rows):
        raise HistoryError("malformed settled event")
    _validate_events(rows, start, end)
    available = sorted((r for r in rows if r["resolved_time_ns"] < end * NS),
                       key=lambda r: (r["resolved_time_ns"], r["created_index"]))
    prefix = [r for r in available if r["resolved_time_ns"] < split * NS]
    test = sorted((r for r in available if r["received_time_ns"] >= split * NS),
                  key=lambda r: (r["received_time_ns"], r["created_index"]))
    metadata = {"schema_version": 1, "source": "canonical_archive_only",
                "scope": "adaptive_incoming_adjacency_conditional_on_settlement",
                "start": start, "split": split, "end": end,
                "share": SHARE, "historical_cap": HISTORICAL_CAP,
                "half_life_days": HALF_LIFE_DAYS, "shrinkage": SHRINKAGE,
                "amount_boundaries_msat": list(AMOUNT_BOUNDARIES_MSAT),
                "bootstrap_events": len(prefix), "test_events": len(test),
                "late_prefix_events": len(available) - len(prefix) - len(test),
                "unresolved_at_end_events": len(rows) - len(available),
                "alphabet": "shared_prefix_then_settlement_available_expansion"}
    if not prefix or not test:
        return {**metadata, "status": "insufficient_evidence", "scores": None}
    warm, cold = _Model(start * NS), _Model(start * NS)
    vocabulary = {r["in_channel"] for r in prefix}
    for row in prefix:
        warm.update(row)
    gates = {"adaptive": {key: _Gate() for key in CONTEXTS},
             "adaptive_capped": {key: _Gate(HISTORICAL_CAP) for key in CONTEXTS}}
    weights = {arm: {key: {"sum": 0.0, "min": 1.0, "max": 0.0}
                     for key in CONTEXTS} for arm in gates}
    losses = {arm: Counter() for arm in ARMS}
    days = defaultdict(lambda: {"count": 0, **{arm: Counter() for arm in ARMS}})
    cursor = len(prefix)
    pending = {}
    max_pending = 0
    unknown = 0
    for row in test:
        while cursor < len(available) and available[cursor]["resolved_time_ns"] < row["received_time_ns"]:
            observed = available[cursor]
            forecast = pending.pop(observed["created_index"], None)
            if forecast is not None:
                for group in gates.values():
                    for key, gate in group.items():
                        gate.observe(forecast["warm"][key], forecast["cold"][key])
            warm.update(observed)
            cold.update(observed)
            vocabulary.add(observed["in_channel"])
            cursor += 1
        unknown += row["in_channel"] not in vocabulary
        forecast = {"warm": warm.predict(row, vocabulary), "cold": cold.predict(row, vocabulary)}
        pending[row["created_index"]] = forecast
        max_pending = max(max_pending, len(pending))
        daily = days[row["received_time_ns"] // (DAY * NS) * DAY]
        daily["count"] += 1
        for key in CONTEXTS:
            warm_p, cold_p = forecast["warm"][key], forecast["cold"][key]
            probabilities = {"warm": warm_p, "cold": cold_p, "fixed_half": (warm_p + cold_p) / 2}
            for arm, group in gates.items():
                weight = group[key].weight
                probabilities[arm] = weight * warm_p + (1 - weight) * cold_p
                summary = weights[arm][key]
                summary["sum"] += weight
                summary["min"] = min(summary["min"], weight)
                summary["max"] = max(summary["max"], weight)
            for arm, probability in probabilities.items():
                loss = -math.log2(probability)
                losses[arm][key] += loss
                daily[arm][key] += loss
    scores = {arm: {key: losses[arm][key] / len(test) for key in CONTEXTS} for arm in ARMS}
    return {**metadata, "status": "evaluated", "scores": scores,
            "unknown_incoming_events": unknown,
            "max_pending_forecasts": max_pending,
            "pending_at_last_prediction": len(pending),
            "updates_before_last_prediction": {"warm": warm.updates, "cold": cold.updates,
                "gate": gates["adaptive"]["pooled"].updates},
            "mean_historical_weights": {arm: {key: summary["sum"] / len(test)
                for key, summary in group.items()} for arm, group in weights.items()},
            "historical_weight_ranges": {arm: {key: [summary["min"], summary["max"]]
                for key, summary in group.items()} for arm, group in weights.items()},
            "minus_cold_bits": {arm: {key: scores[arm][key] - scores["cold"][key]
                for key in CONTEXTS} for arm in ARMS if arm != "cold"},
            "daily": [{"day": day, "events": value["count"],
                "minus_cold_bits": {arm: {key: (value[arm][key] - value["cold"][key]) / value["count"]
                    for key in CONTEXTS} for arm in ARMS if arm != "cold"}}
                for day, value in sorted(days.items())]}
