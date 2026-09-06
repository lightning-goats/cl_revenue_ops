"""Offline prequential warm-start ablation; no runtime import, RPC or writes.

Predict incoming adjacency conditional on an eventually settled forward.
Settlement timestamps approximate availability, not actual notification receipt.
This is neither demand/price learning nor an economic qualification.
"""

from collections import Counter, defaultdict
import math

from tools.historical_route_context_replay import (
    DAY, NS, MAX_ROWS, HALF_LIFE_DAYS, SHRINKAGE, AMOUNT_BOUNDARIES_MSAT,
    HistoryError, _validate_events, validate_bounds,
)


class _Model:
    """Lazy exponential decay with bounded (<=400 day) time coordinates."""

    def __init__(self, origin_ns):
        self.origin_ns = origin_ns
        self.pooled = Counter()
        self.outgoing = defaultdict(Counter)
        self.amount = defaultdict(Counter)
        self.total = 0.0
        self.out_totals = Counter()
        self.amount_totals = Counter()
        self.updates = 0

    @staticmethod
    def key(row):
        return (row["out_channel"], sum(row["out_msat"] >= boundary
                                      for boundary in AMOUNT_BOUNDARIES_MSAT))

    def scale(self, when_ns):
        return 2 ** ((when_ns - self.origin_ns) / (DAY * NS * HALF_LIFE_DAYS))

    def update(self, row):
        weight = self.scale(row["resolved_time_ns"])
        incoming, outgoing = row["in_channel"], row["out_channel"]
        key = self.key(row)
        self.pooled[incoming] += weight
        self.outgoing[outgoing][incoming] += weight
        self.amount[key][incoming] += weight
        self.total += weight
        self.out_totals[outgoing] += weight
        self.amount_totals[key] += weight
        self.updates += 1

    def predict(self, row, vocabulary):
        # Both arms use the SAME causally available alphabet. Otherwise an
        # empty cold model could score every label as probability-one UNKNOWN,
        # spuriously beating a warm model predicting a finer outcome space.
        scale = self.scale(row["received_time_ns"])
        incoming, outgoing = row["in_channel"], row["out_channel"]
        pooled = (self.pooled.get(incoming, 0) / scale + 1) / (
            self.total / scale + len(vocabulary) + 1)
        out_p = (self.outgoing.get(outgoing, {}).get(incoming, 0) / scale
                 + SHRINKAGE * pooled) / (self.out_totals.get(outgoing, 0) / scale + SHRINKAGE)
        key = self.key(row)
        amount_p = (self.amount.get(key, {}).get(incoming, 0) / scale
                    + SHRINKAGE * out_p) / (self.amount_totals.get(key, 0) / scale + SHRINKAGE)
        return {"pooled": pooled, "outgoing": out_p, "outgoing_amount": amount_p}


def evaluate_online(rows, start, split, end):
    """Score before updating; equal-time outcomes are unavailable to prediction.

    Warm counts use all settled prefix evidence; cold counts start at zero.
    Both arms receive identical post-split outcomes (including prefix arrivals
    settling after split) and share prefix-known then causally expanding labels.
    Thus this isolates historical COUNTS, not all historical side information.
    No inventory or fee exposure is reconstructed from present-day values.
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
                "scope": "prequential_incoming_adjacency_conditional_on_settlement",
                "start": start, "split": split, "end": end,
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
    cursor = len(prefix)
    losses = {arm: Counter() for arm in ("warm", "cold")}
    day_losses = defaultdict(lambda: {"count": 0, "warm": Counter(), "cold": Counter()})
    unknown = 0
    for row in test:
        while cursor < len(available) and available[cursor]["resolved_time_ns"] < row["received_time_ns"]:
            observed = available[cursor]
            warm.update(observed)
            cold.update(observed)
            vocabulary.add(observed["in_channel"])
            cursor += 1
        unknown += row["in_channel"] not in vocabulary
        daily = day_losses[row["received_time_ns"] // (DAY * NS) * DAY]
        daily["count"] += 1
        for arm, model in (("warm", warm), ("cold", cold)):
            for context, probability in model.predict(row, vocabulary).items():
                loss = -math.log2(probability)
                losses[arm][context] += loss
                daily[arm][context] += loss
    scores = {arm: {context: total / len(test) for context, total in values.items()}
              for arm, values in losses.items()}
    return {**metadata, "status": "evaluated", "scores": scores,
            "unknown_incoming_events": unknown,
            "updates_before_last_prediction": {"warm": warm.updates, "cold": cold.updates},
            "warm_minus_cold_bits": {key: scores["warm"][key] - scores["cold"][key]
                                     for key in scores["warm"]},
            "daily": [{"day": day, "events": value["count"],
                       "warm_minus_cold_bits": {
                           key: (value["warm"][key] - value["cold"][key]) / value["count"]
                           for key in value["warm"]}}
                      for day, value in sorted(day_losses.items())]}
