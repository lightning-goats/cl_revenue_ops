"""Read-only historical role-credit overlap, not a marginal-profit estimator.

Incoming/source and outgoing/destination credits refer to the same fee on an
intersecting forward. Count each receipt once in a pair's historical union.
Do not sum pair unions across a node, or treat them as rebalance-attributable
income. This research helper has no RPC, runtime caller or database writes.
"""

from collections import Counter
import statistics

from tools.historical_route_context_replay import (
    DAY, NS, MAX_ROWS, HistoryError, _validate_events,
)


def _amount(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise HistoryError("credit must be a nonnegative bounded integer")
    return value


def paired_credit(destination_msat, source_msat, overlap_msat):
    """Union of SAME-horizon, SAME-unit marginal credits, if intersection known.

    Caller must establish the common evidence basis. Unknown intersection is
    an interval, not zero. Historical overlap is not causal forecast overlap.
    """
    destination, source = _amount(destination_msat), _amount(source_msat)
    summed = _amount(destination + source)
    if overlap_msat is None:
        return {"status": "unknown_overlap", "unique_credit_msat": None,
                "lower_bound_msat": max(destination, source), "upper_bound_msat": summed}
    overlap = _amount(overlap_msat)
    if overlap > min(destination, source):
        raise HistoryError("intersection exceeds a marginal credit")
    return {"status": "known_overlap", "unique_credit_msat": summed - overlap,
            "lower_bound_msat": summed - overlap, "upper_bound_msat": summed - overlap}


def audit_pair_credits(rows, start, end):
    """Aggregate-only audit over exact archive channel labels, not live pairs.

    Includes historical closed channels; no live eligibility, alias continuity,
    action exposure or prediction is inferred. Unknown current channels do not
    silently become test controls. Only already settled-by-end events count.
    """
    if (any(type(v) is not int or v < 0 or v % DAY for v in (start, end))
            or not start < end or end - start > 400 * DAY):
        raise HistoryError("bounded whole UTC days required")
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise HistoryError("bounded event list required")
    if any(not isinstance(row, dict) for row in rows):
        raise HistoryError("malformed settled event")
    _validate_events(rows, start, end)
    received, sent, corridors = Counter(), Counter(), Counter()
    received_count, sent_count, corridor_count = Counter(), Counter(), Counter()
    total_fee = 0
    withheld = 0
    for row in rows:
        if row["resolved_time_ns"] >= end * NS:
            withheld += 1
            continue
        incoming, outgoing, fee = row["in_channel"], row["out_channel"], row["fee_msat"]
        total_fee = _amount(total_fee + fee)
        received[incoming] += fee
        sent[outgoing] += fee
        corridors[incoming, outgoing] += fee
        received_count[incoming] += 1
        sent_count[outgoing] += 1
        corridor_count[incoming, outgoing] += 1
    shares, count_fee_gaps = [], []
    dominant_both = count = zero_fee = self_pairs = 0
    for (incoming, outgoing), fee in corridors.items():
        if incoming == outgoing:
            self_pairs += 1
            continue
        count += 1
        source_fee, dest_fee = received[incoming], sent[outgoing]
        credit = paired_credit(dest_fee, source_fee, fee)
        assert credit["unique_credit_msat"] >= max(source_fee, dest_fee)
        # A zero-fee corridor can still have a well-defined fee share when
        # another corridor earned fees on this destination. Do not exclude
        # those observations from the count-vs-fee comparison.
        if dest_fee > 0:
            count_fraction = corridor_count[incoming, outgoing] / sent_count[outgoing]
            fee_fraction = fee / dest_fee
            count_fee_gaps.append(abs(count_fraction - fee_fraction))
        if not fee:
            zero_fee += 1
            continue
        shares.append(fee / (source_fee + dest_fee))
        dominant_both += (2 * fee >= source_fee and 2 * fee >= dest_fee)
    return {"schema_version": 1, "scope": "observed_pair_historical_fee_credit_union",
            "source": "canonical_archive_only_exact_channel_labels",
            "start": start, "end": end, "settled_events": len(rows) - withheld,
            "unresolved_at_end_events": withheld, "total_fee_msat": total_fee,
            "observed_distinct_channel_pairs": count, "excluded_self_pairs": self_pairs,
            "zero_fee_pairs": zero_fee, "positive_overlap_pairs": len(shares),
            "overlap_fraction_of_summed_role_credits": None if not shares else {
                "min": min(shares), "median": statistics.median(shares), "max": max(shares),
                "at_least_10_percent_pairs": sum(v >= 0.1 for v in shares),
                "at_least_25_percent_pairs": sum(v >= 0.25 for v in shares)},
            "overlap_at_least_half_of_each_role_pairs": dominant_both,
            "event_fee_share_defined_pairs": len(count_fee_gaps),
            "absolute_event_share_minus_fee_share": None if not count_fee_gaps else {
                "median": statistics.median(count_fee_gaps), "max": max(count_fee_gaps)},
            "status": "audited" if len(rows) > withheld else "insufficient_evidence",
            "causal_rebalance_value_estimated": False, "production_earnings_loss_estimated": False}
