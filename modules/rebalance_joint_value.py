"""Pure bounds on joint fee credit; no source, forecast or action authority.

The bounds assume nonnegative credits on one common future evidence basis.
They do not certify the incumbent's marginal forecasts or causal attribution.
Unknown overlap is not silently set to zero; disjointness needs separate proof.
"""

import math

MODEL_VERSION = "v3-joint-lower-bound"


def joint_credit_bounds(destination_sats, source_sats):
    """Return conservative lower/upper/overlap bounds, or None for bad evidence."""
    values = []
    for value in (destination_sats, source_sats):
        if type(value) not in (int, float):
            return None
        try:
            number = float(value)
        except (ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number < 0:
            return None
        values.append(number)
    destination, source = values
    upper = destination + source
    if not math.isfinite(upper):
        return None
    return {"lower_sats": max(destination, source), "upper_sats": upper,
            "overlap_upper_sats": min(destination, source),
            "overlap_status": "zero" if min(destination, source) == 0 else "unknown"}
