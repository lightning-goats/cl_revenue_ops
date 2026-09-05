"""Pure research model of CLN v26.06.7 fee/HTLC policy checks.

Source: ElementsProject/lightning at 9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911,
peer_control.c:set_channel_config and peer_htlcs.c forwarding checks. Not wired into
runtime decisions. Inputs require a complete, ordered CLN-instance timeline
and both requested fields and effective policies, not wall-clock guesses.
Passing these checks does not prove liquidity, a successful route, or which
advertised policy the payer saw. There is no RPC, clock or learning mutation.
"""

from dataclasses import dataclass


CLN_SOURCE_REF = "9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911"
DEFAULT_ENFORCE_DELAY_SECONDS = 600
MAX_MSAT = 2**63 - 1  # Local SQLite evidence domain, narrower than CLN's u64.


def _integer(value, name, maximum=MAX_MSAT):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a nonnegative bounded integer")
    return value


@dataclass(frozen=True)
class FeePolicy:
    fee_ppm: int
    base_fee_msat: int
    htlc_min_msat: int
    htlc_max_msat: int

    def __post_init__(self):
        _integer(self.fee_ppm, "fee_ppm", 2**32 - 1)
        _integer(self.base_fee_msat, "base_fee_msat", 2**32 - 1)
        _integer(self.htlc_min_msat, "htlc_min_msat")
        _integer(self.htlc_max_msat, "htlc_max_msat")
        if self.htlc_min_msat > self.htlc_max_msat:
            raise ValueError("inverted HTLC range")

    def minimum_fee_msat(self, amount_msat: int) -> int:
        _integer(amount_msat, "amount_msat")
        return _integer(self.base_fee_msat + amount_msat * self.fee_ppm // 1_000_000,
                        "minimum_fee_msat")

    def permits_amount(self, amount_msat: int) -> bool:
        _integer(amount_msat, "amount_msat")
        return self.htlc_min_msat <= amount_msat <= self.htlc_max_msat


@dataclass(frozen=True)
class FeeAcceptanceState:
    current: FeePolicy
    at_ns: int
    previous: FeePolicy | None = None
    previous_until_ns: int | None = None
    previous_known: bool = False

    def __post_init__(self):
        if not isinstance(self.current, FeePolicy):
            raise ValueError("effective current policy required")
        _integer(self.at_ns, "at_ns")
        if type(self.previous_known) is not bool:
            raise ValueError("previous_known must be boolean")
        if self.previous is not None:
            if not self.previous_known or not isinstance(self.previous, FeePolicy):
                raise ValueError("previous policy requires known provenance")
            _integer(self.previous_until_ns, "previous_until_ns")
        elif self.previous_until_ns is not None:
            raise ValueError("expiry without previous policy")


@dataclass(frozen=True)
class FeePolicyRequest:
    """Known setchannel arguments: None means omitted, never unknown."""

    fee_ppm: int | None = None
    base_fee_msat: int | None = None
    htlc_min_msat: int | None = None
    htlc_max_msat: int | None = None

    def __post_init__(self):
        for name in ("fee_ppm", "base_fee_msat", "htlc_min_msat", "htlc_max_msat"):
            value = getattr(self, name)
            if value is not None:
                maximum = 2**32 - 1 if name in ("fee_ppm", "base_fee_msat") else MAX_MSAT
                _integer(value, name, maximum)


def apply_setchannel(
    state: FeeAcceptanceState, policy: FeePolicy, at_ns: int,
    *, request: FeePolicyRequest,
    enforce_delay_seconds: int = DEFAULT_ENFORCE_DELAY_SECONDS,
) -> FeeAcceptanceState:
    """Replay one successful setchannel transition, never perform it.

    CLN has ONE previous-policy slot. Requested fee/base/minimum increases or
    maximum decreases replace that slot before peer/capacity clamping. Looking
    only at effective values can therefore reconstruct the wrong grace state.
    Permissive requests preserve the existing slot and deadline.
    Callers must account for external changes and RPC uncertainty;
    this model cannot fill missing history with today's state.
    """
    if (not isinstance(state, FeeAcceptanceState) or not isinstance(policy, FeePolicy)
            or not isinstance(request, FeePolicyRequest)):
        raise ValueError("validated state, request and effective policy required")
    _integer(at_ns, "at_ns")
    _integer(enforce_delay_seconds, "enforce_delay_seconds", 2**32 - 1)
    if at_ns < state.at_ns:
        raise ValueError("timeline moved backwards")
    old = state.current
    for name in ("fee_ppm", "base_fee_msat", "htlc_min_msat", "htlc_max_msat"):
        requested, effective = getattr(request, name), getattr(policy, name)
        if requested is None:
            valid = effective == getattr(old, name)
        elif name == "htlc_min_msat":
            valid = effective >= requested
        elif name == "htlc_max_msat":
            valid = effective <= requested
        else:
            valid = effective == requested
        if not valid:
            raise ValueError("effective policy inconsistent with known request")
    restrictive = any(
        getattr(request, name) is not None
        and ((getattr(request, name) < getattr(old, name)) if name == "htlc_max_msat"
             else (getattr(request, name) > getattr(old, name)))
        for name in ("fee_ppm", "base_fee_msat", "htlc_min_msat", "htlc_max_msat")
    )
    if restrictive:
        until = _integer(at_ns + enforce_delay_seconds * 1_000_000_000, "expiry")
        return FeeAcceptanceState(policy, at_ns, old, until, True)
    return FeeAcceptanceState(policy, at_ns, state.previous,
                              state.previous_until_ns, state.previous_known)


def after_cln_restart(policy: FeePolicy, at_ns: int) -> FeeAcceptanceState:
    """Explicit CLN-daemon restart clears grace; a plugin restart does not."""
    return FeeAcceptanceState(policy, at_ns, previous_known=True)


def assess_policy_checks(state: FeeAcceptanceState, *, at_ns: int,
                         out_msat: int, fee_msat: int) -> dict:
    """Assess fee and amount constraints independently, as pinned CLN does.

    fee_msat is actual observed payment, not a reconstructed quote. A known
    current policy can pass without known prior state; a failed current check
    needs known previous state before declaring policy rejection. Fee and
    amount may legitimately pass different slots. Never infer payer exposure.
    """
    if not isinstance(state, FeeAcceptanceState):
        raise ValueError("validated state required")
    _integer(at_ns, "at_ns")
    _integer(out_msat, "out_msat")
    _integer(fee_msat, "fee_msat")
    if at_ns < state.at_ns:
        raise ValueError("use the state that existed at forward time")
    current_min = state.current.minimum_fee_msat(out_msat)
    previous_active = (state.previous is not None and at_ns < state.previous_until_ns)

    def check(current_ok, predicate):
        if current_ok:
            return "current"
        if not state.previous_known:
            return "unknown"
        if previous_active and predicate(state.previous):
            return "previous"
        return "rejected"

    fee_check = check(fee_msat >= current_min,
                      lambda policy: fee_msat >= policy.minimum_fee_msat(out_msat))
    amount_check = check(state.current.permits_amount(out_msat),
                         lambda policy: policy.permits_amount(out_msat))
    checks = (fee_check, amount_check)
    passes = False if "rejected" in checks else None if "unknown" in checks else True
    return {"policy_checks_pass": passes, "fee_check": fee_check,
            "amount_check": amount_check, "actual_fee_msat": fee_msat,
            "current_minimum_fee_msat": current_min,
            "payer_policy_known": False}
