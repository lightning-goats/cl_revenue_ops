"""Pinned local fee-policy mechanics, not a simulated competitor or scorer."""

from dataclasses import replace

import pytest

from modules.fee_acceptance_model import (
    FeeAcceptanceState, FeePolicy, FeePolicyRequest, after_cln_restart,
    apply_setchannel, assess_policy_checks,
)


NS = 1_000_000_000
OLD = FeePolicy(775, 0, 1, 500_000_000)
NEW = replace(OLD, fee_ppm=856)


def initial():
    return after_cln_restart(OLD, 0)


def apply_effective_policy(state, policy, at_ns, **kwargs):
    # These fixtures explicitly request all four values without clamping.
    request = FeePolicyRequest(**vars(policy)) if policy is not None else None
    return apply_setchannel(state, policy, at_ns, request=request, **kwargs)


def assess(state, seconds, amount=250_000_000, fee=193_750):
    return assess_policy_checks(state, at_ns=seconds * NS, out_msat=amount, fee_msat=fee)


def test_old_policy_earnings_are_not_new_quote_revenue():
    state = apply_effective_policy(initial(), NEW, 100 * NS)
    result = assess(state, 105)
    assert result["policy_checks_pass"] is True
    assert result["fee_check"] == "previous"
    assert result["actual_fee_msat"] == 193_750
    assert result["current_minimum_fee_msat"] == 214_000
    assert result["payer_policy_known"] is False


def test_exact_grace_expiry_boundary():
    state = apply_effective_policy(initial(), NEW, 100 * NS)
    assert assess(state, 699)["policy_checks_pass"] is True
    assert assess(state, 700)["policy_checks_pass"] is False


def test_htlc_restriction_replaces_old_fee_slot_even_without_price_change():
    raised = apply_effective_policy(initial(), NEW, 100 * NS)
    assert assess(raised, 109)["fee_check"] == "previous"
    restricted = apply_effective_policy(raised, replace(NEW, htlc_max_msat=300_000_000), 110 * NS)
    assert restricted.previous == NEW
    assert restricted.previous_until_ns == 710 * NS
    assert assess(restricted, 111)["fee_check"] == "rejected"
    # Changing an admission limit can end compatibility with the original fee.
    assert raised.previous == OLD  # Immutable prior replay state is preserved.


@pytest.mark.parametrize("change", [{"fee_ppm": 900}, {"base_fee_msat": 20},
                                   {"htlc_min_msat": 2}, {"htlc_max_msat": 400_000_000}])
def test_each_restrictive_control_resets_the_single_old_slot(change):
    state = apply_effective_policy(initial(), replace(OLD, **change), 100 * NS)
    assert state.previous == OLD and state.previous_until_ns == 700 * NS


def test_second_fee_increase_does_not_keep_arbitrary_older_prices():
    state = apply_effective_policy(initial(), NEW, 100 * NS)
    state = apply_effective_policy(state, replace(NEW, fee_ppm=1000), 110 * NS)
    assert assess(state, 111)["policy_checks_pass"] is False
    assert assess(state, 111, fee=214_000)["fee_check"] == "previous"


def test_permissive_changes_keep_prior_timer_and_overpayment_is_not_identity():
    state = apply_effective_policy(initial(), NEW, 100 * NS)
    state = apply_effective_policy(state, replace(NEW, fee_ppm=700), 110 * NS)
    assert state.previous == OLD and state.previous_until_ns == 700 * NS
    result = assess(state, 111)
    # Payment exactly matching the old 775-ppm quote also passes current 700.
    assert result["fee_check"] == "current"
    assert result["payer_policy_known"] is False


def test_fee_and_amount_can_pass_different_policy_slots():
    old = FeePolicy(800, 0, 1, 500_000_000)
    new = FeePolicy(700, 0, 1, 200_000_000)
    state = apply_effective_policy(after_cln_restart(old, 0), new, 100 * NS)
    result = assess(state, 101, fee=175_000)
    assert result["policy_checks_pass"] is True
    assert result["fee_check"] == "current" and result["amount_check"] == "previous"


def test_restart_semantics_and_no_extra_mutation():
    state = apply_effective_policy(initial(), NEW, 100 * NS)
    assert assess(state, 101)["policy_checks_pass"] is True
    restarted = after_cln_restart(NEW, 101 * NS)
    assert assess(restarted, 102)["policy_checks_pass"] is False
    # Merely observing/reusing state (including plugin restart) cannot clear grace.
    assert assess(state, 102)["policy_checks_pass"] is True


def test_incomplete_prior_state_is_unknown_not_automatically_rejected():
    state = FeeAcceptanceState(NEW, 100 * NS)
    assert assess(state, 101)["policy_checks_pass"] is None
    assert assess(state, 101, fee=214_000)["policy_checks_pass"] is True


def test_base_fee_and_msat_rounding_are_exact():
    policy = FeePolicy(775, 19, 0, 500_000_000)
    assert policy.minimum_fee_msat(123_456) == 114
    state = after_cln_restart(policy, 0)
    assert assess(state, 1, amount=123_456, fee=114)["policy_checks_pass"] is True
    assert assess(state, 1, amount=123_456, fee=113)["policy_checks_pass"] is False


@pytest.mark.parametrize("bad", [None, True, -1, 1.1, "600", float("nan"), 2**40])
def test_unknown_delay_is_not_silently_replaced_with_default(bad):
    with pytest.raises(ValueError):
        apply_effective_policy(initial(), NEW, 100 * NS, enforce_delay_seconds=bad)


def test_explicit_zero_delay_is_modeled_but_never_sent_to_a_node():
    state = apply_effective_policy(initial(), NEW, 100 * NS, enforce_delay_seconds=0)
    assert assess(state, 100)["policy_checks_pass"] is False


@pytest.mark.parametrize("bad", [None, True, -1, 1.1, "100", float("inf"), 2**80])
def test_malformed_amounts_are_not_zero_evidence(bad):
    with pytest.raises(ValueError):
        assess_policy_checks(initial(), at_ns=NS, out_msat=bad, fee_msat=0)


def test_nonmonotonic_timeline_and_unknown_policy_rejected():
    state = apply_effective_policy(initial(), NEW, 100 * NS)
    with pytest.raises(ValueError):
        apply_effective_policy(state, OLD, 99 * NS)
    with pytest.raises(ValueError):
        assess(state, 99)
    with pytest.raises(ValueError):
        apply_effective_policy(state, None, 101 * NS)


@pytest.mark.parametrize("field,requested,effective", [
    ("htlc_min_msat", 1, 2),
    ("htlc_max_msat", 500_000_000, 400_000_000),
])
def test_clamped_effective_restriction_does_not_invent_requested_restriction(
        field, requested, effective):
    state = apply_setchannel(initial(), replace(OLD, **{field: effective}), NS,
                             request=FeePolicyRequest(**{field: requested}))
    assert state.previous is None and state.previous_known is True


def test_omitted_fields_preserve_policy_and_old_slot():
    raised = apply_effective_policy(initial(), NEW, NS)
    state = apply_setchannel(raised, NEW, 2 * NS, request=FeePolicyRequest())
    assert state.previous == OLD and state.previous_until_ns == 601 * NS


@pytest.mark.parametrize("policy,policy_request", [
    (NEW, FeePolicyRequest()),
    (NEW, FeePolicyRequest(fee_ppm=800)),
    (OLD, FeePolicyRequest(htlc_min_msat=2)),
    (OLD, FeePolicyRequest(htlc_max_msat=400_000_000)),
    (OLD, None),
])
def test_inconsistent_or_unknown_request_is_not_reconstructed(policy, policy_request):
    with pytest.raises(ValueError):
        apply_setchannel(initial(), policy, NS, request=policy_request)


@pytest.mark.parametrize("field", ["fee_ppm", "base_fee_msat", "htlc_min_msat", "htlc_max_msat"])
@pytest.mark.parametrize("bad", [True, -1, 1.1, "1", float("nan"), 2**80])
def test_malformed_requested_fields_rejected(field, bad):
    with pytest.raises(ValueError):
        FeePolicyRequest(**{field: bad})
