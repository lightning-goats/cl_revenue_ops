"""Phase 1: typed intent envelope, deterministic idempotency, wire form."""
import pytest

from modules.econ_intents import (
    INTENT_TYPES,
    Explanation,
    IntentEnvelope,
    compute_idempotency_key,
    from_wire,
    is_expired,
    make_intent,
    to_wire,
)
from modules.econ_types import (
    EconArithmeticError,
    Micro,
    Msat,
    SignedMsat,
    UnixTime,
)


def _fields(**over):
    base = dict(
        intent_type="SET_FEE",
        snapshot_id="snap-000001",
        created_at=UnixTime(1_752_400_000),
        expires_at=UnixTime(1_752_400_600),
        target="123x456x0",
        amount_msat=Msat(250),
        expected_benefit_msat=SignedMsat(1_000),
        max_cost_msat=Msat(0),
        capital_committed_msat=Msat(0),
        confidence_micro=Micro(900_000),
        reason_codes=(),
        explanation=Explanation(
            "fee_target", (("baseline_ppm", 200), ("target_ppm", 250))),
        preconditions=("channel_active",),
        priority=50,
        budget_bucket="fees",
        origin_policy="fee_policy",
        reversible=True,
    )
    base.update(over)
    return base


@pytest.mark.parametrize("intent_type", INTENT_TYPES)
def test_all_intent_types_construct(intent_type):
    env = make_intent(**_fields(intent_type=intent_type))
    assert env.intent_type == intent_type
    assert env.intent_id.value.startswith("int-")


def test_invalid_type_rejected():
    with pytest.raises(EconArithmeticError):
        make_intent(**_fields(intent_type="RUN_AWAY"))


def test_expiry_must_follow_creation():
    with pytest.raises(EconArithmeticError):
        make_intent(**_fields(expires_at=UnixTime(1_752_400_000)))


def test_unknown_reason_code_rejected():
    with pytest.raises(EconArithmeticError):
        make_intent(**_fields(reason_codes=("NOT_A_CODE",)))
    env = make_intent(**_fields(reason_codes=("COOLDOWN_ACTIVE",)))
    assert env.reason_codes == ("COOLDOWN_ACTIVE",)


def test_priority_bounds():
    with pytest.raises(EconArithmeticError):
        make_intent(**_fields(priority=101))
    with pytest.raises(EconArithmeticError):
        make_intent(**_fields(priority=-1))


def test_idempotency_key_pinned_and_order_insensitive():
    key1 = compute_idempotency_key(
        intent_type="SET_FEE", target="123x456x0", amount_msat=250,
        snapshot_id="snap-000001", budget_bucket="fees")
    key2 = compute_idempotency_key(
        budget_bucket="fees", snapshot_id="snap-000001", amount_msat=250,
        target="123x456x0", intent_type="SET_FEE")
    assert key1 == key2
    assert len(key1) == 64 and int(key1, 16) >= 0
    # Characterization pin (J3): recorded once, then frozen. Changing the
    # canonical-serialization or subset definition breaks this — that is
    # the point.
    from modules.econ_snapshot import canonical_json
    import hashlib
    expected = hashlib.sha256(canonical_json({
        "amount_msat": 250, "budget_bucket": "fees",
        "intent_type": "SET_FEE", "snapshot_id": "snap-000001",
        "target": "123x456x0",
    }).encode("utf-8")).hexdigest()
    assert key1 == expected


def test_key_differs_by_amount():
    base = dict(intent_type="SET_FEE", target="123x456x0",
                snapshot_id="snap-000001", budget_bucket="fees")
    assert (compute_idempotency_key(amount_msat=250, **base)
            != compute_idempotency_key(amount_msat=251, **base))


def test_same_inputs_same_intent_id():
    a = make_intent(**_fields())
    b = make_intent(**_fields())
    assert a.intent_id == b.intent_id
    assert a.idempotency_key == b.idempotency_key


def test_is_expired_boundary():
    env = make_intent(**_fields())
    assert not is_expired(env, UnixTime(1_752_400_599))
    assert is_expired(env, UnixTime(1_752_400_600))  # now == expires_at
    assert is_expired(env, UnixTime(1_752_400_601))


def test_wire_round_trip():
    env = make_intent(**_fields(reason_codes=("COOLDOWN_ACTIVE",)))
    assert from_wire(to_wire(env)) == env
    wire = to_wire(env)
    assert wire["schema_name"] == "intent"
    # v0 -> v1 emission cutover (2026-08-12 compatibility window)
    assert wire["schema_version"] == 1
    assert wire["amount_msat"] == 250
    assert wire["explanation"]["kind"] == "fee_target"


def test_explanation_render():
    exp = Explanation("fee_target", (("baseline_ppm", 200),
                                     ("target_ppm", 250)))
    assert exp.render() == "fee_target: baseline_ppm=200, target_ppm=250"


def test_none_amount_allowed():
    env = make_intent(**_fields(amount_msat=None))
    assert env.amount_msat is None
    assert to_wire(env)["amount_msat"] is None


def test_wire_form_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    import json
    import pathlib
    schema = json.loads(
        (pathlib.Path(__file__).resolve().parent.parent / "schemas"
         / "intent.v1.schema.json").read_text())
    env = make_intent(**_fields(reason_codes=("COOLDOWN_ACTIVE",)))
    jsonschema.validate(to_wire(env), schema)
