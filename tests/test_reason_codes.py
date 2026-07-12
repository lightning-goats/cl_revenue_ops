"""Phase 1: stable reason-code catalog v0 (wire-contract-spec.md J4)."""
import re

from modules.reason_codes import CATALOG, KINDS, LAYERS, is_valid_code

SPEC_CODES = {
    "BUDGET_EXHAUSTED", "AUTHORITY_LEVEL_BLOCKED", "INTENT_STALE",
    "INTENT_SUPERSEDED", "CHANNEL_PROTECTED", "CONTRACT_OBLIGATION",
    "EV_BELOW_HOLD_MARGIN", "INSUFFICIENT_CONFIDENCE", "FEE_RAIL_CLAMPED",
    "COOLDOWN_ACTIVE", "CONFLICT_CLOSE_REBALANCE",
    "EXTERNAL_CIRCUIT_BREAKER", "EXTERNAL_OUTCOME_UNKNOWN",
    "ARITHMETIC_OVERFLOW", "SCHEMA_INVALID", "PAUSED",
}


def test_all_spec_codes_present():
    assert SPEC_CODES <= set(CATALOG)


def test_catalog_entries_well_formed():
    for code, rc in CATALOG.items():
        assert rc.code == code
        assert re.match(r"^[A-Z][A-Z0-9_]*$", code), code
        assert rc.layer in LAYERS, code
        assert rc.kind in KINDS, code


def test_is_valid_code():
    assert is_valid_code("BUDGET_EXHAUSTED")
    assert not is_valid_code("MADE_UP_CODE")
    assert not is_valid_code("")


def test_specific_ownership():
    assert CATALOG["BUDGET_EXHAUSTED"].layer == "governor"
    assert CATALOG["BUDGET_EXHAUSTED"].kind == "rejection"
    assert CATALOG["INTENT_STALE"].layer == "arbiter"
    assert CATALOG["INTENT_STALE"].kind == "deferral"
    assert CATALOG["EXTERNAL_OUTCOME_UNKNOWN"].kind == "unknown"
    assert CATALOG["PAUSED"].layer == "governor"
    assert CATALOG["PAUSED"].kind == "rejection"
