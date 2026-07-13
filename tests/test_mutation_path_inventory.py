"""Phase 0 pin: every file that can invoke a mutating CLN RPC is inventoried.

A new mutating call site anywhere else — or a new verb in a known file —
fails this test until docs/refactor/phase0/mutation-paths.md and the
allowlist below are updated together. This is the enforcement half of the
Phase 0 mutation-path inventory (docs/planning/refactor.md, deliverable 1).

Scope: direct CLN RPC invocations only. Wrapper *callers* (e.g. modules
calling data_service.set_channel) are documented in the markdown inventory
but not scanned here — the wrapper itself is the choke point we pin.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# CLN RPC verbs that mutate node/network/external state. Read-only verbs
# (listpeerchannels, askrene-listlayers, getroutes, ...) are excluded.
MUTATING_ATTR_VERBS = (
    "setchannel|sendpay|waitsendpay|fundchannel|connect|invoice|delpay"
    "|delinvoice|signmessage|datastore|pay"
)
MUTATING_CALL_VERBS = (
    MUTATING_ATTR_VERBS
    + "|close|askrene-create-layer|askrene-remove-layer"
    + "|askrene-update-channel|askrene-bias-node|askrene-bias-channel"
    + "|askrene-reserve|askrene-unreserve|askrene-inform-channel"
    + "|askrene-age|askrene-disable-node"
)
PAT_ATTR = re.compile(r"rpc\.(" + MUTATING_ATTR_VERBS + r")\s*\(")
PAT_CALL = re.compile(
    r"""(?:\.call|_rpc_call)\(\s*['"](""" + MUTATING_CALL_VERBS + r""")['"]"""
)

# The complete inventory at baseline commit 5e8f747. Keys are repo-relative
# paths; values are the sorted set of mutating verbs the file may invoke.
MUTATING_CALL_SITES = {
    "modules/boltz_manager.py": ["pay"],
    "modules/capacity_planner.py": ["close", "fundchannel"],
    "modules/data_service.py": [
        # Phase 3E (2026-07-13): +connect, +signmessage — adapter grows
        # to absorb the LN+ lifecycle's direct CLN calls (Workstream G).
        "askrene-age", "askrene-bias-channel", "askrene-bias-node",
        "askrene-create-layer", "askrene-disable-node",
        "askrene-inform-channel", "askrene-remove-layer", "askrene-reserve",
        "askrene-unreserve", "askrene-update-channel", "close", "connect",
        "datastore", "delinvoice", "delpay", "fundchannel", "invoice",
        "pay", "sendpay", "setchannel", "signmessage", "waitsendpay",
    ],
    "modules/lnplus_swaps.py": ["connect", "fundchannel", "signmessage"],
    "modules/rebalance_engine_v2.py": [
        "askrene-remove-layer", "datastore", "delpay",
    ],
    "modules/rebalance_native_executor_v2.py": [
        "delinvoice", "delpay", "invoice", "sendpay", "waitsendpay",
    ],
    "modules/rebalance_router_v3.py": [
        "askrene-create-layer", "askrene-remove-layer",
        "askrene-update-channel",
    ],
}


def _scan():
    hits = {}
    files = sorted((REPO / "modules").glob("*.py"))
    files.append(REPO / "cl-revenue-ops.py")
    for f in files:
        text = f.read_text()
        verbs = set(PAT_ATTR.findall(text)) | set(PAT_CALL.findall(text))
        if verbs:
            hits[str(f.relative_to(REPO))] = sorted(verbs)
    return hits


def test_mutating_call_sites_match_inventory():
    actual = _scan()
    assert actual == MUTATING_CALL_SITES, (
        "Mutating CLN RPC call sites changed. Update BOTH this allowlist "
        "AND docs/refactor/phase0/mutation-paths.md, and say why in the "
        "commit message.\n"
        f"scan={actual!r}"
    )


def test_scanner_detects_known_seams():
    """Guard against the scanner regressing into matching nothing."""
    actual = _scan()
    assert "modules/data_service.py" in actual
    assert "setchannel" in actual["modules/data_service.py"]
