#!/usr/bin/env python3
"""Generate rebalance-stack parity fixtures for the Rust port (Phase 5).

Run from the repo root (~/bin/cl_revenue_ops-port, branch `port`):

    python3 tools/port/gen_rebalance_fixtures.py modes > fixtures/rebalance/modes.json

This is the ONE generator every Phase-5 task extends: each suite is a
subcommand printing JSON to stdout (the Rust repo commits the redirected
output under `fixtures/rebalance/<suite>.json`). Later tasks add subcommands
(planner, segstore, router, executor, ev, cooldowns, inbound_fee, defib) —
do not fork a second generator (the `gen_fees_fixtures.py` / `gen_flow_
fixtures.py` precedent).

The real module code is the oracle: `modules.rebalance_modes.MODES` /
`engine_kwargs` are called directly; nothing here reimplements the table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.rebalance_modes import MODES, engine_kwargs  # noqa: E402

MODE_NAMES = ["normal", "hot_protection", "structural_drain", "manual", "diagnostic"]


def gen_modes() -> dict:
    """Dump the MODES dict + engine_kwargs for all 5 modes (Phase 5 Task 1).

    Feeds `crates/revops-rebalance/src/modes.rs` /
    `crates/revops-rebalance/tests/modes.rs` in the Rust port
    (cl-revenue-ops-r), committed there as `fixtures/rebalance/modes.json`.
    """
    assert set(MODES.keys()) == set(MODE_NAMES), (
        f"MODES keys changed: {sorted(MODES.keys())} vs expected {sorted(MODE_NAMES)}"
    )
    modes_out = {}
    kwargs_out = {}
    for name in MODE_NAMES:
        m = MODES[name]
        modes_out[name] = {
            "name": m.name,
            "priority": m.priority,
            "budget_bucket": m.budget_bucket,
            "reserve_on_rail": m.reserve_on_rail,
            "account_costs": m.account_costs,
            "deadline": m.deadline,
            "accounting_owner": m.accounting_owner,
        }
        kwargs_out[name] = engine_kwargs(name)
    return {"modes": modes_out, "engine_kwargs": kwargs_out}


# ---------------------------------------------------------------------------
# router suite (Phase 5 Task 4)
# ---------------------------------------------------------------------------

# Deterministic ids: hex ordering drives CLN's channel direction bit
# (1 if start_node > end_node), so the set below pins both directions.
_OUR = "02" + "aa" * 32
_PEER_A = "02" + "cc" * 32  # source peer > our node  -> first-hop direction 0
_PEER_B = "02" + "44" * 32  # dest peer  < our node   -> final-hop direction 0
_PEER_C = "02" + "11" * 32  # source peer < our node  -> first-hop direction 1
_PEER_D = "02" + "ee" * 32  # dest peer  > our node   -> final-hop direction 1
_M1 = "03" + "11" * 32
_M2 = "03" + "22" * 32
_M3 = "03" + "33" * 32
_SRC_SCID = "800000x1000x1"
_DST_SCID = "800001x2000x1"
_SRC2_SCID = "800002x3000x0"
_DST2_SCID = "800003x4000x0"
_MID1 = "700000x100x0"  # peer_A -> M1
_MID2 = "700001x200x0"  # M1 -> peer_B (2-hop) / M1 -> M2 (3-hop)
_MID3 = "700002x300x0"  # M2 -> peer_B
_MID4 = "700003x400x0"  # peer_C -> peer_D (single-hop middle)


class _RouterRpcStub:
    """Scripted `plugin.rpc` double for RebalanceRouterV3 (data_service=None).

    Serves canned responses and records every askrene/getroutes interaction
    so the fixture pins the payload SHAPES (method, params — final_cltv
    arithmetic, layer lists) alongside the RouteResult outputs.
    """

    def __init__(self, setup):
        self._setup = setup
        self._getroutes_i = 0
        self.reset()

    def reset(self):
        self.getroutes_calls = []
        self.creates = []
        self.updates = {}  # layer -> [scid_dir, ...] in call order
        self.removes = []
        self.listlayers_calls = 0

    def call(self, method, params):
        if method == "askrene-listlayers":
            self.listlayers_calls += 1
            return {
                "layers": [
                    {"layer": n} for n in self._setup.get("live_layers", [])
                ]
            }
        if method == "askrene-create-layer":
            self.creates.append(params["layer"])
            return {}
        if method == "askrene-update-channel":
            self.updates.setdefault(params["layer"], []).append(
                params["short_channel_id_dir"]
            )
            assert params["enabled"] is False
            return {}
        if method == "askrene-remove-layer":
            self.removes.append(params["layer"])
            return {}
        raise AssertionError(f"unexpected rpc.call({method!r})")

    def getroutes(self, **kwargs):
        self.getroutes_calls.append(dict(kwargs))
        script = self._setup["getroutes"]
        entry = script[min(self._getroutes_i, len(script) - 1)]
        self._getroutes_i += 1
        if entry.get("error") is not None:
            raise Exception(entry["error"])
        return entry["result"]

    def listpeerchannels(self, peer_id=None):
        return {
            "channels": list(self._setup.get("peers", {}).get(peer_id, []))
        }

    def listchannels(self, source=None, short_channel_id=None):
        chans = self._setup.get("gossip_channels", [])
        if short_channel_id is not None:
            out = [
                c for c in chans if c.get("short_channel_id") == short_channel_id
            ]
        elif source is not None:
            out = [c for c in chans if c.get("source") == source]
        else:
            out = list(chans)
        return {"channels": out}

    def listconfigs(self):
        return {
            "configs": {
                "cltv-final": {"value_int": self._setup["invoice_final_cltv"]}
            }
        }


class _StubPlugin:
    def __init__(self, rpc):
        self.rpc = rpc

    def log(self, msg, level=None):
        pass


def _nolog(msg, level=None):
    pass


def _norm_layer(name, mapping):
    return mapping.get(name, name)


def _router_case_inputs():
    """Input-only case table; expected outputs come from the real Python."""
    remote_b = {
        "fee_proportional_millionths": 250,
        "fee_base_msat": 0,
        "cltv_expiry_delta": 34,
    }
    peers_b = {
        _PEER_B: [
            {
                "peer_id": _PEER_B,
                "short_channel_id": _DST_SCID,
                "updates": {"remote": dict(remote_b)},
            }
        ]
    }
    gossip_2hop = [
        # peer_A -> M1 (first middle edge; source-peer forwarding policy)
        {
            "short_channel_id": _MID1,
            "source": _PEER_A,
            "destination": _M1,
            "fee_per_millionth": 100,
            "base_fee_millisatoshi": 1000,
            "delay": 34,
        },
        # M1 -> peer_B (reprice edge)
        {
            "short_channel_id": _MID2,
            "source": _M1,
            "destination": _PEER_B,
            "fee_per_millionth": 150,
            "base_fee_millisatoshi": 0,
            "delay": 40,
        },
    ]
    path_2hop = [
        {
            "short_channel_id_dir": f"{_MID1}/1",
            "next_node_id": _M1,
            "amount_msat": 500_200_000,
            "delay": 92,
        },
        {
            "short_channel_id_dir": f"{_MID2}/0",
            "next_node_id": _PEER_B,
            "amount_msat": 500_125_000,
            "delay": 52,
        },
    ]
    route_2hop = {
        "probability_ppm": 875_000,
        "amount_msat": 500_125_000,
        "path": path_2hop,
    }
    base_pair = {
        "source_channel_id": _SRC_SCID,
        "dest_channel_id": _DST_SCID,
        "source_peer_id": _PEER_A,
        "dest_peer_id": _PEER_B,
        "amount_sats": 500_000,
    }
    base_setup = {
        "our_node_id": _OUR,
        "layer_names": ["xpay", "bimodal"],
        "invoice_final_cltv": 18,
        "live_layers": ["xpay", "auto.no_mpp_support"],
        "peers": peers_b,
        "gossip_channels": gossip_2hop,
    }

    def setup(**over):
        s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_setup.items()}
        s.update(over)
        return s

    cases = []

    # 1. Cheapest-route selection: empty-path route (sentinel), expensive
    # route, cheap route — cheapest by (first-hop amt - delivered).
    expensive_path = [
        {
            "short_channel_id_dir": f"{_MID3}/0",
            "next_node_id": _M2,
            "amount_msat": 500_300_000,
            "delay": 100,
        },
        {
            "short_channel_id_dir": f"{_MID2}/0",
            "next_node_id": _PEER_B,
            "amount_msat": 500_125_000,
            "delay": 52,
        },
    ]
    cases.append(
        {
            "name": "multi_route_picks_cheapest_int_msat",
            "setup": setup(
                getroutes=[
                    {
                        "result": {
                            "routes": [
                                {"amount_msat": 500_125_000, "path": []},
                                {
                                    "probability_ppm": 900_000,
                                    "amount_msat": 500_125_000,
                                    "path": expensive_path,
                                },
                                route_2hop,
                            ]
                        }
                    }
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 2. "Nmsat" string amounts everywhere in the getroutes response.
    def _msat_str(route):
        r = json.loads(json.dumps(route))
        r["amount_msat"] = f"{r['amount_msat']}msat"
        for hop in r["path"]:
            hop["amount_msat"] = f"{hop['amount_msat']}msat"
        return r

    cases.append(
        {
            "name": "nmsat_string_amounts",
            "setup": setup(
                getroutes=[{"result": {"routes": [_msat_str(route_2hop)]}}]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 3. Missing probability_ppm defaults to 0.
    no_prob = json.loads(json.dumps(route_2hop))
    del no_prob["probability_ppm"]
    cases.append(
        {
            "name": "missing_probability_defaults_zero",
            "setup": setup(getroutes=[{"result": {"routes": [no_prob]}}]),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 4. Middle path routing through us -> rejected.
    loop_path = json.loads(json.dumps(path_2hop))
    loop_path[0]["next_node_id"] = _OUR
    cases.append(
        {
            "name": "loop_through_us_rejected",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [{"amount_msat": 500_125_000, "path": loop_path}]}}
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 5. Path not terminating at dest peer -> rejected.
    wrong_end = json.loads(json.dumps(path_2hop))
    wrong_end[-1]["next_node_id"] = _M2
    cases.append(
        {
            "name": "wrong_terminus_rejected",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [{"amount_msat": 500_125_000, "path": wrong_end}]}}
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 6. Empty routes list.
    cases.append(
        {
            "name": "empty_routes_list",
            "setup": setup(getroutes=[{"result": {"routes": []}}]),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 7. Only route has an empty path: sentinel keeps it "cheapest", then
    # validation rejects it.
    cases.append(
        {
            "name": "all_paths_empty_rejected",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [{"amount_msat": 1_000, "path": []}]}}
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 8. getroutes error translation (full pipeline).
    cases.append(
        {
            "name": "getroutes_error_unknown_source",
            "setup": setup(
                getroutes=[{"error": f"Unknown source node {_PEER_A}"}]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )
    cases.append(
        {
            "name": "getroutes_error_child_died",
            "setup": setup(
                getroutes=[{"error": "askrene: child died with signal 6"}]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 9. 'Unknown layer' invalidates the cycle caches: exclude layer is torn
    # down, listlayers re-probed, a fresh exclude layer minted on retry.
    cases.append(
        {
            "name": "unknown_layer_invalidates_cycle_caches",
            "setup": setup(
                getroutes=[
                    {"error": "Unknown layer 'rebalance-exclude-3-7'"},
                    {"result": {"routes": [route_2hop]}},
                ]
            ),
            "calls": [
                {"pair": dict(base_pair), "exclude": []},
                {"pair": dict(base_pair), "exclude": []},
            ],
        }
    )

    # 10. Final-hop policy from gossip fallback (no updates.remote).
    cases.append(
        {
            "name": "final_hop_policy_gossip_fallback",
            "setup": setup(
                peers={
                    _PEER_B: [
                        {"peer_id": _PEER_B, "short_channel_id": _DST_SCID}
                    ]
                },
                gossip_channels=gossip_2hop
                + [
                    {
                        "short_channel_id": _DST_SCID,
                        "source": _PEER_B,
                        "destination": _OUR,
                        "fee_per_millionth": 300,
                        "base_fee_millisatoshi": 0,
                        "delay": 34,
                    }
                ],
                getroutes=[{"result": {"routes": [route_2hop]}}],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 11. No final-hop policy anywhere.
    cases.append(
        {
            "name": "final_hop_policy_none",
            "setup": setup(peers={}, gossip_channels=[], getroutes=[]),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 12. Policy without cltv_expiry_delta -> dest-cltv fallback default 40
    # (pins final_cltv arithmetic: 40 + invoice_final_cltv).
    cases.append(
        {
            "name": "dest_cltv_default_40",
            "setup": setup(
                peers={
                    _PEER_B: [
                        {
                            "peer_id": _PEER_B,
                            "short_channel_id": _DST_SCID,
                            "updates": {
                                "remote": {
                                    "fee_proportional_millionths": 250,
                                    "fee_base_msat": 0,
                                }
                            },
                        }
                    ]
                },
                getroutes=[{"result": {"routes": [route_2hop]}}],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 13. Source peer forwarding policy missing -> hard failure.
    cases.append(
        {
            "name": "first_middle_policy_missing",
            "setup": setup(
                gossip_channels=[],
                getroutes=[{"result": {"routes": [route_2hop]}}],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 14. Reprice keeps the router amount when a middle policy is missing
    # (3-hop middle; M1->M2 edge absent from gossip).
    path_3hop = [
        {
            "short_channel_id_dir": f"{_MID1}/1",
            "next_node_id": _M1,
            "amount_msat": 500_260_000,
            "delay": 132,
        },
        {
            "short_channel_id_dir": f"{_MID2}/0",
            "next_node_id": _M2,
            "amount_msat": 500_190_000,
            "delay": 92,
        },
        {
            "short_channel_id_dir": f"{_MID3}/1",
            "next_node_id": _PEER_B,
            "amount_msat": 500_125_000,
            "delay": 52,
        },
    ]
    cases.append(
        {
            "name": "reprice_missing_policy_keeps_router_amount",
            "setup": setup(
                gossip_channels=[
                    dict(gossip_2hop[0]),  # peer_A -> M1 (first middle)
                    {
                        "short_channel_id": _MID3,
                        "source": _M2,
                        "destination": _PEER_B,
                        "fee_per_millionth": 200,
                        "base_fee_millisatoshi": 500,
                        "delay": 30,
                    },
                ],
                getroutes=[
                    {
                        "result": {
                            "routes": [
                                {
                                    "probability_ppm": 640_000,
                                    "amount_msat": 500_125_000,
                                    "path": path_3hop,
                                }
                            ]
                        }
                    }
                ],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 15. Retry exclusions: directional entry passes through as-is, bare SCID
    # disables both directions, local endpoint SCIDs always appended (dest
    # already present -> not duplicated).
    cases.append(
        {
            "name": "exclude_retry_directional_and_bare",
            "setup": setup(getroutes=[{"result": {"routes": [route_2hop]}}]),
            "calls": [
                {
                    "pair": dict(base_pair),
                    "exclude": ["600000x5x0/1", "600001x6x0", _DST_SCID],
                }
            ],
        }
    )

    # 16. Cycle cache: identical exclude sets share ONE throwaway layer.
    cases.append(
        {
            "name": "cycle_reuses_exclude_layer",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [route_2hop]}},
                    {"result": {"routes": [route_2hop]}},
                ]
            ),
            "calls": [
                {"pair": dict(base_pair), "exclude": []},
                {"pair": dict(base_pair), "exclude": []},
            ],
        }
    )

    # 17. Single-hop middle + opposite direction bits + standalone layers
    # (configured []) + final-hop base fee arithmetic.
    cases.append(
        {
            "name": "single_hop_middle_standalone_layers",
            "setup": setup(
                layer_names=[],
                peers={
                    _PEER_D: [
                        {
                            "peer_id": _PEER_D,
                            "short_channel_id": _DST2_SCID,
                            "updates": {
                                "remote": {
                                    "fee_proportional_millionths": 777,
                                    "fee_base_msat": 1500,
                                    "cltv_expiry_delta": 80,
                                }
                            },
                        }
                    ]
                },
                gossip_channels=[
                    {
                        "short_channel_id": _MID4,
                        "source": _PEER_C,
                        "destination": _PEER_D,
                        "fee_per_millionth": 50,
                        "base_fee_millisatoshi": 0,
                        "delay": 14,
                    }
                ],
                getroutes=[
                    {
                        "result": {
                            "routes": [
                                {
                                    "probability_ppm": 999_999,
                                    "amount_msat": 250_195_000,
                                    "path": [
                                        {
                                            "short_channel_id_dir": f"{_MID4}/0",
                                            "next_node_id": _PEER_D,
                                            "amount_msat": 250_195_000,
                                            "delay": 98,
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                ],
            ),
            "calls": [
                {
                    "pair": {
                        "source_channel_id": _SRC2_SCID,
                        "dest_channel_id": _DST2_SCID,
                        "source_peer_id": _PEER_C,
                        "dest_peer_id": _PEER_D,
                        "amount_sats": 250_000,
                    },
                    "exclude": [],
                }
            ],
        }
    )

    return cases


_EXCLUDE_LAYER_RE = __import__("re").compile(r"^rebalance-exclude-\d+-\d+$")


def gen_router() -> dict:
    """Drive the REAL RebalanceRouterV3 over scripted RPC doubles.

    Feeds `crates/revops-rebalance/src/router.rs` /
    `crates/revops-rebalance/tests/router.rs` (Phase 5 Task 4), committed in
    the Rust repo as `fixtures/rebalance/router.json`.
    """
    import dataclasses

    from modules.rebalance_router_v3 import (
        RebalanceRouterV3,
        _configured_layer_names,
        _translate_getroutes_error,
        GETROUTES_RPC_TIMEOUT_SECONDS,
    )
    from modules.rebalance_engine_v2 import RebalanceEngine

    out_cases = []
    for case in _router_case_inputs():
        s = case["setup"]
        stub = _RouterRpcStub(s)
        router = RebalanceRouterV3(
            _StubPlugin(stub), s["our_node_id"], list(s["layer_names"]), _nolog
        )
        # The init-time probe is outside the cycle window; the Rust
        # CycleRouter has no init probe, so capture from begin_cycle on.
        stub.reset()
        router.begin_cycle()
        calls_out = []
        for call in case["calls"]:
            before = len(stub.getroutes_calls)
            p = call["pair"]
            rr = router.price_pair(
                p["source_channel_id"],
                p["dest_channel_id"],
                p["source_peer_id"],
                p["dest_peer_id"],
                p["amount_sats"],
                exclude=list(call["exclude"]) or None,
            )
            new_grs = stub.getroutes_calls[before:]
            assert len(new_grs) <= 1
            calls_out.append(
                {
                    "pair": dict(p),
                    "exclude": list(call["exclude"]),
                    "expected": dataclasses.asdict(rr),
                    "expected_getroutes_params": new_grs[0] if new_grs else None,
                }
            )
        router.end_cycle()

        # Normalize the time+counter throwaway layer names, creation order.
        mapping = {}
        for i, name in enumerate(stub.creates, start=1):
            assert _EXCLUDE_LAYER_RE.match(name), name
            mapping[name] = f"<exclude-{i}>"
        for c in calls_out:
            params = c["expected_getroutes_params"]
            if params is not None:
                params["layers"] = [
                    _norm_layer(n, mapping) for n in params["layers"]
                ]
        out_cases.append(
            {
                "name": case["name"],
                "setup": {
                    k: s[k]
                    for k in (
                        "our_node_id",
                        "layer_names",
                        "invoice_final_cltv",
                        "live_layers",
                        "peers",
                        "gossip_channels",
                        "getroutes",
                    )
                },
                "calls": calls_out,
                "expected_layer_lifecycle": {
                    "creates": [_norm_layer(n, mapping) for n in stub.creates],
                    "removes": [_norm_layer(n, mapping) for n in stub.removes],
                    "updates": {
                        _norm_layer(k, mapping): v
                        for k, v in stub.updates.items()
                    },
                    "listlayers_calls": stub.listlayers_calls,
                },
            }
        )

    # --- error-translation table (real _translate_getroutes_error) ---
    error_inputs = [
        f"Unknown source node {_PEER_A}",
        f"Unknown destination node {_PEER_B}",
        "Unknown layer 'rebalance-exclude-1-2'",
        "askrene: child died with signal 11",
        "askrene: failed to fork child",
        "askrene: child produced no output",
        "askrene: failed to create pipes: EMFILE",
        "We could not find a usable set of paths",
        "RPC timeout after 45s on getroutes",
        "",
    ]
    error_translation = []
    for msg in error_inputs:
        reason, detail = _translate_getroutes_error(msg)
        error_translation.append(
            {"input": msg, "reason": reason, "detail": detail}
        )

    # --- configured-layer-names normalization table ---
    layer_config_inputs = [
        None,
        "",
        "   ",
        "standalone",
        "STANDALONE",
        "none",
        "off",
        "Disabled",
        "false",
        "0",
        "xpay",
        "xpay,bimodal",
        " a , b ,, c ",
    ]
    layer_config = [
        {"raw": raw, "expected": _configured_layer_names(raw)}
        for raw in layer_config_inputs
    ]

    # --- orphan sweep (real RebalanceEngine._sweep_orphan_exclude_layers) ---
    class _SweepHost:
        _data_service = None

        def __init__(self, rpc):
            self.plugin = _StubPlugin(rpc)

        def _log(self, msg, level="debug"):
            pass

    class _SweepRpc:
        def __init__(self, layers, fail_list=False, fail_remove=()):
            self._layers = layers
            self._fail_list = fail_list
            self._fail_remove = set(fail_remove)
            self.removes = []

        def call(self, method, params):
            if method == "askrene-listlayers":
                if self._fail_list:
                    raise Exception("askrene not loaded")
                return {"layers": [{"layer": n} for n in self._layers]}
            if method == "askrene-remove-layer":
                name = params["layer"]
                self.removes.append(name)
                if name in self._fail_remove:
                    raise Exception("Unknown layer")
                return {}
            raise AssertionError(method)

    sweep_inputs = [
        {
            "name": "removes_only_prefix",
            "live_layers": [
                "rebalance-exclude-1700000000-1",
                "xpay",
                "auto.no_mpp_support",
                "rebalance-exclude-1700000099-7",
                "my-rebalance-exclude-not-prefix",
            ],
            "fail_list": False,
            "fail_remove": [],
        },
        {
            "name": "listlayers_failure_returns_zero",
            "live_layers": [],
            "fail_list": True,
            "fail_remove": [],
        },
        {
            "name": "remove_failure_still_counted",
            "live_layers": [
                "rebalance-exclude-1700000000-1",
                "rebalance-exclude-1700000000-2",
            ],
            "fail_list": False,
            "fail_remove": ["rebalance-exclude-1700000000-1"],
        },
        {
            "name": "no_orphans",
            "live_layers": ["xpay"],
            "fail_list": False,
            "fail_remove": [],
        },
    ]
    sweep_cases = []
    for si in sweep_inputs:
        rpc = _SweepRpc(
            si["live_layers"],
            fail_list=si["fail_list"],
            fail_remove=si["fail_remove"],
        )
        count = RebalanceEngine._sweep_orphan_exclude_layers(_SweepHost(rpc))
        sweep_cases.append(
            {
                "name": si["name"],
                "live_layers": si["live_layers"],
                "fail_list": si["fail_list"],
                "fail_remove": si["fail_remove"],
                "expected_removed_count": count,
                "expected_remove_attempts": rpc.removes,
            }
        )

    return {
        "getroutes_rpc_timeout_seconds": GETROUTES_RPC_TIMEOUT_SECONDS,
        "cases": out_cases,
        "error_translation": error_translation,
        "configured_layer_names": layer_config,
        "orphan_sweep": sweep_cases,
    }


SUITES = {
    "modes": gen_modes,
    "router": gen_router,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(SUITES.keys()))
    parser.add_argument(
        "outfile",
        nargs="?",
        default="-",
        help="output path, or '-' for stdout (default)",
    )
    args = parser.parse_args()

    out = SUITES[args.suite]()
    text = json.dumps(out, indent=1) + "\n"
    if args.outfile == "-":
        sys.stdout.write(text)
    else:
        Path(args.outfile).write_text(text)
        print(f"wrote {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
