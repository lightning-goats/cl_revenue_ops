import importlib.util
import sys
from pathlib import Path

import pytest


def load_scorecard():
    path = Path(__file__).resolve().parents[1] / "tools" / "polar_soak_scorecard.py"
    spec = importlib.util.spec_from_file_location("polar_soak_scorecard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(counts, *, paused=True, budget=0, reserved=0, divergences=None):
    routers = {
        name: {
            "settled_count": count,
            "fee_msat": count * 10,
            "volume_msat": count * 1000,
        }
        for name, count in zip(
            ("revenue-node", "cln-competitor", "lnd-competitor"), counts
        )
    }
    return {
        "routers": routers,
        "module_health": {
            "status": {"status": "running"},
            "config": {"config": {"paused": paused, "daily_budget_sats": budget}},
            "fee_debug": {},
            "rebalance_debug": {},
            "profitability": {},
            "budget": {"reserved_24h_sats": reserved},
            "econ_reconcile": {"divergences": [] if divergences is None else divergences},
        },
    }


def traffic(family, amounts=(5_000, 15_000, 35_000)):
    records = []
    for amount in amounts:
        records.extend(
            [
                {
                    "payer": f"{family}-payer",
                    "sink": f"{family}-sink",
                    "amount_sats": amount,
                    "payment": {"success": True},
                },
                {
                    "payer": f"{family}-sink",
                    "sink": f"{family}-payer",
                    "amount_sats": amount,
                    "payment": {"success": True},
                },
            ]
            * 5
        )
    return records


def test_score_phase_requires_both_directions_sizes_and_router_evidence():
    score = load_scorecard()
    result = score.score_phase(
        "lnd",
        snapshot((10, 10, 10)),
        snapshot((20, 20, 20)),
        traffic("lnd"),
        min_per_direction=5,
        min_distinct_amounts=3,
    )

    assert result["passed"] is True
    assert result["settled"] == 30
    assert sum(
        router["settled_count"] for router in result["router_delta"].values()
    ) == 30


def test_score_phase_rejects_one_client_or_ambiguous_payment():
    score = load_scorecard()
    records = traffic("lnd")
    records[0] = {
        "payer": "cln-payer",
        "sink": "cln-sink",
        "amount_sats": 5_000,
        "payment_outcome": "unknown_do_not_retry",
    }
    result = score.score_phase(
        "lnd",
        snapshot((0, 0, 0)),
        snapshot((10, 10, 10)),
        records,
        min_per_direction=5,
        min_distinct_amounts=3,
    )

    assert result["passed"] is False
    assert result["checks"]["all_payments_settled"] is False
    assert result["checks"]["single_expected_client_family"] is False


def test_final_module_checks_require_safety_rails_and_clean_reconcile():
    score = load_scorecard()
    assert all(score.final_module_checks(snapshot((0, 0, 0))).values())
    checks = score.final_module_checks(
        snapshot((0, 0, 0), paused=False, budget=100, reserved=1, divergences=[{}])
    )
    assert checks["paused"] is False
    assert checks["daily_budget_zero"] is False
    assert checks["zero_active_reservations"] is False
    assert checks["econ_reconcile_clean"] is False


def test_aggregators_fail_closed_on_malformed_amounts():
    score = load_scorecard()
    with pytest.raises(score.ScorecardError, match="invalid msat"):
        score.aggregate_cln_forwards(
            {"forwards": [{"status": "settled", "fee_msat": "bad", "out_msat": 1}]}
        )


def test_network_id_rejects_shell_text():
    score = load_scorecard()
    with pytest.raises(score.ScorecardError, match="positive integer"):
        score._network_id("4;touch /tmp/bad")
