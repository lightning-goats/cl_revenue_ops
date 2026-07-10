#!/usr/bin/env python3
"""Run capex/planner scenario checks.

This loop is intentionally synthetic: it exercises the planner's decision
composition so capex regressions can be isolated before live Polar tests add
topology noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


try:
    from pyln.client import Plugin  # noqa: F401
except Exception:
    mock_pyln = MagicMock()
    mock_pyln.Plugin = MagicMock
    mock_pyln.RpcError = Exception
    sys.modules.setdefault("pyln", mock_pyln)
    sys.modules.setdefault("pyln.client", mock_pyln)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.capacity_planner import CapacityPlanner
from modules.capex_budget import CapexBudgetEngine
from modules.config import Config


@dataclass
class ScenarioResult:
    name: str
    mode: str
    passed: bool
    action: str
    ev_sats: float | None
    opens: int
    skipped_reasons: list[str]
    reason: str


BASELINE_PEER = "02" + "a" * 64


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cfg(*, min_annual_roi_pct: float) -> SimpleNamespace:
    return SimpleNamespace(
        planner_enabled=True,
        planner_dry_run=False,
        planner_execute_closes=False,
        planner_max_opens_per_cycle=1,
        planner_max_closes_per_cycle=0,
        planner_max_defibrillations_per_cycle=0,
        planner_min_channel_sats=500_000,
        planner_max_channel_sats=10_000_000,
        planner_max_fee_rate_sat_vb=50.0,
        planner_min_annual_roi_pct=min_annual_roi_pct,
        min_wallet_reserve=500_000,
    )


def _planner(
    *,
    closed_daily_net_sats: float | None,
    feerate_perkb: int = 1000,
    confirmed_sats: int = 50_000_000,
) -> tuple[CapacityPlanner, MagicMock]:
    plugin = MagicMock()
    plugin.rpc.feerates.return_value = {"perkb": {"opening": feerate_perkb}}
    plugin.rpc.listfunds.return_value = {
        "outputs": [{"amount_msat": confirmed_sats * 1000, "status": "confirmed"}],
        "channels": [],
    }
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listnodes.return_value = {"nodes": []}
    plugin.rpc.getinfo.return_value = {"id": "revenue-node"}
    plugin.rpc.call.return_value = {"channel_id": "opened-scenario-channel"}

    profitability = MagicMock()
    profitability.analyze_all_channels.return_value = {}
    profitability.identify_bleeders_v2.return_value = []
    if closed_daily_net_sats is None:
        profitability.database.get_peer_closed_channel_profit_summary.return_value = None
    else:
        profitability.database.get_peer_closed_channel_profit_summary.return_value = {
            "daily_net_est_sats": closed_daily_net_sats,
            "count": 1,
            "marginal_roi_proxy": 0.1,
        }
    profitability.database.get_historical_inbound_fee_ppm.return_value = None
    profitability.database.get_recent_planner_actions.return_value = []
    profitability.database.get_planner_candidates.return_value = []
    profitability.database.record_planner_candidate.return_value = None
    profitability.database.record_planner_action.return_value = 1
    profitability.database.update_planner_action.return_value = True
    profitability.database.reserve_spend.return_value = True
    profitability.database.mark_spend_reservation_spent.return_value = True
    profitability.database.get_channel_rebalance_success_rate.return_value = None
    profitability.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}
    profitability.database.get_fee_strategy_state.return_value = None
    profitability.database.get_peer_reputation.return_value = None
    profitability.database.get_peer_uptime_percent.side_effect = Exception("not available")

    flow = MagicMock()
    flow.analyze_all_channels.return_value = {}

    planner = CapacityPlanner(plugin, profitability, flow)
    planner._update_candidate_pool = MagicMock()
    planner._identify_winners = MagicMock(return_value=[])
    planner._identify_losers = MagicMock(return_value=[])
    planner._discover_peers = MagicMock(return_value=[
        {
            "peer_id": BASELINE_PEER,
            "score": 1.0,
            "source": "winner",
            "reason": "scenario candidate",
        }
    ])
    return planner, plugin


def _ev_scenario(
    *,
    name: str,
    closed_daily_net_sats: float | None,
    channel_size_sats: int,
    min_annual_roi_pct: float,
    feerate_perkb: int,
    expect_positive: bool,
) -> ScenarioResult:
    planner, _plugin = _planner(
        closed_daily_net_sats=closed_daily_net_sats,
        feerate_perkb=feerate_perkb,
    )
    ev = planner._calculate_open_ev(
        BASELINE_PEER,
        channel_size_sats,
        _cfg(min_annual_roi_pct=min_annual_roi_pct),
    )
    positive = ev > 0
    return ScenarioResult(
        name=name,
        mode="disabled",
        passed=positive is expect_positive,
        action="accept" if positive else "reject",
        ev_sats=round(float(ev), 3),
        opens=0,
        skipped_reasons=[],
        reason=f"ev {'positive' if positive else 'negative'} at hurdle={min_annual_roi_pct:.2f}%",
    )


def _cycle_scenario(
    *,
    name: str,
    closed_daily_net_sats: float,
    min_annual_roi_pct: float,
    expect_open: bool,
    mode: str = "disabled",
) -> ScenarioResult:
    planner, _plugin = _planner(
        closed_daily_net_sats=closed_daily_net_sats,
    )
    cfg = _cfg(min_annual_roi_pct=min_annual_roi_pct)
    summary = planner.execute_cycle(cfg)
    opens = len(summary.get("opens", []))
    passed = (opens > 0) is expect_open
    ev = None
    for reason in summary.get("skipped_reasons", []):
        if "Negative EV" in reason:
            ev = None
            break
    return ScenarioResult(
        name=name,
        mode=mode,
        passed=passed,
        action="open" if opens else "skip",
        ev_sats=ev,
        opens=opens,
        skipped_reasons=list(summary.get("skipped_reasons", [])),
        reason="cycle opened candidate" if opens else "cycle skipped candidate",
    )


def _baseline_results(*, min_annual_roi_pct: float) -> list[ScenarioResult]:
    return [
        _ev_scenario(
            name="low_yield_large_open_rejected",
            closed_daily_net_sats=50,
            channel_size_sats=10_000_000,
            min_annual_roi_pct=min_annual_roi_pct,
            feerate_perkb=1000,
            expect_positive=False,
        ),
        _ev_scenario(
            name="legacy_zero_hurdle_accepts_absolute_profit",
            closed_daily_net_sats=50,
            channel_size_sats=10_000_000,
            min_annual_roi_pct=0.0,
            feerate_perkb=1000,
            expect_positive=True,
        ),
        _ev_scenario(
            name="new_peer_fallback_clears_default_hurdle",
            closed_daily_net_sats=None,
            channel_size_sats=5_000_000,
            min_annual_roi_pct=min_annual_roi_pct,
            feerate_perkb=1000,
            expect_positive=True,
        ),
        _ev_scenario(
            name="high_onchain_fee_rejected",
            closed_daily_net_sats=None,
            channel_size_sats=500_000,
            min_annual_roi_pct=min_annual_roi_pct,
            feerate_perkb=500_000,
            expect_positive=False,
        ),
        _cycle_scenario(
            name="cycle_skips_low_yield_candidate",
            closed_daily_net_sats=50,
            min_annual_roi_pct=min_annual_roi_pct,
            expect_open=False,
        ),
        _cycle_scenario(
            name="cycle_opens_hurdle_clearing_candidate",
            closed_daily_net_sats=350,
            min_annual_roi_pct=min_annual_roi_pct,
            expect_open=True,
        ),
    ]


def run_loop(*, min_annual_roi_pct: float) -> list[ScenarioResult]:
    return list(_baseline_results(min_annual_roi_pct=min_annual_roi_pct))


def write_analysis(
    path: Path,
    results: list[ScenarioResult],
    *,
    min_annual_roi_pct: float,
) -> None:
    passed = sum(1 for item in results if item.passed)
    lines = [
        "# Capex Planner Loop Analysis",
        "",
        f"- Planner annual ROI hurdle: {min_annual_roi_pct:.2f}%",
        f"- Scenarios: {len(results)}",
        f"- Passing scenarios: {passed}",
        "",
        "| scenario | mode | pass | action | ev sats | opens | reason |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in results:
        ev = "" if item.ev_sats is None else f"{item.ev_sats:.3f}"
        lines.append(
            f"| {item.name} | {item.mode} | {item.passed} | {item.action} | {ev} | "
            f"{item.opens} | {item.reason} |"
        )
    lines.append("")
    if passed == len(results):
        lines.append("Conclusion: planner open EV respects the capital hurdle and preserves explicit legacy override behavior.")
    else:
        lines.append("Conclusion: one or more planner scenarios failed; inspect loop.json before live Polar comparison.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--planner-min-annual-roi-pct", type=float, default=Config().planner_min_annual_roi_pct)
    args = parser.parse_args()

    started = time.strftime("%Y%m%dT%H%M%S%z")
    out_dir = args.out_dir or REPO_ROOT / "results" / f"capex-planner-loop-{started}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_loop(
        min_annual_roi_pct=args.planner_min_annual_roi_pct,
    )
    payload = {
        "started": started,
        "mode": "capex_planner",
        "planner_min_annual_roi_pct": args.planner_min_annual_roi_pct,
        "results": [asdict(item) for item in results],
    }
    write_json(out_dir / "loop.json", payload)
    write_analysis(
        out_dir / "ANALYSIS.md",
        results,
        min_annual_roi_pct=args.planner_min_annual_roi_pct,
    )
    latest = REPO_ROOT / "results" / "capex-planner-loop-latest"
    try:
        latest.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    try:
        try:
            latest_target = out_dir.relative_to(REPO_ROOT)
        except ValueError:
            latest_target = out_dir
        latest.symlink_to(latest_target)
    except OSError:
        pass
    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
