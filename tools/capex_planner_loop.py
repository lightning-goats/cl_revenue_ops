#!/usr/bin/env python3
"""Run capex/planner scenario checks with optional cl-hive A/B coverage.

This loop is intentionally synthetic: it exercises the planner's decision
composition so capex regressions can be isolated before live Polar tests add
topology noise.  By default it keeps the original cl_revenue_ops-only mode;
``--hive-mode ab`` adds controlled hive hint scenarios.
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


HIVE_PEER = "02" + "b" * 64
BASELINE_PEER = "02" + "a" * 64


class FakeHiveHints:
    """Deterministic hive hint source for synthetic A/B scenarios."""

    def __init__(
        self,
        *,
        members: set[str] | None = None,
        open_hints: dict[str, dict] | None = None,
        rebalance_bias: dict[str, float] | None = None,
        corridor_bias: dict[str, float] | None = None,
        reputation: dict[str, float] | None = None,
        corridor_roles: dict[str, str] | None = None,
    ):
        self.members = members or set()
        self.open_hints = open_hints or {}
        self.rebalance_bias = rebalance_bias or {}
        self.corridor_bias = corridor_bias or {}
        self.reputation = reputation or {}
        self.corridor_roles = corridor_roles or {}

    def get_open_candidates(self):
        return [
            (peer_id, hint.copy())
            for peer_id, hint in self.open_hints.items()
            if hint.get("open_preference") == "open"
        ]

    def get_channel_open_hint(self, peer_id: str) -> dict:
        return self.open_hints.get(peer_id, {}).copy()

    def get_rebalance_bias(self, peer_id: str) -> float:
        return self.rebalance_bias.get(peer_id, 1.0)

    def get_corridor_utilization_bias(self, peer_id: str) -> float:
        return self.corridor_bias.get(peer_id, 1.0)

    def get_reputation_score(self, peer_id: str):
        return self.reputation.get(peer_id)

    def is_hive_member(self, peer_id: str) -> bool:
        return peer_id in self.members

    def get_corridor_role(self, peer_id: str) -> str:
        return self.corridor_roles.get(peer_id, "")

    def is_closure_recommended(self, peer_id: str) -> bool:
        return False


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
    hive_hints=None,
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
    planner.hive_hints = hive_hints
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
    hive_hints=None,
) -> ScenarioResult:
    planner, _plugin = _planner(
        closed_daily_net_sats=closed_daily_net_sats,
        hive_hints=hive_hints,
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


def _hive_discovery_scenario() -> ScenarioResult:
    hive = FakeHiveHints(
        open_hints={
            HIVE_PEER: {
                "open_preference": "open",
                "topology_confidence": 0.8,
                "suggested_size_bucket": "medium",
                "reason": "underserved_corridor",
            }
        }
    )
    disabled, _ = _planner(closed_daily_net_sats=None)
    disabled.hive_hints = None
    enabled, _ = _planner(closed_daily_net_sats=None, hive_hints=hive)

    disabled_candidates = disabled._discover_from_hive()
    enabled_candidates = enabled._discover_from_hive()
    passed = (
        len(disabled_candidates) == 0
        and len(enabled_candidates) == 1
        and enabled_candidates[0]["peer_id"] == HIVE_PEER
        and enabled_candidates[0]["source"] == "hive"
    )
    return ScenarioResult(
        name="hive_open_hint_adds_candidate",
        mode="ab",
        passed=passed,
        action="candidate" if enabled_candidates else "none",
        ev_sats=None,
        opens=len(enabled_candidates),
        skipped_reasons=[],
        reason=(
            f"disabled_candidates={len(disabled_candidates)}, "
            f"enabled_candidates={len(enabled_candidates)}"
        ),
    )


def _hive_score_scenario() -> ScenarioResult:
    hive = FakeHiveHints(
        open_hints={
            HIVE_PEER: {
                "open_preference": "open",
                "topology_confidence": 1.0,
            }
        },
        corridor_bias={HIVE_PEER: 1.1},
        reputation={HIVE_PEER: 90.0},
    )
    disabled, _ = _planner(closed_daily_net_sats=None)
    enabled, _ = _planner(closed_daily_net_sats=None, hive_hints=hive)
    disabled_score = disabled._score_candidate(HIVE_PEER, 1.0)
    enabled_score = enabled._score_candidate(HIVE_PEER, 1.0)
    passed = enabled_score > disabled_score
    return ScenarioResult(
        name="hive_score_bias_prioritizes_stronger_peer",
        mode="ab",
        passed=passed,
        action="boost" if passed else "flat",
        ev_sats=None,
        opens=0,
        skipped_reasons=[],
        reason=f"disabled_score={disabled_score:.3f}, enabled_score={enabled_score:.3f}",
    )


def _hive_ev_bias_scenario(*, min_annual_roi_pct: float) -> ScenarioResult:
    hive = FakeHiveHints(rebalance_bias={BASELINE_PEER: 1.06})
    disabled, _ = _planner(closed_daily_net_sats=300)
    enabled, _ = _planner(closed_daily_net_sats=300, hive_hints=hive)
    cfg = _cfg(min_annual_roi_pct=min_annual_roi_pct)
    disabled_ev = disabled._calculate_open_ev(BASELINE_PEER, 10_000_000, cfg)
    enabled_ev = enabled._calculate_open_ev(BASELINE_PEER, 10_000_000, cfg)
    passed = disabled_ev <= 0 < enabled_ev
    return ScenarioResult(
        name="hive_rebalance_bias_clears_marginal_roi_hurdle",
        mode="ab",
        passed=passed,
        action="accept" if enabled_ev > 0 else "reject",
        ev_sats=round(float(enabled_ev), 3),
        opens=0,
        skipped_reasons=[],
        reason=f"disabled_ev={disabled_ev:.0f}, enabled_ev={enabled_ev:.0f}",
    )


def _hive_cycle_scenario(*, min_annual_roi_pct: float) -> ScenarioResult:
    hive = FakeHiveHints(rebalance_bias={BASELINE_PEER: 1.06})
    disabled = _cycle_scenario(
        name="cycle_disabled_reference",
        closed_daily_net_sats=300,
        min_annual_roi_pct=min_annual_roi_pct,
        expect_open=False,
        mode="disabled",
    )
    enabled = _cycle_scenario(
        name="cycle_enabled_reference",
        closed_daily_net_sats=300,
        min_annual_roi_pct=min_annual_roi_pct,
        expect_open=True,
        mode="enabled",
        hive_hints=hive,
    )
    passed = disabled.passed and enabled.passed and disabled.opens == 0 and enabled.opens > 0
    return ScenarioResult(
        name="hive_bias_changes_cycle_decision",
        mode="ab",
        passed=passed,
        action="open" if enabled.opens else "skip",
        ev_sats=None,
        opens=enabled.opens,
        skipped_reasons=disabled.skipped_reasons + enabled.skipped_reasons,
        reason=f"disabled_opens={disabled.opens}, enabled_opens={enabled.opens}",
    )


def _mock_channel_profitability(*, peer_id: str, days_open: int = 1):
    prof = SimpleNamespace()
    prof.channel_id = "100x1x0"
    prof.peer_id = peer_id
    prof.revenue = SimpleNamespace(
        total_contribution_msat=0,
        fees_earned_msat=0,
        total_forward_count=0,
    )
    prof.days_open = days_open
    prof.capacity_sats = 5_000_000
    prof.classification = "break_even"
    prof.marginal_roi = 0.0
    return prof


def _capex_hive_budget_scenario() -> ScenarioResult:
    cfg = Config().snapshot()
    prof = _mock_channel_profitability(peer_id=HIVE_PEER, days_open=1)

    disabled = CapexBudgetEngine.__new__(CapexBudgetEngine)
    disabled._hive_member_check = lambda pid: False
    disabled._hive_hints = None
    disabled._capital_efficiency = None
    disabled._database = MagicMock()
    disabled._database.get_channel_rebalance_success_rate.return_value = None

    enabled = CapexBudgetEngine.__new__(CapexBudgetEngine)
    enabled._hive_member_check = lambda pid: pid == HIVE_PEER
    enabled._hive_hints = FakeHiveHints(members={HIVE_PEER})
    enabled._capital_efficiency = None
    enabled._database = MagicMock()
    enabled._database.get_channel_rebalance_success_rate.return_value = None

    disabled_budget = disabled._compute_channel_budget(
        ch_id="100x1x0",
        prof=prof,
        total_capex_30d_msat=0,
        bleeder_status="none",
        cfg=cfg,
    )
    enabled_budget = enabled._compute_channel_budget(
        ch_id="100x1x0",
        prof=prof,
        total_capex_30d_msat=0,
        bleeder_status="none",
        cfg=cfg,
    )
    passed = disabled_budget.tier == "blocked" and enabled_budget.tier == "fleet"
    return ScenarioResult(
        name="hive_member_gets_fleet_capex_budget",
        mode="ab",
        passed=passed,
        action=enabled_budget.tier,
        ev_sats=None,
        opens=0,
        skipped_reasons=[],
        reason=(
            f"disabled_tier={disabled_budget.tier}, enabled_tier={enabled_budget.tier}, "
            f"enabled_budget_sats={enabled_budget.budget_sats}"
        ),
    )


def _hive_ab_results(*, min_annual_roi_pct: float) -> list[ScenarioResult]:
    return [
        _hive_discovery_scenario(),
        _hive_score_scenario(),
        _hive_ev_bias_scenario(min_annual_roi_pct=min_annual_roi_pct),
        _hive_cycle_scenario(min_annual_roi_pct=min_annual_roi_pct),
        _capex_hive_budget_scenario(),
    ]


def run_loop(*, min_annual_roi_pct: float, hive_mode: str = "disabled") -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    if hive_mode in ("disabled", "ab"):
        results.extend(_baseline_results(min_annual_roi_pct=min_annual_roi_pct))
    if hive_mode in ("enabled", "ab"):
        results.extend(_hive_ab_results(min_annual_roi_pct=min_annual_roi_pct))
    return results


def write_analysis(
    path: Path,
    results: list[ScenarioResult],
    *,
    min_annual_roi_pct: float,
    hive_mode: str,
) -> None:
    passed = sum(1 for item in results if item.passed)
    lines = [
        "# Capex Planner Loop Analysis",
        "",
        f"- Mode: {hive_mode}",
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
        if hive_mode == "disabled":
            lines.append("Conclusion: planner open EV respects the capital hurdle and preserves explicit legacy override behavior.")
        else:
            lines.append("Conclusion: controlled hive hints improve discovery, score/EV selection, and fleet capex treatment without relaxing the capital hurdle.")
    else:
        lines.append("Conclusion: one or more planner scenarios failed; inspect loop.json before live Polar comparison.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--planner-min-annual-roi-pct", type=float, default=Config().planner_min_annual_roi_pct)
    parser.add_argument(
        "--hive-mode",
        choices=("disabled", "enabled", "ab"),
        default="disabled",
        help="Run baseline only, hive-only synthetic checks, or disabled/enabled A/B checks.",
    )
    args = parser.parse_args()

    started = time.strftime("%Y%m%dT%H%M%S%z")
    out_dir = args.out_dir or REPO_ROOT / "results" / f"capex-planner-loop-{started}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_loop(
        min_annual_roi_pct=args.planner_min_annual_roi_pct,
        hive_mode=args.hive_mode,
    )
    payload = {
        "started": started,
        "mode": "capex_planner",
        "hive_mode": args.hive_mode,
        "planner_min_annual_roi_pct": args.planner_min_annual_roi_pct,
        "results": [asdict(item) for item in results],
    }
    write_json(out_dir / "loop.json", payload)
    write_analysis(
        out_dir / "ANALYSIS.md",
        results,
        min_annual_roi_pct=args.planner_min_annual_roi_pct,
        hive_mode=args.hive_mode,
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
