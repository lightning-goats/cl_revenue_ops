"""
Live cross-plugin contract test against the local cl-hive checkout.

This complements the static golden fixture tests by generating a real
hive-export-hints snapshot from cl-hive and verifying that the adapter
accepts the current producer output.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.hive_hints import HiveHintAdapter


CL_HIVE_PATH = Path(os.environ.get("CL_HIVE_PATH", "/home/sat/bin/cl-hive"))
CL_HIVE_PYTHON = Path(
    os.environ.get("CL_HIVE_PYTHON", "/home/sat/bin/cl-hive/.venv/bin/python")
)

PEER_QUALITY = "02" + "a" * 64
PEER_TRAFFIC = "03" + "b" * 64
PEER_REBAL = "02" + "c" * 64
PEER_PRIMARY = "03" + "d" * 64
PEER_SECONDARY = "02" + "e" * 64
OUR_PUBKEY = "02" + "ff" * 32


def _build_adapter(snapshot: dict) -> HiveHintAdapter:
    plugin = MagicMock()
    plugin.rpc.call.return_value = snapshot
    adapter = HiveHintAdapter(plugin, ttl_override=0)
    adapter.poll()
    return adapter


@pytest.fixture(scope="module")
def live_snapshot() -> dict:
    if not (CL_HIVE_PATH / "modules" / "rpc_commands.py").exists():
        pytest.skip("local cl-hive checkout not available")

    python = str(CL_HIVE_PYTHON if CL_HIVE_PYTHON.exists() else Path(sys.executable))
    script = textwrap.dedent(
        f"""
        import json
        import os
        import sys
        from unittest.mock import MagicMock

        sys.path.insert(0, os.getcwd())

        from modules.fee_coordination import CorridorAssignment, FlowCorridor
        from modules.quality_scorer import PeerQualityResult
        from modules.rpc_commands import HiveContext, export_hints

        PEER_QUALITY = {PEER_QUALITY!r}
        PEER_TRAFFIC = {PEER_TRAFFIC!r}
        PEER_REBAL = {PEER_REBAL!r}
        PEER_PRIMARY = {PEER_PRIMARY!r}
        PEER_SECONDARY = {PEER_SECONDARY!r}
        OUR_PUBKEY = {OUR_PUBKEY!r}


        class Metric:
            def __init__(self, peer_id, flow_direction, volume_routed_sats):
                self.peer_id = peer_id
                self.flow_direction = flow_direction
                self.volume_routed_sats = volume_routed_sats


        def zero_confidence_result(peer_id):
            return PeerQualityResult(
                peer_id=peer_id,
                overall_score=0.0,
                reliability_score=0.0,
                profitability_score=0.0,
                routing_score=0.0,
                consistency_score=0.0,
                confidence=0.0,
                recommendation="neutral",
                factors={{}},
            )


        db = MagicMock()
        db.get_all_members.return_value = []

        scorer = MagicMock()
        quality_result = PeerQualityResult(
            peer_id=PEER_QUALITY,
            overall_score=0.81,
            reliability_score=0.8,
            profitability_score=0.8,
            routing_score=0.8,
            consistency_score=0.8,
            confidence=0.7,
            recommendation="good",
            factors={{}},
        )
        scorer.get_scored_peers.return_value = [quality_result]
        scorer.calculate_score.side_effect = (
            lambda peer_id, days=90: quality_result
            if peer_id == PEER_QUALITY
            else zero_confidence_result(peer_id)
        )

        traffic_mgr = MagicMock()
        traffic_mgr.get_all_profiles.return_value = [{{"peer_id": PEER_TRAFFIC, "confidence": 0.73}}]
        traffic_mgr.get_aggregated_profile.side_effect = (
            lambda peer_id: {{"confidence": 0.73}} if peer_id == PEER_TRAFFIC else None
        )

        yield_mgr = MagicMock()
        yield_mgr.get_channel_yield_metrics.return_value = [
            Metric(PEER_REBAL, "sink", 5_000_000),
            Metric(PEER_REBAL, "source", 1_000_000),
        ]

        corridor_mgr = MagicMock()
        corridor_mgr.get_assignments.return_value = [
            CorridorAssignment(
                corridor=FlowCorridor(
                    source_peer_id="source",
                    destination_peer_id="dest",
                    capable_members=[PEER_PRIMARY, PEER_SECONDARY],
                ),
                primary_member=PEER_PRIMARY,
                secondary_members=[PEER_SECONDARY],
                primary_fee_ppm=500,
                secondary_fee_ppm=900,
                assignment_reason="test",
                confidence=0.9,
            )
        ]

        fee_coordination_mgr = MagicMock(corridor_mgr=corridor_mgr)

        ctx = HiveContext(
            database=db,
            config=MagicMock(),
            safe_plugin=MagicMock(),
            our_pubkey=OUR_PUBKEY,
            quality_scorer=scorer,
            traffic_intel_mgr=traffic_mgr,
            yield_metrics_mgr=yield_mgr,
            fee_coordination_mgr=fee_coordination_mgr,
        )

        print(json.dumps(export_hints(ctx), sort_keys=True))
        """
    )

    completed = subprocess.run(
        [python, "-c", script],
        cwd=str(CL_HIVE_PATH),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


class TestLiveHiveContract:
    def test_quality_and_traffic_only_peers_survive_export(self, live_snapshot):
        adapter = _build_adapter(live_snapshot)
        hints = adapter._snapshot["hints"]

        assert hints[PEER_QUALITY]["peer_quality_score"] == 0.81
        assert hints[PEER_QUALITY]["member"] is False
        assert hints[PEER_TRAFFIC]["traffic_confidence"] == 0.73
        assert hints[PEER_TRAFFIC]["member"] is False

    def test_dominant_rebalance_direction_flows_through(self, live_snapshot):
        adapter = _build_adapter(live_snapshot)

        assert adapter._snapshot["hints"][PEER_REBAL]["rebalance_preference"] == "sink"
        assert adapter.get_rebalance_bias(PEER_REBAL) > 1.0

    def test_secondary_fee_prior_flows_through(self, live_snapshot):
        adapter = _build_adapter(live_snapshot)

        assert adapter.get_corridor_role(PEER_SECONDARY) == "secondary"
        assert adapter.get_fleet_fee_prior(PEER_SECONDARY) == 900
