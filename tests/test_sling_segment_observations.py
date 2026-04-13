"""Tests for local sling segment observation export."""

from modules.sling_segment_observations import SlingSegmentObservationStore


def test_export_snapshot_validates_and_buckets_failure_observations():
    store = SlingSegmentObservationStore(ttl_seconds=900)
    store.record_failure(
        short_channel_id="123x1x0",
        direction=1,
        amount_sats=420_000,
        failure_class="liquidity",
        confidence=0.8,
        observed_at=1_700_000_000,
        source_channel_id="111x1x0",
        dest_channel_id="123x1x0",
        route_policy="hybrid",
        router_kind="v3",
        correlation_id="corr-1",
    )

    snapshot = store.export_snapshot(
        observer_member_id="02" + "a" * 64,
        now=1_700_000_100,
    )

    assert snapshot["observer_member_id"] == "02" + "a" * 64
    assert snapshot["segment_observations"][0]["amount_bucket_sats"] == 250_000
    assert snapshot["segment_observations"][0]["confidence"] == 0.8
    assert snapshot["segment_observations"][0]["outcome"] == "failure"
