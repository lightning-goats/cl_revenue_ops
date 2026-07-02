"""P4-001: _hive_member_zero_fee_active must read the hive-member cache
through the locked helper _cached_hive_membership_active (like its sibling
_classify_channel_role), not via a lock-free _hive_member_set_at.get().

The lock-free read on the money-gate site was the one converted-collection
reader the P2-006 sweep missed. Routing it through the locked helper keeps a
concurrent mutate+read of the set consistent.
"""

import threading
import time

import pytest


def _bare_fee_controller(set_at=None):
    from modules.fee_controller import FeeController

    fc = object.__new__(FeeController)
    fc._hive_member_lock = threading.Lock()
    fc._hive_member_set_at = dict(set_at or {})
    fc._hive_member_released_peers = set()
    fc._hive_member_advisory_peers = set()
    # Presence-only sentinel so the None-guard passes.
    fc.hive_hints = object()
    return fc


def test_p4_001_zero_fee_active_routes_through_locked_helper():
    """On the usable-but-not-member path the money gate must consult the
    locked helper rather than reading the cache dict lock-free."""
    now = int(time.time())
    fc = _bare_fee_controller({"peerZ": now})

    # usable, but hive_hints does not (currently) assert membership -> the
    # decision falls to the cached-membership branch.
    fc._get_hive_membership_status = lambda pid: {"usable": True, "member": False}
    fc._hive_hint_effective_ttl = lambda: 900

    calls = []
    real_helper = type(fc)._cached_hive_membership_active

    def spy(pid):
        calls.append(pid)
        return real_helper(fc, pid)

    fc._cached_hive_membership_active = spy

    # Fresh cache entry -> active.
    assert fc._hive_member_zero_fee_active("peerZ") is True
    assert calls == ["peerZ"], "money gate did not route through the locked helper"


def test_p4_001_stale_entry_evicted_and_inactive():
    """A stale cached entry yields False and is evicted (behaviour preserved)."""
    old = int(time.time()) - 10_000
    fc = _bare_fee_controller({"peerS": old})
    fc._get_hive_membership_status = lambda pid: {"usable": True, "member": False}
    fc._hive_hint_effective_ttl = lambda: 900

    assert fc._hive_member_zero_fee_active("peerS") is False
    assert "peerS" not in fc._hive_member_set_at


def test_p4_001_concurrent_mutate_read_consistent():
    """A writer that transiently exposes a stale timestamp *only while holding
    _hive_member_lock* must never be observed by the reader: the locked read
    always sees the committed (fresh) value, so the gate stays True and the
    entry is never spuriously evicted."""
    fresh = int(time.time())
    stale = fresh - 10_000
    fc = _bare_fee_controller({"p": fresh})
    fc._get_hive_membership_status = lambda pid: {"usable": True, "member": False}
    fc._hive_hint_effective_ttl = lambda: 900

    errors = []
    results = []
    stop = threading.Event()

    def writer():
        try:
            while not stop.is_set():
                with fc._hive_member_lock:
                    # Expose an inconsistent (stale) value, then repair it —
                    # all inside the critical section.
                    fc._hive_member_set_at["p"] = stale
                    fc._hive_member_set_at["p"] = fresh
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            while not stop.is_set():
                results.append(fc._hive_member_zero_fee_active("p"))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"raced: {errors}"
    assert results, "reader never ran"
    assert all(results), "lock-free read observed a stale/torn value -> gate flipped"
    assert "p" in fc._hive_member_set_at, "entry spuriously evicted by a torn read"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
