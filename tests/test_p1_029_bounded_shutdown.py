"""P1-029: atexit executor shutdown is bounded.

A wedged lightningd leaves an in-flight worker blocked forever on recv().
_bounded_executor_shutdown must return within its timeout instead of blocking
process exit indefinitely, while still draining a healthy (idle) pool.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.plugin_test_utils import load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def test_idle_pool_drains_quickly(mod):
    ex = ThreadPoolExecutor(max_workers=2)
    start = time.time()
    assert mod._bounded_executor_shutdown(ex, timeout=2.0) is True
    assert time.time() - start < 1.0


def test_blocked_worker_does_not_block_forever(mod):
    ex = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    ex.submit(release.wait)  # in-flight worker blocked until released

    start = time.time()
    drained = mod._bounded_executor_shutdown(ex, timeout=0.5)
    elapsed = time.time() - start

    assert drained is False           # could not drain the wedged worker
    assert elapsed < 2.0              # but returned promptly, bounded
    release.set()                    # let the daemon drain thread finish


def test_none_executor_is_safe(mod):
    assert mod._bounded_executor_shutdown(None) is True
