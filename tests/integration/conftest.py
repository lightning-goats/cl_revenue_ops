"""Integration-test fixtures that require a live CLN node.

Gated by the CLN_INTEGRATION=1 environment variable. Tests are skipped
when the variable is not set, so `pytest tests/` remains safe to run
on any developer machine without a local lightningd.
"""

import os
import pytest


@pytest.fixture
def live_plugin():
    if os.environ.get("CLN_INTEGRATION", "") != "1":
        pytest.skip("CLN_INTEGRATION not set; live-node tests skipped")

    rpc_path = os.environ.get(
        "LIGHTNING_RPC",
        os.path.expanduser("~/.lightning/bitcoin/lightning-rpc"),
    )
    if not os.path.exists(rpc_path):
        pytest.skip(f"lightning-rpc socket not found at {rpc_path}")

    try:
        from pyln.client import LightningRpc
    except ImportError:
        pytest.skip("pyln.client not available")

    class _FakePlugin:
        def __init__(self, rpc):
            self.rpc = rpc
            self._log_buf = []

        def log(self, msg, level="info"):
            self._log_buf.append((level, msg))

    return _FakePlugin(LightningRpc(rpc_path))
