"""2026-07-03 audit H-2: the dynamic htlc_max flow valve was fully
implemented in fee_controller but unwireable — none of its four config keys
were registered, so `getattr(cfg, 'enable_dynamic_htlcmax', False)` was
always False on a real snapshot and enabling it would have crashed on the
missing pct fields. CLN has no negative inbound fees, so htlc_max throttling
is the only inbound-side flow-control surface this plugin has.
"""

import sys
import os
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import (
    Config,
    ConfigSnapshot,
    PUBLIC_RUNTIME_KEYS,
    CONFIG_FIELD_TYPES,
    CONFIG_FIELD_RANGES,
)


def _snapshot():
    return ConfigSnapshot.from_config(Config())


class TestDynamicHtlcmaxConfig:

    def test_disabled_by_default(self):
        snap = _snapshot()
        assert snap.enable_dynamic_htlcmax is False

    def test_pct_fields_exist_with_sane_defaults(self):
        snap = _snapshot()
        for key in ("htlcmax_source_pct", "htlcmax_sink_pct", "htlcmax_balanced_pct"):
            val = getattr(snap, key)
            assert isinstance(val, float)
            assert 0.0 < val <= 1.0
        # Sinks get the tightest valve: smaller max HTLC slows drain-through.
        assert snap.htlcmax_sink_pct <= snap.htlcmax_balanced_pct <= snap.htlcmax_source_pct

    def test_keys_registered_for_validation(self):
        assert CONFIG_FIELD_TYPES["enable_dynamic_htlcmax"] is bool
        for key in ("htlcmax_source_pct", "htlcmax_sink_pct", "htlcmax_balanced_pct"):
            assert CONFIG_FIELD_TYPES[key] is float
            lo, hi = CONFIG_FIELD_RANGES[key]
            assert 0.0 <= lo < hi <= 1.0

    def test_keys_operator_settable_at_runtime(self):
        assert "enable_dynamic_htlcmax" in PUBLIC_RUNTIME_KEYS
        for key in ("htlcmax_source_pct", "htlcmax_sink_pct", "htlcmax_balanced_pct"):
            assert key in PUBLIC_RUNTIME_KEYS
