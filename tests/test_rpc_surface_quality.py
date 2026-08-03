"""Fail-closed policy gate regression coverage."""

from unittest.mock import MagicMock

class TestPolicyGatesFailClosedWithoutManager:
    """Lazy-eval audit F6: policy_manager is constructed unconditionally at
    init, so None means broken init — and the close gate returned
    allowed, silently discarding ALL tag protection. Fail closed."""

    def test_close_gate_blocks_without_policy_manager(self):
        from modules.capacity_planner import CapacityPlanner
        from modules.config import Config
        planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock(),
                                  config=Config())
        planner.policy_manager = None
        allowed, reason = planner._check_close_allowed("02" + "a" * 64)
        assert allowed is False
        assert "policy" in reason.lower()
