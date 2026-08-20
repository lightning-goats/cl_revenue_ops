from unittest.mock import MagicMock

from modules.rebalance_cycle_capture import RebalanceCycleCaptureManager

import pytest

from modules.config import Config
from tests.plugin_test_utils import load_plugin_module
from tests.test_operator_surface import _run_init_with_stubbed_dependencies


CAPTURE_OPTION = "revenue-ops-rebalance-replay-capture-enabled"


def test_rebalance_capture_defaults_off_and_is_snapshotted():
    cfg = Config()

    assert cfg.rebalance_replay_capture_enabled is False
    assert cfg.snapshot().rebalance_replay_capture_enabled is False


def test_rebalance_capture_option_is_dynamic_and_default_off():
    mod = load_plugin_module()

    option = mod.plugin.options[CAPTURE_OPTION]
    assert option["default"] == "false"
    assert option["dynamic"] is True
    assert option["on_change"] is mod._on_rebalance_replay_capture_change


def test_rebalance_capture_callback_rolls_back_when_manager_cannot_enable():
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = False
    mod.config = Config()
    mod.rebalancer = MagicMock(rebalance_engine_v2=MagicMock(rebalance_capture_manager=manager))

    with pytest.raises(ValueError, match="could not be enabled"):
        mod._on_rebalance_replay_capture_change(mod.plugin, CAPTURE_OPTION, True)

    assert mod.config.rebalance_replay_capture_enabled is False
    manager.set_enabled.assert_called_once_with(True, timeout_seconds=5.0)


def test_rebalance_capture_option_is_parsed_at_init(monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = True
    monkeypatch.setattr(
        capture_module,
        "RebalanceCycleCaptureManager",
        lambda *_args, **_kwargs: manager,
    )

    cfg = _run_init_with_stubbed_dependencies(mod, monkeypatch, {CAPTURE_OPTION: "true"})

    assert cfg.rebalance_replay_capture_enabled is True
    manager.set_enabled.assert_called_once_with(True)



def test_rebalance_capture_disable_failure_rolls_back_config():
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = False
    mod.config = Config(rebalance_replay_capture_enabled=True)
    mod.rebalancer = MagicMock(rebalance_engine_v2=MagicMock(rebalance_capture_manager=manager))

    with pytest.raises(ValueError, match="could not be disabled"):
        mod._on_rebalance_replay_capture_change(mod.plugin, CAPTURE_OPTION, False)

    assert mod.config.rebalance_replay_capture_enabled is True
    manager.set_enabled.assert_called_once_with(False, timeout_seconds=5.0)



def test_disable_timeout_callback_keeps_actual_manager_and_config_active(tmp_path):
    mod = load_plugin_module()
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    reference = manager.begin_cycle({
        "config_version": 1, "target_band_low": 0.35,
        "target_band_high": 0.65, "max_chunk_sats": 100,
        "max_pairs": 1, "pair_fee_cap_ppm": 0,
    }, {"trigger": "automatic"})
    original_set_enabled = manager.set_enabled
    manager.set_enabled = lambda enabled, timeout_seconds=5.0: original_set_enabled(
        enabled, timeout_seconds=0.0,
    )
    mod.config = Config(rebalance_replay_capture_enabled=True)
    mod.rebalancer = MagicMock(
        rebalance_engine_v2=MagicMock(rebalance_capture_manager=manager),
    )

    with pytest.raises(ValueError, match="could not be disabled"):
        mod._on_rebalance_replay_capture_change(
            mod.plugin, CAPTURE_OPTION, False,
        )

    assert mod.config.rebalance_replay_capture_enabled is True
    second_reference = manager.begin_cycle({
        "config_version": 1, "target_band_low": 0.35,
        "target_band_high": 0.65, "max_chunk_sats": 100,
        "max_pairs": 1, "pair_fee_cap_ppm": 0,
    }, {"trigger": "automatic"})
    assert second_reference is not None
    manager.finish_cycle(reference, object(), "failed")
    manager.finish_cycle(second_reference, object(), "failed")
    manager.set_enabled = original_set_enabled
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_init_requested_enable_failure_corrects_config_and_logs(monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    mod = load_plugin_module()
    instances = []

    class RejectingManager:
        def __init__(self, *_args, **_kwargs):
            instances.append(self)

        def set_enabled(self, enabled, timeout_seconds=5.0):
            assert enabled is True
            return False

    monkeypatch.setattr(
        capture_module, "RebalanceCycleCaptureManager", RejectingManager,
    )

    cfg = _run_init_with_stubbed_dependencies(
        mod, monkeypatch, {CAPTURE_OPTION: "true"},
    )

    assert instances
    assert cfg.rebalance_replay_capture_enabled is False
    assert any(
        "rebalance replay capture" in str(call).lower()
        and "enable" in str(call).lower()
        for call in mod._test_fake_proxy.log.call_args_list
    )


def test_rebalance_capture_enable_rejects_missing_runtime_manager():
    mod = load_plugin_module()
    mod.config = Config(rebalance_replay_capture_enabled=False)
    mod.rebalancer = MagicMock(rebalance_engine_v2=MagicMock(
        rebalance_capture_manager=None,
    ))

    with pytest.raises(ValueError, match="manager is unavailable"):
        mod._on_rebalance_replay_capture_change(
            mod.plugin, CAPTURE_OPTION, True,
        )

    assert mod.config.rebalance_replay_capture_enabled is False


def test_init_requested_enable_exception_corrects_config_and_logs(monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    mod = load_plugin_module()

    class RaisingManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_enabled(self, enabled, timeout_seconds=5.0):
            raise RuntimeError("start failed")

    monkeypatch.setattr(
        capture_module, "RebalanceCycleCaptureManager", RaisingManager,
    )

    cfg = _run_init_with_stubbed_dependencies(
        mod, monkeypatch, {CAPTURE_OPTION: "true"},
    )

    assert cfg.rebalance_replay_capture_enabled is False
    assert any(
        "rebalance replay capture" in str(call).lower()
        and "enable" in str(call).lower()
        for call in mod._test_fake_proxy.log.call_args_list
    )
