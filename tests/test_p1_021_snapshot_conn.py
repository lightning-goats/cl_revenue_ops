"""P1-021: snapshot_peers_delayed closes its thread-local DB connection.

The one-shot startup snapshot thread must call database.close_connection()
before it exits so its thread-local SQLite connection is not retained.
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import DummyPlugin, load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def test_snapshot_peers_once_closes_connection(mod):
    mod.plugin = DummyPlugin()
    mod.data_service = MagicMock()
    mod.data_service.get_peers.return_value = {"peers": []}
    mod.database = MagicMock()
    mod.safe_plugin = MagicMock()

    assert hasattr(mod, "_snapshot_peers_once")
    mod._snapshot_peers_once()
    mod.database.close_connection.assert_called_once()


def test_snapshot_peers_once_closes_connection_on_error(mod):
    mod.plugin = DummyPlugin()
    mod.data_service = MagicMock()
    mod.data_service.get_peers.side_effect = RuntimeError("boom")
    mod.database = MagicMock()
    mod.safe_plugin = MagicMock()

    mod._snapshot_peers_once()  # must not raise
    mod.database.close_connection.assert_called_once()
