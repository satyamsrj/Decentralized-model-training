import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.dht_manager import DHTManager
from p2p_ml_swarm.exceptions import DHTConnectionError


@pytest.fixture
def config():
    return SwarmConfig(
        bootnode_addr="/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        run_id="test_run",
        bucket_url="https://test.r2.cloudflarestorage.com",
        bucket_name="test-bucket",
    )


@pytest.fixture
def mock_dht():
    dht = MagicMock()
    dht.peer_id = "QmMockPeer"
    dht.is_alive.return_value = True
    dht.get_visible_maddrs.return_value = {
        "peer1": ["/ip4/1.1.1.1/tcp/9000"],
        "peer2": ["/ip4/2.2.2.2/tcp/9000"],
    }
    return dht


class TestDHTManagerConnect:
    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_connect_success(self, mock_dht_cls, config, mock_dht):
        mock_dht_cls.return_value = mock_dht
        manager = DHTManager(config)
        result = manager.connect()
        assert result == mock_dht
        mock_dht_cls.assert_called_once_with(
            initial_peers=[config.bootnode_addr],
            client_mode=True,
            start=True,
        )

    def test_dht_property_before_connect(self, config):
        manager = DHTManager(config)
        with pytest.raises(DHTConnectionError, match="not connected"):
            _ = manager.dht


class TestTokenCheckout:
    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_checkout_first_token(self, mock_dht_cls, config, mock_dht):
        mock_dht.get.return_value = None
        mock_dht.store.return_value = True
        mock_dht_cls.return_value = mock_dht

        manager = DHTManager(config)
        manager.connect()
        token = manager.checkout_token()

        assert token == 0

    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_checkout_increments(self, mock_dht_cls, config, mock_dht):
        result_mock = MagicMock()
        result_mock.value = 5
        mock_dht.get.return_value = result_mock
        mock_dht.store.return_value = True
        mock_dht_cls.return_value = mock_dht

        manager = DHTManager(config)
        manager.connect()
        token = manager.checkout_token()

        assert token == 5


class TestMarkComplete:
    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_mark_complete(self, mock_dht_cls, config, mock_dht):
        mock_dht.store.return_value = True
        mock_dht_cls.return_value = mock_dht

        manager = DHTManager(config)
        manager.connect()
        manager.mark_complete(3)

        key = "test_run/done/3"
        mock_dht.store.assert_called_with(key, True, expiration_time=pytest.approx(time.time() + 86400, abs=5))


class TestPeerCount:
    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_get_peer_count(self, mock_dht_cls, config, mock_dht):
        mock_dht_cls.return_value = mock_dht
        manager = DHTManager(config)
        manager.connect()
        assert manager.get_peer_count() == 2

    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_peer_count_on_error(self, mock_dht_cls, config, mock_dht):
        mock_dht.get_visible_maddrs.side_effect = RuntimeError("network error")
        mock_dht_cls.return_value = mock_dht
        manager = DHTManager(config)
        manager.connect()
        assert manager.get_peer_count() == 1


class TestShutdown:
    @patch("p2p_ml_swarm.dht_manager.DHT")
    def test_shutdown(self, mock_dht_cls, config, mock_dht):
        mock_dht_cls.return_value = mock_dht
        manager = DHTManager(config)
        manager.connect()
        manager.shutdown()
        mock_dht.shutdown.assert_called_once()
