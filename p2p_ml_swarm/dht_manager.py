from __future__ import annotations

import logging
import time
from typing import Optional

try:
    from hivemind import DHT
except ImportError:
    DHT = None

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import DHTConnectionError, TokenCheckoutError

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BASE_DELAY = 1.0


def _retry(func, attempts: int = _DEFAULT_RETRY_ATTEMPTS, base_delay: float = _DEFAULT_RETRY_BASE_DELAY):
    for attempt in range(attempts):
        try:
            return func()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Retry {attempt + 1}/{attempts} after error: {e}, waiting {delay:.1f}s")
            time.sleep(delay)


class DHTManager:
    def __init__(self, config: SwarmConfig):
        self._config = config
        self._dht: Optional[DHT] = None

    def connect(self) -> DHT:
        logger.info(f"Connecting to DHT bootnode: {self._config.bootnode_addr}")
        self._dht = DHT(
            initial_peers=[self._config.bootnode_addr],
            client_mode=self._config.client_mode,
            start=True,
        )
        logger.info(f"Connected to DHT, peer_id={self._dht.peer_id}")
        return self._dht

    @property
    def dht(self) -> DHT:
        if self._dht is None:
            raise DHTConnectionError("DHT not connected, call connect() first")
        return self._dht

    def checkout_token(self) -> Optional[int]:
        run_id = self._config.run_id
        key = f"{run_id}/next_batch_id"

        def _checkout():
            result = self.dht.get(key, latest=True)
            current_id = 0 if result is None or result.value is None else int(result.value)
            success = self.dht.store(key, current_id + 1, expiration_time=time.time() + self._config.token_timeout_sec)
            if not success:
                raise TokenCheckoutError("Failed to store incremented batch ID in DHT")
            claim_key = f"{run_id}/claimed/{current_id}"
            self.dht.store(claim_key, time.time(), expiration_time=time.time() + self._config.token_timeout_sec)
            logger.info(f"Checked out token: batch_id={current_id}")
            return current_id

        return _retry(_checkout)

    def mark_complete(self, batch_id: int) -> None:
        run_id = self._config.run_id
        done_key = f"{run_id}/done/{batch_id}"

        def _mark():
            success = self.dht.store(done_key, True, expiration_time=time.time() + 86400)
            if not success:
                raise TokenCheckoutError(f"Failed to mark batch {batch_id} as complete")
            logger.info(f"Marked batch {batch_id} as complete")

        _retry(_mark)

    def is_batch_done(self, batch_id: int) -> bool:
        run_id = self._config.run_id
        done_key = f"{run_id}/done/{batch_id}"
        result = self.dht.get(done_key, latest=True)
        return result is not None and result.value is True

    def get_peer_count(self) -> int:
        try:
            peers = self.dht.get_visible_maddrs()
            return len(peers) if peers else 1
        except Exception:
            return 1

    def shutdown(self) -> None:
        if self._dht is not None:
            logger.info("Shutting down DHT connection")
            self._dht.shutdown()
            self._dht = None
