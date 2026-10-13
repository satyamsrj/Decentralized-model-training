from __future__ import annotations

import argparse
import logging

from hivemind import DHT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_bootnode(host: str = "0.0.0.0", port: int = 0) -> None:
    logger.info(f"Starting DHT bootnode on {host}:{port or 'auto'}")
    dht = DHT(initial_peers=[], host_maddrs=[f"/ip4/{host}/tcp/{port}"], start=True)
    logger.info(f"Bootnode running. Peer ID: {dht.peer_id}")
    logger.info(f"Visible multiaddrs: {dht.get_visible_maddrs()}")
    logger.info("Share these addresses with worker peers as initial_peers.")
    logger.info("Press Ctrl+C to stop.")

    try:
        while dht.is_alive():
            dht.run_in_background(timeout=60)
    except KeyboardInterrupt:
        logger.info("Shutting down bootnode")
    finally:
        dht.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a p2p_ml_swarm DHT bootnode")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=0, help="TCP port (0 = auto)")
    args = parser.parse_args()
    run_bootnode(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
