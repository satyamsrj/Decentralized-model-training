"""Decentralized P2P Machine Learning Swarm"""

__version__ = "0.1.0"

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import SwarmError

def __getattr__(name: str):
    if name == "DHTManager":
        from p2p_ml_swarm.dht_manager import DHTManager
        return DHTManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SwarmConfig", "DHTManager", "SwarmError", "__version__"]
