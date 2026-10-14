"""Decentralized P2P Machine Learning Swarm"""

__version__ = "0.1.0"

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import SwarmError


def __getattr__(name: str):
    _lazy_imports = {
        "DHTManager": "p2p_ml_swarm.dht_manager",
        "StorageClient": "p2p_ml_swarm.storage",
        "DataPipeline": "p2p_ml_swarm.data_pipeline",
        "SwarmTrainer": "p2p_ml_swarm.trainer",
        "SwarmWorker": "p2p_ml_swarm.worker",
        "CheckpointManager": "p2p_ml_swarm.checkpoint",
        "SwarmMonitor": "p2p_ml_swarm.monitoring",
        "RunManager": "p2p_ml_swarm.control.run_manager",
        "WorkerRegistry": "p2p_ml_swarm.control.worker_registry",
        "DDPBackend": "p2p_ml_swarm.training.ddp_backend",
        "MainWindow": "p2p_ml_swarm.gui",
        "launch_gui": "p2p_ml_swarm.gui",
    }
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SwarmConfig",
    "SwarmError",
    "DHTManager",
    "StorageClient",
    "DataPipeline",
    "SwarmTrainer",
    "SwarmWorker",
    "CheckpointManager",
    "SwarmMonitor",
    "RunManager",
    "WorkerRegistry",
    "DDPBackend",
    "MainWindow",
    "launch_gui",
    "__version__",
]
