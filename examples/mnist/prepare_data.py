"""Download MNIST, shard it into safetensors files, and optionally upload to S3/R2.

Usage:
    python prepare_data.py --output-dir ./mnist_shards --shard-size-mb 10
    python prepare_data.py --output-dir ./mnist_shards --upload --config ../../config/swarm_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_mnist(data_dir: str = "./mnist_raw") -> datasets.MNIST:
    """Download MNIST dataset and return it as a PyTorch Dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    logger.info(f"MNIST downloaded: {len(dataset)} training samples")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST data for swarm training")
    parser.add_argument("--output-dir", type=str, default="./mnist_shards",
                        help="Directory to save sharded safetensor files")
    parser.add_argument("--shard-size-mb", type=int, default=10,
                        help="Target shard size in MB (default: 10)")
    parser.add_argument("--data-dir", type=str, default="./mnist_raw",
                        help="Directory to download raw MNIST data")
    parser.add_argument("--upload", action="store_true",
                        help="Upload shards to cloud storage after creation")
    parser.add_argument("--config", type=str, default="../../config/swarm_config.yaml",
                        help="Path to swarm_config.yaml (required for --upload)")
    args = parser.parse_args()

    # Step 1: Download MNIST
    dataset = download_mnist(args.data_dir)

    # Step 2: Shard into safetensors files
    from p2p_ml_swarm.config import SwarmConfig
    from p2p_ml_swarm.storage import StorageClient
    from p2p_ml_swarm.data_pipeline import DataPipeline

    # Create a minimal config for sharding (bucket fields not needed unless uploading)
    if args.upload:
        config = SwarmConfig.from_yaml(args.config)
    else:
        config = SwarmConfig(
            bootnode_addr="placeholder",
            run_id="mnist_prep",
            bucket_url="placeholder",
            bucket_name="placeholder",
            shard_size_mb=args.shard_size_mb,
        )

    # Use a dummy storage client for local-only sharding
    if args.upload:
        storage = StorageClient(config)
    else:
        storage = None

    pipeline = DataPipeline(config, storage)
    shard_paths = pipeline.shard_dataset(dataset, args.output_dir, args.shard_size_mb)
    logger.info(f"Created {len(shard_paths)} shard files in {args.output_dir}")

    # Step 3: Upload if requested
    if args.upload:
        if storage is None:
            logger.error("Storage client not initialized — provide a valid --config with bucket credentials")
            return
        uploaded = pipeline.upload_all_shards(args.output_dir)
        logger.info(f"Uploaded {uploaded} shards to cloud storage")

    # Verify: load one shard and print shape
    data, labels = pipeline.load_shard_local(shard_paths[0])
    logger.info(f"Verification — Shard 0: data={data.shape}, labels={labels.shape}")
    logger.info("Done! Data is ready for swarm training.")


if __name__ == "__main__":
    main()
