"""Run a Local ML Cluster training session with PyTorch DDP and synthetic dataset.

Usage:
    python examples/local_cluster_train.py --model mnist_cnn --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch

from p2p_ml_swarm.control.run_manager import RunManager
from p2p_ml_swarm.control.run_state import ConnectionMode, RunState, WorkerIdentity, WorkerTelemetry
from p2p_ml_swarm.control.worker_registry import WorkerRegistry
from p2p_ml_swarm.training.ddp_backend import DDPBackend
from p2p_ml_swarm.training.model_registry import (
    build_synthetic_dataset,
    create_distributed_loaders,
    get_model_template,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local_cluster_train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ML Cluster Training Runner")
    parser.add_argument("--model", type=str, default="mnist_cnn", help="Model template name")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--samples", type=int, default=320, help="Synthetic dataset sample count")
    parser.add_argument("--port", type=int, default=29500, help="TCP rendezvous port")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      [*]  LOCAL ML CLUSTER -- DISTRIBUTED TRAINING RUNNER")
    print("=" * 65)
    print(f"Model Architecture : {args.model}")
    print(f"Total Epochs       : {args.epochs}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Learning Rate      : {args.lr}")
    print(f"PyTorch Version    : {torch.__version__}")
    print(f"CUDA Available     : {torch.cuda.is_available()}")
    print("=" * 65 + "\n")

    # 1. Load model template
    template = get_model_template(args.model)
    model = template.model_factory()
    arch_hash = template.compute_architecture_hash()
    logger.info("Loaded template '%s' (Parameters: %d, Hash: %s...)", template.name, template.parameter_count, arch_hash[:12])

    # 2. Worker Membership & Authoritative Control
    registry = WorkerRegistry()
    coord_id = "coordinator-local-0"
    coord_identity = WorkerIdentity(
        worker_id=coord_id,
        hostname="localhost",
        ip_address="127.0.0.1",
        connection_mode=ConnectionMode.OFFLINE_USB,
        os_name=sys.platform,
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
    )
    registry.register_worker(coord_identity)
    registry.record_heartbeat(
        coord_id,
        telemetry=WorkerTelemetry(disk_free_gb=25.0, ram_total_gb=16.0, ram_used_gb=4.0),
    )

    manager = RunManager(registry=registry)
    rendezvous_uri = f"tcp://127.0.0.1:{args.port}"
    run = manager.create_run(
        connection_mode=ConnectionMode.OFFLINE_USB,
        model_template=args.model,
        model_sha256=arch_hash,
        dataset_manifest_sha256="manifest-sha256-verified",
        world_size=1,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        total_epochs=args.epochs,
        rendezvous_endpoint=rendezvous_uri,
        backend_name="PyTorch DDP",
    )
    logger.info("Initialized Run '%s' (Generation: %s)", run.run_id, run.generation_id)

    # 3. Preflight Safety Gate
    logger.info("Executing cluster preflight checks...")
    report = manager.run_preflight_checks(rank_zero_worker_id=coord_id)
    if not report.is_passed:
        logger.error("Preflight check FAILED: %s", report.blocking_errors)
        sys.exit(1)
    logger.info("Preflight verification PASSED cleanly.")

    # 4. Initialize DDP Backend Engine
    manager.transition_state(RunState.STARTING, reason="Configuring DDP process group")
    backend = DDPBackend(model=model)
    backend.setup(
        world_size=1,
        rank=0,
        rendezvous_endpoint=rendezvous_uri,
        config={"learning_rate": args.lr, "optimizer": "adam", "loss_fn": "cross_entropy"},
        device="cpu",
    )
    manager.transition_state(RunState.RUNNING, reason="Workers synchronized")

    # 5. Build Synthetic Offline Dataset & Distributed DataLoaders
    train_ds, val_ds = build_synthetic_dataset(template.name.split("_")[0], num_samples=args.samples)
    train_loader, val_loader = create_distributed_loaders(
        dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        world_size=1,
        rank=0,
    )
    logger.info("DataLoaders ready: %d training samples across %d batches per epoch", len(train_ds), len(train_loader))

    # 6. Execute Synchronized Distributed Training Loop
    global_step = 0
    print("\n" + "-" * 65)
    print("  Epoch     Step       Loss       Throughput         Status")
    print("-" * 65)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            metrics = backend.train_step(batch, step_idx=global_step)
            manager.record_step_metrics(manager.current_generation, metrics)
            print(
                f"  [{epoch+1}/{args.epochs}]      {global_step:03d}       "
                f"{metrics.train_loss:.4f}     "
                f"{metrics.throughput_samples_per_sec:6.1f} samples/s     RUNNING"
            )

    print("-" * 65 + "\n")

    # 7. Atomic Checkpointing with SHA-256 Manifest
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{args.epochs}_step_{global_step}.pt"
    meta = backend.save_checkpoint(
        checkpoint_file,
        metadata={"epoch": args.epochs, "step": global_step, "model": args.model},
    )
    logger.info("Saved verified checkpoint artifact:")
    logger.info("  Path    : %s", meta.checkpoint_path)
    logger.info("  SHA-256 : %s", meta.sha256_hash)
    logger.info("  Size    : %d bytes", meta.file_size_bytes)

    # 8. Safe Teardown
    backend.teardown()
    manager.stop_run(reason="Training run completed successfully")
    print("=" * 65)
    print(f"  [OK]  TRAINING COMPLETED SUCCESSFULLY  (State: {manager.current_state.value})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
