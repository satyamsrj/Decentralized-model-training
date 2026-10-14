"""Unit tests for online SQLite database persistence layer."""

import os
from pathlib import Path
import pytest
from p2p_ml_swarm.online_server.db import Database


@pytest.fixture
def test_db(tmp_path: Path):
    db_file = tmp_path / "test_cluster.db"
    return Database(str(db_file))


def test_upsert_and_get_worker(test_db: Database):
    test_db.upsert_worker(
        worker_id="w-node-1",
        hostname="laptop-host",
        ip_address="192.168.1.100",
        connection_mode="OFFLINE_USB",
        os_name="Windows 11",
        state="READY",
        telemetry={"cpu_percent": 12.5, "disk_free_gb": 45.0},
        rank=0,
    )

    worker = test_db.get_worker("w-node-1")
    assert worker is not None
    assert worker["hostname"] == "laptop-host"
    assert worker["rank"] == 0
    assert worker["telemetry"]["cpu_percent"] == 12.5

    # Update heartbeat
    test_db.update_heartbeat(
        worker_id="w-node-1",
        telemetry={"cpu_percent": 25.0, "disk_free_gb": 44.8},
        state="TRAINING",
    )
    updated = test_db.get_worker("w-node-1")
    assert updated["state"] == "TRAINING"
    assert updated["telemetry"]["cpu_percent"] == 25.0


def test_insert_run_and_metrics(test_db: Database):
    test_db.insert_run(
        run_id="run-test-123",
        generation_id="gen-1",
        connection_mode="ONLINE_VM",
        model_template="mnist_cnn",
        model_sha256="abc",
        dataset_manifest_sha256="def",
        world_size=2,
        batch_size=32,
        learning_rate=0.001,
        total_epochs=3,
        rendezvous_endpoint="100.64.0.1:29500",
        backend_name="PyTorch DDP",
        state="CREATED",
    )

    run = test_db.get_run("run-test-123")
    assert run is not None
    assert run["generation_id"] == "gen-1"
    assert run["state"] == "CREATED"

    # Update state
    test_db.update_run_state("run-test-123", "RUNNING")
    run_running = test_db.get_run("run-test-123")
    assert run_running["state"] == "RUNNING"

    # Insert metrics
    test_db.insert_metrics(
        run_id="run-test-123",
        generation_id="gen-1",
        worker_id="w-node-1",
        step=1,
        epoch=0,
        train_loss=0.45,
        samples_processed=32,
        throughput=150.0,
    )

    metrics = test_db.get_run_metrics("run-test-123")
    assert len(metrics) == 1
    assert metrics[0]["train_loss"] == 0.45
    assert metrics[0]["throughput"] == 150.0
