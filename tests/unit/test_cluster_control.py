"""Unit tests for cluster control state machines, worker registry, and run manager."""

import pytest
from p2p_ml_swarm.control.run_state import (
    RunState,
    WorkerState,
    ConnectionMode,
    WorkerIdentity,
    WorkerTelemetry,
    validate_run_transition,
    VALID_RUN_TRANSITIONS,
)
from p2p_ml_swarm.control.worker_registry import WorkerRegistry, RegistryError
from p2p_ml_swarm.control.run_manager import RunManager, RunManagerError
from p2p_ml_swarm.training.base import StepMetrics


def test_run_state_transitions():
    assert validate_run_transition(RunState.CREATED, RunState.PREFLIGHT)
    assert validate_run_transition(RunState.PREFLIGHT, RunState.STARTING)
    assert validate_run_transition(RunState.STARTING, RunState.RUNNING)
    assert validate_run_transition(RunState.RUNNING, RunState.DEGRADED)
    assert validate_run_transition(RunState.DEGRADED, RunState.RECOVERING)
    assert validate_run_transition(RunState.RECOVERING, RunState.RUNNING)
    assert validate_run_transition(RunState.RUNNING, RunState.FINALIZING)
    assert validate_run_transition(RunState.FINALIZING, RunState.COMPLETED)

    # Invalid transitions
    assert not validate_run_transition(RunState.CREATED, RunState.RUNNING)
    assert not validate_run_transition(RunState.COMPLETED, RunState.RUNNING)


def test_run_state_properties():
    assert RunState.RUNNING.is_active
    assert RunState.PREFLIGHT.is_active
    assert not RunState.COMPLETED.is_active
    assert RunState.COMPLETED.is_terminal
    assert RunState.FAILED.is_terminal
    assert not RunState.RUNNING.is_terminal


def test_worker_registry_pairing_and_enrollment():
    registry = WorkerRegistry()
    token = registry.generate_pairing_token("worker-1")
    assert len(token) == 6
    assert registry.verify_pairing(token, "worker-1")
    assert not registry.verify_pairing(token, "worker-1")  # Token is single-use

    identity_1 = WorkerIdentity(
        worker_id="worker-1",
        hostname="laptop-alpha",
        ip_address="192.168.1.50",
        connection_mode=ConnectionMode.OFFLINE_USB,
        os_name="Windows 11",
        python_version="3.11.0",
        pytorch_version="2.2.0",
    )
    record = registry.register_worker(identity_1)
    assert record.identity.worker_id == "worker-1"
    assert record.state == WorkerState.READY

    assert len(registry.list_workers()) == 1
    assert registry.get_worker("worker-1") is not None


def test_worker_registry_heartbeat_and_liveness():
    registry = WorkerRegistry()
    identity = WorkerIdentity(
        worker_id="worker-node-x",
        hostname="worker-x",
        ip_address="10.0.0.2",
        connection_mode=ConnectionMode.ONLINE_VM,
        os_name="Linux",
        python_version="3.10.12",
        pytorch_version="2.1.2",
    )
    registry.register_worker(identity)

    # Heartbeat update
    telemetry = WorkerTelemetry(
        cpu_percent=45.2,
        ram_used_gb=8.0,
        ram_total_gb=16.0,
        disk_free_gb=120.0,
    )
    updated = registry.record_heartbeat("worker-node-x", telemetry=telemetry)
    assert updated.telemetry.cpu_percent == 45.2
    assert updated.telemetry.disk_free_gb == 120.0

    # Heartbeat from unregistered worker raises RegistryError
    with pytest.raises(RegistryError):
        registry.record_heartbeat("unknown-worker")


def test_run_manager_lifecycle_and_preflight():
    registry = WorkerRegistry()
    manager = RunManager(registry=registry, min_disk_free_gb=2.0)

    # Register 2 workers
    for i in range(2):
        ident = WorkerIdentity(
            worker_id=f"worker-{i}",
            hostname=f"node-{i}",
            ip_address=f"10.0.0.{i+1}",
            connection_mode=ConnectionMode.OFFLINE_USB,
            os_name="Linux",
            python_version="3.10.12",
            pytorch_version="2.1.2",
        )
        registry.register_worker(ident)
        registry.record_heartbeat(
            f"worker-{i}",
            telemetry=WorkerTelemetry(disk_free_gb=10.0, ram_total_gb=16.0, ram_used_gb=4.0),
        )

    run = manager.create_run(
        connection_mode=ConnectionMode.OFFLINE_USB,
        model_template="mnist_cnn",
        model_sha256="abc123hash",
        dataset_manifest_sha256="def456hash",
        world_size=2,
        batch_size=64,
        learning_rate=0.001,
        total_epochs=5,
        rendezvous_endpoint="10.0.0.1:29500",
    )
    assert run.run_id.startswith("run-")
    assert run.generation_id == "gen-1"
    assert manager.current_state == RunState.CREATED

    report = manager.run_preflight_checks(rank_zero_worker_id="worker-0")
    assert report.is_passed
    assert len(report.blocking_errors) == 0

    manager.transition_state(RunState.STARTING)
    manager.transition_state(RunState.RUNNING)
    assert manager.current_state == RunState.RUNNING

    manager.record_step_metrics(
        generation_id=run.generation_id,
        metrics=StepMetrics(
            step=1,
            epoch=0,
            train_loss=0.55,
            samples_processed=64,
            throughput_samples_per_sec=250.0,
            learning_rate=0.001,
            step_duration_seconds=0.25,
        ),
    )
    assert manager._latest_loss == 0.55
    assert manager._total_samples == 64

    manager.stop_run()
    assert manager.current_state == RunState.STOPPED
