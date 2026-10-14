"""Authoritative run and worker state machines for Local ML Cluster.

Enforces strict lifecycle transitions, unique generation tracking,
and immutable run identity.
"""

from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RunState(str, enum.Enum):
    """Authoritative state machine for a distributed training run."""

    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"

    @property
    def is_active(self) -> bool:
        """Return True if the run is currently active or transitioning."""
        return self in {
            RunState.PREFLIGHT,
            RunState.STARTING,
            RunState.RUNNING,
            RunState.FINALIZING,
            RunState.DEGRADED,
            RunState.RECOVERING,
            RunState.STOPPING,
        }

    @property
    def is_terminal(self) -> bool:
        """Return True if the run has reached an unalterable terminal state."""
        return self in {RunState.COMPLETED, RunState.STOPPED, RunState.FAILED}


class WorkerState(str, enum.Enum):
    """Authoritative state machine for an individual worker node."""

    DISCOVERING = "DISCOVERING"
    PAIRING = "PAIRING"
    READY = "READY"
    STARTING = "STARTING"
    TRAINING = "TRAINING"
    RECONNECTING = "RECONNECTING"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"

    @property
    def is_healthy(self) -> bool:
        """Return True if the worker is ready or actively computing."""
        return self in {WorkerState.READY, WorkerState.TRAINING}


class ConnectionMode(str, enum.Enum):
    """Connection topology for multi-laptop communication."""

    OFFLINE_USB = "OFFLINE_USB"
    ONLINE_VM = "ONLINE_VM"


@dataclass(frozen=True)
class WorkerIdentity:
    """Immutable identity and metadata for a worker node."""

    worker_id: str
    hostname: str
    ip_address: str
    connection_mode: ConnectionMode
    os_name: str
    python_version: str
    pytorch_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class WorkerTelemetry:
    """Dynamic hardware and training telemetry reported by a worker."""

    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    disk_free_gb: float = 0.0
    throughput_samples_per_sec: float = 0.0
    round_trip_latency_ms: float = 0.0
    is_direct_path: bool = True
    last_heartbeat: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class WorkerRecord:
    """Full registry record for a participating worker node."""

    identity: WorkerIdentity
    state: WorkerState = WorkerState.DISCOVERING
    rank: Optional[int] = None
    telemetry: WorkerTelemetry = field(default_factory=WorkerTelemetry)
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class RunDescriptor:
    """Immutable metadata and configuration hashes for a training run."""

    run_id: str
    generation_id: str
    connection_mode: ConnectionMode
    model_template: str
    model_sha256: str
    dataset_manifest_sha256: str
    world_size: int
    batch_size: int
    learning_rate: float
    total_epochs: int
    rendezvous_endpoint: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    backend_name: str = "PyTorch DDP"


# Valid state transitions for run lifecycle
VALID_RUN_TRANSITIONS: Dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PREFLIGHT, RunState.STOPPING, RunState.FAILED},
    RunState.PREFLIGHT: {RunState.STARTING, RunState.STOPPING, RunState.FAILED},
    RunState.STARTING: {RunState.RUNNING, RunState.DEGRADED, RunState.STOPPING, RunState.FAILED},
    RunState.RUNNING: {RunState.FINALIZING, RunState.DEGRADED, RunState.STOPPING, RunState.FAILED},
    RunState.DEGRADED: {RunState.RECOVERING, RunState.STOPPING, RunState.FAILED},
    RunState.RECOVERING: {RunState.RUNNING, RunState.DEGRADED, RunState.STOPPING, RunState.FAILED},
    RunState.FINALIZING: {RunState.COMPLETED, RunState.STOPPING, RunState.FAILED},
    RunState.STOPPING: {RunState.STOPPED, RunState.FAILED},
    RunState.STOPPED: set(),
    RunState.COMPLETED: set(),
    RunState.FAILED: set(),
}


def validate_run_transition(current: RunState, target: RunState) -> bool:
    """Check whether a run state transition is permitted by the state machine."""
    allowed = VALID_RUN_TRANSITIONS.get(current, set())
    is_valid = target in allowed
    if not is_valid:
        logger.warning(
            "Invalid run state transition rejected: %s -> %s (allowed: %s)",
            current.value,
            target.value,
            [s.value for s in allowed],
        )
    return is_valid
