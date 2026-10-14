"""Authoritative Run Manager for Local ML Cluster.

Coordinates run lifecycle, monotonic generation IDs, preflight validation,
worker failure detection, and metrics aggregation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from p2p_ml_swarm.control.run_state import (
    ConnectionMode,
    RunDescriptor,
    RunState,
    WorkerRecord,
    WorkerState,
    validate_run_transition,
)
from p2p_ml_swarm.control.worker_registry import WorkerRegistry
from p2p_ml_swarm.exceptions import SwarmError
from p2p_ml_swarm.training.base import StepMetrics

logger = logging.getLogger(__name__)


class RunManagerError(SwarmError):
    """Raised on illegal state transitions or preflight failures."""

    pass


@dataclass
class PreflightReport:
    """Consolidated preflight validation report across all workers."""

    is_passed: bool
    blocking_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    worker_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunManager:
    """Authoritative coordinator managing run lifecycle and worker synchronization.

    Enforces immutable run identity, generation ID tracking to prevent stale messages,
    preflight gates, and fault detection.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        min_disk_free_gb: float = 2.0,
    ) -> None:
        """Initialize RunManager with worker registry and safety thresholds.

        Args:
            registry: Authoritative WorkerRegistry instance.
            min_disk_free_gb: Minimum free disk required on each worker (default 2GB).
        """
        self._registry = registry
        self._min_disk_free_gb = min_disk_free_gb
        self._current_run: Optional[RunDescriptor] = None
        self._current_state: RunState = RunState.CREATED
        self._generation_counter: int = 0
        self._metrics_history: List[StepMetrics] = []
        self._latest_loss: Optional[float] = None
        self._total_samples: int = 0
        logger.info("RunManager initialized with min_disk_free=%.1fGB", min_disk_free_gb)

    @property
    def current_state(self) -> RunState:
        """Return the current authoritative state of the active run."""
        return self._current_state

    @property
    def current_run(self) -> Optional[RunDescriptor]:
        """Return active RunDescriptor if a run has been created."""
        return self._current_run

    @property
    def current_generation(self) -> str:
        """Return current monotonic generation ID."""
        return f"gen-{self._generation_counter}"

    def create_run(
        self,
        connection_mode: ConnectionMode,
        model_template: str,
        model_sha256: str,
        dataset_manifest_sha256: str,
        world_size: int,
        batch_size: int,
        learning_rate: float,
        total_epochs: int,
        rendezvous_endpoint: str,
        backend_name: str = "PyTorch DDP",
    ) -> RunDescriptor:
        """Create and register a new distributed training run.

        Args:
            connection_mode: OFFLINE_USB or ONLINE_VM.
            model_template: Name or path of the model architecture.
            model_sha256: Cryptographic hash of model code/weights.
            dataset_manifest_sha256: Hash of the dataset manifest.
            world_size: Expected worker count.
            batch_size: Local batch size.
            learning_rate: Base learning rate.
            total_epochs: Epoch count.
            rendezvous_endpoint: TCP rendezvous URI.
            backend_name: Training engine identifier.

        Returns:
            Immutable RunDescriptor.

        Raises:
            RunManagerError: If a run is already active.
        """
        if self._current_state.is_active:
            raise RunManagerError(
                f"Cannot create run: active run '{self._current_run.run_id if self._current_run else 'unknown'}' "
                f"is currently in state {self._current_state.value}"
            )

        self._generation_counter += 1
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        gen_id = self.current_generation

        descriptor = RunDescriptor(
            run_id=run_id,
            generation_id=gen_id,
            connection_mode=connection_mode,
            model_template=model_template,
            model_sha256=model_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            world_size=world_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            total_epochs=total_epochs,
            rendezvous_endpoint=rendezvous_endpoint,
            backend_name=backend_name,
        )

        self._current_run = descriptor
        self._current_state = RunState.CREATED
        self._metrics_history.clear()
        self._latest_loss = None
        self._total_samples = 0

        logger.info(
            "Created run '%s' [%s, mode=%s, world_size=%d, backend=%s]",
            run_id,
            gen_id,
            connection_mode.value,
            world_size,
            backend_name,
        )
        return descriptor

    def transition_state(self, target_state: RunState, reason: str = "") -> None:
        """Execute validated state machine transition for the active run.

        Args:
            target_state: Desired destination RunState.
            reason: Optional explanation for audit logging.

        Raises:
            RunManagerError: If the transition is disallowed.
        """
        if not validate_run_transition(self._current_state, target_state):
            raise RunManagerError(
                f"Illegal run state transition: {self._current_state.value} -> {target_state.value}"
            )

        prev = self._current_state
        self._current_state = target_state
        logger.info(
            "Run '%s' state transition: %s -> %s %s",
            self._current_run.run_id if self._current_run else "none",
            prev.value,
            target_state.value,
            f"({reason})" if reason else "",
        )

    def run_preflight_checks(self, rank_zero_worker_id: str) -> PreflightReport:
        """Execute comprehensive preflight verification across all workers.

        Blocks launch if:
        - Ready worker count does not match expected world size.
        - Free disk space on any worker is below safety threshold.
        - Rank 0 coordinator is missing or not ready.

        Returns:
            PreflightReport detailing all checks, errors, and warnings.
        """
        if self._current_run is None:
            raise RunManagerError("No active run descriptor to validate preflight against")

        self.transition_state(RunState.PREFLIGHT, reason="Running preflight checks")
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, Dict[str, Any]] = {}

        workers = self._registry.list_workers()
        ready_workers = [w for w in workers if w.state.is_healthy]

        # 1. World size validation
        expected_size = self._current_run.world_size
        if len(ready_workers) < expected_size:
            errors.append(
                f"Insufficient ready workers: expected {expected_size}, but only found {len(ready_workers)}"
            )

        # 2. Rank 0 coordinator check
        r0 = self._registry.get_worker(rank_zero_worker_id)
        if not r0:
            errors.append(f"Designated rank 0 coordinator '{rank_zero_worker_id}' is not registered")
        elif not r0.state.is_healthy:
            errors.append(f"Rank 0 coordinator '{rank_zero_worker_id}' is in unhealthy state: {r0.state.value}")

        # 3. Per-worker hardware & disk checks
        has_gpu = False
        has_cpu_only = False

        for w in workers:
            w_checks: Dict[str, Any] = {"status": "PASS", "details": []}
            disk_free = w.telemetry.disk_free_gb
            if disk_free < self._min_disk_free_gb:
                msg = f"Worker '{w.identity.hostname}' disk space too low: {disk_free:.1f}GB < {self._min_disk_free_gb:.1f}GB required"
                errors.append(msg)
                w_checks["status"] = "FAIL"
                w_checks["details"].append(msg)

            if w.telemetry.gpu_name:
                has_gpu = True
            else:
                has_cpu_only = True

            checks[w.identity.worker_id] = w_checks

        # 4. Heterogeneous warning (CPU mixed with GPU)
        if has_gpu and has_cpu_only and self._current_run.backend_name == "PyTorch DDP":
            warnings.append(
                "Heterogeneous hardware warning: Mixing CPU-only workers with GPU workers in PyTorch DDP "
                "may cause synchronized step bottlenecks."
            )

        passed = len(errors) == 0
        report = PreflightReport(
            is_passed=passed,
            blocking_errors=errors,
            warnings=warnings,
            worker_checks=checks,
        )

        if passed:
            self._registry.assign_ranks(rank_zero_worker_id)
            logger.info("Preflight validation PASSED for run '%s'", self._current_run.run_id)
        else:
            logger.error(
                "Preflight validation FAILED for run '%s' with %d errors: %s",
                self._current_run.run_id,
                len(errors),
                errors,
            )

        return report

    def record_step_metrics(self, generation_id: str, metrics: StepMetrics) -> None:
        """Record step metrics, verifying that message belongs to current generation.

        Args:
            generation_id: Generation identifier sent by worker.
            metrics: StepMetrics object.

        Raises:
            RunManagerError: If message is from a stale generation.
        """
        if generation_id != self.current_generation:
            logger.warning(
                "Rejected stale step metrics from generation '%s' (current: '%s')",
                generation_id,
                self.current_generation,
            )
            return

        self._metrics_history.append(metrics)
        self._latest_loss = metrics.train_loss
        self._total_samples += metrics.samples_processed

    def check_health(self) -> None:
        """Periodically evaluate worker health and update run state accordingly."""
        if self._current_state not in {RunState.RUNNING, RunState.DEGRADED}:
            return

        affected = self._registry.check_liveness()
        failed_nodes = [w for w in affected if w.state == WorkerState.FAILED]
        degraded_nodes = [w for w in affected if w.state == WorkerState.DEGRADED]

        if failed_nodes:
            self.transition_state(
                RunState.DEGRADED,
                reason=f"{len(failed_nodes)} worker(s) failed or lost heartbeat",
            )
        elif degraded_nodes and self._current_state == RunState.RUNNING:
            self.transition_state(
                RunState.DEGRADED,
                reason=f"{len(degraded_nodes)} worker(s) experiencing high latency or missed pings",
            )

    def stop_run(self, reason: str = "User requested stop") -> None:
        """Request immediate safe termination of the active run."""
        if self._current_state.is_terminal:
            logger.info("Run already terminated: %s", self._current_state.value)
            return

        self.transition_state(RunState.STOPPING, reason=reason)
        # Transition all active workers to STOPPED
        for w in self._registry.list_workers():
            w.state = WorkerState.STOPPED

        self.transition_state(RunState.STOPPED, reason="All workers notified of termination")
        logger.info("Run '%s' stopped successfully", self._current_run.run_id if self._current_run else "")
