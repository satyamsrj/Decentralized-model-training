"""Authoritative worker membership registry for Local ML Cluster.

Tracks connected worker nodes, pair tokens, heartbeat liveness,
state transitions, and hardware telemetry.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional

from p2p_ml_swarm.control.run_state import (
    ConnectionMode,
    WorkerIdentity,
    WorkerRecord,
    WorkerState,
    WorkerTelemetry,
)
from p2p_ml_swarm.exceptions import SwarmError

logger = logging.getLogger(__name__)

# Heartbeat timeout constants (seconds)
HEARTBEAT_DEGRADED_THRESHOLD_SEC = 15.0
HEARTBEAT_LOST_THRESHOLD_SEC = 45.0


class RegistryError(SwarmError):
    """Raised on worker registration, pairing, or lookup errors."""

    pass


class WorkerRegistry:
    """In-memory authoritative worker registry with heartbeat tracking.

    Supports pairing verification, rank assignment, and automatic
    degraded/lost node state transitions.
    """

    def __init__(self) -> None:
        """Initialize empty registry with pairing token store."""
        self._workers: Dict[str, WorkerRecord] = {}
        self._pairing_tokens: Dict[str, str] = {}  # token -> worker_id
        logger.debug("WorkerRegistry initialized")

    def generate_pairing_token(self, worker_id: str) -> str:
        """Generate a cryptographically secure 6-character local pairing code.

        Args:
            worker_id: Target worker node identifier.

        Returns:
            Alphanumeric pairing code (e.g. 'K9X2P4').
        """
        token = secrets.token_hex(3).upper()
        self._pairing_tokens[token] = worker_id
        logger.info("Generated pairing token for worker '%s': %s", worker_id, token)
        return token

    def verify_pairing(self, token: str, candidate_worker_id: str) -> bool:
        """Validate a pairing token presented by a prospective worker.

        Args:
            token: The 6-character pairing code.
            candidate_worker_id: Expected worker identifier.

        Returns:
            True if pairing token matches and is consumed.
        """
        expected = self._pairing_tokens.get(token.strip().upper())
        if expected and expected == candidate_worker_id:
            del self._pairing_tokens[token.strip().upper()]
            logger.info("Pairing token validated and consumed for worker '%s'", candidate_worker_id)
            return True
        logger.warning(
            "Pairing token validation failed for worker '%s' (token: %s)",
            candidate_worker_id,
            token,
        )
        return False

    def register_worker(self, identity: WorkerIdentity) -> WorkerRecord:
        """Register or re-enroll a worker node.

        Args:
            identity: Immutable WorkerIdentity.

        Returns:
            Active WorkerRecord.
        """
        if identity.worker_id in self._workers:
            record = self._workers[identity.worker_id]
            logger.info("Re-enrolled existing worker: '%s' (%s)", identity.worker_id, identity.hostname)
            record.state = WorkerState.READY
            return record

        record = WorkerRecord(identity=identity, state=WorkerState.READY)
        self._workers[identity.worker_id] = record
        logger.info(
            "Registered new worker '%s' on %s (IP: %s, Mode: %s)",
            identity.worker_id,
            identity.hostname,
            identity.ip_address,
            identity.connection_mode.value,
        )
        return record

    def record_heartbeat(
        self,
        worker_id: str,
        telemetry: Optional[WorkerTelemetry] = None,
        state: Optional[WorkerState] = None,
    ) -> WorkerRecord:
        """Process incoming heartbeat ping and update node metrics.

        Args:
            worker_id: Target worker node ID.
            telemetry: Optional updated telemetry report.
            state: Optional updated worker state.

        Returns:
            Updated WorkerRecord.

        Raises:
            RegistryError: If worker_id is not registered.
        """
        if worker_id not in self._workers:
            raise RegistryError(f"Heartbeat received from unregistered worker '{worker_id}'")

        record = self._workers[worker_id]
        if telemetry:
            record.telemetry = telemetry
        record.telemetry.last_heartbeat = datetime.now(timezone.utc).isoformat()

        if state and state != record.state:
            logger.debug(
                "Worker '%s' state updated via heartbeat: %s -> %s",
                worker_id,
                record.state.value,
                state.value,
            )
            record.state = state

        return record

    def check_liveness(self) -> List[WorkerRecord]:
        """Scan registered workers and transition timed-out nodes to DEGRADED or FAILED.

        Returns:
            List of degraded or lost WorkerRecords.
        """
        now = datetime.now(timezone.utc)
        affected: List[WorkerRecord] = []

        for worker_id, record in self._workers.items():
            if record.state in {WorkerState.STOPPED, WorkerState.FAILED}:
                continue

            try:
                last_dt = datetime.fromisoformat(record.telemetry.last_heartbeat)
                delta_sec = (now - last_dt).total_seconds()
            except Exception:
                delta_sec = 999.0

            if delta_sec > HEARTBEAT_LOST_THRESHOLD_SEC:
                if record.state != WorkerState.FAILED:
                    logger.error(
                        "Worker '%s' lost (no heartbeat for %.1fs > threshold %.1fs). Marking FAILED.",
                        worker_id,
                        delta_sec,
                        HEARTBEAT_LOST_THRESHOLD_SEC,
                    )
                    record.state = WorkerState.FAILED
                    record.failure_reason = f"Heartbeat timed out after {delta_sec:.1f}s"
                    affected.append(record)
            elif delta_sec > HEARTBEAT_DEGRADED_THRESHOLD_SEC:
                if record.state not in {WorkerState.DEGRADED, WorkerState.RECONNECTING}:
                    logger.warning(
                        "Worker '%s' suspect (no heartbeat for %.1fs > threshold %.1fs). Marking DEGRADED.",
                        worker_id,
                        delta_sec,
                        HEARTBEAT_DEGRADED_THRESHOLD_SEC,
                    )
                    record.state = WorkerState.DEGRADED
                    affected.append(record)

        return affected

    def assign_ranks(self, rank_zero_worker_id: str) -> Dict[str, int]:
        """Assign deterministic DDP ranks to all currently READY workers.

        Args:
            rank_zero_worker_id: ID of the designated rank 0 coordinator node.

        Returns:
            Mapping of worker_id to rank integer.
        """
        if rank_zero_worker_id not in self._workers:
            raise RegistryError(f"Rank 0 worker '{rank_zero_worker_id}' is not registered")

        assigned: Dict[str, int] = {rank_zero_worker_id: 0}
        self._workers[rank_zero_worker_id].rank = 0

        current_rank = 1
        for worker_id, record in sorted(self._workers.items()):
            if worker_id != rank_zero_worker_id and record.state in {
                WorkerState.READY,
                WorkerState.TRAINING,
            }:
                record.rank = current_rank
                assigned[worker_id] = current_rank
                current_rank += 1

        logger.info("Assigned ranks across %d workers: %s", len(assigned), assigned)
        return assigned

    def get_worker(self, worker_id: str) -> Optional[WorkerRecord]:
        """Lookup worker record by ID."""
        return self._workers.get(worker_id)

    def list_workers(self) -> List[WorkerRecord]:
        """Return list of all registered worker records."""
        return list(self._workers.values())

    @property
    def total_count(self) -> int:
        """Return count of all registered workers."""
        return len(self._workers)

    @property
    def active_count(self) -> int:
        """Return count of healthy/ready workers."""
        return sum(1 for w in self._workers.values() if w.state.is_healthy)
