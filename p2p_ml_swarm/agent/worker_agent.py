"""Worker Agent managing local process lifecycle and heartbeat reporting.

Runs on each node, collecting telemetry, handling start/stop training requests,
and reporting status to the authoritative RunManager.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from p2p_ml_swarm.agent.hardware_probe import probe_hardware
from p2p_ml_swarm.control.run_state import (
    ConnectionMode,
    WorkerIdentity,
    WorkerState,
    WorkerTelemetry,
)
from p2p_ml_swarm.exceptions import SwarmError

logger = logging.getLogger(__name__)


class AgentError(SwarmError):
    """Raised when worker process management or communication fails."""

    pass


class WorkerAgent:
    """Local node agent managing hardware telemetry, process execution, and heartbeats."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        connection_mode: ConnectionMode = ConnectionMode.OFFLINE_USB,
        bind_ip: str = "127.0.0.1",
        heartbeat_interval_sec: float = 5.0,
    ) -> None:
        """Initialize WorkerAgent.

        Args:
            worker_id: Unique node identifier (auto-generated if None).
            connection_mode: OFFLINE_USB or ONLINE_VM.
            bind_ip: Primary network interface IP address.
            heartbeat_interval_sec: Heartbeat loop frequency.
        """
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._connection_mode = connection_mode
        self._bind_ip = bind_ip
        self._heartbeat_interval_sec = heartbeat_interval_sec

        self._state: WorkerState = WorkerState.DISCOVERING
        self._current_run_id: Optional[str] = None
        self._current_generation: Optional[str] = None
        self._training_process: Optional[subprocess.Popen] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._log_buffer: List[str] = []
        self._heartbeat_callback: Optional[Callable[[WorkerTelemetry, WorkerState], None]] = None

        logger.info(
            "WorkerAgent created: id='%s', host='%s', mode=%s, bind_ip=%s",
            self._worker_id,
            platform.node(),
            connection_mode.value,
            bind_ip,
        )

    @property
    def identity(self) -> WorkerIdentity:
        """Return immutable identity descriptor for this agent."""
        return WorkerIdentity(
            worker_id=self._worker_id,
            hostname=platform.node(),
            ip_address=self._bind_ip,
            connection_mode=self._connection_mode,
            os_name=platform.system(),
            python_version=sys.version.split()[0],
            pytorch_version="2.x",
        )

    @property
    def state(self) -> WorkerState:
        """Return current state of this agent."""
        return self._state

    def set_heartbeat_callback(
        self, callback: Callable[[WorkerTelemetry, WorkerState], None]
    ) -> None:
        """Register a callback for local in-process heartbeat delivery."""
        self._heartbeat_callback = callback

    def collect_telemetry(self) -> WorkerTelemetry:
        """Probe local hardware and construct a WorkerTelemetry payload."""
        hw = probe_hardware()
        gpu_name = hw.gpus[0].name if hw.gpus else None
        vram_used = (hw.gpus[0].vram_used_mb / 1024.0) if hw.gpus else None
        vram_total = (hw.gpus[0].vram_total_mb / 1024.0) if hw.gpus else None

        return WorkerTelemetry(
            cpu_percent=hw.cpu_percent,
            ram_used_gb=hw.ram_used_gb,
            ram_total_gb=hw.ram_total_gb,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            gpu_name=gpu_name,
            disk_free_gb=hw.disk_free_gb,
            is_direct_path=True,
        )

    def start_heartbeat_loop(self) -> None:
        """Start asynchronous heartbeat background thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.debug("Heartbeat thread already active")
            return

        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat_loop, daemon=True, name="WorkerAgent-Heartbeat"
        )
        self._heartbeat_thread.start()
        logger.debug("Started heartbeat thread (interval: %.1fs)", self._heartbeat_interval_sec)

    def _run_heartbeat_loop(self) -> None:
        """Background loop dispatching telemetry heartbeats."""
        while not self._stop_event.is_set():
            try:
                telemetry = self.collect_telemetry()
                if self._heartbeat_callback:
                    self._heartbeat_callback(telemetry, self._state)
            except Exception as e:
                logger.warning("Heartbeat telemetry collection error: %s", e)

            self._stop_event.wait(self._heartbeat_interval_sec)

    def start_training(
        self,
        run_id: str,
        generation_id: str,
        command: List[str],
        env_vars: Optional[Dict[str, str]] = None,
    ) -> None:
        """Launch local training subprocess.

        Args:
            run_id: Active run ID.
            generation_id: Active generation ID.
            command: Command array to execute.
            env_vars: Optional environment variables to merge.

        Raises:
            AgentError: If training is already active or process launch fails.
        """
        if self._state == WorkerState.TRAINING:
            raise AgentError(f"Worker '{self._worker_id}' is already training")

        self._current_run_id = run_id
        self._current_generation = generation_id
        self._state = WorkerState.STARTING
        logger.info("Worker '%s' starting training process: %s", self._worker_id, " ".join(command))

        try:
            import os

            merged_env = os.environ.copy()
            if env_vars:
                merged_env.update(env_vars)

            self._training_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=merged_env,
            )
            self._state = WorkerState.TRAINING

            # Thread to capture local stdout/stderr
            threading.Thread(
                target=self._capture_process_output,
                args=(self._training_process,),
                daemon=True,
                name="WorkerAgent-LogCapture",
            ).start()

        except Exception as e:
            self._state = WorkerState.FAILED
            msg = f"Failed to launch training process on worker '{self._worker_id}': {e}"
            logger.error(msg)
            raise AgentError(msg) from e

    def _capture_process_output(self, proc: subprocess.Popen) -> None:
        """Stream process stdout to local buffer and logging."""
        if proc.stdout is None:
            return
        for line in proc.stdout:
            clean_line = line.strip()
            if clean_line:
                self._log_buffer.append(clean_line)
                if len(self._log_buffer) > 2000:
                    self._log_buffer.pop(0)
                logger.debug("[%s] %s", self._worker_id, clean_line)

        proc.wait()
        if proc.returncode != 0 and self._state == WorkerState.TRAINING:
            logger.error("Training process exited with error code %d", proc.returncode)
            self._state = WorkerState.FAILED

    def stop_training(self) -> None:
        """Terminate active local training process."""
        self._state = WorkerState.STOPPING
        if self._training_process and self._training_process.poll() is None:
            logger.info("Terminating training process for worker '%s'", self._worker_id)
            self._training_process.terminate()
            try:
                self._training_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not exit; sending SIGKILL")
                self._training_process.kill()

        self._state = WorkerState.STOPPED
        logger.info("Worker '%s' training process stopped cleanly", self._worker_id)

    def shutdown(self) -> None:
        """Stop heartbeat loop and training processes."""
        self._stop_event.set()
        self.stop_training()
        logger.info("WorkerAgent '%s' shut down complete", self._worker_id)

    def get_logs(self, limit: int = 100) -> List[str]:
        """Return the most recent log lines from the training process."""
        return self._log_buffer[-limit:]
