"""Real-time training monitoring with Rich CLI dashboard and TensorBoard logging."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import psutil
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

# TensorBoard import is optional
try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment, misc]
    _TB_AVAILABLE = False


class SwarmMonitor:
    """Monitors and displays swarm training metrics via Rich console and TensorBoard."""

    def __init__(
        self,
        run_id: str,
        tensorboard_dir: Optional[str | Path] = None,
        enable_rich: bool = True,
    ) -> None:
        self._run_id = run_id
        self._console = Console() if enable_rich else None
        self._start_time = time.time()
        self._step_count = 0
        self._total_samples = 0
        self._recent_losses: list[float] = []
        self._max_loss_history = 100

        # TensorBoard
        self._tb_writer: Optional[SummaryWriter] = None  # type: ignore[assignment]
        if tensorboard_dir and _TB_AVAILABLE:
            tb_path = Path(tensorboard_dir) / run_id
            tb_path.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_path))
            logger.info(f"TensorBoard logging to {tb_path}")
        elif tensorboard_dir and not _TB_AVAILABLE:
            logger.warning("TensorBoard requested but torch.utils.tensorboard not available")

        # Current state for dashboard
        self._current_epoch = 0
        self._current_batch = 0
        self._current_loss = 0.0
        self._current_lr = 0.0
        self._current_peers = 0
        self._current_throughput = 0.0

    def log_step(
        self,
        epoch: int,
        batch_id: int,
        loss: float,
        lr: float,
        peer_count: int,
        batch_size: int = 0,
    ) -> None:
        """Log a single training step."""
        self._step_count += 1
        self._total_samples += batch_size
        self._recent_losses.append(loss)
        if len(self._recent_losses) > self._max_loss_history:
            self._recent_losses.pop(0)

        elapsed = time.time() - self._start_time
        throughput = self._total_samples / elapsed if elapsed > 0 else 0

        # Update current state
        self._current_epoch = epoch
        self._current_batch = batch_id
        self._current_loss = loss
        self._current_lr = lr
        self._current_peers = peer_count
        self._current_throughput = throughput

        # Log to console
        if self._step_count % 10 == 0:
            avg_loss = sum(self._recent_losses[-10:]) / min(10, len(self._recent_losses))
            logger.info(
                f"[Epoch {epoch}] batch={batch_id} loss={loss:.4f} "
                f"avg_loss={avg_loss:.4f} lr={lr:.2e} peers={peer_count} "
                f"throughput={throughput:.1f} samples/s"
            )

        # TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss", loss, self._step_count)
            self._tb_writer.add_scalar("train/learning_rate", lr, self._step_count)
            self._tb_writer.add_scalar("swarm/peer_count", peer_count, self._step_count)
            self._tb_writer.add_scalar("swarm/throughput_samples_sec", throughput, self._step_count)
            self._tb_writer.add_scalar("system/cpu_percent", psutil.cpu_percent(), self._step_count)
            mem = psutil.virtual_memory()
            self._tb_writer.add_scalar("system/memory_used_gb", mem.used / 1e9, self._step_count)

    def log_epoch_summary(self, epoch: int, avg_loss: float, elapsed_sec: float) -> None:
        """Log a summary at the end of an epoch."""
        msg = (
            f"═══ Epoch {epoch} Complete ═══  "
            f"avg_loss={avg_loss:.4f}  elapsed={elapsed_sec:.1f}s  "
            f"total_steps={self._step_count}  total_samples={self._total_samples}"
        )
        logger.info(msg)

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)
            self._tb_writer.add_scalar("train/epoch_time_sec", elapsed_sec, epoch)

        if self._console:
            self._console.print(
                Panel(
                    f"[bold green]Epoch {epoch}[/] completed\n"
                    f"  Average Loss: [cyan]{avg_loss:.4f}[/]\n"
                    f"  Elapsed: [yellow]{elapsed_sec:.1f}s[/]\n"
                    f"  Total Steps: {self._step_count}\n"
                    f"  Total Samples: {self._total_samples}",
                    title="📊 Epoch Summary",
                    border_style="green",
                )
            )

    def print_status_table(self) -> None:
        """Print a rich status table to the console."""
        if not self._console:
            return

        elapsed = time.time() - self._start_time

        table = Table(title=f"🌐 Swarm [{self._run_id}]", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white", width=20)
        table.add_column("Value", style="green", width=25)

        table.add_row("Epoch", str(self._current_epoch))
        table.add_row("Current Batch", str(self._current_batch))
        table.add_row("Loss", f"{self._current_loss:.4f}")
        table.add_row("Learning Rate", f"{self._current_lr:.2e}")
        table.add_row("Active Peers", str(self._current_peers))
        table.add_row("Throughput", f"{self._current_throughput:.1f} samples/s")
        table.add_row("Total Steps", str(self._step_count))
        table.add_row("Total Samples", str(self._total_samples))
        table.add_row("Elapsed", f"{elapsed:.0f}s")
        table.add_row("CPU Usage", f"{psutil.cpu_percent():.1f}%")
        mem = psutil.virtual_memory()
        table.add_row("Memory", f"{mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB")

        self._console.print(table)

    def close(self) -> None:
        """Flush and close logging backends."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            logger.info("TensorBoard writer closed")

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def avg_recent_loss(self) -> float:
        if not self._recent_losses:
            return 0.0
        return sum(self._recent_losses) / len(self._recent_losses)
