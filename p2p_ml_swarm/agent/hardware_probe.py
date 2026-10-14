"""Hardware and environment telemetry probe for Local ML Cluster.

Collects real-time OS, CPU, RAM, VRAM, and storage metrics to power
worker profiling and preflight validation checks.
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuDeviceReport:
    """Telemetry report for an individual GPU accelerator."""

    device_index: int
    name: str
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    compute_capability: str
    cuda_driver_version: Optional[str] = None


@dataclass
class HardwareReport:
    """Comprehensive hardware and runtime capability report for a machine."""

    hostname: str
    os_name: str
    os_release: str
    os_architecture: str
    python_version: str
    pytorch_version: str
    cpu_count_physical: int
    cpu_count_logical: int
    cpu_percent: float
    ram_total_gb: float
    ram_used_gb: float
    ram_free_gb: float
    disk_total_gb: float
    disk_free_gb: float
    has_cuda: bool
    cuda_device_count: int
    gpus: List[GpuDeviceReport] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "hostname": self.hostname,
            "os_name": self.os_name,
            "os_release": self.os_release,
            "os_architecture": self.os_architecture,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cpu_count_physical": self.cpu_count_physical,
            "cpu_count_logical": self.cpu_count_logical,
            "cpu_percent": self.cpu_percent,
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ram_free_gb": round(self.ram_free_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "has_cuda": self.has_cuda,
            "cuda_device_count": self.cuda_device_count,
            "gpus": [
                {
                    "device_index": g.device_index,
                    "name": g.name,
                    "vram_total_mb": round(g.vram_total_mb, 1),
                    "vram_used_mb": round(g.vram_used_mb, 1),
                    "vram_free_mb": round(g.vram_free_mb, 1),
                    "compute_capability": g.compute_capability,
                }
                for g in self.gpus
            ],
            "timestamp": self.timestamp,
        }


def probe_hardware(disk_path: str = ".") -> HardwareReport:
    """Collect comprehensive hardware telemetry and runtime capabilities.

    Args:
        disk_path: Target directory to measure disk space on (default current dir).

    Returns:
        HardwareReport containing current CPU, RAM, GPU, and disk metrics.
    """
    logger.debug("Executing hardware probe on host: %s", platform.node())

    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_phys = psutil.cpu_count(logical=False) or 1
    cpu_log = psutil.cpu_count(logical=True) or 1

    # RAM metrics
    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / (1024**3)
    ram_used_gb = vm.used / (1024**3)
    ram_free_gb = vm.available / (1024**3)

    # Disk metrics
    usage = shutil.disk_usage(disk_path)
    disk_total_gb = usage.total / (1024**3)
    disk_free_gb = usage.free / (1024**3)

    # GPU / CUDA probe
    has_cuda = torch.cuda.is_available()
    gpu_reports: List[GpuDeviceReport] = []
    if has_cuda:
        for idx in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                vram_total = props.total_memory / (1024**2)
                vram_allocated = torch.cuda.memory_allocated(idx) / (1024**2)
                vram_free = vram_total - vram_allocated
                cc = f"{props.major}.{props.minor}"
                gpu_reports.append(
                    GpuDeviceReport(
                        device_index=idx,
                        name=name,
                        vram_total_mb=vram_total,
                        vram_used_mb=vram_allocated,
                        vram_free_mb=vram_free,
                        compute_capability=cc,
                    )
                )
            except Exception as e:
                logger.warning("Failed to probe CUDA device %d: %s", idx, e)

    report = HardwareReport(
        hostname=platform.node(),
        os_name=platform.system(),
        os_release=platform.release(),
        os_architecture=platform.machine(),
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        cpu_count_physical=cpu_phys,
        cpu_count_logical=cpu_log,
        cpu_percent=cpu_percent,
        ram_total_gb=ram_total_gb,
        ram_used_gb=ram_used_gb,
        ram_free_gb=ram_free_gb,
        disk_total_gb=disk_total_gb,
        disk_free_gb=disk_free_gb,
        has_cuda=has_cuda,
        cuda_device_count=len(gpu_reports),
        gpus=gpu_reports,
    )
    logger.debug(
        "Probe complete: CPU=%.1f%%, RAM=%.1f/%.1fGB, Disk=%.1fGB free, CUDA=%s",
        report.cpu_percent,
        report.ram_used_gb,
        report.ram_total_gb,
        report.disk_free_gb,
        report.has_cuda,
    )
    return report
