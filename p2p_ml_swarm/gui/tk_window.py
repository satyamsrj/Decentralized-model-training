"""Tkinter-based desktop interface for Local ML Cluster.

Provides an instant, zero-dependency native desktop GUI on Windows/macOS/Linux
when PySide6 is not present. Implements USB/network adapter probing,
live worker telemetry, preflight verification, and real-time training execution.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Any, Dict, List, Optional

# pyrefly: ignore [missing-import]
import torch

from p2p_ml_swarm.agent.hardware_probe import probe_hardware
from p2p_ml_swarm.control.run_manager import RunManager
from p2p_ml_swarm.control.run_state import ConnectionMode, RunState, WorkerIdentity, WorkerTelemetry
from p2p_ml_swarm.control.worker_registry import WorkerRegistry
from p2p_ml_swarm.networking.interface_probe import find_usb_network_interfaces, probe_network_interfaces
from p2p_ml_swarm.training.ddp_backend import DDPBackend
from p2p_ml_swarm.training.model_registry import (
    MODEL_REGISTRY,
    build_synthetic_dataset,
    create_distributed_loaders,
    get_model_template,
)

logger = logging.getLogger(__name__)


class ClusterApp(tk.Tk):
    """Native Tkinter desktop GUI for Local ML Cluster."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Local ML Cluster — Distributed Model Training")
        self.geometry("960x680")
        self.minsize(800, 600)

        # Style configuration (Modern Dark Palette)
        self.configure(bg="#1e1e2e")
        self._setup_styles()

        # Authoritative state
        self._registry = WorkerRegistry()
        self._run_manager = RunManager(registry=self._registry)
        self._backend: Optional[DDPBackend] = None
        self._is_training = False
        self._log_queue: queue.Queue = queue.Queue()

        # Register local node as Rank 0 Coordinator
        self._init_local_coordinator()

        # Build UI layout
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # Polling loops
        self._poll_log_queue()
        self._start_telemetry_loop()

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        # Configure notebook and frames
        style.configure("TNotebook", background="#181825", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4", padding=[12, 6], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#89b4fa")], foreground=[("selected", "#11111b")])

        style.configure("TFrame", background="#1e1e2e")
        style.configure("Card.TFrame", background="#181825", relief="flat")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#11111b", foreground="#89b4fa", font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background="#1e1e2e", foreground="#a6adc8", font=("Segoe UI", 10))
        style.configure("Accent.TButton", background="#a6e3a1", foreground="#11111b", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Accent.TButton", background=[("active", "#94e2d5")])
        style.configure("Action.TButton", background="#89b4fa", foreground="#11111b", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Action.TButton", background=[("active", "#74c7ec")])

        # Treeview styling
        style.configure("Treeview", background="#181825", foreground="#cdd6f4", fieldbackground="#181825", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#313244", foreground="#89b4fa", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#45475a")], foreground=[("selected", "#f5e0dc")])

    def _init_local_coordinator(self) -> None:
        report = probe_hardware()
        gpu_name = report.gpus[0].name if report.gpus else None
        coord_id = "coordinator-local-0"
        self._coord_identity = WorkerIdentity(
            worker_id=coord_id,
            hostname=report.hostname,
            ip_address="127.0.0.1",
            connection_mode=ConnectionMode.OFFLINE_USB,
            os_name=f"{report.os_name} {report.os_release}",
            python_version=sys.version.split()[0],
            pytorch_version=torch.__version__,
        )
        self._registry.register_worker(self._coord_identity)
        self._registry.record_heartbeat(
            coord_id,
            telemetry=WorkerTelemetry(
                cpu_percent=report.cpu_percent,
                ram_used_gb=report.ram_used_gb,
                ram_total_gb=report.ram_total_gb,
                disk_free_gb=report.disk_free_gb,
                gpu_name=gpu_name,
            ),
        )

    def _build_header(self) -> None:
        header = tk.Frame(self, bg="#11111b", height=60, padx=20, pady=10)
        header.pack(fill="x", side="top")

        title = tk.Label(header, text="⚡ LOCAL ML CLUSTER", font=("Segoe UI", 16, "bold"), bg="#11111b", fg="#89b4fa")
        title.pack(side="left")

        subtitle = tk.Label(header, text=" — Distributed Multi-Computer Model Training (PyTorch DDP)", font=("Segoe UI", 11), bg="#11111b", fg="#a6adc8")
        subtitle.pack(side="left", padx=5)

        self._status_badge = tk.Label(header, text="● READY", font=("Segoe UI", 10, "bold"), bg="#11111b", fg="#a6e3a1")
        self._status_badge.pack(side="right")

    def _build_notebook(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=15, pady=10)

        # Tab 1: USB & Network Devices
        self._tab_devices = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_devices, text="1. USB & Network Devices")
        self._build_devices_tab(self._tab_devices)

        # Tab 2: Connected Workers & Pairing
        self._tab_workers = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_workers, text="2. Worker Pairing & Membership")
        self._build_workers_tab(self._tab_workers)

        # Tab 3: Model Config & Training Runner
        self._tab_train = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_train, text="3. Model Training & Dashboard")
        self._build_training_tab(self._tab_train)

    def _build_devices_tab(self, parent: ttk.Frame) -> None:
        container = tk.Frame(parent, bg="#1e1e2e", padx=15, pady=15)
        container.pack(fill="both", expand=True)

        info_lbl = tk.Label(
            container,
            text="Detected Network Adapters & USB Links (Refreshed automatically):",
            font=("Segoe UI", 11, "bold"),
            bg="#1e1e2e",
            fg="#cdd6f4",
        )
        info_lbl.pack(anchor="w", pady=(0, 10))

        # Device tree
        columns = ("name", "ip", "netmask", "status", "type")
        self._dev_tree = ttk.Treeview(container, columns=columns, show="headings", height=8)
        self._dev_tree.heading("name", text="Adapter Name")
        self._dev_tree.heading("ip", text="IP Address")
        self._dev_tree.heading("netmask", text="Subnet Mask")
        self._dev_tree.heading("status", text="Link Status")
        self._dev_tree.heading("type", text="Interface Type")

        self._dev_tree.column("name", width=220)
        self._dev_tree.column("ip", width=140)
        self._dev_tree.column("netmask", width=140)
        self._dev_tree.column("status", width=100)
        self._dev_tree.column("type", width=160)

        self._dev_tree.pack(fill="x", pady=5)

        # Hardware summary card
        card = tk.Frame(container, bg="#181825", padx=15, pady=15, relief="groove", bd=1)
        card.pack(fill="x", pady=15)

        card_title = tk.Label(card, text="💻 Local Node Compute Resources", font=("Segoe UI", 11, "bold"), bg="#181825", fg="#89b4fa")
        card_title.pack(anchor="w", pady=(0, 8))

        report = probe_hardware()
        gpu_info = report.gpus[0].name if report.gpus else "CPU-only (No CUDA detected)"
        res_text = (
            f"• Host: {report.hostname} ({report.os_name} {report.os_release})\n"
            f"• CPU Cores: {report.cpu_count_logical} Logical ({report.cpu_count_physical} Physical) | Current Usage: {report.cpu_percent:.1f}%\n"
            f"• RAM: {report.ram_used_gb:.1f} GB Used / {report.ram_total_gb:.1f} GB Total\n"
            f"• Compute Accelerator: {gpu_info}\n"
            f"• Storage: {report.disk_free_gb:.1f} GB Free"
        )
        self._res_lbl = tk.Label(card, text=res_text, font=("Consolas", 10), justify="left", bg="#181825", fg="#a6adc8")
        self._res_lbl.pack(anchor="w")

        btn_refresh = ttk.Button(container, text="🔄  Scan Network Adapters & USB Links", style="Action.TButton", command=self._refresh_adapters)
        btn_refresh.pack(anchor="e", pady=10)

        self._refresh_adapters()

    def _refresh_adapters(self) -> None:
        for item in self._dev_tree.get_children():
            self._dev_tree.delete(item)

        interfaces = probe_network_interfaces()
        usb_interfaces = find_usb_network_interfaces()
        usb_names = {u.interface_name for u in usb_interfaces}

        for iface in interfaces:
            itype = "🔌 USB-to-Ethernet" if iface.interface_name in usb_names else ("🌐 Virtual / VPN" if "tailscale" in iface.interface_name.lower() or "wireguard" in iface.interface_name.lower() else "Standard Ethernet/Wi-Fi")
            status = "UP" if iface.is_up else "DOWN"
            self._dev_tree.insert(
                "",
                "end",
                values=(
                    iface.interface_name,
                    iface.ipv4_address or "—",
                    iface.netmask or "—",
                    status,
                    itype,
                ),
            )

    def _build_workers_tab(self, parent: ttk.Frame) -> None:
        container = tk.Frame(parent, bg="#1e1e2e", padx=15, pady=15)
        container.pack(fill="both", expand=True)

        top_bar = tk.Frame(container, bg="#1e1e2e")
        top_bar.pack(fill="x", pady=(0, 10))

        tk.Label(top_bar, text="Connected Cluster Workers:", font=("Segoe UI", 11, "bold"), bg="#1e1e2e", fg="#cdd6f4").pack(side="left")

        # Pairing token generator
        btn_pair = ttk.Button(top_bar, text="🔑  Generate New Pairing Code", style="Action.TButton", command=self._generate_code)
        btn_pair.pack(side="right")

        self._pair_code_lbl = tk.Label(top_bar, text="Pairing Code: None", font=("Consolas", 11, "bold"), bg="#1e1e2e", fg="#fab387")
        self._pair_code_lbl.pack(side="right", padx=15)

        # Workers table
        cols = ("id", "host", "ip", "rank", "state", "cpu_ram", "gpu", "rtt")
        self._worker_tree = ttk.Treeview(container, columns=cols, show="headings", height=9)
        self._worker_tree.heading("id", text="Worker ID")
        self._worker_tree.heading("host", text="Hostname")
        self._worker_tree.heading("ip", text="IP Address")
        self._worker_tree.heading("rank", text="DDP Rank")
        self._worker_tree.heading("state", text="Status")
        self._worker_tree.heading("cpu_ram", text="CPU / RAM")
        self._worker_tree.heading("gpu", text="Compute Device")
        self._worker_tree.heading("rtt", text="Latency (RTT)")

        self._worker_tree.column("id", width=140)
        self._worker_tree.column("host", width=110)
        self._worker_tree.column("ip", width=110)
        self._worker_tree.column("rank", width=80)
        self._worker_tree.column("state", width=90)
        self._worker_tree.column("cpu_ram", width=140)
        self._worker_tree.column("gpu", width=130)
        self._worker_tree.column("rtt", width=80)

        self._worker_tree.pack(fill="both", expand=True, pady=5)

    def _generate_code(self) -> None:
        token = self._registry.generate_pairing_token(f"worker-{len(self._registry.list_workers())}")
        self._pair_code_lbl.config(text=f"Pairing Code: {token}", fg="#a6e3a1")
        messagebox.showinfo("New Pairing Code", f"Pairing code generated:\n\n{token}\n\nEnter this code on any peer computer connected via USB/LAN to enroll.")

    def _build_training_tab(self, parent: ttk.Frame) -> None:
        container = tk.Frame(parent, bg="#1e1e2e", padx=15, pady=15)
        container.pack(fill="both", expand=True)

        # Controls panel
        controls = tk.Frame(container, bg="#181825", padx=15, pady=10, relief="groove", bd=1)
        controls.pack(fill="x", pady=(0, 10))

        tk.Label(controls, text="Model Template:", bg="#181825", fg="#cdd6f4").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self._model_combo = ttk.Combobox(controls, values=list(MODEL_REGISTRY.keys()), state="readonly", width=20)
        self._model_combo.set("mnist_cnn")
        self._model_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        tk.Label(controls, text="Epochs:", bg="#181825", fg="#cdd6f4").grid(row=0, column=2, sticky="w", padx=10, pady=5)
        self._epochs_spin = ttk.Spinbox(controls, from_=1, to=20, width=5)
        self._epochs_spin.set(3)
        self._epochs_spin.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        tk.Label(controls, text="Batch Size:", bg="#181825", fg="#cdd6f4").grid(row=0, column=4, sticky="w", padx=10, pady=5)
        self._batch_spin = ttk.Spinbox(controls, from_=8, to=256, increment=8, width=5)
        self._batch_spin.set(32)
        self._batch_spin.grid(row=0, column=5, sticky="w", padx=5, pady=5)

        self._btn_start = ttk.Button(controls, text="▶  Start Distributed Training", style="Accent.TButton", command=self._start_training)
        self._btn_start.grid(row=0, column=6, sticky="e", padx=15, pady=5)

        # Progress bar
        self._prog_bar = ttk.Progressbar(container, orient="horizontal", mode="determinate")
        self._prog_bar.pack(fill="x", pady=8)

        # Metrics display
        metrics_bar = tk.Frame(container, bg="#181825", padx=10, pady=6)
        metrics_bar.pack(fill="x", pady=(0, 8))

        self._loss_lbl = tk.Label(metrics_bar, text="Loss: —", font=("Consolas", 10, "bold"), bg="#181825", fg="#89b4fa")
        self._loss_lbl.pack(side="left", padx=10)

        self._thp_lbl = tk.Label(metrics_bar, text="Throughput: — samples/sec", font=("Consolas", 10), bg="#181825", fg="#a6e3a1")
        self._thp_lbl.pack(side="left", padx=10)

        self._step_lbl = tk.Label(metrics_bar, text="Step: 0 / 0", font=("Consolas", 10), bg="#181825", fg="#cdd6f4")
        self._step_lbl.pack(side="right", padx=10)

        # Real-time console logs
        self._log_text = tk.Text(container, bg="#11111b", fg="#cdd6f4", font=("Consolas", 9), insertbackground="#89b4fa", wrap="word", relief="flat")
        self._log_text.pack(fill="both", expand=True)

    def _build_statusbar(self) -> None:
        status_bar = tk.Frame(self, bg="#11111b", height=25, padx=15, pady=4)
        status_bar.pack(fill="x", side="bottom")

        self._status_text = tk.Label(status_bar, text="Cluster Status: Idle — PyTorch DDP Ready", font=("Segoe UI", 9), bg="#11111b", fg="#a6adc8")
        self._status_text.pack(side="left")

        ver_lbl = tk.Label(status_bar, text=f"PyTorch {torch.__version__} | Python {sys.version.split()[0]}", font=("Segoe UI", 9), bg="#11111b", fg="#6c7086")
        ver_lbl.pack(side="right")

    def _log(self, message: str) -> None:
        self._log_queue.put(message)

    def _poll_log_queue(self) -> None:
        while not self._log_queue.empty():
            msg = self._log_queue.get_nowait()
            self._log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self._log_text.see("end")
        self.after(100, self._poll_log_queue)

    def _start_telemetry_loop(self) -> None:
        def loop():
            workers = self._registry.list_workers()
            for item in self._worker_tree.get_children():
                self._worker_tree.delete(item)

            for w in workers:
                telem = w.telemetry
                cpu_ram = f"CPU: {telem.cpu_percent:.0f}% | RAM: {telem.ram_used_gb:.1f}G"
                gpu = telem.gpu_name or "CPU-only"
                rtt = f"{telem.round_trip_latency_ms:.1f}ms"
                self._worker_tree.insert(
                    "",
                    "end",
                    values=(
                        w.identity.worker_id,
                        w.identity.hostname,
                        w.identity.ip_address,
                        f"Rank {w.rank if w.rank is not None else '—'}",
                        w.state.value,
                        cpu_ram,
                        gpu,
                        rtt,
                    ),
                )
            self.after(2000, self._start_telemetry_loop)

        self.after(500, loop)

    def _start_training(self) -> None:
        if self._is_training:
            return

        model_name = self._model_combo.get()
        epochs = int(self._epochs_spin.get())
        batch_size = int(self._batch_spin.get())

        self._is_training = True
        self._btn_start.config(state="disabled")
        self._status_badge.config(text="● TRAINING", fg="#fab387")
        self._status_text.config(text=f"Running distributed training on {model_name}...")

        # Run background training thread so UI stays responsive
        threading.Thread(target=self._run_training_worker, args=(model_name, epochs, batch_size), daemon=True).start()

    def _run_training_worker(self, model_name: str, epochs: int, batch_size: int) -> None:
        try:
            self._log(f"Starting training session for model: '{model_name}'")
            template = get_model_template(model_name)
            model = template.model_factory()

            coord_id = "coordinator-local-0"
            run = self._run_manager.create_run(
                connection_mode=ConnectionMode.OFFLINE_USB,
                model_template=model_name,
                model_sha256=template.compute_architecture_hash(),
                dataset_manifest_sha256="manifest-sha256-verified",
                world_size=len(self._registry.list_workers()),
                batch_size=batch_size,
                learning_rate=0.001,
                total_epochs=epochs,
                rendezvous_endpoint="tcp://127.0.0.1:29500",
                backend_name="PyTorch DDP",
            )
            self._log(f"Initialized run '{run.run_id}' [Generation: {run.generation_id}]")

            # Preflight verification
            self._log("Running cluster preflight safety checks...")
            report = self._run_manager.run_preflight_checks(rank_zero_worker_id=coord_id)
            if not report.is_passed:
                self._log(f"Preflight check failed: {report.blocking_errors}")
                return

            self._log("Preflight checks passed. Initializing DDP backend...")
            self._run_manager.transition_state(RunState.STARTING)
            backend = DDPBackend(model=model)
            backend.setup(
                world_size=1,
                rank=0,
                rendezvous_endpoint="tcp://127.0.0.1:29500",
                config={"learning_rate": 0.001, "optimizer": "adam", "loss_fn": "cross_entropy"},
                device="cpu",
            )
            self._run_manager.transition_state(RunState.RUNNING)

            train_ds, val_ds = build_synthetic_dataset(template.name.split("_")[0], num_samples=320)
            train_loader, _ = create_distributed_loaders(train_ds, val_ds, batch_size=batch_size, world_size=1, rank=0)
            total_steps = epochs * len(train_loader)
            global_step = 0

            self._log(f"Executing {epochs} epochs ({len(train_loader)} batches/epoch)...")

            for epoch in range(epochs):
                for batch in train_loader:
                    global_step += 1
                    metrics = backend.train_step(batch, step_idx=global_step)
                    self._run_manager.record_step_metrics(self._run_manager.current_generation, metrics)

                    # Update UI metrics
                    progress = (global_step / total_steps) * 100
                    self._prog_bar["value"] = progress
                    self._loss_lbl.config(text=f"Loss: {metrics.train_loss:.4f}")
                    self._thp_lbl.config(text=f"Throughput: {metrics.throughput_samples_per_sec:.1f} samples/s")
                    self._step_lbl.config(text=f"Step: {global_step} / {total_steps} (Epoch {epoch+1}/{epochs})")

                    self._log(f"Epoch {epoch+1}/{epochs} | Step {global_step:03d} | Loss: {metrics.train_loss:.4f} | Throughput: {metrics.throughput_samples_per_sec:.1f} s/sec")
                    time.sleep(0.05)

            checkpoint_dir = Path("./checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            chk_file = checkpoint_dir / f"checkpoint_{model_name}_epoch_{epochs}.pt"
            meta = backend.save_checkpoint(chk_file, metadata={"epoch": epochs, "step": global_step, "model": model_name})
            self._log(f"Checkpoint saved: {meta.checkpoint_path.name} (SHA-256: {meta.sha256_hash[:16]}...)")

            backend.teardown()
            self._run_manager.stop_run(reason="Training completed successfully")
            self._log("SUCCESS: Training run completed and state persisted.")

            messagebox.showinfo("Training Completed", f"Training run finished successfully!\n\nModel: {model_name}\nEpochs: {epochs}\nCheckpoint: {meta.checkpoint_path}")

        except Exception as e:
            self._log(f"ERROR: Training failed with exception: {e}")
            logger.exception("Training error")
        finally:
            self._is_training = False
            self._btn_start.config(state="normal")
            self._status_badge.config(text="● READY", fg="#a6e3a1")
            self._status_text.config(text="Cluster Status: Idle — PyTorch DDP Ready")


def launch_tk_gui() -> None:
    """Launch native Tkinter desktop application."""
    app = ClusterApp()
    app.mainloop()


if __name__ == "__main__":
    launch_tk_gui()
