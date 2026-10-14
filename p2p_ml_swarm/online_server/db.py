"""SQLite persistence layer for Local ML Cluster Online Control API.

Provides ACID transactional guarantees for run metadata, worker registrations,
heartbeats, and checkpoint manifests without requiring an external database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Database:
    """Thread-safe SQLite database manager for authoritative cluster state."""

    def __init__(self, db_path: str = "cluster_state.db") -> None:
        """Initialize database and execute migrations."""
        self._db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Create necessary database schema tables."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    connection_mode TEXT NOT NULL,
                    os_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    rank INTEGER,
                    telemetry_json TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    generation_id TEXT NOT NULL,
                    connection_mode TEXT NOT NULL,
                    model_template TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    dataset_manifest_sha256 TEXT NOT NULL,
                    world_size INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    learning_rate REAL NOT NULL,
                    total_epochs INTEGER NOT NULL,
                    rendezvous_endpoint TEXT NOT NULL,
                    backend_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL NOT NULL,
                    samples_processed INTEGER NOT NULL,
                    throughput REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    global_step INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    val_loss REAL,
                    is_committed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                """
            )
        logger.info("Initialized cluster SQLite schema at '%s'", self._db_path)

    def upsert_worker(
        self,
        worker_id: str,
        hostname: str,
        ip_address: str,
        connection_mode: str,
        os_name: str,
        state: str,
        telemetry: Dict[str, Any],
        rank: Optional[int] = None,
    ) -> None:
        """Insert or update a worker record."""
        now = datetime.now(timezone.utc).isoformat()
        telemetry_str = json.dumps(telemetry)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO workers (worker_id, hostname, ip_address, connection_mode, os_name, state, rank, telemetry_json, last_heartbeat, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    state = excluded.state,
                    rank = COALESCE(excluded.rank, workers.rank),
                    telemetry_json = excluded.telemetry_json,
                    last_heartbeat = excluded.last_heartbeat;
                """,
                (
                    worker_id,
                    hostname,
                    ip_address,
                    connection_mode,
                    os_name,
                    state,
                    rank,
                    telemetry_str,
                    now,
                    now,
                ),
            )

    def update_heartbeat(
        self, worker_id: str, telemetry: Dict[str, Any], state: Optional[str] = None
    ) -> None:
        """Update last heartbeat timestamp and telemetry."""
        now = datetime.now(timezone.utc).isoformat()
        telemetry_str = json.dumps(telemetry)
        with self._get_connection() as conn:
            if state:
                conn.execute(
                    "UPDATE workers SET telemetry_json = ?, last_heartbeat = ?, state = ? WHERE worker_id = ?",
                    (telemetry_str, now, state, worker_id),
                )
            else:
                conn.execute(
                    "UPDATE workers SET telemetry_json = ?, last_heartbeat = ? WHERE worker_id = ?",
                    (telemetry_str, now, worker_id),
                )

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single worker by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            data["telemetry"] = json.loads(data["telemetry_json"])
            del data["telemetry_json"]
            return data

    def list_workers(self) -> List[Dict[str, Any]]:
        """List all registered workers."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM workers ORDER BY created_at ASC").fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["telemetry"] = json.loads(d["telemetry_json"])
                del d["telemetry_json"]
                results.append(d)
            return results

    def insert_run(
        self,
        run_id: str,
        generation_id: str,
        connection_mode: str,
        model_template: str,
        model_sha256: str,
        dataset_manifest_sha256: str,
        world_size: int,
        batch_size: int,
        learning_rate: float,
        total_epochs: int,
        rendezvous_endpoint: str,
        backend_name: str,
        state: str = "CREATED",
    ) -> None:
        """Insert a new run descriptor into the database."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, generation_id, connection_mode, model_template,
                    model_sha256, dataset_manifest_sha256, world_size,
                    batch_size, learning_rate, total_epochs, rendezvous_endpoint,
                    backend_name, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    generation_id,
                    connection_mode,
                    model_template,
                    model_sha256,
                    dataset_manifest_sha256,
                    world_size,
                    batch_size,
                    learning_rate,
                    total_epochs,
                    rendezvous_endpoint,
                    backend_name,
                    state,
                    now,
                    now,
                ),
            )

    def update_run_state(self, run_id: str, state: str) -> None:
        """Update authoritative state of a run."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (state, now, run_id),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch run descriptor by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def insert_metrics(
        self,
        run_id: str,
        generation_id: str,
        worker_id: str,
        step: int,
        epoch: int,
        train_loss: float,
        samples_processed: int,
        throughput: float,
    ) -> None:
        """Record step training metrics."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics (
                    run_id, generation_id, worker_id, step, epoch,
                    train_loss, samples_processed, throughput, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    generation_id,
                    worker_id,
                    step,
                    epoch,
                    train_loss,
                    samples_processed,
                    throughput,
                    now,
                ),
            )

    def get_run_metrics(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Retrieve recent metrics for a given run."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY step DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
