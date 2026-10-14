"""FastAPI Control Plane service for Local ML Cluster Online (VM) mode.

Provides authenticated endpoints for worker enrollment, heartbeats,
run creation, preflight coordination, and metrics streaming.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field

from p2p_ml_swarm.online_server.db import Database

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local ML Cluster Control API",
    version="1.0.0",
    description="Control plane for VM-assisted and local multi-laptop distributed training.",
)

# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Dependency provider for database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


# ── Pydantic Request/Response Models ───────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class WorkerEnrollRequest(BaseModel):
    worker_id: str
    hostname: str
    ip_address: str
    connection_mode: str = "ONLINE_VM"
    os_name: str
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    state: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    run_id: str
    generation_id: str
    connection_mode: str = "ONLINE_VM"
    model_template: str
    model_sha256: str
    dataset_manifest_sha256: str
    world_size: int
    batch_size: int
    learning_rate: float
    total_epochs: int
    rendezvous_endpoint: str
    backend_name: str = "PyTorch DDP"


class PreflightRequest(BaseModel):
    rank_zero_worker_id: str


class RecordMetricsRequest(BaseModel):
    generation_id: str
    worker_id: str
    step: int
    epoch: int
    train_loss: float
    samples_processed: int
    throughput: float


# ── API Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Service health probe."""
    return HealthResponse()


@app.post("/v1/workers/enroll", status_code=status.HTTP_201_CREATED)
def enroll_worker(req: WorkerEnrollRequest, db: Database = Depends(get_db)) -> Dict[str, Any]:
    """Register or re-enroll a worker node."""
    logger.info("Enrolling worker '%s' from %s (%s)", req.worker_id, req.hostname, req.ip_address)
    db.upsert_worker(
        worker_id=req.worker_id,
        hostname=req.hostname,
        ip_address=req.ip_address,
        connection_mode=req.connection_mode,
        os_name=req.os_name,
        state="READY",
        telemetry=req.telemetry,
    )
    return {"status": "enrolled", "worker_id": req.worker_id}


@app.post("/v1/workers/heartbeat")
def worker_heartbeat(
    req: WorkerHeartbeatRequest, db: Database = Depends(get_db)
) -> Dict[str, Any]:
    """Receive heartbeat ping with updated hardware telemetry."""
    worker = db.get_worker(req.worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker '{req.worker_id}' is not enrolled",
        )
    db.update_heartbeat(req.worker_id, req.telemetry, state=req.state)
    return {"status": "acknowledged"}


@app.get("/v1/workers")
def list_workers(db: Database = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all registered workers and their latest telemetry."""
    return db.list_workers()


@app.post("/v1/runs", status_code=status.HTTP_201_CREATED)
def create_run(req: CreateRunRequest, db: Database = Depends(get_db)) -> Dict[str, Any]:
    """Create a new distributed training run in the registry."""
    existing = db.get_run(req.run_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{req.run_id}' already exists",
        )

    db.insert_run(
        run_id=req.run_id,
        generation_id=req.generation_id,
        connection_mode=req.connection_mode,
        model_template=req.model_template,
        model_sha256=req.model_sha256,
        dataset_manifest_sha256=req.dataset_manifest_sha256,
        world_size=req.world_size,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        total_epochs=req.total_epochs,
        rendezvous_endpoint=req.rendezvous_endpoint,
        backend_name=req.backend_name,
        state="CREATED",
    )
    logger.info("Created run '%s' via Control API", req.run_id)
    return {"status": "created", "run_id": req.run_id}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, db: Database = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve details and state of a training run."""
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


@app.post("/v1/runs/{run_id}/start")
def start_run(run_id: str, db: Database = Depends(get_db)) -> Dict[str, Any]:
    """Transition run to RUNNING state."""
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    db.update_run_state(run_id, "RUNNING")
    logger.info("Run '%s' marked as RUNNING", run_id)
    return {"status": "started", "run_id": run_id}


@app.post("/v1/runs/{run_id}/stop")
def stop_run(run_id: str, db: Database = Depends(get_db)) -> Dict[str, Any]:
    """Transition run to STOPPED state."""
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    db.update_run_state(run_id, "STOPPED")
    logger.info("Run '%s' marked as STOPPED", run_id)
    return {"status": "stopped", "run_id": run_id}


@app.post("/v1/runs/{run_id}/metrics")
def record_metrics(
    run_id: str, req: RecordMetricsRequest, db: Database = Depends(get_db)
) -> Dict[str, Any]:
    """Record a worker's step training metrics."""
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if req.generation_id != run["generation_id"]:
        logger.warning(
            "Rejected stale metrics for run '%s' from gen '%s' (active: '%s')",
            run_id,
            req.generation_id,
            run["generation_id"],
        )
        return {"status": "rejected_stale_generation"}

    db.insert_metrics(
        run_id=run_id,
        generation_id=req.generation_id,
        worker_id=req.worker_id,
        step=req.step,
        epoch=req.epoch,
        train_loss=req.train_loss,
        samples_processed=req.samples_processed,
        throughput=req.throughput,
    )
    return {"status": "recorded"}


@app.get("/v1/runs/{run_id}/metrics")
def get_metrics(
    run_id: str, limit: int = 100, db: Database = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Retrieve recent metrics history for dashboard plotting."""
    return db.get_run_metrics(run_id, limit=limit)
