from __future__ import annotations

import os

from p2p_ml_swarm.exceptions import ConfigError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SwarmConfig:
    bootnode_addr: str
    run_id: str
    bucket_url: str
    bucket_name: str
    batch_size_per_step: int = 32
    target_batch_size: int = 256
    shard_size_mb: int = 50
    matchmaking_time: float = 15.0
    averaging_timeout: float = 60.0
    token_timeout_sec: float = 300.0
    offload_optimizer: bool = False
    auxiliary: bool = False
    client_mode: bool = True
    max_upload_mbps: Optional[float] = None
    max_download_mbps: Optional[float] = None
    checkpoint_dir: Optional[str] = None
    learning_rate: float = 1e-4
    s3_endpoint_url: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: str = "auto"
    extra: dict = field(default_factory=dict)

    _ENV_MAP: dict = field(default_factory=lambda: {
        "SWARM_BOOTNODE_ADDR": "bootnode_addr",
        "SWARM_RUN_ID": "run_id",
        "SWARM_BUCKET_URL": "bucket_url",
        "SWARM_BUCKET_NAME": "bucket_name",
        "SWARM_S3_ENDPOINT_URL": "s3_endpoint_url",
        "SWARM_S3_ACCESS_KEY": "s3_access_key",
        "SWARM_S3_SECRET_KEY": "s3_secret_key",
        "SWARM_S3_REGION": "s3_region",
        "SWARM_CHECKPOINT_DIR": "checkpoint_dir",
    }, init=False, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SwarmConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        known_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != "_ENV_MAP"}
        init_kwargs = {k: v for k, v in raw.items() if k in known_fields}
        extra = {k: v for k, v in raw.items() if k not in known_fields}
        if extra:
            init_kwargs["extra"] = extra

        config = cls(**init_kwargs)
        config._apply_env_overrides()
        config.validate()
        return config

    def _apply_env_overrides(self) -> None:
        for env_key, field_name in self._ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                current = getattr(self, field_name)
                if isinstance(current, (int, float)) and not isinstance(current, bool):
                    val = type(current)(val)
                elif isinstance(current, bool):
                    val = val.lower() in ("1", "true", "yes")
                setattr(self, field_name, val)

    def validate(self) -> None:
        if not self.bootnode_addr:
            raise ConfigError("bootnode_addr is required")
        if not self.run_id:
            raise ConfigError("run_id is required")
        if not self.bucket_url:
            raise ConfigError("bucket_url is required")
        if not self.bucket_name:
            raise ConfigError("bucket_name is required")
        if self.batch_size_per_step <= 0:
            raise ConfigError("batch_size_per_step must be positive")
        if self.target_batch_size < self.batch_size_per_step:
            raise ConfigError("target_batch_size must be >= batch_size_per_step")
        if self.matchmaking_time <= 0:
            raise ConfigError("matchmaking_time must be positive")
        if self.averaging_timeout <= 0:
            raise ConfigError("averaging_timeout must be positive")
