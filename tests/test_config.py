import os
import tempfile
from pathlib import Path

import pytest
import yaml

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import ConfigError


@pytest.fixture
def minimal_yaml(tmp_path):
    data = {
        "bootnode_addr": "/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        "run_id": "test_run",
        "bucket_url": "https://test.r2.cloudflarestorage.com",
        "bucket_name": "test-bucket",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_minimal(minimal_yaml):
    config = SwarmConfig.from_yaml(minimal_yaml)
    assert config.bootnode_addr == "/ip4/1.2.3.4/tcp/9000/p2p/QmTest"
    assert config.run_id == "test_run"
    assert config.batch_size_per_step == 32
    assert config.target_batch_size == 256


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        SwarmConfig.from_yaml("/nonexistent/path.yaml")


def test_missing_required_field(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"run_id": "test"}))
    with pytest.raises(TypeError):
        SwarmConfig.from_yaml(path)


def test_validation_bad_batch_size(tmp_path):
    data = {
        "bootnode_addr": "/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        "run_id": "test",
        "bucket_url": "https://test.com",
        "bucket_name": "bucket",
        "batch_size_per_step": -1,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(data))
    with pytest.raises(ConfigError, match="batch_size_per_step must be positive"):
        SwarmConfig.from_yaml(path)


def test_env_override(minimal_yaml, monkeypatch):
    monkeypatch.setenv("SWARM_RUN_ID", "overridden_run")
    config = SwarmConfig.from_yaml(minimal_yaml)
    assert config.run_id == "overridden_run"


def test_extra_fields_preserved(tmp_path):
    data = {
        "bootnode_addr": "/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        "run_id": "test",
        "bucket_url": "https://test.com",
        "bucket_name": "bucket",
        "custom_field": "custom_value",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    config = SwarmConfig.from_yaml(path)
    assert config.extra["custom_field"] == "custom_value"
