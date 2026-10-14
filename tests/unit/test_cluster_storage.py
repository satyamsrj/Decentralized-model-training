"""Unit tests for artifact integrity, streaming SHA-256 digests, and manifests."""

import json
from pathlib import Path
import pytest

from p2p_ml_swarm.storage.integrity import (
    ArtifactManifest,
    FileDigest,
    IntegrityError,
    compute_sha256,
    create_directory_manifest,
    verify_file_integrity,
)


def test_compute_sha256(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"hello world decentralized ml")

    digest = compute_sha256(test_file)
    assert len(digest) == 64  # Hex SHA-256 is 64 characters

    # Same content produces identical hash
    test_file_2 = tmp_path / "sample_copy.txt"
    test_file_2.write_bytes(b"hello world decentralized ml")
    assert compute_sha256(test_file_2) == digest


def test_compute_sha256_missing_file(tmp_path: Path):
    missing_file = tmp_path / "non_existent.bin"
    with pytest.raises(FileNotFoundError):
        compute_sha256(missing_file)


def test_verify_file_integrity_success_and_failure(tmp_path: Path):
    test_file = tmp_path / "weights.pt"
    test_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

    expected_hash = compute_sha256(test_file)
    assert verify_file_integrity(test_file, expected_hash)

    # Corrupt the file
    test_file.write_bytes(b"\x00\x01\x02\x03\x04\xFF")
    with pytest.raises(IntegrityError):
        verify_file_integrity(test_file, expected_hash)


def test_artifact_manifest_serialization(tmp_path: Path):
    dir_path = tmp_path / "artifacts"
    dir_path.mkdir()

    f1 = dir_path / "model.pt"
    f1.write_bytes(b"fake model tensor data")
    f2 = dir_path / "config.json"
    f2.write_text('{"lr": 0.001}')

    manifest = create_directory_manifest(
        directory=dir_path,
        manifest_id="run-test-manifest",
        metadata={"architecture": "mnist_cnn"},
    )

    assert manifest.manifest_id == "run-test-manifest"
    assert len(manifest.files) == 2
    assert manifest.metadata["architecture"] == "mnist_cnn"

    # Test JSON round-trip
    json_data = manifest.to_json()
    reconstructed = ArtifactManifest.from_json(json_data)

    assert reconstructed.manifest_id == manifest.manifest_id
    assert len(reconstructed.files) == len(manifest.files)
    assert reconstructed.files[0].sha256 == manifest.files[0].sha256
