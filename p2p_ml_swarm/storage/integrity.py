"""Artifact integrity and SHA-256 verification utilities for Local ML Cluster.

Ensures that model definitions, dataset manifests, and checkpoint files
match exact cryptographic hashes before preflight or training begins.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from p2p_ml_swarm.exceptions import SwarmError

logger = logging.getLogger(__name__)


class IntegrityError(SwarmError):
    """Raised when an artifact's cryptographic hash does not match expectation."""

    pass


@dataclass(frozen=True)
class FileDigest:
    """Record of a validated file artifact."""

    file_path: str
    sha256: str
    size_bytes: int


@dataclass
class ArtifactManifest:
    """Cryptographic manifest of expected files for a run or dataset."""

    manifest_version: str = "1.0"
    manifest_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    files: List[FileDigest] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize manifest to deterministic JSON."""
        return json.dumps(
            {
                "manifest_version": self.manifest_version,
                "manifest_id": self.manifest_id,
                "created_at": self.created_at,
                "files": [
                    {"file_path": f.file_path, "sha256": f.sha256, "size_bytes": f.size_bytes}
                    for f in self.files
                ],
                "metadata": self.metadata,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, json_str: str) -> ArtifactManifest:
        """Deserialize manifest from JSON."""
        data = json.loads(json_str)
        files = [
            FileDigest(
                file_path=item["file_path"],
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
            )
            for item in data.get("files", [])
        ]
        return cls(
            manifest_version=data.get("manifest_version", "1.0"),
            manifest_id=data.get("manifest_id", ""),
            created_at=data.get("created_at", ""),
            files=files,
            metadata=data.get("metadata", {}),
        )


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file using streaming reads.

    Args:
        file_path: Absolute or relative path to the target file.
        chunk_size: Byte size for buffer chunks (default 64KB).

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the target file does not exist.
        PermissionError: If the file cannot be accessed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute hash: file not found at '{path}'")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_integrity(file_path: Path, expected_sha256: str) -> bool:
    """Verify that a file's SHA-256 matches an expected digest.

    Args:
        file_path: Path to the target file.
        expected_sha256: Target SHA-256 string (case-insensitive).

    Returns:
        True if the hash matches.

    Raises:
        IntegrityError: If the computed hash does not match expected_sha256.
    """
    path = Path(file_path)
    computed = compute_sha256(path)
    if computed.lower() != expected_sha256.lower():
        msg = (
            f"Integrity check failed for '{path.name}': "
            f"computed SHA-256={computed[:12]}... does not match expected={expected_sha256[:12]}..."
        )
        logger.error(msg)
        raise IntegrityError(msg)
    logger.debug("Integrity verified for '%s' (SHA-256: %s...)", path.name, computed[:12])
    return True


def create_directory_manifest(
    directory: Path,
    manifest_id: str = "",
    pattern: str = "*",
    metadata: Optional[Dict[str, Any]] = None,
) -> ArtifactManifest:
    """Scan a directory and generate an ArtifactManifest with SHA-256 digests.

    Args:
        directory: Path to the directory to scan.
        manifest_id: Optional unique identifier for the manifest.
        pattern: Glob pattern to filter files (default '*').
        metadata: Optional metadata dictionary to attach.

    Returns:
        Populated ArtifactManifest object.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: '{dir_path}'")

    file_digests: List[FileDigest] = []
    for file_path in sorted(dir_path.glob(pattern)):
        if file_path.is_file():
            digest = compute_sha256(file_path)
            size = file_path.stat().st_size
            rel_path = file_path.relative_to(dir_path).as_posix()
            file_digests.append(FileDigest(file_path=rel_path, sha256=digest, size_bytes=size))

    manifest = ArtifactManifest(
        manifest_id=manifest_id or dir_path.name,
        files=file_digests,
        metadata=metadata or {},
    )
    logger.info(
        "Created manifest for '%s': %d files indexed (manifest_id: %s)",
        dir_path.name,
        len(file_digests),
        manifest.manifest_id,
    )
    return manifest
