"""Custom exception hierarchy for p2p_ml_swarm."""


class SwarmError(Exception):
    """Base exception for all swarm operations."""


class ConfigError(SwarmError):
    """Invalid YAML, missing required fields, or bad values."""


class DHTConnectionError(SwarmError):
    """Bootnode unreachable, DHT join failed, or peer timeout."""


class TokenCheckoutError(SwarmError):
    """Atomic increment failed, token contested, or DHT write rejected."""


class StorageError(SwarmError):
    """Bucket unreachable, upload/download failed, or auth error."""


class ShardUploadError(StorageError):
    """Failed to upload a shard to cloud storage."""


class ShardDownloadError(StorageError):
    """Failed to download a shard from cloud storage."""


class ShardingError(SwarmError):
    """Dataset too small, serialization failed, or corrupt safetensor."""


class TrainingError(SwarmError):
    """Forward/backward crash or OOM."""


class NaNLossError(TrainingError):
    """Loss became NaN or Inf during training."""


class PeerSyncError(TrainingError):
    """Gossip averaging or peer synchronization failed."""


class CheckpointError(SwarmError):
    """Checkpoint save/load failed or corrupt state file."""
