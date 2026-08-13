"""Quantization manifest: provenance and per-tensor metadata for a checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import torch

from ..config import FORMAT_VERSION, QuantConfig

MANIFEST_FILENAME = "quantization.json"
SOURCE_FILENAME = "source-checkpoint.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class QuantEntry:
    """Metadata for one quantized tensor."""

    tensor_name: str
    shape: list[int]
    bits: int
    group_size: int
    storage_dtype: str
    compute_dtype: str
    source_tensor: Optional[str] = None
    sha256: Optional[str] = None


@dataclass
class QuantizationManifest:
    """Versioned, human-readable record of everything that produced a checkpoint."""

    format_version: int = FORMAT_VERSION
    framework_version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_checkpoint: Optional[str] = None
    config: Optional[QuantConfig] = None
    entries: list[QuantEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "framework_version": self.framework_version,
            "created_at": self.created_at,
            "source_checkpoint": self.source_checkpoint,
            "config": self.config.to_dict() if self.config else None,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationManifest":
        cfg = QuantConfig.from_dict(data["config"]) if data.get("config") else None
        entries = [QuantEntry(**e) for e in data.get("entries", [])]
        return cls(
            format_version=data.get("format_version", FORMAT_VERSION),
            framework_version=data.get("framework_version", "unknown"),
            created_at=data.get("created_at", ""),
            source_checkpoint=data.get("source_checkpoint"),
            config=cfg,
            entries=entries,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save(self, directory: str) -> None:
        with open(os.path.join(directory, MANIFEST_FILENAME), "w") as f:
            f.write(self.to_json())


def tensor_hash(t: torch.Tensor) -> str:
    """Deterministic hash of a tensor's values (CPU, byteswap-safe)."""
    t = t.detach().float().contiguous()
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()


def write_source_checkpoint_meta(directory: str, source: Optional[str]) -> None:
    """Record the unquantized source so artifacts are provably traceable."""
    meta = {"source_checkpoint": source}
    if source and os.path.isfile(source):
        meta["source_sha256"] = _sha256(source)
    with open(os.path.join(directory, SOURCE_FILENAME), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)