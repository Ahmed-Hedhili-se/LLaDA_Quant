"""Quantization manifest: provenance, targeting audit trail, per-tensor metadata.

The manifest is what makes a run reproducible *and* auditable. Beyond the
config it records exactly which modules were converted, their shapes, and how
many bytes each one cost before and after — so a memory claim can be checked
against the artifact instead of taken on trust.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
class TargetedModule:
    """One module a quantization run actually converted.

    ``source_bytes`` and ``quantized_bytes`` are measured, not estimated, and
    are what :mod:`LLaDA_Quant.memory` cross-checks its accounting against.
    """

    name: str
    kind: str
    module_type: str
    shapes: Dict[str, List[int]]
    bits: int
    group_size: int
    packed: bool
    execution_mode: str
    source_bytes: int
    quantized_bytes: int

    @property
    def compression_ratio(self) -> float:
        """Quantized bytes divided by source bytes — lower is better."""
        return self.quantized_bytes / self.source_bytes if self.source_bytes else 1.0


@dataclass
class QuantEntry:
    """Metadata for one quantized tensor."""

    tensor_name: str
    shape: List[int]
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
    framework_version: str = "0.2.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_checkpoint: Optional[str] = None
    config: Optional[QuantConfig] = None
    entries: List[QuantEntry] = field(default_factory=list)
    targets: List[TargetedModule] = field(default_factory=list)

    @property
    def source_bytes(self) -> int:
        return sum(t.source_bytes for t in self.targets)

    @property
    def quantized_bytes(self) -> int:
        return sum(t.quantized_bytes for t in self.targets)

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "framework_version": self.framework_version,
            "created_at": self.created_at,
            "source_checkpoint": self.source_checkpoint,
            "config": self.config.to_dict() if self.config else None,
            "entries": [asdict(e) for e in self.entries],
            "targets": [asdict(t) for t in self.targets],
            "totals": {
                "source_bytes": self.source_bytes,
                "quantized_bytes": self.quantized_bytes,
                "module_count": len(self.targets),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationManifest":
        cfg = QuantConfig.from_dict(data["config"]) if data.get("config") else None
        entries = [QuantEntry(**e) for e in data.get("entries", [])]
        targets = [TargetedModule(**t) for t in data.get("targets", [])]
        return cls(
            format_version=data.get("format_version", FORMAT_VERSION),
            framework_version=data.get("framework_version", "unknown"),
            created_at=data.get("created_at", ""),
            source_checkpoint=data.get("source_checkpoint"),
            config=cfg,
            entries=entries,
            targets=targets,
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
