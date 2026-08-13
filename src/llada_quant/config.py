"""Serializable quantization configuration and target-selection policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple, Union

FORMAT_VERSION = 1

DEFAULT_EXCLUDES = ("router", "norm", "embed_tokens", "lm_head", "gate")


@dataclass(frozen=True)
class QuantConfig:
    """Everything needed to reproduce a quantization run.

    Attributes:
        bits: Bit-width of the stored integer weights (8 or 4).
        group_size: Number of consecutive K-dimension elements sharing one
            scale, along the last axis of a weight tensor. Use ``-1`` (or any
            value not dividing the dimension) for per-tensor scaling.
        targets: Which component kinds to quantize. See ``ComponentTarget``.
        exclude: Substring patterns; any module whose name contains one of
            these is never quantized.
        compute_dtype: Dtype activations are kept in (weights are dequantized
            into this dtype for the reference matmul).
        scale_dtype: Storage dtype for scales.
        source_checkpoint: Path (or URL) of the unquantized source checkpoint
            the quantized artifact was derived from.
    """

    bits: int = 8
    group_size: int = 128
    targets: Tuple[str, ...] = ("expert",)
    exclude: Tuple[str, ...] = DEFAULT_EXCLUDES
    compute_dtype: str = "bfloat16"
    scale_dtype: str = "float32"
    source_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.bits not in (8, 4):
            raise ValueError(f"bits must be 8 or 4, got {self.bits}")
        if self.group_size <= 0 and self.group_size != -1:
            raise ValueError(f"group_size must be > 0 or -1, got {self.group_size}")

    def to_dict(self) -> dict:
        return {"format_version": FORMAT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "QuantConfig":
        data = dict(data)
        for key in ("targets", "exclude"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        cfg = cls(**{k: v for k, v in data.items() if k != "format_version"})
        return cfg

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


COMPONENT_LINEAR = "linear"
COMPONENT_EXPERT = "expert"


def target_matches_targets(module_name: str, target: str) -> bool:
    if target == COMPONENT_EXPERT:
        return any(t in module_name for t in ("expert", "mlp"))
    if target == COMPONENT_LINEAR:
        return not any(t in module_name for t in ("expert", "mlp"))
    return target in module_name


def is_excluded(module_name: str, config: QuantConfig) -> bool:
    parts = module_name.lower().split(".")
    return any(pat.lower() in parts or pat.lower() in module_name.lower() for pat in config.exclude)


def matches(module_name: str, config: QuantConfig) -> bool:
    if is_excluded(module_name, config):
        return False
    return any(target_matches_targets(module_name, t) for t in config.targets)