"""Serializable quantization configuration and target-selection policies.

Two things here decide whether a quantization run is honest:

**Execution mode** — whether the dequantized BF16 weights stay resident
alongside the packed ones. Only ``PACKED`` reduces memory; ``REFERENCE``
exists for validation and costs *more* than not quantizing at all. Nothing
picks between them implicitly.

**Targeting** — which modules get touched. Matching is structural (for MoE
experts) or explicitly named (for linears), never a substring guess, because
a quantizer that silently converts the wrong module produces a model that
looks fine and is wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Optional, Tuple

FORMAT_VERSION = 2

#: Never quantized. Patterns are ``fnmatch`` globs tested against each
#: dot-separated component of a module path, so ``*norm*`` catches
#: ``input_layernorm`` without also catching an unrelated ``normalizer.proj``
#: further up a path.
DEFAULT_EXCLUDES = ("router", "gate", "*norm*", "embed_tokens", "lm_head")


class ExecutionMode(str, Enum):
    """What is resident after quantization.

    ``REFERENCE``
        Packed weights *and* dequantized BF16 Parameters both stay resident.
        Uses ~1.5x the memory of the unquantized model and runs at BF16
        speed. Its only purpose is to hold the numerical contract still while
        validating something else. It is not a deployment mode and must never
        be reported as a memory saving.

    ``PACKED``
        Only packed integers and scales stay resident; the BF16 Parameters
        are removed and reconstructed on attribute access. This is the mode
        that actually reduces model memory. It trades compute for memory:
        every forward pass dequantizes, so it is *slower* than BF16 until a
        kernel consumes the packed weights directly.
    """

    REFERENCE = "reference"
    PACKED = "packed"


COMPONENT_LINEAR = "linear"
COMPONENT_EXPERT = "expert"
KNOWN_TARGETS = (COMPONENT_EXPERT, COMPONENT_LINEAR)


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 8
    group_size: int = 128
    targets: Tuple[str, ...] = (COMPONENT_EXPERT,)
    execution_mode: str = ExecutionMode.PACKED.value
    linear_include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = DEFAULT_EXCLUDES
    compute_dtype: str = "bfloat16"
    scale_dtype: str = "float32"
    scale_search: str = "amax"
    search_grid: int = 24
    compile_dequant: bool = False
    source_checkpoint: Optional[str] = None
    expect_expert_blocks: Optional[int] = None
    expect_linears: Optional[int] = None
    allow_no_matches: bool = False

    def __post_init__(self) -> None:
        if self.bits not in (8, 4):
            raise ValueError(f"bits must be 8 or 4, got {self.bits}")
        if self.group_size <= 0 and self.group_size != -1:
            raise ValueError(f"group_size must be > 0 or -1, got {self.group_size}")
        if self.bits == 4 and self.group_size != -1 and self.group_size % 2 != 0:
            raise ValueError(
                f"bits=4 needs an even group_size so int4 groups stay byte-aligned, "
                f"got {self.group_size}"
            )
        if self.scale_search not in ("amax", "mse"):
            raise ValueError(
                f"scale_search must be 'amax' or 'mse', got {self.scale_search!r}"
            )
        if self.search_grid < 1:
            raise ValueError(f"search_grid must be >= 1, got {self.search_grid}")
        unknown = [t for t in self.targets if t not in KNOWN_TARGETS]
        if unknown:
            raise ValueError(f"unknown targets {unknown}; known: {list(KNOWN_TARGETS)}")
        try:
            ExecutionMode(self.execution_mode)
        except ValueError:
            raise ValueError(
                f"execution_mode must be one of "
                f"{[m.value for m in ExecutionMode]}, got {self.execution_mode!r}"
            ) from None

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode(self.execution_mode)

    @property
    def reduces_memory(self) -> bool:
        return self.mode is ExecutionMode.PACKED

    def to_dict(self) -> dict:
        return {"format_version": FORMAT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "QuantConfig":
        data = dict(data)
        for key in ("targets", "exclude", "linear_include"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def is_excluded(module_name: str, config: QuantConfig) -> bool:
    parts = module_name.split(".")
    return any(fnmatch(part, pattern) for part in parts for pattern in config.exclude)


def matches_linear(module_name: str, config: QuantConfig) -> bool:
    if COMPONENT_LINEAR not in config.targets or not config.linear_include:
        return False
    if is_excluded(module_name, config):
        return False
    leaf = module_name.rsplit(".", 1)[-1]
    return any(
        fnmatch(leaf, pattern) or fnmatch(module_name, pattern)
        for pattern in config.linear_include
    )
