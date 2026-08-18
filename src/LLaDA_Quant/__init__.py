"""LLaDA_Quant: quantization stack for LLaDA-MoE diffusion LLM inference.

Scope is weight quantization plus the measurement layer needed to trust it.
Decoding, KV caching and serving stay in the inference repository.

v0.2 — what is implemented and measured:
  * symmetric groupwise INT8 and genuinely packed INT4 (two values per byte)
  * two explicit residency modes: PACKED (real memory reduction) and
    REFERENCE (validation only, uses more memory than not quantizing)
  * structural expert-block targeting with a full audit trail
  * self-contained checkpoints with no redundant BF16 copies
  * resident-memory accounting measured from live tensors
  * diffusion-trajectory validation with an offline trace/replay path

Not implemented: a kernel that consumes packed weights directly. Until that
exists, quantized execution is dequantize-then-matmul, which trades latency
for memory. Nothing here reports a speedup.
"""

from .api import (
    QuantConfig,
    QuantizationManifest,
    QuantizationResult,
    TargetingError,
    load_quantized_weights,
    quantize_and_measure,
    quantize_model,
    quantized_model,
    save_quantized_checkpoint,
)
from .config import ExecutionMode
from .memory import (
    MemoryComparison,
    MemoryReport,
    compare_resident_memory,
    resident_memory,
)

__version__ = "0.2.0"

__all__ = [
    "QuantConfig",
    "ExecutionMode",
    "QuantizationManifest",
    "QuantizationResult",
    "TargetingError",
    "quantize_model",
    "quantized_model",
    "quantize_and_measure",
    "save_quantized_checkpoint",
    "load_quantized_weights",
    "MemoryReport",
    "MemoryComparison",
    "resident_memory",
    "compare_resident_memory",
    "__version__",
]
