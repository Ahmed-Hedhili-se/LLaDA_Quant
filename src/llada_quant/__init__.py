"""llada-quant: independent quantization framework for LLaDA-style MoE models.

v0.1: symmetric groupwise INT8 weight quantization, reference QuantLinear +
fused-expert (w1/w2) adapter, versioned checkpoint format, validation tools.
"""

from .api import (
    QuantConfig,
    QuantizationManifest,
    load_quantized_weights,
    quantize_model,
    quantized_model,
    save_quantized_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    "QuantConfig",
    "QuantizationManifest",
    "quantize_model",
    "quantized_model",
    "save_quantized_checkpoint",
    "load_quantized_weights",
    "__version__",
]