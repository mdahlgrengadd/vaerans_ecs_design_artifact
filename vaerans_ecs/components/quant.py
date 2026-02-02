"""Quantization components."""

from pydantic import BaseModel, Field

from vaerans_ecs.core.arena import TensorRef


class Component(BaseModel):
    """Base class for all ECS components."""

    model_config = {"arbitrary_types_allowed": True}


class QuantParams(Component):
    """Quantization parameters for inverse transform.

    Attributes:
        scales: TensorRef to per-band scale factors (float32)
        offsets: TensorRef to per-band offset factors (float32)
        quality: Quality level used (1-100)
        per_band: Whether quantization is per-band or per-channel
    """

    scales: TensorRef
    offsets: TensorRef
    quality: int = Field(ge=1, le=100)
    per_band: bool = Field(default=True)


class SymbolsU8(Component):
    """Quantized symbols after quantization (uint8).

    Attributes:
        data: TensorRef to quantized symbols (uint8)
        params: Reference to QuantParams component for dequantization
    """

    data: TensorRef
    params: QuantParams
