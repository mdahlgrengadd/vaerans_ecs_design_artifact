"""Wavelet decomposition components."""

from pydantic import BaseModel, Field

from vaerans_ecs.core.arena import TensorRef


class Component(BaseModel):
    """Base class for all ECS components."""

    model_config = {"arbitrary_types_allowed": True}


class WaveletPyr(Component):
    """Wavelet pyramid representation after decomposition.

    Attributes:
        packed: TensorRef to packed wavelet coefficients (C, H, W) float32
        index: TensorRef to decomposition index structure for unpacking
        levels: Number of decomposition levels
        wavelet: Wavelet name (e.g., 'bior2.2', 'haar')
    """

    packed: TensorRef
    index: TensorRef
    levels: int = Field(ge=1, le=10)
    wavelet: str = Field(default="bior2.2")
