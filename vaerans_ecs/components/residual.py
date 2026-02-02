"""Residual coding components."""

from pydantic import BaseModel

from vaerans_ecs.core.arena import TensorRef


class Component(BaseModel):
    """Base class for all ECS components."""

    model_config = {"arbitrary_types_allowed": True}


class Residual(Component):
    """Residual (difference) tensor for coding or quality improvement.

    Attributes:
        tensor: TensorRef to residual data (same shape as parent)
        scale: Scale factor for residual contribution
    """

    tensor: TensorRef
    scale: float = 1.0
