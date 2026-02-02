"""Latent space components: Latent4, YUVW4."""

from pydantic import BaseModel, Field

from vaerans_ecs.core.arena import TensorRef


class Component(BaseModel):
    """Base class for all ECS components."""

    model_config = {"arbitrary_types_allowed": True}


class Latent4(Component):
    """VAE latent representation (4 channels).

    Attributes:
        z: TensorRef to latent tensor (4, h, w) float32
    """

    z: TensorRef


class YUVW4(Component):
    """Hadamard-transformed latent space (Y, U, V, W channels).

    Attributes:
        t: TensorRef to transformed tensor (4, h, w) float32
    """

    t: TensorRef
