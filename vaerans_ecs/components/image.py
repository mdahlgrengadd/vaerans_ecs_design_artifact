"""Image components: RGB, ReconRGB, BlurRGB."""

from pydantic import BaseModel, Field

from vaerans_ecs.core.arena import TensorRef


class Component(BaseModel):
    """Base class for all ECS components.

    Components are data containers using Pydantic for validation and type safety.
    All tensor data is stored as TensorRef handles pointing into the arena.
    """

    model_config = {"arbitrary_types_allowed": True}


class RGB(Component):
    """Original RGB image component.

    Attributes:
        pix: TensorRef to RGB pixel data (H, W, 3) uint8 or float32
        colorspace: Colorspace identifier (default 'sRGB')
    """

    pix: TensorRef
    colorspace: str = Field(default="sRGB")


class ReconRGB(Component):
    """Reconstructed RGB image after decode.

    Attributes:
        pix: TensorRef to RGB pixel data (H, W, 3) uint8 or float32
        colorspace: Colorspace identifier (default 'sRGB')
    """

    pix: TensorRef
    colorspace: str = Field(default="sRGB")


class BlurRGB(Component):
    """Blurred RGB image for residual coding.

    Attributes:
        pix: TensorRef to blurred RGB pixel data (H, W, 3)
        sigma: Gaussian blur sigma parameter
    """

    pix: TensorRef
    sigma: float = Field(gt=0.0)
