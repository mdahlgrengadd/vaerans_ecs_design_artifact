"""Entropy coding components (ANS bitstream)."""

from pydantic import BaseModel, Field

from vaerans_ecs.core.arena import TensorRef


class Component(BaseModel):
    """Base class for all ECS components."""

    model_config = {"arbitrary_types_allowed": True}


class ANSBitstream(Component):
    """Compressed ANS bitstream ready for serialization.

    Attributes:
        data: TensorRef to bitstream bytes (uint8)
        probs: TensorRef to probability table for decoding
        initial_state: Initial rANS state for decoding (uint32)
    """

    data: TensorRef
    probs: TensorRef
    initial_state: int = Field(ge=0)
