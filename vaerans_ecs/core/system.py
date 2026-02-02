"""System base class for ECS transformations.

Systems are the "logic" layer of the ECS architecture. They operate on
components attached to entities, reading required components and producing
new components.

Systems support two modes:
- 'encode' or 'forward': Transforms data forward (compression direction)
- 'decode' or 'inverse': Transforms data backward (decompression direction)

Example:
    >>> class MySystem(System):
    ...     def required_components(self):
    ...         return [InputComponent]
    ...     def produced_components(self):
    ...         return [OutputComponent]
    ...     def run(self, world, eids):
    ...         for eid in eids:
    ...             input = world.get_component(eid, InputComponent)
    ...             # Process...
    ...             output = OutputComponent(...)
    ...             world.add_component(eid, output)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from vaerans_ecs.core.world import World


class System(ABC):
    """Base class for all ECS systems.

    Systems transform components attached to entities. They declare:
    - required_components(): What inputs they need
    - produced_components(): What outputs they create
    - run(): The actual transformation logic

    Attributes:
        mode: Transformation direction ('encode'/'forward'/'decode'/'inverse')
    """

    def __init__(
        self,
        mode: Literal["encode", "decode", "forward", "inverse"] = "encode",
    ) -> None:
        """Initialize system with transformation mode.

        Args:
            mode: Direction of transformation
                - 'encode', 'forward': Compression direction
                - 'decode', 'inverse': Decompression direction
        """
        self.mode = mode

    @abstractmethod
    def required_components(self) -> list[type]:
        """Return list of component types this system requires as input.

        Returns:
            List of Component classes needed by this system

        Example:
            >>> def required_components(self):
            ...     if self.mode == 'encode':
            ...         return [RGB]
            ...     else:
            ...         return [ReconRGB]
        """
        pass

    @abstractmethod
    def produced_components(self) -> list[type]:
        """Return list of component types this system produces as output.

        Returns:
            List of Component classes created by this system

        Example:
            >>> def produced_components(self):
            ...     if self.mode == 'encode':
            ...         return [Latent4]
            ...     else:
            ...         return [ReconRGB]
        """
        pass

    @abstractmethod
    def run(self, world: World, eids: list[int]) -> None:
        """Execute system on given entities.

        Args:
            world: World instance with entities and components
            eids: List of entity IDs to process

        Note:
            - Should handle batching efficiently
            - May group entities by shape for batch operations
            - Should add produced_components to each entity
        """
        pass

    def can_run(self, world: World, eid: int) -> bool:
        """Check if entity has all required components.

        Args:
            eid: Entity ID to check

        Returns:
            True if entity has all required components

        Example:
            >>> if system.can_run(world, eid):
            ...     system.run(world, [eid])
        """
        return all(world.has_component(eid, ct) for ct in self.required_components())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode})"
