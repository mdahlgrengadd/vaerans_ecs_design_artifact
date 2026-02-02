"""World: Entity-Component-System manager.

The World is the central ECS registry that manages:
- Entity creation (integer IDs)
- Component storage (type -> entity -> component mapping)
- Component queries (find entities with specific component combinations)
- Arena memory management

Example:
    >>> world = World()
    >>> eid = world.new_entity()
    >>> world.add_component(eid, RGB(pix=ref))
    >>> entities = world.query(RGB, Latent4)  # Entities with both RGB and Latent4
    >>> world.clear()  # Reset for next batch
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from vaerans_ecs.core.arena import Arena

# Import Component base class
from pydantic import BaseModel

# Component base class (can be imported from components.image if needed)
Component = BaseModel

T = TypeVar("T", bound=Component)


class World:
    """Central ECS registry managing entities, components, and memory.

    The World owns:
    - Arena: Zero-copy memory allocator
    - Entity registry: Integer entity IDs
    - Component stores: Mappings from (component_type, entity_id) to component
    - Metadata: Arbitrary key-value data per entity

    Attributes:
        arena: Memory arena for tensor allocation
        metadata: Per-entity metadata dict

    Example:
        >>> world = World(arena_bytes=512 << 20)  # 512 MB
        >>> eid = world.spawn_image(np.zeros((512, 512, 3), dtype=np.uint8))
        >>> world.add_component(eid, Latent4(z=latent_ref))
        >>> has_latent = world.has_component(eid, Latent4)
        >>> world.clear()  # Cleanup for next batch
    """

    def __init__(self, arena_bytes: int = 512 << 20):
        """Create World with specified arena size.

        Args:
            arena_bytes: Arena size in bytes (default 512 MB)
        """
        self.arena = Arena(size_bytes=arena_bytes)
        self._next_eid = 0
        self._components: dict[type[Component], dict[int, Component]] = {}
        self.metadata: dict[int, dict[str, Any]] = {}

    def new_entity(self) -> int:
        """Create a new entity and return its ID.

        Returns:
            Entity ID (monotonically increasing integer)
        """
        eid = self._next_eid
        self._next_eid += 1
        self.metadata[eid] = {}
        return eid

    def spawn_image(self, img: np.ndarray) -> int:
        """Ingest an RGB image into the world.

        Args:
            img: RGB image array (H, W, 3) with dtype uint8 or float32

        Returns:
            Entity ID with RGB component attached

        Raises:
            ValueError: If image shape or dtype is invalid
        """
        # Import here to avoid circular dependency
        from vaerans_ecs.components.image import RGB

        # Validate image
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected image with shape (H, W, 3), got {img.shape}"
            )
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError(
                f"Expected dtype uint8 or float32, got {img.dtype}"
            )

        # Create entity
        eid = self.new_entity()

        # Copy image to arena
        pix_ref = self.arena.copy_tensor(img)

        # Add RGB component
        rgb = RGB(pix=pix_ref, colorspace='sRGB')
        self.add_component(eid, rgb)

        # Store metadata
        self.metadata[eid]['image_shape'] = img.shape
        self.metadata[eid]['image_dtype'] = str(img.dtype)

        return eid

    def spawn_batch_images(self, images: list[np.ndarray]) -> list[int]:
        """Ingest multiple RGB images, allocating contiguously for batching.

        Args:
            images: List of RGB image arrays with same shape and dtype

        Returns:
            List of entity IDs with RGB components attached

        Raises:
            ValueError: If images have different shapes or dtypes
        """
        # Import here to avoid circular dependency
        from vaerans_ecs.components.image import RGB

        if not images:
            return []

        # Validate all images have same shape and dtype
        ref_shape = images[0].shape
        ref_dtype = images[0].dtype

        for i, img in enumerate(images):
            if img.shape != ref_shape:
                raise ValueError(
                    f"Image {i} has shape {img.shape}, expected {ref_shape}"
                )
            if img.dtype != ref_dtype:
                raise ValueError(
                    f"Image {i} has dtype {img.dtype}, expected {ref_dtype}"
                )
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Image {i} has invalid shape {img.shape}, expected (H, W, 3)"
                )

        # Allocate contiguous batch tensor
        batch_shape = (len(images),) + ref_shape
        batch_ref = self.arena.alloc_tensor(batch_shape, ref_dtype)
        batch = self.arena.view(batch_ref)

        # Copy images into batch
        for i, img in enumerate(images):
            batch[i] = img

        # Create entities with RGB components (using subrefs)
        eids = []
        for i in range(len(images)):
            eid = self.new_entity()

            # Create subref for this image
            img_ref = batch_ref.subref((i, slice(None), slice(None), slice(None)))

            # Add RGB component
            rgb = RGB(pix=img_ref, colorspace='sRGB')
            self.add_component(eid, rgb)

            # Store metadata
            self.metadata[eid]['image_shape'] = ref_shape
            self.metadata[eid]['image_dtype'] = str(ref_dtype)
            self.metadata[eid]['batch_index'] = i

            eids.append(eid)

        return eids

    def clear(self) -> None:
        """Reset arena and clear all entities/components for reuse.

        After clear(), all TensorRefs from previous entities are invalidated.
        The World can then be reused for a new batch of compression operations.

        Example:
            >>> world = World()
            >>> for img_batch in image_batches:
            ...     eids = world.spawn_batch_images(img_batch)
            ...     # ... process entities ...
            ...     world.clear()  # Prepare for next batch
        """
        self.arena.reset()
        self._next_eid = 0
        self._components.clear()
        self.metadata.clear()

    def add_component(self, eid: int, component: Component) -> None:
        """Attach a component to an entity.

        Args:
            eid: Entity ID
            component: Component instance

        Raises:
            ValueError: If entity does not exist
        """
        if eid not in self.metadata:
            raise ValueError(f"Entity {eid} does not exist")

        comp_type = type(component)
        if comp_type not in self._components:
            self._components[comp_type] = {}

        self._components[comp_type][eid] = component

    def get_component(self, eid: int, comp_type: type[T]) -> T:
        """Retrieve a component from an entity.

        Args:
            eid: Entity ID
            comp_type: Component class to retrieve

        Returns:
            Component instance

        Raises:
            KeyError: If entity does not have the component
        """
        if comp_type not in self._components:
            raise KeyError(f"No entities have component type {comp_type.__name__}")
        if eid not in self._components[comp_type]:
            raise KeyError(f"Entity {eid} does not have component {comp_type.__name__}")

        return self._components[comp_type][eid]  # type: ignore

    def has_component(self, eid: int, comp_type: type[Component]) -> bool:
        """Check if entity has a specific component type.

        Args:
            eid: Entity ID
            comp_type: Component class to check

        Returns:
            True if entity has the component
        """
        return (
            comp_type in self._components
            and eid in self._components[comp_type]
        )

    def remove_component(self, eid: int, comp_type: type[Component]) -> None:
        """Remove a component from an entity.

        Args:
            eid: Entity ID
            comp_type: Component class to remove

        Raises:
            KeyError: If entity does not have the component
        """
        if not self.has_component(eid, comp_type):
            raise KeyError(f"Entity {eid} does not have component {comp_type.__name__}")

        del self._components[comp_type][eid]

    def query(self, *comp_types: type[Component]) -> list[int]:
        """Query entities that have ALL specified component types.

        Args:
            *comp_types: Component classes to query for

        Returns:
            List of entity IDs that have all specified components

        Example:
            >>> # Find entities with both RGB and Latent4
            >>> eids = world.query(RGB, Latent4)
        """
        if not comp_types:
            # No filter: return all entities
            return list(self.metadata.keys())

        # Start with entities that have the first component type
        result_set = set(self._components.get(comp_types[0], {}).keys())

        # Intersect with entities that have remaining component types
        for comp_type in comp_types[1:]:
            if comp_type not in self._components:
                return []  # No entities have this component
            result_set &= set(self._components[comp_type].keys())

        return sorted(result_set)

    def destroy_entity(self, eid: int) -> None:
        """Remove entity and all its components.

        Args:
            eid: Entity ID to destroy

        Note:
            This does not free arena memory (use clear() for that).
            Component data remains in arena until next clear().
        """
        if eid not in self.metadata:
            raise ValueError(f"Entity {eid} does not exist")

        # Remove from all component stores
        for comp_store in self._components.values():
            comp_store.pop(eid, None)

        # Remove metadata
        del self.metadata[eid]

    def pipe(self, entity: int) -> Any:
        """Create a pipeline for the given entity.

        Starts a fluent pipeline builder that chains systems together.
        Systems are executed in order when `.out()` is called.

        Args:
            entity: Entity ID to build pipeline for

        Returns:
            Pipe instance for fluent API

        Example:
            >>> result = (
            ...     world.pipe(entity)
            ...     .to(OnnxVAEEncode())
            ...     .to(Hadamard4())
            ...     .out(YUVW4)
            ... )
        """
        from vaerans_ecs.core.pipeline import Pipe

        return Pipe(world=self, entity=entity)

    def __repr__(self) -> str:
        num_entities = len(self.metadata)
        num_comp_types = len(self._components)
        return (
            f"World(entities={num_entities}, component_types={num_comp_types}, "
            f"arena={self.arena})"
        )
