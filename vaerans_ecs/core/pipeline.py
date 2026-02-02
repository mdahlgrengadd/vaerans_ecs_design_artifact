"""Pipeline and scheduling (Phase 11).

Implements fluent API for composing systems into pipelines with
dependency resolution and branching support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from vaerans_ecs.core.system import System
    from vaerans_ecs.core.world import World

T = TypeVar("T", bound=BaseModel)


class Pipe:
    """Type-safe fluent pipeline builder with dependency resolution.

    Enables chaining systems with method `.to()` or pipe operator `|`,
    branching with `.select()`/`.use()`, and execution with `.out()`.

    Example:
        >>> world = World()
        >>> entity = world.spawn_image(img)
        >>> latent = (
        ...     world.pipe(entity)
        ...     .to(OnnxVAEEncode(model='sdxl-vae'))
        ...     .out(Latent4)
        ... )
    """

    def __init__(self, world: "World", entity: int) -> None:
        """Initialize Pipe with world and entity.

        Args:
            world: The ECS world
            entity: Entity ID to apply pipeline to
        """
        self.world: Any = world
        self.entities = [entity]
        self.systems: list[Any] = []
        self._current_component_type: type[BaseModel] | None = None

    def to(self, system: "System") -> "Pipe":
        """Add system to pipeline.

        Args:
            system: System to add

        Returns:
            Self for method chaining
        """
        self.systems.append(system)
        return self

    def __or__(self, system: "System") -> "Pipe":
        """Pipe operator for chaining systems.

        Equivalent to `.to(system)`.

        Args:
            system: System to add

        Returns:
            Self for method chaining
        """
        return self.to(system)

    def select(self, component_type: type[T]) -> "Pipe":
        """Switch to a different component branch.

        This allows building non-linear pipelines by selecting which
        component to operate on next. Useful when an entity has multiple
        components and you want to apply systems to a specific one.

        Args:
            component_type: The component type to select

        Returns:
            Self for method chaining
        """
        self._current_component_type = component_type
        return self

    def use(self, component_type: type[T]) -> "Pipe":
        """Alias for select() for clearer intent.

        Args:
            component_type: The component type to select

        Returns:
            Self for method chaining
        """
        return self.select(component_type)

    def out(self, component_type: type[T]) -> T:
        """Execute pipeline and return component of specified type.

        Runs all systems in the pipeline in order, then retrieves
        the final component from the entity.

        Args:
            component_type: The component type to retrieve

        Returns:
            The component instance

        Raises:
            RuntimeError: If any system cannot run (missing dependencies)
            KeyError: If entity doesn't have the requested component after execution
        """
        self.execute()
        return self.world.get_component(self.entities[0], component_type)  # type: ignore[no-any-return]

    def execute(self) -> None:
        """Run all systems in order with dependency checking.

        Executes systems sequentially. For each system, checks that all
        entities have required components, runs the system, then moves
        to the next. Raises an error if any system cannot run.

        Raises:
            RuntimeError: If any system cannot run on any entity
        """
        for system in self.systems:
            # Check which entities can run this system
            runnable = [
                eid for eid in self.entities if system.can_run(self.world, eid)
            ]

            if not runnable:
                # Get component names for better error message
                required = [ct.__name__ for ct in system.required_components()]
                raise RuntimeError(
                    f"System {type(system).__name__} cannot run: "
                    f"entities missing required components {required}. "
                    f"Available entities: {self.entities}"
                )

            # Run system on runnable entities
            system.run(self.world, runnable)
