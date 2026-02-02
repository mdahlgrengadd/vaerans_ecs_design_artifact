"""Tests for World and entity management."""

import numpy as np
import pytest

from vaerans_ecs.components.image import Component, RGB
from vaerans_ecs.core.world import World


# Mock component for testing
class MockComponent(Component):
    """Mock component for testing."""

    value: int


class TestWorld:
    """Tests for World ECS manager."""

    def test_creation(self) -> None:
        """Test World creation."""
        world = World(arena_bytes=1024)
        assert world.arena.size == 1024
        assert len(world.metadata) == 0

    def test_new_entity(self) -> None:
        """Test entity creation."""
        world = World()
        eid1 = world.new_entity()
        eid2 = world.new_entity()

        assert eid1 == 0
        assert eid2 == 1
        assert eid1 in world.metadata
        assert eid2 in world.metadata

    def test_add_component(self) -> None:
        """Test adding component to entity."""
        world = World()
        eid = world.new_entity()

        comp = MockComponent(value=42)
        world.add_component(eid, comp)

        assert world.has_component(eid, MockComponent)

    def test_add_component_nonexistent_entity(self) -> None:
        """Test adding component to non-existent entity raises error."""
        world = World()

        with pytest.raises(ValueError, match="Entity .* does not exist"):
            world.add_component(999, MockComponent(value=42))

    def test_get_component(self) -> None:
        """Test retrieving component from entity."""
        world = World()
        eid = world.new_entity()

        comp = MockComponent(value=42)
        world.add_component(eid, comp)

        retrieved = world.get_component(eid, MockComponent)
        assert retrieved.value == 42

    def test_get_component_not_present(self) -> None:
        """Test retrieving non-existent component raises KeyError."""
        world = World()
        eid = world.new_entity()

        with pytest.raises(KeyError, match="(does not have component|No entities have component)"):
            world.get_component(eid, MockComponent)

    def test_has_component(self) -> None:
        """Test checking if entity has component."""
        world = World()
        eid = world.new_entity()

        assert not world.has_component(eid, MockComponent)

        world.add_component(eid, MockComponent(value=42))
        assert world.has_component(eid, MockComponent)

    def test_remove_component(self) -> None:
        """Test removing component from entity."""
        world = World()
        eid = world.new_entity()

        world.add_component(eid, MockComponent(value=42))
        assert world.has_component(eid, MockComponent)

        world.remove_component(eid, MockComponent)
        assert not world.has_component(eid, MockComponent)

    def test_remove_component_not_present(self) -> None:
        """Test removing non-existent component raises KeyError."""
        world = World()
        eid = world.new_entity()

        with pytest.raises(KeyError, match="does not have component"):
            world.remove_component(eid, MockComponent)

    def test_query_no_filter(self) -> None:
        """Test query with no filter returns all entities."""
        world = World()
        eid1 = world.new_entity()
        eid2 = world.new_entity()

        result = world.query()
        assert set(result) == {eid1, eid2}

    def test_query_single_component(self) -> None:
        """Test query for single component type."""
        world = World()
        eid1 = world.new_entity()
        eid2 = world.new_entity()
        eid3 = world.new_entity()

        world.add_component(eid1, MockComponent(value=1))
        world.add_component(eid3, MockComponent(value=3))

        result = world.query(MockComponent)
        assert set(result) == {eid1, eid3}

    def test_query_multiple_components(self) -> None:
        """Test query for multiple component types (intersection)."""
        world = World()

        class MockComponent2(Component):
            name: str

        eid1 = world.new_entity()
        eid2 = world.new_entity()
        eid3 = world.new_entity()

        # eid1 has both components
        world.add_component(eid1, MockComponent(value=1))
        world.add_component(eid1, MockComponent2(name="one"))

        # eid2 has only MockComponent
        world.add_component(eid2, MockComponent(value=2))

        # eid3 has only MockComponent2
        world.add_component(eid3, MockComponent2(name="three"))

        # Query for both components
        result = world.query(MockComponent, MockComponent2)
        assert result == [eid1]

    def test_query_nonexistent_component(self) -> None:
        """Test query for component type that no entities have."""
        world = World()
        eid = world.new_entity()

        result = world.query(MockComponent)
        assert result == []

    def test_destroy_entity(self) -> None:
        """Test destroying entity removes it and all components."""
        world = World()
        eid = world.new_entity()

        world.add_component(eid, MockComponent(value=42))
        assert eid in world.metadata
        assert world.has_component(eid, MockComponent)

        world.destroy_entity(eid)

        assert eid not in world.metadata
        assert not world.has_component(eid, MockComponent)

    def test_destroy_entity_nonexistent(self) -> None:
        """Test destroying non-existent entity raises error."""
        world = World()

        with pytest.raises(ValueError, match="Entity .* does not exist"):
            world.destroy_entity(999)

    def test_spawn_image(self) -> None:
        """Test spawning RGB image into world."""
        world = World()

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        eid = world.spawn_image(img)

        assert world.has_component(eid, RGB)

        rgb = world.get_component(eid, RGB)
        img_view = world.arena.view(rgb.pix)

        assert img_view.shape == img.shape
        assert img_view.dtype == img.dtype
        assert np.array_equal(img_view, img)

    def test_spawn_image_invalid_shape(self) -> None:
        """Test spawning image with invalid shape raises error."""
        world = World()

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected image with shape"):
            world.spawn_image(np.zeros((128, 128), dtype=np.uint8))

        # Wrong number of channels
        with pytest.raises(ValueError, match="Expected image with shape"):
            world.spawn_image(np.zeros((128, 128, 4), dtype=np.uint8))

    def test_spawn_image_invalid_dtype(self) -> None:
        """Test spawning image with invalid dtype raises error."""
        world = World()

        img = np.zeros((128, 128, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="Expected dtype"):
            world.spawn_image(img)

    def test_spawn_batch_images(self) -> None:
        """Test spawning batch of images."""
        world = World()

        images = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            for _ in range(4)
        ]

        eids = world.spawn_batch_images(images)

        assert len(eids) == 4

        # Verify each entity has RGB component
        for i, eid in enumerate(eids):
            assert world.has_component(eid, RGB)
            rgb = world.get_component(eid, RGB)
            img_view = world.arena.view(rgb.pix)
            assert img_view.shape == images[i].shape
            assert np.array_equal(img_view, images[i])

        # Verify batch metadata
        for i, eid in enumerate(eids):
            assert world.metadata[eid]['batch_index'] == i

    def test_spawn_batch_images_empty(self) -> None:
        """Test spawning empty batch returns empty list."""
        world = World()
        eids = world.spawn_batch_images([])
        assert eids == []

    def test_spawn_batch_images_different_shapes(self) -> None:
        """Test spawning batch with different shapes raises error."""
        world = World()

        images = [
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.zeros((128, 128, 3), dtype=np.uint8),  # Different shape
        ]

        with pytest.raises(ValueError, match="has shape .*, expected"):
            world.spawn_batch_images(images)

    def test_spawn_batch_images_different_dtypes(self) -> None:
        """Test spawning batch with different dtypes raises error."""
        world = World()

        images = [
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.zeros((64, 64, 3), dtype=np.float32),  # Different dtype
        ]

        with pytest.raises(ValueError, match="has dtype .*, expected"):
            world.spawn_batch_images(images)

    def test_spawn_batch_images_contiguous_allocation(self) -> None:
        """Test that batch images are allocated contiguously."""
        world = World()

        images = [
            np.full((8, 8, 3), i, dtype=np.uint8)
            for i in range(4)
        ]

        eids = world.spawn_batch_images(images)

        # Get RGB components
        rgb_components = [world.get_component(eid, RGB) for eid in eids]

        # Check that offsets are sequential
        offsets = [rgb.pix.offset for rgb in rgb_components]

        # Images should be contiguous (offset differences should be image size)
        img_size = 8 * 8 * 3 * 1  # H * W * C * itemsize
        for i in range(1, len(offsets)):
            assert offsets[i] == offsets[i - 1] + img_size

    def test_clear(self) -> None:
        """Test clearing world resets state."""
        world = World()

        # Create entities and components
        eid1 = world.new_entity()
        eid2 = world.new_entity()
        world.add_component(eid1, MockComponent(value=1))
        world.add_component(eid2, MockComponent(value=2))

        # Allocate some arena memory
        ref = world.arena.alloc_tensor((100,), np.float32)
        initial_offset = world.arena.offset
        initial_generation = world.arena.generation

        assert initial_offset > 0
        assert len(world.metadata) == 2

        # Clear world
        world.clear()

        # Verify reset
        assert world.arena.offset == 0
        assert world.arena.generation == initial_generation + 1
        assert len(world.metadata) == 0
        assert world.query(MockComponent) == []

        # Old ref should be stale
        with pytest.raises(ValueError, match="Stale TensorRef"):
            world.arena.view(ref)

        # New allocations should work
        new_eid = world.new_entity()
        assert new_eid == 0  # EID counter reset

    def test_clear_with_spawn_image(self) -> None:
        """Test clear after spawning images."""
        world = World()

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        eid = world.spawn_image(img)

        rgb = world.get_component(eid, RGB)
        old_ref = rgb.pix

        # Clear world
        world.clear()

        # Old ref should be stale
        with pytest.raises(ValueError, match="Stale TensorRef"):
            world.arena.view(old_ref)

        # Can spawn new images
        eid2 = world.spawn_image(img)
        assert eid2 == 0
        assert world.has_component(eid2, RGB)

    def test_multiple_clear_cycles(self) -> None:
        """Test multiple clear cycles work correctly."""
        world = World()

        for cycle in range(3):
            # Create entities
            eid = world.new_entity()
            world.add_component(eid, MockComponent(value=cycle))

            # Verify
            assert world.has_component(eid, MockComponent)
            assert world.get_component(eid, MockComponent).value == cycle

            # Clear for next cycle
            world.clear()

            # Verify clean slate
            assert len(world.metadata) == 0
            assert world.arena.offset == 0
            assert world.arena.generation == cycle + 1

    def test_repr(self) -> None:
        """Test World repr."""
        world = World()
        eid1 = world.new_entity()
        eid2 = world.new_entity()
        world.add_component(eid1, MockComponent(value=1))

        repr_str = repr(world)
        assert "World" in repr_str
        assert "entities=2" in repr_str
        assert "component_types=1" in repr_str
