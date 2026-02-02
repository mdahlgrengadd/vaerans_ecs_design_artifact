"""Tests for System base class."""

import numpy as np
import pytest

from vaerans_ecs.components.image import Component, RGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.arena import Arena
from vaerans_ecs.core.system import System
from vaerans_ecs.core.world import World


# Mock component for testing
class MockInput(Component):
    """Mock input component."""

    value: int


class MockOutput(Component):
    """Mock output component."""

    result: int


# Mock system implementation
class MockSystem(System):
    """Mock system for testing."""

    def required_components(self) -> list[type]:
        """Return required components."""
        if self.mode == "encode":
            return [MockInput]
        else:  # decode
            return [MockOutput]

    def produced_components(self) -> list[type]:
        """Return produced components."""
        if self.mode == "encode":
            return [MockOutput]
        else:  # decode
            return [MockInput]

    def run(self, world: World, eids: list[int]) -> None:
        """Execute mock transformation."""
        if self.mode == "encode":
            for eid in eids:
                input_comp = world.get_component(eid, MockInput)
                output = MockOutput(result=input_comp.value * 2)
                world.add_component(eid, output)
        else:  # decode
            for eid in eids:
                output_comp = world.get_component(eid, MockOutput)
                input_comp = MockInput(value=output_comp.result // 2)
                world.add_component(eid, input_comp)


class TestSystemBase:
    """Tests for System base class."""

    def test_system_creation_default_mode(self) -> None:
        """Test system creation with default mode."""
        system = MockSystem()
        assert system.mode == "encode"

    def test_system_creation_encode_mode(self) -> None:
        """Test system creation with encode mode."""
        system = MockSystem(mode="encode")
        assert system.mode == "encode"

    def test_system_creation_decode_mode(self) -> None:
        """Test system creation with decode mode."""
        system = MockSystem(mode="decode")
        assert system.mode == "decode"

    def test_system_creation_forward_mode(self) -> None:
        """Test system creation with forward mode (alias for encode)."""
        system = MockSystem(mode="forward")
        assert system.mode == "forward"

    def test_system_creation_inverse_mode(self) -> None:
        """Test system creation with inverse mode (alias for decode)."""
        system = MockSystem(mode="inverse")
        assert system.mode == "inverse"

    def test_required_components_encode(self) -> None:
        """Test required_components for encode mode."""
        system = MockSystem(mode="encode")
        required = system.required_components()
        assert required == [MockInput]

    def test_required_components_decode(self) -> None:
        """Test required_components for decode mode."""
        system = MockSystem(mode="decode")
        required = system.required_components()
        assert required == [MockOutput]

    def test_produced_components_encode(self) -> None:
        """Test produced_components for encode mode."""
        system = MockSystem(mode="encode")
        produced = system.produced_components()
        assert produced == [MockOutput]

    def test_produced_components_decode(self) -> None:
        """Test produced_components for decode mode."""
        system = MockSystem(mode="decode")
        produced = system.produced_components()
        assert produced == [MockInput]

    def test_can_run_true(self) -> None:
        """Test can_run returns True when all components present."""
        world = World()
        eid = world.new_entity()
        world.add_component(eid, MockInput(value=42))

        system = MockSystem(mode="encode")
        assert system.can_run(world, eid)

    def test_can_run_false(self) -> None:
        """Test can_run returns False when required component missing."""
        world = World()
        eid = world.new_entity()

        system = MockSystem(mode="encode")
        assert not system.can_run(world, eid)

    def test_can_run_multiple_requirements(self) -> None:
        """Test can_run with multiple required components."""

        class MultiReqSystem(System):
            """System with multiple required components."""

            def required_components(self) -> list[type]:
                return [MockInput, MockOutput]

            def produced_components(self) -> list[type]:
                return []

            def run(self, world: World, eids: list[int]) -> None:
                pass

        world = World()
        eid = world.new_entity()

        # Missing both components
        system = MultiReqSystem()
        assert not system.can_run(world, eid)

        # Missing MockOutput
        world.add_component(eid, MockInput(value=1))
        assert not system.can_run(world, eid)

        # Has both components
        world.add_component(eid, MockOutput(result=2))
        assert system.can_run(world, eid)

    def test_run_encode(self) -> None:
        """Test run in encode mode."""
        world = World()
        eid = world.new_entity()
        world.add_component(eid, MockInput(value=10))

        system = MockSystem(mode="encode")
        system.run(world, [eid])

        # Verify output was produced
        assert world.has_component(eid, MockOutput)
        output = world.get_component(eid, MockOutput)
        assert output.result == 20  # 10 * 2

    def test_run_decode(self) -> None:
        """Test run in decode mode."""
        world = World()
        eid = world.new_entity()
        world.add_component(eid, MockOutput(result=20))

        system = MockSystem(mode="decode")
        system.run(world, [eid])

        # Verify output was produced
        assert world.has_component(eid, MockInput)
        input_comp = world.get_component(eid, MockInput)
        assert input_comp.value == 10  # 20 // 2

    def test_run_multiple_entities(self) -> None:
        """Test run on multiple entities."""
        world = World()

        eids = []
        for i in range(5):
            eid = world.new_entity()
            world.add_component(eid, MockInput(value=i * 10))
            eids.append(eid)

        system = MockSystem(mode="encode")
        system.run(world, eids)

        # Verify all outputs were produced
        for i, eid in enumerate(eids):
            assert world.has_component(eid, MockOutput)
            output = world.get_component(eid, MockOutput)
            assert output.result == i * 20

    def test_repr(self) -> None:
        """Test system repr."""
        system = MockSystem(mode="encode")
        repr_str = repr(system)
        assert "MockSystem" in repr_str
        assert "encode" in repr_str

    def test_abstract_methods(self) -> None:
        """Test that abstract methods must be implemented."""

        class IncompleteSystem(System):
            """System missing required_components implementation."""

            def produced_components(self) -> list[type]:
                return []

            def run(self, world: World, eids: list[int]) -> None:
                pass

        # Should raise TypeError when instantiating
        with pytest.raises(TypeError):
            IncompleteSystem()  # type: ignore


class TestSystemModes:
    """Tests for system mode handling."""

    def test_mode_encoding_aliases(self) -> None:
        """Test that encode and forward modes are recognized."""
        system_encode = MockSystem(mode="encode")
        system_forward = MockSystem(mode="forward")

        # Both should be encoding modes
        assert system_encode.mode == "encode"
        assert system_forward.mode == "forward"
        # Note: modes can be semantically equivalent but have different logic in derived classes

    def test_mode_decoding_aliases(self) -> None:
        """Test that decode and inverse modes work identically."""
        world = World()
        eid = world.new_entity()
        world.add_component(eid, MockOutput(result=10))

        system_decode = MockSystem(mode="decode")
        system_inverse = MockSystem(mode="inverse")

        assert system_decode.required_components() == system_inverse.required_components()
        assert system_decode.produced_components() == system_inverse.produced_components()


class TestSystemIntegration:
    """Integration tests with real components."""

    def test_system_with_real_components(self) -> None:
        """Test system with real RGB and Latent components."""

        class RGBToLatentSystem(System):
            """Mock system converting RGB to Latent."""

            def required_components(self) -> list[type]:
                return [RGB]

            def produced_components(self) -> list[type]:
                return [Latent4]

            def run(self, world: World, eids: list[int]) -> None:
                for eid in eids:
                    rgb = world.get_component(eid, RGB)
                    # Create latent tensor (4 channels, downsampled)
                    rgb_view = world.arena.view(rgb.pix)
                    latent_shape = (4, rgb_view.shape[0] // 8, rgb_view.shape[1] // 8)
                    latent_ref = world.arena.alloc_tensor(latent_shape, np.float32)
                    world.add_component(eid, Latent4(z=latent_ref))

        world = World()
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        eid = world.spawn_image(img)

        system = RGBToLatentSystem()
        assert system.can_run(world, eid)

        system.run(world, [eid])

        assert world.has_component(eid, Latent4)
        latent = world.get_component(eid, Latent4)
        latent_data = world.arena.view(latent.z)
        assert latent_data.shape[0] == 4  # 4 channels
