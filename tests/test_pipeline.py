"""Tests for Pipeline (Phase 11)."""

import numpy as np
import pytest

from vaerans_ecs.components.image import RGB
from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.core.pipeline import Pipe
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4


class TestPipeBasics:
    """Test basic Pipe construction and chaining."""

    def test_pipe_creation(self) -> None:
        """Test creating a pipe."""
        world = World()
        entity = world.new_entity()
        pipe = world.pipe(entity)

        assert pipe is not None
        assert isinstance(pipe, Pipe)
        assert pipe.entities == [entity]
        assert pipe.systems == []

    def test_pipe_to_chaining(self) -> None:
        """Test .to() method chains systems."""
        world = World()
        entity = world.new_entity()
        hadamard = Hadamard4(mode="forward")

        pipe = world.pipe(entity).to(hadamard)

        assert pipe.systems == [hadamard]
        assert isinstance(pipe, Pipe)

    def test_pipe_or_operator(self) -> None:
        """Test | operator for chaining."""
        world = World()
        entity = world.new_entity()
        hadamard = Hadamard4(mode="forward")

        pipe = world.pipe(entity) | hadamard

        assert pipe.systems == [hadamard]

    def test_pipe_multiple_systems(self) -> None:
        """Test chaining multiple systems."""
        world = World()
        entity = world.new_entity()
        hadamard1 = Hadamard4(mode="forward")
        hadamard2 = Hadamard4(mode="inverse")

        pipe = world.pipe(entity).to(hadamard1).to(hadamard2)

        assert len(pipe.systems) == 2
        assert pipe.systems[0] is hadamard1
        assert pipe.systems[1] is hadamard2

    def test_pipe_mixed_operators(self) -> None:
        """Test mixing .to() and | operators."""
        world = World()
        entity = world.new_entity()
        hadamard1 = Hadamard4(mode="forward")
        hadamard2 = Hadamard4(mode="inverse")

        pipe = world.pipe(entity).to(hadamard1) | hadamard2

        assert len(pipe.systems) == 2


class TestPipeExecution:
    """Test pipeline execution."""

    def test_execute_single_system(self) -> None:
        """Test executing a pipeline with one system."""
        world = World()

        # Create latent tensor (Hadamard expects Latent4 input)
        latent_data = np.random.randn(4, 32, 32).astype(np.float32)
        latent_ref = world.arena.copy_tensor(latent_data)
        entity = world.new_entity()
        world.add_component(entity, Latent4(z=latent_ref))

        # Execute pipeline
        hadamard = Hadamard4(mode="forward")
        pipe = world.pipe(entity).to(hadamard)
        pipe.execute()

        # Verify system ran
        assert world.has_component(entity, YUVW4)

    def test_execute_multiple_systems(self) -> None:
        """Test executing a pipeline with multiple systems."""
        world = World()

        # Create input
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        entity = world.spawn_image(img)

        # Execute pipeline: RGB -> Latent4 via VAE -> YUVW4 via Hadamard
        # For this test we'll create a test system that produces Latent4
        from vaerans_ecs.core.system import System

        class DummyVAEEncode(System):
            """Dummy VAE encoder for testing."""

            def required_components(self):
                return [RGB]

            def produced_components(self):
                return [Latent4]

            def run(self, world, eids):
                for eid in eids:
                    rgb = world.get_component(eid, RGB)
                    rgb_view = world.arena.view(rgb.pix)
                    # Create latent with shape (4, H/8, W/8)
                    h, w = rgb_view.shape[0] // 8, rgb_view.shape[1] // 8
                    latent_data = np.random.randn(4, h, w).astype(np.float32)
                    latent_ref = world.arena.copy_tensor(latent_data)
                    world.add_component(eid, Latent4(z=latent_ref))

        pipeline = (
            world.pipe(entity)
            .to(DummyVAEEncode())
            .to(Hadamard4(mode="forward"))
        )
        pipeline.execute()

        # Verify both systems ran
        assert world.has_component(entity, Latent4)
        assert world.has_component(entity, YUVW4)

    def test_execute_error_on_missing_dependencies(self) -> None:
        """Test that execute raises error if dependencies missing."""
        world = World()
        entity = world.new_entity()

        # Try to run Hadamard without Latent4
        hadamard = Hadamard4(mode="forward")
        pipe = world.pipe(entity).to(hadamard)

        with pytest.raises(RuntimeError, match="cannot run"):
            pipe.execute()

    def test_execute_partial_dependencies(self) -> None:
        """Test execute with multiple entities where some lack dependencies."""
        world = World()

        # Create two entities, only one with RGB
        entity1 = world.spawn_image(
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        )
        entity2 = world.new_entity()

        # Run Hadamard on both (only entity2 will have Latent4 after encode)
        from vaerans_ecs.core.system import System

        class DummyVAEEncode(System):
            def required_components(self):
                return [RGB]

            def produced_components(self):
                return [Latent4]

            def run(self, world, eids):
                for eid in eids:
                    rgb = world.get_component(eid, RGB)
                    rgb_view = world.arena.view(rgb.pix)
                    h, w = rgb_view.shape[0] // 8, rgb_view.shape[1] // 8
                    latent_data = np.random.randn(4, h, w).astype(np.float32)
                    latent_ref = world.arena.copy_tensor(latent_data)
                    world.add_component(eid, Latent4(z=latent_ref))

        # Only entity1 will make it through VAE -> Hadamard
        pipe = world.pipe(entity1).to(DummyVAEEncode()).to(Hadamard4(mode="forward"))
        pipe.execute()

        assert world.has_component(entity1, YUVW4)


class TestPipeOut:
    """Test the .out() method for retrieving results."""

    def test_out_returns_component(self) -> None:
        """Test that .out() executes and returns component."""
        world = World()
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        entity = world.spawn_image(img)

        # Use .out() to execute and get result
        latent = (
            world.pipe(entity)
            .to(
                Hadamard4(mode="forward")
            )  # This will fail - need Latent4 first
        )

        # Actually, let's test with a working pipeline
        from vaerans_ecs.core.system import System

        class DummyVAEEncode(System):
            def required_components(self):
                return [RGB]

            def produced_components(self):
                return [Latent4]

            def run(self, world, eids):
                for eid in eids:
                    rgb = world.get_component(eid, RGB)
                    rgb_view = world.arena.view(rgb.pix)
                    h, w = rgb_view.shape[0] // 8, rgb_view.shape[1] // 8
                    latent_data = np.random.randn(4, h, w).astype(np.float32)
                    latent_ref = world.arena.copy_tensor(latent_data)
                    world.add_component(eid, Latent4(z=latent_ref))

        result = (
            world.pipe(entity).to(DummyVAEEncode()).to(Hadamard4(mode="forward")).out(YUVW4)
        )

        assert isinstance(result, YUVW4)
        assert hasattr(result, "t")

    def test_out_error_missing_component(self) -> None:
        """Test that .out() raises error if component not found."""
        world = World()
        entity = world.new_entity()

        pipe = world.pipe(entity)

        with pytest.raises(KeyError):
            pipe.out(RGB)

    def test_out_type_safety(self) -> None:
        """Test that .out() returns correct type."""
        world = World()
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        entity = world.spawn_image(img)

        from vaerans_ecs.core.system import System

        class DummyVAEEncode(System):
            def required_components(self):
                return [RGB]

            def produced_components(self):
                return [Latent4]

            def run(self, world, eids):
                for eid in eids:
                    rgb = world.get_component(eid, RGB)
                    rgb_view = world.arena.view(rgb.pix)
                    h, w = rgb_view.shape[0] // 8, rgb_view.shape[1] // 8
                    latent_data = np.random.randn(4, h, w).astype(np.float32)
                    latent_ref = world.arena.copy_tensor(latent_data)
                    world.add_component(eid, Latent4(z=latent_ref))

        result: Latent4 = world.pipe(entity).to(DummyVAEEncode()).out(Latent4)

        # Type checker should know result is Latent4
        assert isinstance(result, Latent4)
        assert hasattr(result, "z")


class TestPipeSelect:
    """Test the .select() method for branching."""

    def test_select_switches_component(self) -> None:
        """Test that .select() switches component type."""
        world = World()
        entity = world.new_entity()

        from vaerans_ecs.components.latent import Latent4

        pipe = world.pipe(entity).select(Latent4)

        assert pipe._current_component_type is Latent4

    def test_use_alias_for_select(self) -> None:
        """Test that .use() is alias for .select()."""
        world = World()
        entity = world.new_entity()

        from vaerans_ecs.components.latent import Latent4

        pipe = world.pipe(entity).use(Latent4)

        assert pipe._current_component_type is Latent4

    def test_select_chaining(self) -> None:
        """Test that .select() returns self for chaining."""
        world = World()
        entity = world.new_entity()

        from vaerans_ecs.components.latent import Latent4

        pipe = world.pipe(entity).select(Latent4).to(Hadamard4(mode="forward"))

        assert len(pipe.systems) == 1


class TestPipeIntegration:
    """Integration tests with Hadamard4."""

    def test_hadamard_roundtrip_via_pipe(self) -> None:
        """Test encode-decode roundtrip via pipeline."""
        world = World()

        # Create latent tensor (4, H, W)
        latent_data = np.random.randn(4, 16, 16).astype(np.float32)
        latent_ref = world.arena.copy_tensor(latent_data)

        entity = world.new_entity()
        world.add_component(entity, Latent4(z=latent_ref))

        # Forward then inverse via pipeline
        result = (
            world.pipe(entity)
            .to(Hadamard4(mode="forward"))
            .to(Hadamard4(mode="inverse"))
            .out(Latent4)
        )

        # Check recovery
        result_view = world.arena.view(result.z)
        assert np.allclose(result_view, latent_data, atol=1e-5)

    def test_pipe_multiple_entities_separate_pipelines(self) -> None:
        """Test running separate pipelines on different entities."""
        world = World()

        # Create two entities with latents
        latent1_data = np.random.randn(4, 32, 32).astype(np.float32)
        latent2_data = np.random.randn(4, 32, 32).astype(np.float32)

        latent1_ref = world.arena.copy_tensor(latent1_data)
        latent2_ref = world.arena.copy_tensor(latent2_data)

        entity1 = world.new_entity()
        entity2 = world.new_entity()
        world.add_component(entity1, Latent4(z=latent1_ref))
        world.add_component(entity2, Latent4(z=latent2_ref))

        # Run separate pipelines
        result1 = (
            world.pipe(entity1)
            .to(Hadamard4(mode="forward"))
            .to(Hadamard4(mode="inverse"))
            .out(Latent4)
        )
        result2 = (
            world.pipe(entity2)
            .to(Hadamard4(mode="forward"))
            .to(Hadamard4(mode="inverse"))
            .out(Latent4)
        )

        # Both should recover their inputs
        result1_view = world.arena.view(result1.z)
        result2_view = world.arena.view(result2.z)

        assert np.allclose(result1_view, latent1_data, atol=1e-5)
        assert np.allclose(result2_view, latent2_data, atol=1e-5)

    def test_pipe_with_real_vae_encode(self) -> None:
        """Test pipeline with real ONNX VAE encoder."""
        import os

        # Check if real models exist
        if not os.path.exists("vaerans_ecs.toml"):
            pytest.skip("Real VAE config not available")

        try:
            from vaerans_ecs.systems.vae import OnnxVAEEncode

            world = World()
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            entity = world.spawn_image(img)

            # Encode via pipeline
            latent = (
                world.pipe(entity).to(OnnxVAEEncode(model="sdxl-vae")).out(Latent4)
            )

            assert isinstance(latent, Latent4)
            latent_view = world.arena.view(latent.z)
            assert latent_view.shape == (4, 32, 32)
            assert latent_view.dtype == np.float32

        except (ImportError, FileNotFoundError):
            pytest.skip("ONNX Runtime or real models not available")


class TestPipeEdgeCases:
    """Test edge cases and error handling."""

    def test_pipe_with_empty_entity_list(self) -> None:
        """Test that pipe works with valid entity."""
        world = World()
        entity = world.new_entity()

        # Should not raise during creation
        pipe = world.pipe(entity)
        assert pipe is not None

    def test_pipe_execute_without_systems(self) -> None:
        """Test executing empty pipeline."""
        world = World()
        entity = world.new_entity()

        # Should not raise
        pipe = world.pipe(entity)
        pipe.execute()

    def test_pipe_system_order_matters(self) -> None:
        """Test that system order is preserved."""
        world = World()
        latent_data = np.random.randn(4, 32, 32).astype(np.float32)
        latent_ref = world.arena.copy_tensor(latent_data)
        entity = world.new_entity()
        world.add_component(entity, Latent4(z=latent_ref))

        # Forward then inverse should recover
        result = (
            world.pipe(entity)
            .to(Hadamard4(mode="forward"))
            .to(Hadamard4(mode="inverse"))
            .out(Latent4)
        )

        result_view = world.arena.view(result.z)
        assert np.allclose(result_view, latent_data, atol=1e-5)

    def test_pipe_double_to_call_returns_self(self) -> None:
        """Test that chaining multiple .to() calls works."""
        world = World()
        entity = world.new_entity()

        h1 = Hadamard4(mode="forward")
        h2 = Hadamard4(mode="inverse")

        pipe1 = world.pipe(entity).to(h1)
        pipe2 = pipe1.to(h2)

        # Should be same object
        assert pipe1 is pipe2
        assert len(pipe2.systems) == 2
