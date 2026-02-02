"""Tests for wavelet decomposition systems."""

from __future__ import annotations

import numpy as np
import pytest

from vaerans_ecs.components.latent import YUVW4
from vaerans_ecs.components.wavelet import WaveletPyr
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.wavelet import WaveletCDF53, WaveletHaar


class TestWaveletCDF53:
    """Test CDF 5/3 wavelet system."""

    def test_init(self):
        """Test wavelet system initialization."""
        system = WaveletCDF53(levels=3, mode="forward")
        assert system.levels == 3
        assert system.mode == "forward"
        assert system.wavelet == "bior2.2"

    def test_init_invalid_levels(self):
        """Test initialization with invalid levels."""
        with pytest.raises(ValueError, match="levels must be in"):
            WaveletCDF53(levels=0)
        with pytest.raises(ValueError, match="levels must be in"):
            WaveletCDF53(levels=11)

    def test_required_components_forward(self):
        """Test required components in forward mode."""
        system = WaveletCDF53(mode="forward")
        required = system.required_components()
        assert required == [YUVW4]

    def test_required_components_inverse(self):
        """Test required components in inverse mode."""
        system = WaveletCDF53(mode="inverse")
        required = system.required_components()
        assert required == [WaveletPyr]

    def test_produced_components_forward(self):
        """Test produced components in forward mode."""
        system = WaveletCDF53(mode="forward")
        produced = system.produced_components()
        assert produced == [WaveletPyr]

    def test_produced_components_inverse(self):
        """Test produced components in inverse mode."""
        system = WaveletCDF53(mode="inverse")
        produced = system.produced_components()
        assert produced == [YUVW4]

    def test_forward_decomposition(self):
        """Test forward wavelet decomposition."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Create YUVW4 component with random data
        t_data = np.random.randn(4, 32, 32).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Run forward decomposition
        system = WaveletCDF53(levels=2, mode="forward")
        system.run(world, [entity])

        # Check WaveletPyr was created
        assert world.has_component(entity, WaveletPyr)
        pyr = world.get_component(entity, WaveletPyr)

        # Verify component fields
        assert pyr.levels == 2
        assert pyr.wavelet == "bior2.2"
        assert pyr.packed is not None
        assert pyr.index is not None

        # Verify packed data exists in arena
        packed = world.arena.view(pyr.packed)
        assert packed.dtype == np.float32
        assert len(packed) > 0

        # Verify index exists
        index = world.arena.view(pyr.index)
        assert index.dtype == np.int32
        assert len(index) > 0

    def test_inverse_reconstruction(self):
        """Test inverse wavelet reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Create YUVW4 component
        t_data = np.random.randn(4, 32, 32).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Forward decomposition
        forward = WaveletCDF53(levels=2, mode="forward")
        forward.run(world, [entity])

        # Inverse reconstruction
        inverse = WaveletCDF53(levels=2, mode="inverse")
        world.remove_component(entity, YUVW4)  # Remove original
        inverse.run(world, [entity])

        # Check YUVW4 was reconstructed
        assert world.has_component(entity, YUVW4)
        yuvw = world.get_component(entity, YUVW4)
        t_recon = world.arena.view(yuvw.t)

        # Verify shape matches
        assert t_recon.shape == t_data.shape

    def test_round_trip_accuracy(self):
        """Test round-trip decomposition and reconstruction accuracy."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Create YUVW4 component with random data
        t_original = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_original)
        world.add_component(entity, YUVW4(t=t_ref))

        # Forward then inverse
        forward = WaveletCDF53(levels=3, mode="forward")
        forward.run(world, [entity])

        inverse = WaveletCDF53(levels=3, mode="inverse")
        world.remove_component(entity, YUVW4)
        inverse.run(world, [entity])

        # Get reconstructed data
        yuvw = world.get_component(entity, YUVW4)
        t_recon = world.arena.view(yuvw.t)

        # Verify near-exact reconstruction (wavelet transform is nearly lossless)
        assert t_recon.shape == t_original.shape
        np.testing.assert_allclose(t_recon, t_original, rtol=1e-5, atol=1e-6)

    def test_multiple_entities(self):
        """Test processing multiple entities."""
        world = World(arena_bytes=20 << 20)
        entities = []

        # Create multiple entities
        for _ in range(3):
            entity = world.new_entity()
            t_data = np.random.randn(4, 32, 32).astype(np.float32)
            t_ref = world.arena.copy_tensor(t_data)
            world.add_component(entity, YUVW4(t=t_ref))
            entities.append(entity)

        # Process all entities
        system = WaveletCDF53(levels=2, mode="forward")
        system.run(world, entities)

        # Verify all have WaveletPyr
        for entity in entities:
            assert world.has_component(entity, WaveletPyr)

    def test_different_levels(self):
        """Test decomposition with different level counts."""
        world = World(arena_bytes=10 << 20)

        for levels in [1, 2, 4, 6]:
            entity = world.new_entity()
            t_data = np.random.randn(4, 64, 64).astype(np.float32)
            t_ref = world.arena.copy_tensor(t_data)
            world.add_component(entity, YUVW4(t=t_ref))

            # Forward
            forward = WaveletCDF53(levels=levels, mode="forward")
            forward.run(world, [entity])

            pyr = world.get_component(entity, WaveletPyr)
            assert pyr.levels == levels

            # Inverse
            inverse = WaveletCDF53(levels=levels, mode="inverse")
            world.remove_component(entity, YUVW4)
            inverse.run(world, [entity])

            # Verify reconstruction
            yuvw = world.get_component(entity, YUVW4)
            t_recon = world.arena.view(yuvw.t)
            np.testing.assert_allclose(t_recon, t_data, rtol=1e-5, atol=1e-6)

    def test_large_image(self):
        """Test with larger image sizes."""
        world = World(arena_bytes=50 << 20)
        entity = world.new_entity()

        # Larger image
        t_data = np.random.randn(4, 256, 256).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Forward and inverse
        forward = WaveletCDF53(levels=4, mode="forward")
        forward.run(world, [entity])

        inverse = WaveletCDF53(levels=4, mode="inverse")
        world.remove_component(entity, YUVW4)
        inverse.run(world, [entity])

        # Verify reconstruction
        yuvw = world.get_component(entity, YUVW4)
        t_recon = world.arena.view(yuvw.t)
        assert t_recon.shape == t_data.shape
        np.testing.assert_allclose(t_recon, t_data, rtol=1e-5, atol=1e-6)


class TestWaveletHaar:
    """Test Haar wavelet system."""

    def test_init(self):
        """Test Haar wavelet initialization."""
        system = WaveletHaar(levels=3, mode="forward")
        assert system.levels == 3
        assert system.mode == "forward"
        assert system.wavelet == "haar"

    def test_forward_decomposition(self):
        """Test Haar forward decomposition."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        t_data = np.random.randn(4, 32, 32).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        system = WaveletHaar(levels=2, mode="forward")
        system.run(world, [entity])

        assert world.has_component(entity, WaveletPyr)
        pyr = world.get_component(entity, WaveletPyr)
        assert pyr.wavelet == "haar"

    def test_round_trip_accuracy(self):
        """Test Haar round-trip accuracy."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        t_original = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_original)
        world.add_component(entity, YUVW4(t=t_ref))

        # Forward then inverse
        forward = WaveletHaar(levels=3, mode="forward")
        forward.run(world, [entity])

        inverse = WaveletHaar(levels=3, mode="inverse")
        world.remove_component(entity, YUVW4)
        inverse.run(world, [entity])

        # Get reconstructed data
        yuvw = world.get_component(entity, YUVW4)
        t_recon = world.arena.view(yuvw.t)

        # Haar should also have near-exact reconstruction
        assert t_recon.shape == t_original.shape
        np.testing.assert_allclose(t_recon, t_original, rtol=1e-5, atol=1e-6)

    def test_haar_vs_cdf53(self):
        """Test that Haar and CDF53 produce different results."""
        world = World(arena_bytes=20 << 20)

        # Same input for both
        t_data = np.random.randn(4, 64, 64).astype(np.float32)

        # Process with CDF53
        entity1 = world.new_entity()
        t_ref1 = world.arena.copy_tensor(t_data)
        world.add_component(entity1, YUVW4(t=t_ref1))
        cdf53 = WaveletCDF53(levels=2, mode="forward")
        cdf53.run(world, [entity1])
        pyr1 = world.get_component(entity1, WaveletPyr)

        # Process with Haar
        entity2 = world.new_entity()
        t_ref2 = world.arena.copy_tensor(t_data)
        world.add_component(entity2, YUVW4(t=t_ref2))
        haar = WaveletHaar(levels=2, mode="forward")
        haar.run(world, [entity2])
        pyr2 = world.get_component(entity2, WaveletPyr)

        # Verify different wavelet types
        assert pyr1.wavelet == "bior2.2"
        assert pyr2.wavelet == "haar"

        # Both should have same levels
        assert pyr1.levels == pyr2.levels

        # Packed arrays may have different sizes due to boundary handling
        # but both should have valid data
        packed1 = world.arena.view(pyr1.packed)
        packed2 = world.arena.view(pyr2.packed)
        assert len(packed1) > 0
        assert len(packed2) > 0


class TestWaveletEdgeCases:
    """Test edge cases for wavelet systems."""

    def test_minimum_levels(self):
        """Test with minimum level count (1)."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        t_data = np.random.randn(4, 32, 32).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        forward = WaveletCDF53(levels=1, mode="forward")
        forward.run(world, [entity])

        inverse = WaveletCDF53(levels=1, mode="inverse")
        world.remove_component(entity, YUVW4)
        inverse.run(world, [entity])

        yuvw = world.get_component(entity, YUVW4)
        t_recon = world.arena.view(yuvw.t)
        np.testing.assert_allclose(t_recon, t_data, rtol=1e-5, atol=1e-6)

    def test_maximum_levels(self):
        """Test with maximum level count (10)."""
        world = World(arena_bytes=100 << 20)
        entity = world.new_entity()

        # Need large enough image for 10 levels
        t_data = np.random.randn(4, 1024, 1024).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        forward = WaveletCDF53(levels=10, mode="forward")
        forward.run(world, [entity])

        assert world.has_component(entity, WaveletPyr)

    def test_non_power_of_two_size(self):
        """Test with non-power-of-2 image sizes."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Non-power-of-2 dimensions
        t_data = np.random.randn(4, 48, 60).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        forward = WaveletCDF53(levels=2, mode="forward")
        forward.run(world, [entity])

        inverse = WaveletCDF53(levels=2, mode="inverse")
        world.remove_component(entity, YUVW4)
        inverse.run(world, [entity])

        yuvw = world.get_component(entity, YUVW4)
        t_recon = world.arena.view(yuvw.t)

        # Shape should match (PyWavelets handles non-power-of-2)
        assert t_recon.shape == t_data.shape
