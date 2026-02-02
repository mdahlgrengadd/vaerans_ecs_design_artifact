"""Tests for quantization systems."""

from __future__ import annotations

import numpy as np
import pytest

from vaerans_ecs.components.latent import YUVW4
from vaerans_ecs.components.quant import QuantParams, SymbolsU8
from vaerans_ecs.components.wavelet import WaveletPyr
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.quantize import QuantizeU8
from vaerans_ecs.systems.wavelet import WaveletCDF53


class TestQuantizeU8:
    """Test QuantizeU8 system."""

    def test_init(self):
        """Test quantization system initialization."""
        system = QuantizeU8(quality=80, per_band=True, mode="forward")
        assert system.quality == 80
        assert system.per_band is True
        assert system.mode == "forward"

    def test_init_invalid_quality(self):
        """Test initialization with invalid quality."""
        with pytest.raises(ValueError, match="quality must be in"):
            QuantizeU8(quality=0)
        with pytest.raises(ValueError, match="quality must be in"):
            QuantizeU8(quality=101)

    def test_required_components_forward(self):
        """Test required components in forward mode."""
        system = QuantizeU8(mode="forward")
        required = system.required_components()
        assert required == [WaveletPyr]

    def test_required_components_inverse(self):
        """Test required components in inverse mode."""
        system = QuantizeU8(mode="inverse")
        required = system.required_components()
        assert required == [SymbolsU8]

    def test_produced_components_forward(self):
        """Test produced components in forward mode."""
        system = QuantizeU8(mode="forward")
        produced = system.produced_components()
        assert produced == [SymbolsU8]

    def test_produced_components_inverse(self):
        """Test produced components in inverse mode."""
        system = QuantizeU8(mode="inverse")
        produced = system.produced_components()
        assert produced == [WaveletPyr]

    def test_forward_quantization(self):
        """Test forward quantization."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Create wavelet pyramid via actual decomposition
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        # Now quantize
        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Check SymbolsU8 was created
        assert world.has_component(entity, SymbolsU8)
        symbols = world.get_component(entity, SymbolsU8)

        # Verify data is uint8
        data = world.arena.view(symbols.data)
        assert data.dtype == np.uint8
        assert len(data) > 0

        # Verify parameters exist
        assert symbols.params.quality == 50
        scales = world.arena.view(symbols.params.scales)
        offsets = world.arena.view(symbols.params.offsets)
        assert len(scales) > 0
        assert len(offsets) > 0

    def test_inverse_dequantization(self):
        """Test inverse dequantization."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Create wavelet pyramid
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        # Quantize
        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Dequantize (pyramid still exists for metadata)
        dequantize = QuantizeU8(quality=50, mode="inverse")
        dequantize.run(world, [entity])

        # Should produce WaveletPyr again
        assert world.has_component(entity, WaveletPyr)
        pyr = world.get_component(entity, WaveletPyr)
        packed = world.arena.view(pyr.packed)
        assert packed.dtype == np.float32

    def test_round_trip_quality(self):
        """Test round-trip quantization quality."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Create wavelet pyramid
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Wavelet decomposition
        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])
        pyr_original = world.get_component(entity, WaveletPyr)
        packed_original = world.arena.view(pyr_original.packed).copy()

        # Quantize and dequantize
        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        dequantize = QuantizeU8(quality=50, mode="inverse")
        dequantize.run(world, [entity])

        # Get reconstructed
        pyr_recon = world.get_component(entity, WaveletPyr)
        packed_recon = world.arena.view(pyr_recon.packed)

        # Should be similar but not exact (lossy compression)
        # Higher quality should have lower error
        mse = np.mean((packed_original - packed_recon) ** 2)
        assert mse < 10.0  # Reasonable error bound for quality=50

    def test_quality_levels(self):
        """Test different quality levels produce different quantization."""
        world = World(arena_bytes=40 << 20)

        # Same input for all tests
        t_data = np.random.randn(4, 64, 64).astype(np.float32)

        errors = []
        for quality in [10, 50, 90]:
            entity = world.new_entity()
            t_ref = world.arena.copy_tensor(t_data)
            world.add_component(entity, YUVW4(t=t_ref))

            # Wavelet
            wavelet = WaveletCDF53(levels=2, mode="forward")
            wavelet.run(world, [entity])
            pyr = world.get_component(entity, WaveletPyr)
            packed_original = world.arena.view(pyr.packed).copy()

            # Quantize + dequantize
            quantize = QuantizeU8(quality=quality, mode="forward")
            quantize.run(world, [entity])

            dequantize = QuantizeU8(quality=quality, mode="inverse")
            dequantize.run(world, [entity])

            # Measure error
            pyr_recon = world.get_component(entity, WaveletPyr)
            packed_recon = world.arena.view(pyr_recon.packed)
            mse = np.mean((packed_original - packed_recon) ** 2)
            errors.append(mse)

        # Higher quality should have lower error
        # errors should be monotonically decreasing
        assert errors[2] < errors[1]  # quality 90 < quality 50
        assert errors[1] < errors[0]  # quality 50 < quality 10

    def test_per_band_vs_global(self):
        """Test per-band vs global quantization."""
        world = World(arena_bytes=40 << 20)
        t_data = np.random.randn(4, 64, 64).astype(np.float32)

        # Test per_band=True
        entity1 = world.new_entity()
        t_ref1 = world.arena.copy_tensor(t_data)
        world.add_component(entity1, YUVW4(t=t_ref1))
        wavelet1 = WaveletCDF53(levels=2, mode="forward")
        wavelet1.run(world, [entity1])
        quantize1 = QuantizeU8(quality=50, per_band=True, mode="forward")
        quantize1.run(world, [entity1])
        symbols1 = world.get_component(entity1, SymbolsU8)
        scales1 = world.arena.view(symbols1.params.scales)

        # Test per_band=False
        entity2 = world.new_entity()
        t_ref2 = world.arena.copy_tensor(t_data)
        world.add_component(entity2, YUVW4(t=t_ref2))
        wavelet2 = WaveletCDF53(levels=2, mode="forward")
        wavelet2.run(world, [entity2])
        quantize2 = QuantizeU8(quality=50, per_band=False, mode="forward")
        quantize2.run(world, [entity2])
        symbols2 = world.get_component(entity2, SymbolsU8)
        scales2 = world.arena.view(symbols2.params.scales)

        # Per-band should have more scale factors
        assert len(scales1) >= len(scales2)
        # Global should have just one scale
        assert len(scales2) == 1

    def test_multiple_entities(self):
        """Test quantizing multiple entities."""
        world = World(arena_bytes=40 << 20)
        entities = []

        for _ in range(3):
            entity = world.new_entity()
            t_data = np.random.randn(4, 32, 32).astype(np.float32)
            t_ref = world.arena.copy_tensor(t_data)
            world.add_component(entity, YUVW4(t=t_ref))

            wavelet = WaveletCDF53(levels=2, mode="forward")
            wavelet.run(world, [entity])
            entities.append(entity)

        # Quantize all
        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, entities)

        # All should have SymbolsU8
        for entity in entities:
            assert world.has_component(entity, SymbolsU8)

    def test_extreme_quality(self):
        """Test extreme quality values."""
        world = World(arena_bytes=20 << 20)

        # Test quality=1 (worst)
        entity1 = world.new_entity()
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref1 = world.arena.copy_tensor(t_data)
        world.add_component(entity1, YUVW4(t=t_ref1))
        wavelet1 = WaveletCDF53(levels=2, mode="forward")
        wavelet1.run(world, [entity1])
        quantize1 = QuantizeU8(quality=1, mode="forward")
        quantize1.run(world, [entity1])
        assert world.has_component(entity1, SymbolsU8)

        # Test quality=100 (best)
        entity2 = world.new_entity()
        t_ref2 = world.arena.copy_tensor(t_data)
        world.add_component(entity2, YUVW4(t=t_ref2))
        wavelet2 = WaveletCDF53(levels=2, mode="forward")
        wavelet2.run(world, [entity2])
        quantize2 = QuantizeU8(quality=100, mode="forward")
        quantize2.run(world, [entity2])
        assert world.has_component(entity2, SymbolsU8)

    def test_uniform_data(self):
        """Test quantization with uniform data."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # All zeros
        t_data = np.zeros((4, 64, 64), dtype=np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Should not crash and produce valid output
        assert world.has_component(entity, SymbolsU8)
        symbols = world.get_component(entity, SymbolsU8)
        data = world.arena.view(symbols.data)
        # All zeros should quantize to mostly zeros
        assert np.mean(data) < 10.0
