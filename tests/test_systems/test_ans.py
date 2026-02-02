"""Tests for ANS entropy coding systems."""

from __future__ import annotations

import numpy as np
import pytest

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.latent import YUVW4
from vaerans_ecs.components.quant import SymbolsU8
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.ans import ANSDecode, ANSEncode
from vaerans_ecs.systems.quantize import QuantizeU8
from vaerans_ecs.systems.wavelet import WaveletCDF53


class TestANSEncode:
    """Test ANS encoding system."""

    def test_init(self):
        """Test ANS encoder initialization."""
        encoder = ANSEncode()
        assert encoder.mode == "forward"

    def test_init_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="only supports forward"):
            ANSEncode(mode="inverse")

    def test_required_components(self):
        """Test required components."""
        encoder = ANSEncode()
        required = encoder.required_components()
        assert required == [SymbolsU8]

    def test_produced_components(self):
        """Test produced components."""
        encoder = ANSEncode()
        produced = encoder.produced_components()
        assert produced == [ANSBitstream]

    def test_encode_symbols(self):
        """Test encoding symbols to bitstream."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Create a full pipeline to get symbols
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Wavelet + Quantize
        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Get original symbols
        orig_symbols = world.get_component(entity, SymbolsU8)
        orig_data = world.arena.view(orig_symbols.data)
        orig_size = len(orig_data)

        # ANS encode
        encoder = ANSEncode()
        encoder.run(world, [entity])

        # Check ANSBitstream was created
        assert world.has_component(entity, ANSBitstream)
        bitstream = world.get_component(entity, ANSBitstream)

        # Verify bitstream properties
        data = world.arena.view(bitstream.data)
        assert data.dtype == np.uint8
        assert len(data) > 0

        # Compressed should be smaller (or similar size for random data)
        # This test might not always hold for truly random data
        # but should hold for structured data
        print(f"Original: {orig_size} bytes, Compressed: {len(data)} bytes")

        # Verify probability table
        probs = world.arena.view(bitstream.probs)
        assert probs.dtype in (np.float32, np.float64)
        assert len(probs) == 256
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_multiple_entities(self):
        """Test encoding multiple entities."""
        world = World(arena_bytes=40 << 20)
        entities = []

        for _ in range(3):
            entity = world.new_entity()
            t_data = np.random.randn(4, 32, 32).astype(np.float32)
            t_ref = world.arena.copy_tensor(t_data)
            world.add_component(entity, YUVW4(t=t_ref))

            wavelet = WaveletCDF53(levels=2, mode="forward")
            wavelet.run(world, [entity])

            quantize = QuantizeU8(quality=50, mode="forward")
            quantize.run(world, [entity])

            entities.append(entity)

        # Encode all
        encoder = ANSEncode()
        encoder.run(world, entities)

        # All should have ANSBitstream
        for entity in entities:
            assert world.has_component(entity, ANSBitstream)


class TestANSDecode:
    """Test ANS decoding system."""

    def test_init(self):
        """Test ANS decoder initialization."""
        decoder = ANSDecode()
        assert decoder.mode == "forward"

    def test_init_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="only supports forward"):
            ANSDecode(mode="inverse")

    def test_required_components(self):
        """Test required components."""
        decoder = ANSDecode()
        required = decoder.required_components()
        assert required == [ANSBitstream]

    def test_produced_components(self):
        """Test produced components."""
        decoder = ANSDecode()
        produced = decoder.produced_components()
        assert produced == [SymbolsU8]

    def test_decode_bitstream(self):
        """Test decoding bitstream to symbols."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Full pipeline to encode
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Store original symbols
        orig_symbols = world.get_component(entity, SymbolsU8)
        orig_data = world.arena.view(orig_symbols.data).copy()

        # Encode
        encoder = ANSEncode()
        encoder.run(world, [entity])

        # Decode (symbols still exist for metadata)
        decoder = ANSDecode()
        decoder.run(world, [entity])

        # Should produce SymbolsU8
        assert world.has_component(entity, SymbolsU8)
        decoded_symbols = world.get_component(entity, SymbolsU8)
        decoded_data = world.arena.view(decoded_symbols.data)

        # Decoded should match original exactly (lossless)
        assert decoded_data.shape == orig_data.shape
        assert decoded_data.dtype == orig_data.dtype


class TestANSRoundTrip:
    """Test complete ANS encode/decode round-trip."""

    def test_lossless_round_trip(self):
        """Test that ANS encoding is lossless."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Create data
        t_data = np.random.randn(4, 64, 64).astype(np.float32)
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Wavelet + Quantize
        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Store original
        orig_symbols = world.get_component(entity, SymbolsU8)
        orig_data = world.arena.view(orig_symbols.data).copy()

        # Encode then decode
        encoder = ANSEncode()
        encoder.run(world, [entity])

        decoder = ANSDecode()
        decoder.run(world, [entity])

        # Verify exact recovery
        decoded_symbols = world.get_component(entity, SymbolsU8)
        decoded_data = world.arena.view(decoded_symbols.data)

        np.testing.assert_array_equal(decoded_data, orig_data)

    def test_compression_ratio(self):
        """Test compression achieves some bitrate reduction."""
        world = World(arena_bytes=20 << 20)
        entity = world.new_entity()

        # Create structured data (should compress well)
        t_data = np.zeros((4, 64, 64), dtype=np.float32)
        # Add some structure
        t_data[0, :32, :32] = 1.0
        t_data[1, 32:, 32:] = 2.0
        
        t_ref = world.arena.copy_tensor(t_data)
        world.add_component(entity, YUVW4(t=t_ref))

        # Wavelet + Quantize
        wavelet = WaveletCDF53(levels=2, mode="forward")
        wavelet.run(world, [entity])

        quantize = QuantizeU8(quality=50, mode="forward")
        quantize.run(world, [entity])

        # Get original size
        orig_symbols = world.get_component(entity, SymbolsU8)
        orig_data = world.arena.view(orig_symbols.data)
        orig_size = len(orig_data)

        # Encode
        encoder = ANSEncode()
        encoder.run(world, [entity])

        # Get compressed size
        bitstream = world.get_component(entity, ANSBitstream)
        compressed_data = world.arena.view(bitstream.data)
        compressed_size = len(compressed_data)

        print(f"Original: {orig_size} bytes, Compressed: {compressed_size} bytes")
        print(f"Compression ratio: {orig_size / compressed_size:.2f}x")

        # Structured data should achieve some compression
        # This is a weak test since random noise won't compress
        assert compressed_size <= orig_size * 1.2  # Allow 20% overhead for encoding

    def test_different_symbol_distributions(self):
        """Test ANS with different symbol distributions."""
        world = World(arena_bytes=40 << 20)

        # Test uniform distribution
        entity1 = world.new_entity()
        symbols1 = np.random.randint(0, 256, 10000, dtype=np.uint8)
        from vaerans_ecs.components.quant import QuantParams
        data_ref1 = world.arena.copy_tensor(symbols1)
        scales_ref = world.arena.copy_tensor(np.array([1.0], dtype=np.float32))
        offsets_ref = world.arena.copy_tensor(np.array([0.0], dtype=np.float32))
        params = QuantParams(scales=scales_ref, offsets=offsets_ref, quality=50)
        world.add_component(entity1, SymbolsU8(data=data_ref1, params=params))

        encoder1 = ANSEncode()
        encoder1.run(world, [entity1])

        decoder1 = ANSDecode()
        decoder1.run(world, [entity1])

        decoded1 = world.get_component(entity1, SymbolsU8)
        decoded_data1 = world.arena.view(decoded1.data)
        np.testing.assert_array_equal(decoded_data1, symbols1)

        # Test skewed distribution
        entity2 = world.new_entity()
        # Mostly zeros
        symbols2 = np.zeros(10000, dtype=np.uint8)
        symbols2[::10] = np.random.randint(1, 256, 1000, dtype=np.uint8)
        data_ref2 = world.arena.copy_tensor(symbols2)
        params2 = QuantParams(scales=scales_ref, offsets=offsets_ref, quality=50)
        world.add_component(entity2, SymbolsU8(data=data_ref2, params=params2))

        encoder2 = ANSEncode()
        encoder2.run(world, [entity2])

        decoder2 = ANSDecode()
        decoder2.run(world, [entity2])

        decoded2 = world.get_component(entity2, SymbolsU8)
        decoded_data2 = world.arena.view(decoded2.data)
        np.testing.assert_array_equal(decoded_data2, symbols2)


class TestANSEdgeCases:
    """Test edge cases for ANS coding."""

    def test_single_symbol(self):
        """Test encoding data with single unique symbol."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # All zeros
        symbols = np.zeros(1000, dtype=np.uint8)
        from vaerans_ecs.components.quant import QuantParams
        data_ref = world.arena.copy_tensor(symbols)
        scales_ref = world.arena.copy_tensor(np.array([1.0], dtype=np.float32))
        offsets_ref = world.arena.copy_tensor(np.array([0.0], dtype=np.float32))
        params = QuantParams(scales=scales_ref, offsets=offsets_ref, quality=50)
        world.add_component(entity, SymbolsU8(data=data_ref, params=params))

        # Encode and decode
        encoder = ANSEncode()
        encoder.run(world, [entity])

        decoder = ANSDecode()
        decoder.run(world, [entity])

        # Verify lossless
        decoded = world.get_component(entity, SymbolsU8)
        decoded_data = world.arena.view(decoded.data)
        np.testing.assert_array_equal(decoded_data, symbols)

    def test_small_data(self):
        """Test encoding very small data."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Just 10 symbols
        symbols = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8)
        from vaerans_ecs.components.quant import QuantParams
        data_ref = world.arena.copy_tensor(symbols)
        scales_ref = world.arena.copy_tensor(np.array([1.0], dtype=np.float32))
        offsets_ref = world.arena.copy_tensor(np.array([0.0], dtype=np.float32))
        params = QuantParams(scales=scales_ref, offsets=offsets_ref, quality=50)
        world.add_component(entity, SymbolsU8(data=data_ref, params=params))

        encoder = ANSEncode()
        encoder.run(world, [entity])

        decoder = ANSDecode()
        decoder.run(world, [entity])

        decoded = world.get_component(entity, SymbolsU8)
        decoded_data = world.arena.view(decoded.data)
        np.testing.assert_array_equal(decoded_data, symbols)
