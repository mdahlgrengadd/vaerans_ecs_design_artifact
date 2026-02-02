"""Tests for Bitstream Serialization (Phase 9)."""

import json
import struct

import numpy as np
import pytest

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.quant import QuantParams
from vaerans_ecs.core.arena import Arena
from vaerans_ecs.core.serialization import (
    HEADER_SIZE,
    MAGIC,
    VERSION_MAJOR,
    VERSION_MINOR,
    deserialize_bitstream,
    get_serialized_size,
    serialize_bitstream,
)


@pytest.fixture
def arena() -> Arena:
    """Create a test arena."""
    return Arena(size_bytes=10 << 20)  # 10 MB


@pytest.fixture
def valid_bitstream(arena: Arena) -> ANSBitstream:
    """Create a valid test bitstream."""
    # Create data tensor (bitstream bytes)
    data = np.random.randint(0, 256, 1000, dtype=np.uint8)
    data_ref = arena.copy_tensor(data)

    # Create probability tensor
    probs = np.random.rand(256).astype(np.float32)
    probs = probs / probs.sum()  # Normalize
    probs_ref = arena.copy_tensor(probs)

    return ANSBitstream(data=data_ref, probs=probs_ref, initial_state=12345)


@pytest.fixture
def valid_quant_params(arena: Arena) -> QuantParams:
    """Create valid quantization parameters."""
    scales = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offsets = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    scales_ref = arena.copy_tensor(scales)
    offsets_ref = arena.copy_tensor(offsets)

    return QuantParams(scales=scales_ref, offsets=offsets_ref, quality=50)


class TestSerializeBasics:
    """Test basic serialization functionality."""

    def test_serialize_valid_bitstream(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test serializing a valid bitstream."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        assert isinstance(data, bytes)
        assert len(data) > HEADER_SIZE
        assert data[:4] == MAGIC

    def test_serialize_returns_bytes(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that serialization returns bytes."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(512, 512, 3),
            quant_params=valid_quant_params,
        )

        assert isinstance(data, bytes)


class TestDeserializeBasics:
    """Test basic deserialization functionality."""

    def test_deserialize_valid_data(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test deserializing valid data."""
        original_data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        metadata, probs, bitstream_data = deserialize_bitstream(original_data)

        assert isinstance(metadata, dict)
        assert isinstance(probs, np.ndarray)
        assert isinstance(bitstream_data, np.ndarray)
        assert metadata["model"] == "test-vae"
        assert metadata["wavelet_levels"] == 4
        assert metadata["image_shape"] == [256, 256, 3]

    def test_deserialize_metadata_preserved(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that metadata is correctly deserialized."""
        original_data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="sdxl-vae",
            levels=3,
            image_shape=(512, 512, 3),
            quant_params=valid_quant_params,
        )

        metadata, _, _ = deserialize_bitstream(original_data)

        assert metadata["model"] == "sdxl-vae"
        assert metadata["wavelet_levels"] == 3
        assert metadata["image_shape"] == [512, 512, 3]
        assert "initial_state" in metadata


class TestRoundTrip:
    """Test round-trip serialization/deserialization."""

    def test_roundtrip_metadata(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that metadata is preserved in round-trip."""
        model_name = "test-vae"
        levels = 4
        image_shape = (256, 256, 3)

        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model=model_name,
            levels=levels,
            image_shape=image_shape,
            quant_params=valid_quant_params,
        )

        metadata, _, _ = deserialize_bitstream(data)

        assert metadata["model"] == model_name
        assert metadata["wavelet_levels"] == levels
        assert metadata["image_shape"] == list(image_shape)

    def test_roundtrip_probability_table(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that probability table is preserved."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        _, probs, _ = deserialize_bitstream(data)

        # Get original probs
        original_probs = arena.view(valid_bitstream.probs)

        assert np.allclose(probs, original_probs, atol=1e-6)

    def test_roundtrip_bitstream_data(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that bitstream data is preserved."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        _, _, bitstream_data = deserialize_bitstream(data)

        # Get original data
        original_data = arena.view(valid_bitstream.data)

        assert np.array_equal(bitstream_data, original_data)


class TestVersionChecking:
    """Test version checking."""

    def test_version_in_header(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that version is correctly stored in header."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        # Parse header manually
        magic, version_byte, _, _ = struct.unpack("<4sHII", data[:HEADER_SIZE])

        ver_major = (version_byte >> 8) & 0xFF
        ver_minor = version_byte & 0xFF

        assert magic == MAGIC
        assert ver_major == VERSION_MAJOR
        assert ver_minor == VERSION_MINOR

    def test_reject_unsupported_version(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that unsupported version is rejected."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        # Modify version to unsupported value
        modified_data = bytearray(data)
        modified_data[4:6] = struct.pack("<H", 2)  # Version 2.0

        with pytest.raises(ValueError, match="Unsupported version"):
            deserialize_bitstream(bytes(modified_data))

    def test_reject_invalid_magic(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that invalid magic is rejected."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        # Modify magic number
        modified_data = bytearray(data)
        modified_data[0:4] = b"XXXX"

        with pytest.raises(ValueError, match="Invalid file format"):
            deserialize_bitstream(bytes(modified_data))


class TestErrorHandling:
    """Test error handling for invalid data."""

    def test_deserialize_empty_data(self) -> None:
        """Test deserialization of empty data."""
        with pytest.raises(ValueError, match="Data too short"):
            deserialize_bitstream(b"")

    def test_deserialize_truncated_header(self) -> None:
        """Test deserialization of truncated header."""
        with pytest.raises(ValueError, match="Data too short"):
            deserialize_bitstream(b"VAE\x00")

    def test_deserialize_truncated_metadata(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test deserialization of truncated metadata."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        # Truncate metadata
        with pytest.raises(ValueError, match="extends beyond data"):
            deserialize_bitstream(data[:20])

    def test_serialize_invalid_bitstream_type(
        self, arena: Arena, valid_quant_params: QuantParams
    ) -> None:
        """Test serialization with invalid bitstream type."""
        with pytest.raises(TypeError, match="Expected ANSBitstream"):
            serialize_bitstream(
                "not a bitstream",  # type: ignore
                arena=arena,
                model="test-vae",
                levels=4,
                image_shape=(256, 256, 3),
                quant_params=valid_quant_params,
            )


class TestFileSize:
    """Test serialized file size calculations."""

    def test_get_serialized_size_basic(self) -> None:
        """Test getting serialized size."""
        bitstream_data = np.random.randint(0, 256, 1000, dtype=np.uint8)
        probs_data = np.random.rand(256).astype(np.float32)

        size = get_serialized_size(bitstream_data, probs_data)
        assert isinstance(size, int)
        assert size > HEADER_SIZE

    def test_serialized_size_matches_actual(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test that calculated size matches actual serialized size."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        bitstream_data = arena.view(valid_bitstream.data)
        probs_data = arena.view(valid_bitstream.probs)

        metadata = {
            "model": "test-vae",
            "wavelet_levels": 4,
            "image_shape": [256, 256, 3],
            "initial_state": 12345,
        }

        calculated_size = get_serialized_size(bitstream_data, probs_data, metadata)
        assert calculated_size == len(data)

    def test_size_scales_with_data_length(self) -> None:
        """Test that size scales with bitstream data length."""
        probs = np.random.rand(256).astype(np.float32)

        data_small = np.random.randint(0, 256, 100, dtype=np.uint8)
        data_large = np.random.randint(0, 256, 1000, dtype=np.uint8)

        size_small = get_serialized_size(data_small, probs)
        size_large = get_serialized_size(data_large, probs)

        # Large should be 900 bytes bigger (1000 - 100 elements)
        assert size_large - size_small == 900


class TestEdgeCases:
    """Test edge cases."""

    def test_serialize_minimum_bitstream(
        self, arena: Arena, valid_quant_params: QuantParams
    ) -> None:
        """Test serialization with minimum valid bitstream."""
        data = np.array([0, 1, 2], dtype=np.uint8)
        data_ref = arena.copy_tensor(data)

        probs = np.ones(256, dtype=np.float32) / 256
        probs_ref = arena.copy_tensor(probs)

        bs = ANSBitstream(data=data_ref, probs=probs_ref, initial_state=0)

        serialized = serialize_bitstream(
            bs,
            arena=arena,
            model="x",
            levels=1,
            image_shape=(1, 1, 3),
            quant_params=valid_quant_params,
        )

        metadata, probs_out, data_out = deserialize_bitstream(serialized)
        assert len(data_out) == 3
        assert len(probs_out) == 256

    def test_serialize_with_unicode_model_name(
        self,
        arena: Arena,
        valid_bitstream: ANSBitstream,
        valid_quant_params: QuantParams,
    ) -> None:
        """Test serialization with unicode characters in model name."""
        data = serialize_bitstream(
            valid_bitstream,
            arena=arena,
            model="test-vae-日本語",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        metadata, _, _ = deserialize_bitstream(data)
        assert metadata["model"] == "test-vae-日本語"

    def test_roundtrip_with_zero_probability_symbols(
        self, arena: Arena, valid_quant_params: QuantParams
    ) -> None:
        """Test round-trip with zero probability for some symbols."""
        data = np.array([0, 0, 0], dtype=np.uint8)
        data_ref = arena.copy_tensor(data)

        probs = np.zeros(256, dtype=np.float32)
        probs[0] = 1.0  # Only symbol 0 has probability
        probs_ref = arena.copy_tensor(probs)

        bs = ANSBitstream(data=data_ref, probs=probs_ref, initial_state=0)

        serialized = serialize_bitstream(
            bs,
            arena=arena,
            model="test-vae",
            levels=4,
            image_shape=(256, 256, 3),
            quant_params=valid_quant_params,
        )

        _, probs_out, data_out = deserialize_bitstream(serialized)
        assert np.allclose(probs_out, probs, atol=1e-6)
