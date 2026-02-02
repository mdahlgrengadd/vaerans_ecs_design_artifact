"""Tests for High-Level API (Phase 13)."""

import os

import numpy as np
import pytest

from vaerans_ecs.api import (
    compress,
    decompress,
    get_compression_info,
    get_compression_ratio,
)


@pytest.fixture
def test_image_small() -> np.ndarray:
    """Create a small test image."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def test_image_medium() -> np.ndarray:
    """Create a medium test image."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def test_image_large() -> np.ndarray:
    """Create a large test image."""
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


class TestCompressBasics:
    """Test basic compression functionality."""

    def test_compress_returns_bytes(self, test_image_small: np.ndarray) -> None:
        """Test that compress returns bytes."""
        try:
            result = compress(test_image_small)
            assert isinstance(result, bytes)
            assert len(result) > 0
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compress_with_model_name(self, test_image_small: np.ndarray) -> None:
        """Test compress with specific model name."""
        try:
            result = compress(test_image_small, model="sdxl-vae")
            assert isinstance(result, bytes)
            assert len(result) > 0
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compress_with_hadamard(self, test_image_small: np.ndarray) -> None:
        """Test compress with Hadamard transform enabled."""
        try:
            result = compress(test_image_small, use_hadamard=True)
            assert isinstance(result, bytes)
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compress_without_hadamard(self, test_image_small: np.ndarray) -> None:
        """Test compress with Hadamard transform disabled."""
        try:
            result = compress(test_image_small, use_hadamard=False)
            assert isinstance(result, bytes)
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compress_with_quality(self, test_image_small: np.ndarray) -> None:
        """Test compress with different quality settings."""
        try:
            for quality in [10, 50, 90]:
                result = compress(test_image_small, quality=quality)
                assert isinstance(result, bytes)
        except FileNotFoundError:
            pytest.skip("Test models not available")


class TestCompressErrors:
    """Test compression error handling."""

    def test_compress_invalid_dtype(self) -> None:
        """Test compress with invalid dtype."""
        img = np.random.rand(64, 64, 3).astype(np.float64)  # Should be uint8
        with pytest.raises(ValueError, match="dtype"):
            compress(img)

    def test_compress_invalid_shape_2d(self) -> None:
        """Test compress with 2D image."""
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            compress(img)

    def test_compress_invalid_shape_4d(self) -> None:
        """Test compress with 4D array."""
        img = np.random.randint(0, 256, (1, 64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            compress(img)

    def test_compress_invalid_channels(self) -> None:
        """Test compress with wrong number of channels."""
        img = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            compress(img)

    def test_compress_invalid_type(self) -> None:
        """Test compress with non-array input."""
        with pytest.raises(TypeError, match="ndarray"):
            compress([1, 2, 3])  # type: ignore


class TestDecompressBasics:
    """Test basic decompression functionality."""

    def test_decompress_returns_array(self, test_image_small: np.ndarray) -> None:
        """Test that decompress returns numpy array."""
        try:
            compressed = compress(test_image_small)
            result = decompress(compressed)

            assert isinstance(result, np.ndarray)
            assert result.ndim == 3
            assert result.shape[2] == 3
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_decompress_output_dtype(self, test_image_small: np.ndarray) -> None:
        """Test decompressed image dtype."""
        try:
            compressed = compress(test_image_small)
            result = decompress(compressed)

            assert result.dtype == np.float32
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_decompress_value_range(self, test_image_small: np.ndarray) -> None:
        """Test decompressed image values are in valid range."""
        try:
            compressed = compress(test_image_small)
            result = decompress(compressed)

            # VAE reconstruction should be in [0, 1] range
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)
        except FileNotFoundError:
            pytest.skip("Test models not available")


class TestRoundTrip:
    """Test round-trip compression/decompression."""

    def test_roundtrip_small_image(self, test_image_small: np.ndarray) -> None:
        """Test compression roundtrip with small image."""
        try:
            compressed = compress(test_image_small)
            reconstructed = decompress(compressed)

            assert reconstructed.shape[:2] == test_image_small.shape[:2]
            assert reconstructed.shape[2] == 3
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_roundtrip_medium_image(self, test_image_medium: np.ndarray) -> None:
        """Test compression roundtrip with medium image."""
        try:
            compressed = compress(test_image_medium)
            reconstructed = decompress(compressed)

            assert reconstructed.shape == test_image_medium.shape
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_roundtrip_with_hadamard(self, test_image_small: np.ndarray) -> None:
        """Test roundtrip with Hadamard transform."""
        try:
            compressed = compress(test_image_small, use_hadamard=True)
            reconstructed = decompress(compressed)

            assert reconstructed.shape == test_image_small.shape
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_roundtrip_without_hadamard(self, test_image_small: np.ndarray) -> None:
        """Test roundtrip without Hadamard transform."""
        try:
            compressed = compress(test_image_small, use_hadamard=False)
            reconstructed = decompress(compressed)

            assert reconstructed.shape == test_image_small.shape
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_roundtrip_different_qualities(self, test_image_small: np.ndarray) -> None:
        """Test that different quality settings produce different compressed sizes."""
        try:
            compressed_q10 = compress(test_image_small, quality=10)
            compressed_q50 = compress(test_image_small, quality=50)
            compressed_q90 = compress(test_image_small, quality=90)

            # All should decompress without error
            _ = decompress(compressed_q10)
            _ = decompress(compressed_q50)
            _ = decompress(compressed_q90)

        except FileNotFoundError:
            pytest.skip("Test models not available")


class TestCompressionInfo:
    """Test metadata extraction."""

    def test_get_compression_info(self, test_image_small: np.ndarray) -> None:
        """Test extracting compression info from compressed data."""
        try:
            compressed = compress(test_image_small, model="sdxl-vae")
            info = get_compression_info(compressed)

            assert isinstance(info, dict)
            assert "model" in info
            assert "image_shape" in info
            assert "wavelet_levels" in info
            assert info["model"] == "sdxl-vae"
            assert info["image_shape"] == [64, 64, 3]

        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compression_info_invalid_data(self) -> None:
        """Test get_compression_info with invalid data."""
        with pytest.raises(ValueError):
            get_compression_info(b"invalid data")

    def test_compression_info_truncated_data(self) -> None:
        """Test get_compression_info with truncated data."""
        with pytest.raises(ValueError):
            get_compression_info(b"VAE\x00")


class TestCompressionRatio:
    """Test compression ratio calculations."""

    def test_compression_ratio_basic(self, test_image_small: np.ndarray) -> None:
        """Test compression ratio calculation."""
        try:
            compressed = compress(test_image_small)
            ratio = get_compression_ratio(test_image_small, compressed)

            assert isinstance(ratio, float)
            assert ratio > 0
        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compression_ratio_different_sizes(self) -> None:
        """Test compression ratio for different image sizes."""
        try:
            img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

            compressed1 = compress(img1)
            compressed2 = compress(img2)

            ratio1 = get_compression_ratio(img1, compressed1)
            ratio2 = get_compression_ratio(img2, compressed2)

            assert ratio1 > 0
            assert ratio2 > 0

        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_compression_ratio_empty_compressed(self, test_image_small: np.ndarray) -> None:
        """Test compression ratio with empty compressed data."""
        ratio = get_compression_ratio(test_image_small, b"")
        assert ratio == float("inf")


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_simple_workflow(self, test_image_small: np.ndarray) -> None:
        """Test simple compress-decompress workflow."""
        try:
            # Compress
            print("Compressing image...")
            compressed = compress(test_image_small, model="sdxl-vae", quality=50)
            print(f"Compressed size: {len(compressed)} bytes")

            # Get info
            info = get_compression_info(compressed)
            print(f"Model: {info['model']}")
            print(f"Image shape: {info['image_shape']}")

            # Get ratio
            ratio = get_compression_ratio(test_image_small, compressed)
            print(f"Compression ratio: {ratio:.2f}x")

            # Decompress
            print("Decompressing image...")
            reconstructed = decompress(compressed)
            print(f"Reconstructed shape: {reconstructed.shape}")
            print(f"Reconstructed dtype: {reconstructed.dtype}")
            print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

            # Verify
            assert reconstructed.shape[2] == 3
            assert reconstructed.dtype == np.float32
            assert np.all(reconstructed >= 0.0)
            assert np.all(reconstructed <= 1.0)

        except FileNotFoundError:
            pytest.skip("Test models not available")

    def test_multiple_compressions(self, test_image_small: np.ndarray) -> None:
        """Test compressing multiple images sequentially."""
        try:
            images = [
                test_image_small,
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            ]

            for i, img in enumerate(images):
                compressed = compress(img)
                reconstructed = decompress(compressed)

                assert reconstructed.shape == img.shape

        except FileNotFoundError:
            pytest.skip("Test models not available")
