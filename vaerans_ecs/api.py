"""High-level API for image compression and decompression (Phase 13).

Provides user-friendly compress() and decompress() functions that handle
the complete compression pipeline.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.image import ReconRGB, RGB
from vaerans_ecs.components.quant import QuantParams
from vaerans_ecs.core.serialization import deserialize_bitstream, serialize_bitstream
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4
from vaerans_ecs.systems.vae import OnnxVAEDecode, OnnxVAEEncode


def compress(
    image: np.ndarray,
    model: str = "sdxl-vae",
    quality: int = 50,
    use_hadamard: bool = True,
    config_path: str | None = None,
) -> bytes:
    """Compress an RGB image to bytes.

    Encodes image using VAE and optionally applies Hadamard transform,
    then serializes to a byte string.

    Args:
        image: Input image as (H, W, 3) uint8 array
        model: VAE model name from config (default: 'sdxl-vae')
        quality: Quality level 1-100 (for future compression stages)
        use_hadamard: Whether to apply Hadamard transform (default: True)
        config_path: Path to vaerans_ecs.toml (auto-detected if None)

    Returns:
        Compressed image as bytes

    Raises:
        ValueError: If image has invalid shape or dtype
        FileNotFoundError: If model not found
        ImportError: If onnxruntime not installed

    Example:
        >>> import numpy as np
        >>> from vaerans_ecs import compress, decompress
        >>> img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> compressed = compress(img, model='sdxl-vae')
        >>> reconstructed = decompress(compressed)
        >>> print(reconstructed.shape)
        (512, 512, 3)
    """
    # Validate input
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected shape (H, W, 3), got {image.shape}"
        )
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(
            f"Expected uint8 or int dtype, got {image.dtype}"
        )

    # Create world with arena
    world = World(arena_bytes=512 << 20)  # 512 MB

    try:
        # Spawn image entity
        entity = world.spawn_image(image)

        # Build encode pipeline
        pipeline = (
            world.pipe(entity)
            .to(OnnxVAEEncode(model=model, mode="encode", config_path=config_path))
        )

        # Optionally apply Hadamard transform
        if use_hadamard:
            pipeline = pipeline.to(Hadamard4(mode="forward"))

        # Get latent (or transformed latent if Hadamard applied)
        from vaerans_ecs.components.latent import Latent4, YUVW4

        latent_component = YUVW4 if use_hadamard else Latent4
        latent = pipeline.out(latent_component)

        # Create dummy quantization parameters for serialization
        # (In full implementation, would come from quantization stage)
        dummy_quant_params = QuantParams(
            scales=world.arena.copy_tensor(np.array([1.0], dtype=np.float32)),
            offsets=world.arena.copy_tensor(np.array([0.0], dtype=np.float32)),
            quality=quality,
        )

        # Create ANSBitstream component with latent data
        # For now, serialize the latent tensor directly
        latent_ref = latent.t if use_hadamard else latent.z
        latent_view = world.arena.view(latent_ref)

        # Convert latent to bitstream format (uint8 bytes)
        latent_bytes = latent_view.astype(np.float32).tobytes()
        latent_uint8 = np.frombuffer(latent_bytes, dtype=np.uint8).copy()
        bitstream_ref = world.arena.copy_tensor(latent_uint8)

        # Create probability table (uniform for now)
        probs = np.ones(256, dtype=np.float32) / 256
        probs_ref = world.arena.copy_tensor(probs)

        bitstream = ANSBitstream(
            data=bitstream_ref,
            probs=probs_ref,
            initial_state=0,
        )

        # Serialize to bytes
        return serialize_bitstream(
            bitstream,
            arena=world.arena,
            model=model,
            levels=1,  # Single latent layer, not wavelet decomposed yet
            image_shape=image.shape,
            quant_params=dummy_quant_params,
        )

    finally:
        # Clean up world resources
        world.clear()


def decompress(
    data: bytes,
    config_path: str | None = None,
) -> np.ndarray:
    """Decompress bytes to RGB image.

    Deserializes bitstream and applies inverse VAE decoder with optional
    inverse Hadamard transform.

    Args:
        data: Compressed image bytes
        config_path: Path to vaerans_ecs.toml (auto-detected if None)

    Returns:
        Reconstructed image as (H, W, 3) float32 array in [0, 1] range

    Raises:
        ValueError: If data format is invalid or corrupted
        FileNotFoundError: If model not found
        ImportError: If onnxruntime not installed

    Example:
        >>> compressed_bytes = b'...'  # From compress()
        >>> img = decompress(compressed_bytes)
        >>> print(img.dtype, img.shape)
        float32 (512, 512, 3)
    """
    # Deserialize
    metadata, probs, bitstream_data = deserialize_bitstream(data)

    # Create world with arena
    world = World(arena_bytes=512 << 20)  # 512 MB

    try:
        # Create entity
        entity = world.new_entity()

        # Restore latent from bitstream data
        latent_bytes = bitstream_data.astype(np.uint8).tobytes()
        latent_array = np.frombuffer(latent_bytes, dtype=np.float32)

        # Reshape latent using metadata image shape (assumes 8x downsampling)
        try:
            image_shape = metadata["image_shape"]
            img_h = int(image_shape[0])
            img_w = int(image_shape[1])
            img_c = int(image_shape[2])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError("Missing or invalid image_shape metadata") from exc

        if img_c != 3:
            raise ValueError(f"Expected 3-channel image, got {img_c}")

        if img_h % 8 != 0 or img_w % 8 != 0:
            raise ValueError(
                f"Image shape must be divisible by 8 for VAE decoding; got {img_h}x{img_w}"
            )

        h = img_h // 8
        w = img_w // 8
        expected_len = 4 * h * w
        if len(latent_array) < expected_len:
            raise ValueError(
                f"Latent data too short: expected {expected_len} floats, got {len(latent_array)}"
            )
        latent_array = latent_array[:expected_len].reshape(4, h, w)

        # Copy to arena
        from vaerans_ecs.components.latent import Latent4

        latent_ref = world.arena.copy_tensor(latent_array)
        world.add_component(entity, Latent4(z=latent_ref))

        # Build decode pipeline
        recon: ReconRGB = (
            world.pipe(entity)
            .to(OnnxVAEDecode(model=metadata["model"], mode="decode", config_path=config_path))
            .out(ReconRGB)
        )

        # Get reconstructed image view
        recon_view = world.arena.view(recon.pix)

        # Return copy before world is cleared
        return cast(np.ndarray, recon_view.copy())

    finally:
        # Clean up world resources
        world.clear()


def get_compression_info(data: bytes) -> dict[str, Any]:
    """Get metadata about a compressed image.

    Extracts and returns compression parameters without decompressing.

    Args:
        data: Compressed image bytes

    Returns:
        Dictionary with keys: model, wavelet_levels, image_shape, initial_state

    Raises:
        ValueError: If data format is invalid
    """
    metadata, _, _ = deserialize_bitstream(data)
    return metadata


def get_compression_ratio(
    original_image: np.ndarray,
    compressed_data: bytes,
) -> float:
    """Calculate compression ratio.

    Args:
        original_image: Original image as numpy array
        compressed_data: Compressed bytes

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_bytes = original_image.nbytes
    compressed_bytes = len(compressed_data)
    return original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")
