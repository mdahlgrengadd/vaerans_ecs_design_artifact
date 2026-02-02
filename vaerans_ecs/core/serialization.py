"""Bitstream serialization and deserialization (Phase 9).

Implements file format for compressed bitstreams with metadata.

File format:
  [Header: 16 bytes]
    - Magic: 4 bytes ('VAE\x00')
    - Version: 2 bytes (major, minor)
    - Metadata length: 4 bytes
    - Reserved: 6 bytes
  [Metadata: variable JSON]
    - model name, wavelet levels, image shape, quantization params
  [Probability table: variable]
    - Float32 array (256 entries)
  [Bitstream data: variable]
    - UInt8 array of compressed data
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any, Tuple, cast

import numpy as np

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.quant import QuantParams

if TYPE_CHECKING:
    from vaerans_ecs.core.arena import Arena

# File format constants
MAGIC = b"VAE\x00"
VERSION_MAJOR = 1
VERSION_MINOR = 0
HEADER_SIZE = 14  # 4s (4) + H (2) + I (4) + I (4) = 14 bytes


def serialize_bitstream(
    bitstream: ANSBitstream,
    arena: Arena,
    model: str,
    levels: int,
    image_shape: tuple[int, int, int],
    quant_params: QuantParams,
    use_hadamard: bool = False,
) -> bytes:
    """Serialize ANSBitstream component to bytes.

    Args:
        bitstream: ANSBitstream component to serialize
        arena: Arena containing the bitstream data
        model: Model name (e.g., 'sdxl-vae')
        levels: Number of wavelet decomposition levels
        image_shape: Original image shape (H, W, 3)
        quant_params: Quantization parameters
        use_hadamard: Whether Hadamard transform was applied (default: False)

    Returns:
        Serialized bitstream as bytes

    Raises:
        ValueError: If bitstream is invalid
        TypeError: If components have wrong types
    """
    if not isinstance(bitstream, ANSBitstream):
        raise TypeError(f"Expected ANSBitstream, got {type(bitstream)}")

    if bitstream.probs is None:
        raise ValueError("Bitstream has no probability table")

    if bitstream.data is None:
        raise ValueError("Bitstream has no data")

    # Get data from arena
    probs_data = arena.view(bitstream.probs)
    bitstream_data = arena.view(bitstream.data)

    # Build metadata dictionary
    metadata: dict[str, Any] = {
        "model": model,
        "wavelet_levels": levels,
        "image_shape": list(image_shape),
        "initial_state": bitstream.initial_state,
        "use_hadamard": use_hadamard,
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")

    if len(metadata_bytes) > 2**32 - 1:
        raise ValueError("Metadata too large (>4GB)")

    # Build header
    # Pack version as single byte (major in upper nibble, minor in lower)
    version_byte = (VERSION_MAJOR << 8) | VERSION_MINOR
    header = struct.pack(
        "<4sHII",
        MAGIC,  # Magic (4 bytes)
        version_byte,  # Version (2 bytes: major/minor)
        len(metadata_bytes),  # Metadata length (4 bytes)
        0,  # Reserved (4 bytes)
    )

    # Serialize probability table (ensure float32)
    probs_array = np.asarray(probs_data, dtype=np.float32)
    probs_bytes = probs_array.tobytes()

    # Serialize bitstream data (uint8)
    data_array = np.asarray(bitstream_data, dtype=np.uint8)
    data_bytes = data_array.tobytes()

    return cast(bytes, header + metadata_bytes + probs_bytes + data_bytes)


def deserialize_bitstream(
    data: bytes,
) -> Tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Deserialize bytes to bitstream components and metadata.

    Args:
        data: Serialized bitstream bytes

    Returns:
        Tuple of (metadata_dict, probs_array, bitstream_data_array)

    Raises:
        ValueError: If data format is invalid or corrupted
        struct.error: If header cannot be parsed
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Data too short: need {HEADER_SIZE} bytes, got {len(data)}")

    # Parse header
    try:
        magic, version_byte, meta_len, _ = struct.unpack(
            "<4sHII", data[:HEADER_SIZE]
        )
    except struct.error as e:
        raise ValueError(f"Failed to parse header: {e}") from e

    # Validate magic number
    if magic != MAGIC:
        raise ValueError(
            f"Invalid file format: expected {MAGIC!r}, got {magic!r}"
        )

    # Extract version
    ver_major = (version_byte >> 8) & 0xFF
    ver_minor = version_byte & 0xFF

    # Check version
    if ver_major != VERSION_MAJOR:
        raise ValueError(
            f"Unsupported version {ver_major}.{ver_minor}. "
            f"Expected {VERSION_MAJOR}.{VERSION_MINOR}"
        )

    # Parse metadata
    meta_start = HEADER_SIZE
    meta_end = meta_start + meta_len

    if meta_end > len(data):
        raise ValueError(
            f"Metadata region extends beyond data: "
            f"need {meta_end} bytes, got {len(data)}"
        )

    try:
        metadata = cast(
            dict[str, Any], json.loads(data[meta_start:meta_end].decode("utf-8"))
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse metadata JSON: {e}") from e

    # Parse probability table (256 float32 entries = 1024 bytes)
    probs_start = meta_end
    probs_size = 256 * 4  # 256 float32 values
    probs_end = probs_start + probs_size

    if probs_end > len(data):
        raise ValueError(
            f"Probability table region extends beyond data: "
            f"need {probs_end} bytes, got {len(data)}"
        )

    probs = np.frombuffer(data[probs_start:probs_end], dtype=np.float32).copy()

    # Parse bitstream data (uint8 array, variable length)
    bitstream_start = probs_end

    if bitstream_start >= len(data):
        raise ValueError("No bitstream data in file")

    bitstream_data = np.frombuffer(data[bitstream_start:], dtype=np.uint8).copy()

    if len(bitstream_data) == 0:
        raise ValueError("Bitstream data is empty")

    return metadata, probs, bitstream_data


def get_serialized_size(
    bitstream_data: np.ndarray,
    probs_data: np.ndarray,
    metadata_dict: dict[str, Any] | None = None,
) -> int:
    """Get the size of a serialized bitstream in bytes.

    Args:
        bitstream_data: Bitstream data array
        probs_data: Probability table array
        metadata_dict: Optional pre-serialized metadata dict

    Returns:
        Expected serialized size in bytes
    """
    # Header
    size = HEADER_SIZE

    # Metadata
    if metadata_dict is not None:
        metadata_bytes = json.dumps(metadata_dict).encode("utf-8")
        size += len(metadata_bytes)
    else:
        # Typical metadata: ~150 bytes
        size += 150

    # Probability table
    size += len(probs_data) * probs_data.itemsize

    # Bitstream data
    size += len(bitstream_data) * bitstream_data.itemsize

    return size
