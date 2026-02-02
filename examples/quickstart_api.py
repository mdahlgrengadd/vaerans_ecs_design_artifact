#!/usr/bin/env python3
"""Quickstart example using the high-level compress/decompress API (Phase 13).

This example demonstrates the simplest way to use the SDK:
- Load an image (or generate a random one)
- Compress it with the SDXL VAE model using compress()
- Decompress it back to RGB using decompress()
- Compute quality metrics

The high-level API hides all the ECS complexity and provides a simple,
one-line interface for compression and decompression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.api import (
    compress,
    decompress,
    get_compression_info,
    get_compression_ratio,
)


def _load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    image = Image.open(path).convert("RGB")
    return np.array(image)


def _save_image(path: Path, image: np.ndarray) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    Image.fromarray(image).save(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Quickstart API example")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path (defaults to examples/23.png if present)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/reconstruction_api.png"),
        help="Output path for reconstructed image",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Random image size if no input image is available",
    )
    parser.add_argument(
        "--model",
        default="sdxl-vae",
        help="Model name from config (default: sdxl-vae)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=50,
        help="Quality setting (1-100)",
    )
    parser.add_argument(
        "--use-hadamard",
        action="store_true",
        help="Enable Hadamard transform (currently has reconstruction issues)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to vaerans_ecs.toml (defaults to repo root)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (repo_root / "vaerans_ecs.toml")
    config_arg = str(config_path) if config_path.exists() else None

    input_path = args.input or (repo_root / "examples" / "23.png")
    image = _load_image(input_path)

    if image is None:
        print("No readable input image found; generating random image instead")
        image = np.random.randint(
            0,
            256,
            (args.size, args.size, 3),
            dtype=np.uint8,
        )
    else:
        print(f"Loaded image: {input_path}")

    print("Compressing...")
    compressed = compress(
        image,
        model=args.model,
        quality=args.quality,
        use_hadamard=args.use_hadamard,  # Default: False (disabled until fix is complete)
        config_path=config_arg,
    )

    info = get_compression_info(compressed)
    ratio = get_compression_ratio(image, compressed)
    print(f"Compressed size: {len(compressed)} bytes")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Metadata: model={info['model']} shape={info['image_shape']}")

    print("Decompressing...")
    recon = decompress(compressed, config_path=config_arg)

    # Compute simple reconstruction metrics
    original_float = image.astype(np.float32) / 255.0
    mse = np.mean((original_float - recon) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB")

    # Save output if possible
    recon_uint8 = (recon * 255.0).clip(0, 255).astype(np.uint8)
    if _save_image(args.output, recon_uint8):
        print(f"Reconstruction saved to: {args.output}")
    else:
        print("Pillow not installed; skipping image save")


if __name__ == "__main__":
    main()
