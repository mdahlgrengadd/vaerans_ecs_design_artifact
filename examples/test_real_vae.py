#!/usr/bin/env python3
"""Test script for real SDXL VAE models.

This script demonstrates encoding and decoding with real SDXL VAE models.

Usage:
    python examples/test_real_vae.py [--output image.png]
"""

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.image import ReconRGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.vae import OnnxVAEDecode, OnnxVAEEncode


def main() -> None:
    """Test VAE encoding and decoding with real models."""
    parser = argparse.ArgumentParser(
        description="Test VAE with real SDXL models"
    )
    parser.add_argument(
        "--output",
        default="reconstruction.png",
        help="Output path for reconstructed image",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image size (will be square)",
    )
    parser.add_argument(
        "--model",
        default="sdxl-vae",
        help="Model name from config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to vaerans_ecs.toml (defaults to repo root)",
    )
    args = parser.parse_args()

    # Resolve config path
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (repo_root / "vaerans_ecs.toml")
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        print("Please ensure vaerans_ecs.toml exists with model paths")
        return

    # Create world
    world = World()

    # Create random test image
    print(f"Creating {args.size}x{args.size} random test image...")
    img = np.random.randint(0, 256, (args.size, args.size, 3), dtype=np.uint8)
    eid = world.spawn_image(img)

    # Encode
    print(f"Encoding with model '{args.model}'...")
    try:
        encoder = OnnxVAEEncode(model=args.model, config_path=str(config_path))
        encoder.run(world, [eid])
        print("✓ Encoding successful")
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        return

    # Check latent
    if world.has_component(eid, Latent4):
        latent = world.get_component(eid, Latent4)
        latent_view = world.arena.view(latent.z)
        print(f"  Latent shape: {latent_view.shape}")
        print(f"  Latent dtype: {latent_view.dtype}")
        print(f"  Latent range: [{latent_view.min():.2f}, {latent_view.max():.2f}]")
    else:
        print("✗ Latent4 component not created")
        return

    # Decode
    print(f"Decoding with model '{args.model}'...")
    try:
        decoder = OnnxVAEDecode(model=args.model, config_path=str(config_path))
        decoder.run(world, [eid])
        print("✓ Decoding successful")
    except Exception as e:
        print(f"✗ Decoding failed: {e}")
        return

    # Check reconstruction
    if world.has_component(eid, ReconRGB):
        recon = world.get_component(eid, ReconRGB)
        recon_view = world.arena.view(recon.pix)
        print(f"  Reconstruction shape: {recon_view.shape}")
        print(f"  Reconstruction dtype: {recon_view.dtype}")
        print(f"  Reconstruction range: [{recon_view.min():.2f}, {recon_view.max():.2f}]")

        # Save if PIL available
        try:
            from PIL import Image

            # Convert to uint8
            recon_uint8 = (recon_view * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(recon_uint8).save(args.output)
            print(f"  Saved to {args.output}")
        except ImportError:
            print("  (PIL not available for saving, skipping image export)")
    else:
        print("✗ ReconRGB component not created")
        return

    # Calculate metrics
    print("\nMetrics:")
    mse = np.mean((img.astype(np.float32) / 255 - recon_view) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()
