#!/usr/bin/env python3
"""Test script for real SDXL VAE models.

This script demonstrates encoding and decoding with real SDXL VAE models.

Usage:
    python examples/test_real_vae.py [--output image.png]
"""

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.image import RGB, ReconRGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.metrics import MetricPSNR, MetricSSIM
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
        print("[OK] Encoding successful")
    except Exception as e:
        print(f"[X] Encoding failed: {e}")
        return

    # Check latent
    if world.has_component(eid, Latent4):
        latent = world.get_component(eid, Latent4)
        latent_view = world.arena.view(latent.z)
        print(f"  Latent shape: {latent_view.shape}")
        print(f"  Latent dtype: {latent_view.dtype}")
        print(
            f"  Latent range: [{latent_view.min():.2f}, {latent_view.max():.2f}]")
    else:
        print("[X] Latent4 component not created")
        return

    # Decode
    print(f"Decoding with model '{args.model}'...")
    try:
        decoder = OnnxVAEDecode(model=args.model, config_path=str(config_path))
        decoder.run(world, [eid])
        print("[OK] Decoding successful")
    except Exception as e:
        print(f"[X] Decoding failed: {e}")
        return

    # Check reconstruction
    if world.has_component(eid, ReconRGB):
        recon = world.get_component(eid, ReconRGB)
        recon_view = world.arena.view(recon.pix)
        print(f"  Reconstruction shape: {recon_view.shape}")
        print(f"  Reconstruction dtype: {recon_view.dtype}")
        print(
            f"  Reconstruction range: [{recon_view.min():.2f}, {recon_view.max():.2f}]")

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
        print("[X] ReconRGB component not created")
        return

    # Calculate metrics using the metrics systems (Phase 12)
    print("\nComputing quality metrics...")

    # Create RGB component from original image for metric computation
    rgb_ref = world.arena.copy_tensor(img.astype(np.float32) / 255.0)
    world.add_component(eid, RGB(pix=rgb_ref))

    # Compute PSNR - result stored in world.metadata[eid]["psnr"]
    psnr_system = MetricPSNR(
        src_component=RGB, recon_component=ReconRGB, data_range=1.0)
    psnr_system.run(world, [eid])

    # Compute SSIM - result stored in world.metadata[eid]["ssim"]
    ssim_system = MetricSSIM(
        src_component=RGB, recon_component=ReconRGB, data_range=1.0)
    ssim_system.run(world, [eid])

    # Also compute MSE manually for comparison
    mse = np.mean((img.astype(np.float32) / 255 - recon_view) ** 2)

    print("\nMetrics:")
    print(
        f"  PSNR: {world.metadata[eid]['psnr']:.2f} dB (via MetricPSNR system)")
    print(f"  SSIM: {world.metadata[eid]['ssim']:.4f} (via MetricSSIM system)")
    print(f"  MSE:  {mse:.6f} (computed directly)")

    print("\n[OK] Test completed successfully!")


if __name__ == "__main__":
    main()
