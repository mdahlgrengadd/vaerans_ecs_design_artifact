#!/usr/bin/env python3
"""Batch pipeline example with real SDXL VAE models.

This example shows how to batch-encode and decode multiple images using
World + System APIs (OnnxVAEEncode/OnnxVAEDecode).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.image import RGB, ReconRGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4
from vaerans_ecs.systems.metrics import MetricPSNR, MetricSSIM
from vaerans_ecs.systems.vae import OnnxVAEDecode, OnnxVAEEncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch VAE pipeline example")
    parser.add_argument("--count", type=int, default=3, help="Number of images")
    parser.add_argument("--size", type=int, default=256, help="Square image size")
    parser.add_argument(
        "--model",
        default="sdxl-vae",
        help="Model name from config (default: sdxl-vae)",
    )
    parser.add_argument(
        "--no-hadamard",
        action="store_true",
        help="Disable Hadamard forward/inverse transforms",
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
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. Set --config or VAERANS_CONFIG."
        )

    world = World()
    images = [
        np.random.randint(0, 256, (args.size, args.size, 3), dtype=np.uint8)
        for _ in range(args.count)
    ]
    eids = world.spawn_batch_images(images)
    print(f"Spawned {len(eids)} images with shape {(args.size, args.size, 3)}")

    encoder = OnnxVAEEncode(model=args.model, config_path=str(config_path))
    decoder = OnnxVAEDecode(model=args.model, config_path=str(config_path))

    print("Encoding batch...")
    encoder.run(world, eids)

    if not args.no_hadamard:
        print("Applying Hadamard forward/inverse...")
        Hadamard4(mode="forward").run(world, eids)
        Hadamard4(mode="inverse").run(world, eids)

    print("Decoding batch...")
    decoder.run(world, eids)

    # Add original RGB components for metric computation
    for i, eid in enumerate(eids):
        rgb_ref = world.arena.copy_tensor(images[i].astype(np.float32) / 255.0)
        world.add_component(eid, RGB(pix=rgb_ref))

    # Compute quality metrics using Phase 12 systems
    # Metrics are stored in world.metadata[eid]
    print("Computing quality metrics...")
    psnr_system = MetricPSNR(src_component=RGB, recon_component=ReconRGB, data_range=1.0)
    ssim_system = MetricSSIM(src_component=RGB, recon_component=ReconRGB, data_range=1.0)
    
    psnr_system.run(world, eids)
    ssim_system.run(world, eids)

    # Report shapes and quality metrics
    print(f"\n{'Entity':<10} {'Latent Shape':<15} {'Recon Shape':<15} {'PSNR (dB)':<12} {'SSIM':<8}")
    print("-" * 70)
    for i, eid in enumerate(eids):
        latent = world.get_component(eid, Latent4)
        latent_view = world.arena.view(latent.z)
        recon = world.get_component(eid, ReconRGB)
        recon_view = world.arena.view(recon.pix)
        psnr_value = world.metadata[eid]["psnr"]
        ssim_value = world.metadata[eid]["ssim"]

        print(
            f"{eid:<10} {str(latent_view.shape):<15} {str(recon_view.shape):<15} "
            f"{psnr_value:<12.2f} {ssim_value:<8.4f}"
        )
    
    # Summary statistics
    psnr_values = [world.metadata[eid]["psnr"] for eid in eids]
    ssim_values = [world.metadata[eid]["ssim"] for eid in eids]
    print("-" * 70)
    print(f"{'Average:':<40} {np.mean(psnr_values):<12.2f} {np.mean(ssim_values):<8.4f}")
    print(f"{'Std Dev:':<40} {np.std(psnr_values):<12.2f} {np.std(ssim_values):<8.4f}")


if __name__ == "__main__":
    main()
