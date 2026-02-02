#!/usr/bin/env python3
"""Batch pipeline example with real SDXL VAE models.

This example shows how to batch-encode and decode multiple images using
World + System APIs (OnnxVAEEncode/OnnxVAEDecode).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.image import ReconRGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4
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

    # Report shapes and basic metrics
    for i, eid in enumerate(eids):
        latent = world.get_component(eid, Latent4)
        latent_view = world.arena.view(latent.z)
        recon = world.get_component(eid, ReconRGB)
        recon_view = world.arena.view(recon.pix)

        original = images[i].astype(np.float32) / 255.0
        mse = float(np.mean((original - recon_view) ** 2))
        print(
            f"Entity {eid}: latent={latent_view.shape} recon={recon_view.shape} MSE={mse:.6f}"
        )


if __name__ == "__main__":
    main()
