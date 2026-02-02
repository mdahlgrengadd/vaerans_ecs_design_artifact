#!/usr/bin/env python3
"""Quality metrics example demonstrating PSNR, SSIM, MSE, and MS-SSIM.

This example shows how to use the quality metrics systems (Phase 12) to evaluate
reconstruction quality at different compression levels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.image import RGB, ReconRGB
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4
from vaerans_ecs.systems.metrics import MetricMSE, MetricMSSSIM, MetricPSNR, MetricSSIM
from vaerans_ecs.systems.vae import OnnxVAEDecode, OnnxVAEEncode


def _load_image(path: Path) -> np.ndarray | None:
    """Load image from path using PIL if available."""
    if not path.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    image = Image.open(path).convert("RGB")
    return np.array(image)


def compress_decompress(
    world: World,
    image: np.ndarray,
    model: str,
    config_arg: str | None,
    use_hadamard: bool = True,
) -> int:
    """Compress and decompress an image, return entity ID."""
    # Spawn entity
    entity = world.spawn_image(image)

    # Encode
    encoder = OnnxVAEEncode(model=model, config_path=config_arg, mode="encode")
    encoder.run(world, [entity])

    # Optional Hadamard
    if use_hadamard:
        Hadamard4(mode="forward").run(world, [entity])
        Hadamard4(mode="inverse").run(world, [entity])

    # Decode
    decoder = OnnxVAEDecode(model=model, config_path=config_arg, mode="decode")
    decoder.run(world, [entity])

    return entity


def compute_all_metrics(world: World, entity: int, data_range: float = 1.0) -> dict[str, float]:
    """Compute all available quality metrics for an entity.

    Metrics are stored in world.metadata[entity] as:
    - 'psnr': Peak Signal-to-Noise Ratio (dB)
    - 'ssim': Structural Similarity Index
    - 'mse': Mean Squared Error
    - 'ms_ssim': Multi-Scale SSIM
    """
    # PSNR
    psnr_system = MetricPSNR(
        src_component=RGB, recon_component=ReconRGB, data_range=data_range)
    psnr_system.run(world, [entity])

    # SSIM
    ssim_system = MetricSSIM(
        src_component=RGB, recon_component=ReconRGB, data_range=data_range)
    ssim_system.run(world, [entity])

    # MSE
    mse_system = MetricMSE(src_component=RGB, recon_component=ReconRGB)
    mse_system.run(world, [entity])

    # MS-SSIM (Multi-Scale SSIM)
    msssim_system = MetricMSSSIM(
        src_component=RGB, recon_component=ReconRGB, data_range=data_range)
    msssim_system.run(world, [entity])

    # Retrieve from metadata
    return {
        "psnr": world.metadata[entity]["psnr"],
        "ssim": world.metadata[entity]["ssim"],
        "mse": world.metadata[entity]["mse"],
        "msssim": world.metadata[entity]["ms_ssim"],
    }


def main() -> None:
    """Run metrics demonstration."""
    parser = argparse.ArgumentParser(description="Quality metrics example")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path (defaults to examples/23.png if present)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Random image size if no input image is available",
    )
    parser.add_argument(
        "--model",
        default="sdxl-vae",
        help="Model name from config (default: sdxl-vae)",
    )
    parser.add_argument(
        "--no-hadamard",
        action="store_true",
        help="Disable Hadamard transform",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to vaerans_ecs.toml (defaults to repo root)",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (repo_root / "vaerans_ecs.toml")
    config_arg = str(config_path) if config_path.exists() else None

    input_path = args.input or (repo_root / "examples" / "23.png")
    image = _load_image(input_path)

    if image is None:
        print(f"Generating {args.size}x{args.size} random image")
        image = np.random.randint(
            0, 256, (args.size, args.size, 3), dtype=np.uint8)
    else:
        print(f"[OK] Loaded image: {input_path} {image.shape}")

    # Create world
    world = World(arena_bytes=512 << 20)
    print(f"[OK] Created World with 512 MB arena\n")

    print("=" * 70)
    print("QUALITY METRICS DEMONSTRATION")
    print("=" * 70)
    print()

    # Compress and decompress
    print("Running VAE encode → decode pipeline...")
    entity = compress_decompress(
        world, image, args.model, config_arg, use_hadamard=not args.no_hadamard
    )
    print("[OK] Compression/decompression complete\n")

    # Compute all metrics
    print("Computing quality metrics...\n")
    metrics = compute_all_metrics(world, entity, data_range=1.0)

    # Display results
    print("-" * 70)
    print("QUALITY METRICS")
    print("-" * 70)
    print(f"PSNR (Peak Signal-to-Noise Ratio):      {metrics['psnr']:8.2f} dB")
    print(f"  -> Higher is better (typical range: 20-50 dB)")
    print()
    print(f"SSIM (Structural Similarity Index):     {metrics['ssim']:8.4f}")
    print(f"  -> Range: [-1, 1], where 1 = perfect match")
    print()
    print(f"MSE (Mean Squared Error):                {metrics['mse']:8.6f}")
    print(f"  -> Lower is better (0 = perfect reconstruction)")
    print()
    print(f"MS-SSIM (Multi-Scale SSIM):              {metrics['msssim']:8.4f}")
    print(f"  -> Range: [0, 1], considers multiple scales")
    print("-" * 70)

    # Quality assessment
    print()
    print("QUALITY ASSESSMENT:")
    if metrics['psnr'] >= 40:
        quality = "Excellent"
    elif metrics['psnr'] >= 35:
        quality = "Very Good"
    elif metrics['psnr'] >= 30:
        quality = "Good"
    elif metrics['psnr'] >= 25:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"  Overall quality: {quality} (based on PSNR)")

    if metrics['ssim'] >= 0.95:
        print(f"  Structural similarity: Excellent (SSIM >= 0.95)")
    elif metrics['ssim'] >= 0.90:
        print(f"  Structural similarity: Very Good (SSIM >= 0.90)")
    elif metrics['ssim'] >= 0.80:
        print(f"  Structural similarity: Good (SSIM >= 0.80)")
    else:
        print(f"  Structural similarity: Fair (SSIM < 0.80)")

    # Interpretation guide
    print()
    print("=" * 70)
    print("METRIC INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("PSNR (Peak Signal-to-Noise Ratio):")
    print("  * 40+ dB:    Excellent quality, nearly imperceptible differences")
    print("  * 35-40 dB:  Very good quality, minor artifacts")
    print("  * 30-35 dB:  Good quality, some visible artifacts")
    print("  * 25-30 dB:  Fair quality, noticeable degradation")
    print("  * <25 dB:    Poor quality, significant artifacts")
    print()
    print("SSIM (Structural Similarity Index):")
    print("  * 0.95-1.0:  Excellent structural preservation")
    print("  * 0.90-0.95: Very good structural quality")
    print("  * 0.80-0.90: Good structural quality")
    print("  * <0.80:     Structural degradation visible")
    print()
    print("MSE (Mean Squared Error):")
    print("  * Lower values indicate better reconstruction")
    print("  * MSE = 0 means perfect reconstruction")
    print("  * Compare MSE across different compression settings")
    print()
    print("MS-SSIM (Multi-Scale SSIM):")
    print("  * Considers structural similarity at multiple scales")
    print("  * Often correlates better with human perception than SSIM")
    print("  * Range: [0, 1], higher is better")
    print()

    # Component information
    print("=" * 70)
    print("IMPLEMENTATION NOTES")
    print("=" * 70)
    print()
    print("The metrics systems (Phase 12) integrate with the ECS architecture:")
    print("  * Input:  RGB (source) + ReconRGB (reconstruction) components")
    print("  * Output: Metric* components (MetricPSNR, MetricSSIM, etc.)")
    print("  * Usage:  Can be inserted at any point in pipeline for evaluation")
    print()
    print("Systems used:")
    print(f"  * MetricPSNR:   {MetricPSNR.__module__}.{MetricPSNR.__name__}")
    print(f"  * MetricSSIM:   {MetricSSIM.__module__}.{MetricSSIM.__name__}")
    print(f"  * MetricMSE:    {MetricMSE.__module__}.{MetricMSE.__name__}")
    print(
        f"  * MetricMSSSIM: {MetricMSSSIM.__module__}.{MetricMSSSIM.__name__}")
    print()
    print("Results are stored in world.metadata[entity] dictionary:")
    print("  * world.metadata[entity]['psnr']")
    print("  * world.metadata[entity]['ssim']")
    print("  * world.metadata[entity]['mse']")
    print("  * world.metadata[entity]['ms_ssim']")
    print()

    print("[OK] Metrics demonstration complete!")


if __name__ == "__main__":
    main()
