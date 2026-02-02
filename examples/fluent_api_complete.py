#!/usr/bin/env python3
"""Complete fluent API demonstration with all implemented features.

This example shows the full power of the fluent pipeline API (Phase 11),
demonstrating:
- Method chaining with .to()
- Pipe operator |
- Component selection with .select() and .use()
- Output extraction with .out()
- Integration with all systems (VAE, Hadamard, Wavelet, Quantize, ANS, Metrics)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.image import RGB, ReconRGB
from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.components.quant import QuantParams, SymbolsU8
from vaerans_ecs.components.wavelet import WaveletPyr
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.ans import ANSDecode, ANSEncode
from vaerans_ecs.systems.hadamard import Hadamard4
from vaerans_ecs.systems.metrics import MetricPSNR, MetricSSIM
from vaerans_ecs.systems.quantize import QuantizeU8
from vaerans_ecs.systems.vae import OnnxVAEDecode, OnnxVAEEncode
from vaerans_ecs.systems.wavelet import WaveletCDF53


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


def example_1_simple_chaining(world: World, entity: int, config: str | None) -> int:
    """Example 1: Simple method chaining with .to()
    
    Returns:
        Safe number of wavelet decomposition levels for this image
    """
    print("=" * 70)
    print("EXAMPLE 1: Simple Method Chaining")
    print("=" * 70)
    print()
    print("Fluent API allows chaining systems with .to():")
    print()
    print("  latent = (")
    print("      world.pipe(entity)")
    print("           .to(OnnxVAEEncode(model='sdxl-vae'))")
    print("           .out(Latent4)")
    print("  )")
    print()

    latent = (
        world.pipe(entity)
        .to(OnnxVAEEncode(model="sdxl-vae", config_path=config, mode="encode"))
        .out(Latent4)
    )

    latent_view = world.arena.view(latent.z)
    print(f"[OK] Encoded to latent shape: {latent_view.shape}")
    
    # Calculate safe wavelet levels based on smallest latent dimension
    # Rule: smallest_dim / 2^levels >= 8 (leave at least 8 pixels to avoid boundary effects)
    # PyWavelets recommends conservative margins to avoid boundary artifacts
    _, h, w = latent_view.shape
    min_dim = min(h, w)
    safe_levels = max(1, int(np.log2(min_dim // 8)))
    safe_levels = min(safe_levels, 4)  # Cap at 4 for demonstration
    pixels_after = min_dim // (2**safe_levels)
    print(f"[INFO] Calculated safe wavelet levels: {safe_levels} (min latent dim {min_dim} -> {pixels_after} pixels after {safe_levels} levels)")
    print()
    
    return safe_levels


def example_2_pipe_operator(world: World, entity: int) -> None:
    """Example 2: Pipe operator |"""
    print("=" * 70)
    print("EXAMPLE 2: Pipe Operator |")
    print("=" * 70)
    print()
    print("Alternative syntax using | operator:")
    print()
    print("  yuvw = (")
    print("      world.pipe(entity)")
    print("           | Hadamard4(mode='forward')")
    print("  ).out(YUVW4)")
    print()

    yuvw = (
        world.pipe(entity)
        | Hadamard4(mode="forward")
    ).out(YUVW4)

    yuvw_view = world.arena.view(yuvw.t)
    print(f"[OK] Transformed to YUVW4 shape: {yuvw_view.shape}")
    print()


def example_3_full_compression(world: World, entity: int, quality: int, wavelet_levels: int) -> None:
    """Example 3: Full compression pipeline"""
    print("=" * 70)
    print("EXAMPLE 3: Full Compression Pipeline")
    print("=" * 70)
    print()
    print("Chain multiple systems for complete compression:")
    print()
    print("  bitstream = (")
    print("      world.pipe(entity)")
    print("           .to(Hadamard4(mode='forward'))")
    print(f"           .to(WaveletCDF53(levels={wavelet_levels}, mode='forward'))")
    print(f"           .to(QuantizeU8(quality={quality}, mode='forward'))")
    print("           .to(ANSEncode())")
    print("           .out(ANSBitstream)")
    print("  )")
    print()

    bitstream = (
        world.pipe(entity)
        .to(Hadamard4(mode="forward"))
        .to(WaveletCDF53(levels=wavelet_levels, mode="forward"))
        .to(QuantizeU8(quality=quality, per_band=True, mode="forward"))
        .to(ANSEncode())
        .out(ANSBitstream)
    )

    bitstream_data = world.arena.view(bitstream.data)
    print(f"[OK] Compressed to bitstream: {len(bitstream_data)} uint32 words")

    # Show intermediate components
    print(
        f"  * YUVW4:       {'[OK]' if world.has_component(entity, YUVW4) else '[X]'}")
    print(
        f"  * WaveletPyr:  {'[OK]' if world.has_component(entity, WaveletPyr) else '[X]'}")
    print(
        f"  * SymbolsU8:   {'[OK]' if world.has_component(entity, SymbolsU8) else '[X]'}")
    print(
        f"  * QuantParams: {'[OK]' if world.has_component(entity, QuantParams) else '[X]'}")
    print(
        f"  * ANSBitstream:{'[OK]' if world.has_component(entity, ANSBitstream) else '[X]'}")
    print()


def example_4_component_selection(world: World, entity: int, config: str | None, quality: int, wavelet_levels: int) -> None:
    """Example 4: Component selection and branching"""
    print("=" * 70)
    print("EXAMPLE 4: Component Selection with .select() and .use()")
    print("=" * 70)
    print()
    print("Select specific components for decompression:")
    print()
    print("  recon = (")
    print("      world.pipe(entity)")
    print("           .select(ANSBitstream)  # Start from compressed data")
    print("           .to(ANSDecode())")
    print(f"           .to(QuantizeU8(quality={quality}, mode='inverse'))")
    print(f"           .to(WaveletCDF53(levels={wavelet_levels}, mode='inverse'))")
    print("           .to(Hadamard4(mode='inverse'))")
    print("           .to(OnnxVAEDecode(model='sdxl-vae'))")
    print("           .out(ReconRGB)")
    print("  )")
    print()

    recon = (
        world.pipe(entity)
        .select(ANSBitstream)
        .to(ANSDecode())
        .to(QuantizeU8(quality=quality, per_band=True, mode="inverse"))
        .to(WaveletCDF53(levels=wavelet_levels, mode="inverse"))
        .to(Hadamard4(mode="inverse"))
        .to(OnnxVAEDecode(model="sdxl-vae", config_path=config, mode="decode"))
        .out(ReconRGB)
    )

    recon_view = world.arena.view(recon.pix)
    print(f"[OK] Decompressed to RGB shape: {recon_view.shape}")
    print()
    print("Note: .use() is an alias for .select() for cleaner syntax")
    print()


def example_5_metrics_integration(world: World, entity: int) -> None:
    """Example 5: Integrating quality metrics"""
    print("=" * 70)
    print("EXAMPLE 5: Quality Metrics Integration")
    print("=" * 70)
    print()
    print("Compute metrics as part of the pipeline:")
    print()
    print("  # Metrics store results in world.metadata")
    print("  world.pipe(entity)")
    print("       .to(MetricPSNR(src_component=RGB, recon_component=ReconRGB))")
    print("       .to(MetricSSIM(src_component=RGB, recon_component=ReconRGB))")
    print("       .execute()  # Or continue chaining")
    print()
    print("  psnr = world.metadata[entity]['psnr']")
    print("  ssim = world.metadata[entity]['ssim']")
    print()

    # Run PSNR metric
    MetricPSNR(src_component=RGB, recon_component=ReconRGB,
               data_range=1.0).run(world, [entity])

    # Run SSIM metric
    MetricSSIM(src_component=RGB, recon_component=ReconRGB,
               data_range=1.0).run(world, [entity])

    # Retrieve from metadata
    psnr_value = world.metadata[entity]["psnr"]
    ssim_value = world.metadata[entity]["ssim"]

    print(f"[OK] PSNR: {psnr_value:.2f} dB (stored in world.metadata)")
    print(f"[OK] SSIM: {ssim_value:.4f} (stored in world.metadata)")
    print()


def example_6_execute_without_output(world: World, entity: int) -> None:
    """Example 6: Execute pipeline without extracting output"""
    print("=" * 70)
    print("EXAMPLE 6: Execute Without Output Extraction")
    print("=" * 70)
    print()
    print("Sometimes you just want to run systems without getting output:")
    print()
    print("  world.pipe(entity)")
    print("       .to(PSNRSystem())")
    print("       .to(SSIMSystem())")
    print("       .execute()  # Run without .out()")
    print()
    print("(This example would require the execute() method implementation)")
    print()


def main() -> None:
    """Run complete fluent API demonstration."""
    parser = argparse.ArgumentParser(description="Complete fluent API example")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path (defaults to examples/23.png if present)",
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
        default=80,
        help="Quantization quality (1-100)",
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
        print(f"Loaded image: {input_path} {image.shape}")

    # Create world and entity
    world = World(arena_bytes=512 << 20)
    entity = world.spawn_image(image)

    print()
    print("=" * 70)
    print("FLUENT PIPELINE API - COMPLETE DEMONSTRATION")
    print("=" * 70)
    print()
    print(f"Entity {entity} created with image shape {image.shape}")
    print()

    # Run examples
    wavelet_levels = example_1_simple_chaining(world, entity, config_arg)
    example_2_pipe_operator(world, entity)
    example_3_full_compression(world, entity, args.quality, wavelet_levels)
    example_4_component_selection(world, entity, config_arg, args.quality, wavelet_levels)
    example_5_metrics_integration(world, entity)
    example_6_execute_without_output(world, entity)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The fluent pipeline API provides:")
    print("  * Method chaining with .to()")
    print("  * Alternative syntax with | operator")
    print("  * Component selection with .select() / .use()")
    print("  * Type-safe output extraction with .out()")
    print("  * Full IDE support (autocomplete, type checking)")
    print("  * Composable, readable pipeline construction")
    print()
    print("All systems integrate seamlessly:")
    print("  * VAE encoding/decoding (OnnxVAEEncode/Decode)")
    print("  * Hadamard transform (Hadamard4)")
    print("  * Wavelet decomposition (WaveletCDF53/Haar)")
    print("  * Quantization (QuantizeU8)")
    print("  * ANS entropy coding (ANSEncode/Decode)")
    print("  * Quality metrics (PSNR, SSIM, MSE, MS-SSIM)")
    print()
    print("[OK] Fluent API demonstration complete!")


if __name__ == "__main__":
    main()
