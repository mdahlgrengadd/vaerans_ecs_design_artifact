#!/usr/bin/env python3
"""Full compression pipeline example with all transform stages.

This example demonstrates the complete compression pipeline:
  RGB → VAE Encode → Hadamard → Wavelet → Quantize → ANS → Bitstream

And the reverse decompression pipeline:
  Bitstream → ANS Decode → Dequantize → Wavelet Inverse → Hadamard Inverse → VAE Decode → RGB

Features demonstrated:
- All compression stages (Phases 1-8)
- Serialization and deserialization (Phase 9)
- Quality metrics (Phase 12)
- Fluent pipeline API (Phase 11)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.image import RGB, ReconRGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.serialization import deserialize_bitstream, serialize_bitstream
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


def _save_image(path: Path, image: np.ndarray) -> bool:
    """Save image to path using PIL if available."""
    try:
        from PIL import Image
    except ImportError:
        return False
    Image.fromarray(image).save(path)
    return True


def main() -> None:
    """Run full compression pipeline demonstration."""
    parser = argparse.ArgumentParser(
        description="Full compression pipeline example")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path (defaults to examples/23.png if present)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/reconstruction_full.png"),
        help="Output path for reconstructed image",
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
        "--quality",
        type=int,
        default=80,
        help="Quantization quality (1-100, higher = better quality)",
    )
    parser.add_argument(
        "--wavelet-levels",
        type=int,
        default=4,
        help="Number of wavelet decomposition levels",
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
        print(
            f"No readable input image found; generating {args.size}x{args.size} random image")
        image = np.random.randint(
            0, 256, (args.size, args.size, 3), dtype=np.uint8)
    else:
        print(f"[OK] Loaded image: {input_path} {image.shape}")

    # Create world with sufficient arena
    world = World(arena_bytes=512 << 20)  # 512 MB
    print(f"[OK] Created World with 512 MB arena\n")

    # Spawn image entity
    entity = world.spawn_image(image)
    print(f"=== COMPRESSION PIPELINE ===")
    print(f"Entity {entity} created with shape {image.shape}\n")

    # ==== COMPRESSION PIPELINE ====

    print("Step 1: VAE Encode (RGB → Latent4)")
    encoder = OnnxVAEEncode(
        model=args.model, config_path=config_arg, mode="encode")
    encoder.run(world, [entity])
    latent = world.get_component(entity, Latent4)
    latent_view = world.arena.view(latent.z)
    print(
        f"  [OK] Latent shape: {latent_view.shape}, range: [{latent_view.min():.2f}, {latent_view.max():.2f}]\n")

    if not args.no_hadamard:
        print("Step 2: Hadamard Transform (Latent4 -> YUVW4)")
        hadamard_fwd = Hadamard4(mode="forward")
        hadamard_fwd.run(world, [entity])
        print(f"  [OK] Applied Hadamard forward transform\n")
    else:
        print("Step 2: Hadamard Transform [SKIPPED]\n")

    print(
        f"Step 3: Wavelet Decomposition (CDF 5/3, levels={args.wavelet_levels})")
    wavelet_fwd = WaveletCDF53(levels=args.wavelet_levels, mode="forward")
    wavelet_fwd.run(world, [entity])
    print(f"  [OK] Wavelet decomposition complete\n")

    print(f"Step 4: Quantization (quality={args.quality})")
    quantize = QuantizeU8(quality=args.quality, per_band=True, mode="forward")
    quantize.run(world, [entity])
    print(f"  [OK] Quantized to uint8 symbols\n")

    print("Step 5: ANS Entropy Encoding")
    ans_encode = ANSEncode()
    ans_encode.run(world, [entity])
    bitstream = world.get_component(entity, ANSBitstream)
    bitstream_data = world.arena.view(bitstream.data)
    print(f"  [OK] Encoded to bitstream: {len(bitstream_data)} uint32 words\n")

    print("Step 6: Serialize to bytes")
    serialized_bytes = serialize_bitstream(world, entity)
    print(f"  [OK] Serialized size: {len(serialized_bytes)} bytes")

    # Calculate compression metrics
    original_bytes = image.size * image.itemsize
    compression_ratio = original_bytes / len(serialized_bytes)
    bits_per_pixel = (len(serialized_bytes) * 8) / \
        (image.shape[0] * image.shape[1])
    print(f"  [OK] Original size: {original_bytes:,} bytes")
    print(f"  [OK] Compression ratio: {compression_ratio:.2f}x")
    print(f"  [OK] Bits per pixel: {bits_per_pixel:.2f} bpp\n")

    # ==== DECOMPRESSION PIPELINE ====

    print("=== DECOMPRESSION PIPELINE ===\n")

    # Create new entity for decompression
    decode_entity = world.new_entity()

    print("Step 1: Deserialize from bytes")
    deserialize_bitstream(world, decode_entity, serialized_bytes)
    print(f"  [OK] Deserialized bitstream to entity {decode_entity}\n")

    print("Step 2: ANS Entropy Decoding")
    ans_decode = ANSDecode()
    ans_decode.run(world, [decode_entity])
    print(f"  [OK] Decoded to quantized symbols\n")

    print(f"Step 3: Dequantization")
    dequantize = QuantizeU8(quality=args.quality,
                            per_band=True, mode="inverse")
    dequantize.run(world, [decode_entity])
    print(f"  [OK] Dequantized symbols to float32\n")

    print(f"Step 4: Wavelet Reconstruction (inverse)")
    wavelet_inv = WaveletCDF53(levels=args.wavelet_levels, mode="inverse")
    wavelet_inv.run(world, [decode_entity])
    print(f"  [OK] Wavelet reconstruction complete\n")

    if not args.no_hadamard:
        print("Step 5: Hadamard Inverse Transform (YUVW4 -> Latent4)")
        hadamard_inv = Hadamard4(mode="inverse")
        hadamard_inv.run(world, [decode_entity])
        print(f"  [OK] Applied Hadamard inverse transform\n")
    else:
        print("Step 5: Hadamard Inverse [SKIPPED]\n")

    print("Step 6: VAE Decode (Latent4 -> ReconRGB)")
    decoder = OnnxVAEDecode(
        model=args.model, config_path=config_arg, mode="decode")
    decoder.run(world, [decode_entity])
    recon = world.get_component(decode_entity, ReconRGB)
    recon_view = world.arena.view(recon.pix)
    print(
        f"  [OK] Decoded to RGB shape: {recon_view.shape}, range: [{recon_view.min():.2f}, {recon_view.max():.2f}]\n")

    # ==== QUALITY METRICS ====

    print("=== QUALITY METRICS ===\n")

    # Add original RGB to decode entity for metric computation
    rgb_ref = world.arena.copy_tensor(image.astype(np.float32) / 255.0)
    world.add_component(decode_entity, RGB(pix=rgb_ref))

    # Compute PSNR
    psnr_system = MetricPSNR(
        src_component=RGB, recon_component=ReconRGB, data_range=1.0)
    psnr_system.run(world, [decode_entity])
    psnr_value = world.metadata[decode_entity]["psnr"]
    print(f"PSNR: {psnr_value:.2f} dB")

    # Compute SSIM
    ssim_system = MetricSSIM(
        src_component=RGB, recon_component=ReconRGB, data_range=1.0)
    ssim_system.run(world, [decode_entity])
    ssim_value = world.metadata[decode_entity]["ssim"]
    print(f"SSIM: {ssim_value:.4f}")

    # Compute MSE
    original_float = image.astype(np.float32) / 255.0
    mse = np.mean((original_float - recon_view) ** 2)
    print(f"MSE:  {mse:.6f}\n")

    # ==== SAVE OUTPUT ====

    recon_uint8 = (recon_view * 255.0).clip(0, 255).astype(np.uint8)
    if _save_image(args.output, recon_uint8):
        print(f"[OK] Reconstruction saved to: {args.output}")
    else:
        print("[!] Pillow not installed; skipping image save")

    # ==== SUMMARY ====

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Input image:        {image.shape}")
    print(f"Latent shape:       {latent_view.shape}")
    print(f"Original size:      {original_bytes:,} bytes")
    print(f"Compressed size:    {len(serialized_bytes):,} bytes")
    print(f"Compression ratio:  {compression_ratio:.2f}x")
    print(f"Bits per pixel:     {bits_per_pixel:.2f} bpp")
    print(f"Quality setting:    {args.quality}/100")
    print(f"Wavelet levels:     {args.wavelet_levels}")
    print(f"Hadamard:           {'Yes' if not args.no_hadamard else 'No'}")
    print(f"PSNR:               {psnr_value:.2f} dB")
    print(f"SSIM:               {ssim_value:.4f}")
    print("=" * 60)
    print("\n[OK] Full pipeline demonstration complete!")


if __name__ == "__main__":
    main()
