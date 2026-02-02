"""Generate Hadamard channel images for examples/23.png (notebook-matched)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World
from vaerans_ecs.eval.hadamard import (
    hadamard_4ch_forward,
    hadamard_decorrelation_stats,
    hadamard_energy_stats,
    quantize_latent_u8,
)
from vaerans_ecs.systems.vae import OnnxVAEEncode


def _load_image(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def _resolve_model_path(config_path: Path, model_name: str) -> Path | None:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore[import-not-found]

    if not config_path.exists():
        return None
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    models = config.get("models", {})
    model_cfg = models.get(model_name, {})
    encoder_rel = model_cfg.get("encoder")
    if not encoder_rel:
        return None
    encoder_path = Path(encoder_rel)
    if not encoder_path.is_absolute():
        encoder_path = config_path.parent / encoder_path
    return encoder_path if encoder_path.exists() else None


def _to_u8(channel: np.ndarray, shift: bool = False) -> np.ndarray:
    data = channel.astype(np.float32)
    if shift:
        data = data + 128.0
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def test_hadamard_channel_outputs() -> None:
    """Run VAE encode + Hadamard on examples/23.png and save channel images."""
    pytest.importorskip("onnxruntime")
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    ImageFont = pytest.importorskip("PIL.ImageFont")

    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "examples" / "23.png"
    config_path = repo_root / "vaerans_ecs.toml"

    image = _load_image(image_path)
    if image is None:
        pytest.skip("examples/23.png not available or Pillow missing.")

    encoder_path = _resolve_model_path(config_path, "sdxl-vae")
    if encoder_path is None:
        pytest.skip("SDXL VAE encoder model not configured or missing.")

    # Encode to latent
    world = World(arena_bytes=512 << 20)
    eid = world.spawn_image(image)
    encoder = OnnxVAEEncode(model="sdxl-vae", config_path=str(config_path))
    encoder.run(world, [eid])

    latent = world.get_component(eid, Latent4)
    latent_view = world.arena.view(latent.z)

    # Notebook-matched quantization before Hadamard
    latent_u8 = quantize_latent_u8(latent_view, bits=8, clip=4.0)
    y, u, v, w = hadamard_4ch_forward(latent_u8)

    # Prepare output images (match notebook display: shift U/V/W)
    y_img = _to_u8(y, shift=False)
    u_img = _to_u8(u, shift=True)
    v_img = _to_u8(v, shift=True)
    w_img = _to_u8(w, shift=True)

    output_dir = Path(
        os.environ.get("HADAMARD_OUTPUT_DIR", repo_root / "examples" / "hadamard_channels")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original latent channels (C0-C3) for reference
    for idx in range(4):
        Image.fromarray(latent_u8[idx], mode="L").save(
            output_dir / f"latent_C{idx}.png"
        )

    Image.fromarray(y_img, mode="L").save(output_dir / "hadamard_Y.png")
    Image.fromarray(u_img, mode="L").save(output_dir / "hadamard_U.png")
    Image.fromarray(v_img, mode="L").save(output_dir / "hadamard_V.png")
    Image.fromarray(w_img, mode="L").save(output_dir / "hadamard_W.png")

    # Build a notebook-style grid (C0-3 on top, Y/U/V/W on bottom)
    stats = hadamard_energy_stats(latent_view, domain="notebook_u8", bits=8, clip=4.0)
    decor_float = hadamard_decorrelation_stats(latent_view, domain="float")
    decor_u8 = hadamard_decorrelation_stats(latent_view, domain="notebook_u8", bits=8, clip=4.0)

    up = int(os.environ.get("HADAMARD_UPSCALE", "8"))
    pad = int(os.environ.get("HADAMARD_PAD", "8"))
    w_px, h_px = int(latent_u8.shape[2]), int(latent_u8.shape[1])
    tile_w, tile_h = w_px * up, h_px * up

    font = ImageFont.load_default()
    label_h = font.getbbox("Ag")[3] + 2
    tile_total_h = label_h + tile_h

    grid_w = pad + 4 * (tile_w + pad)
    grid_h = pad + 2 * (tile_total_h + pad)
    grid = Image.new("L", (grid_w, grid_h), color=255)
    draw = ImageDraw.Draw(grid)

    def _paste_tile(col: int, row: int, img_arr: np.ndarray, label: str) -> None:
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_total_h + pad)
        draw.text((x, y), label, font=font, fill=0)
        img = Image.fromarray(img_arr, mode="L").resize((tile_w, tile_h), Image.NEAREST)
        grid.paste(img, (x, y + label_h))

    # Top row: original channels
    for col in range(4):
        c_name = f"C{col}"
        c_stats = stats["original"][c_name]
        label = f"{c_name}\\nvar={c_stats['variance']:.1f}"
        _paste_tile(col, 0, latent_u8[col], label)

    # Bottom row: transformed channels
    trans = [("Y", y_img), ("U", u_img), ("V", v_img), ("W", w_img)]
    for col, (name, img_arr) in enumerate(trans):
        t_stats = stats["transformed"][name]
        label = f"{name}\\nvar={t_stats['variance']:.1f}, {t_stats['energy_pct']:.1f}%"
        _paste_tile(col, 1, img_arr, label)

    grid.save(output_dir / "hadamard_grid.png")

    # Write decorrelation metrics (true measure on float latents + notebook domain)
    def _fmt_mat(mat: np.ndarray) -> str:
        return np.array2string(mat, precision=4, suppress_small=True)

    stats_path = output_dir / "hadamard_decorrelation.txt"
    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write("Hadamard Decorrelation Stats (float latent, mean-centered)\n")
        handle.write("Before: " + str(decor_float["before"]) + "\n")
        handle.write("After:  " + str(decor_float["after"]) + "\n\n")
        handle.write("Correlation before:\n" + _fmt_mat(decor_float["corr_before"]) + "\n")
        handle.write("Correlation after:\n" + _fmt_mat(decor_float["corr_after"]) + "\n\n")

        handle.write("Notebook-domain Decorrelation Stats (u8 quantized)\n")
        handle.write("Before: " + str(decor_u8["before"]) + "\n")
        handle.write("After:  " + str(decor_u8["after"]) + "\n")
        handle.write("Correlation before:\n" + _fmt_mat(decor_u8["corr_before"]) + "\n")
        handle.write("Correlation after:\n" + _fmt_mat(decor_u8["corr_after"]) + "\n")

    assert (output_dir / "hadamard_Y.png").exists()
    assert (output_dir / "hadamard_U.png").exists()
    assert (output_dir / "hadamard_V.png").exists()
    assert (output_dir / "hadamard_W.png").exists()
    assert (output_dir / "hadamard_grid.png").exists()
    assert stats_path.exists()
