#!/usr/bin/env python3
"""Learn a 4x4 KLT matrix from Latent4 across a batch of images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World
from vaerans_ecs.eval import (
    compose_matrices,
    hadamard4_matrix,
    klt_from_covariance,
    permutation_matrix,
    reorder_indices_by_variance,
    variance_normalization_matrix,
)
from vaerans_ecs.systems.vae import OnnxVAEEncode


def _load_image(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    return np.array(image)


def _gather_images(path: Path, pattern: str, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    if recursive:
        return sorted(path.rglob(pattern))
    return sorted(path.glob(pattern))


def _covariance_accumulate(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = data.reshape(4, -1).astype(np.float64)
    sum_vec = flat.sum(axis=1)
    sum_outer = flat @ flat.T
    count = flat.shape[1]
    return sum_vec, sum_outer, count


def _covariance_finalize(sum_vec: np.ndarray, sum_outer: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    mean = sum_vec / count
    cov = (sum_outer - count * (mean[:, None] * mean[None, :])) / (count - 1)
    return mean, cov


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    denom = std[:, None] * std[None, :]
    denom[denom == 0] = 1.0
    return cov / denom


def _decorrelation_metrics(cov: np.ndarray, corr: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(4, dtype=bool)
    offdiag_cov_energy = float(np.sum(cov[mask] ** 2))
    total_cov_energy = float(np.sum(cov ** 2))
    offdiag_energy_ratio = 0.0 if total_cov_energy == 0.0 else offdiag_cov_energy / total_cov_energy
    offdiag_abs_corr_mean = float(np.mean(np.abs(corr[mask]))) if mask.any() else 0.0
    max_abs_corr = float(np.max(np.abs(corr[mask]))) if mask.any() else 0.0
    return {
        "offdiag_energy_ratio": offdiag_energy_ratio,
        "offdiag_abs_corr_mean": offdiag_abs_corr_mean,
        "max_abs_corr": max_abs_corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn KLT from Latent4.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/23.png"),
        help="Image file or directory (default: examples/23.png)",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern when --input is a directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=32,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image list before sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--model",
        default="sdxl-vae",
        help="Model name from config (default: sdxl-vae)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to vaerans_ecs.toml (defaults to repo root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/klt_latent4.npz"),
        help="Output .npz path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (repo_root / "vaerans_ecs.toml")
    config_arg = str(config_path) if config_path.exists() else None

    images = _gather_images(args.input, args.pattern, args.recursive)
    if not images:
        raise SystemExit("No images found.")

    if args.shuffle:
        random.Random(args.seed).shuffle(images)
    images = images[: args.max_images]

    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit("onnxruntime is required to encode latents.") from exc

    encoder = OnnxVAEEncode(model=args.model, config_path=config_arg, mode="encode")

    sum_vec = np.zeros(4, dtype=np.float64)
    sum_outer = np.zeros((4, 4), dtype=np.float64)
    count = 0
    processed = 0

    for path in images:
        img = _load_image(path)
        if img is None:
            continue

        world = World(arena_bytes=512 << 20)
        eid = world.spawn_image(img)
        encoder.run(world, [eid])
        latent = world.get_component(eid, Latent4)
        data = world.arena.view(latent.z)

        s_vec, s_outer, n = _covariance_accumulate(data)
        sum_vec += s_vec
        sum_outer += s_outer
        count += n
        processed += 1

    if processed == 0:
        raise SystemExit("No images could be loaded.")

    mean, cov = _covariance_finalize(sum_vec, sum_outer, count)
    corr = _corr_from_cov(cov)

    klt_fwd, klt_inv, eigvals = klt_from_covariance(cov)

    # Optional: reorder + variance normalization + Hadamard
    order = reorder_indices_by_variance(cov, descending=True)
    P = permutation_matrix(order)
    D = variance_normalization_matrix(cov)
    H = hadamard4_matrix(normalized=True)
    hadamard_combo = compose_matrices(P, D, H)

    # Decorrel metrics
    cov_klt = klt_fwd @ cov @ klt_fwd.T
    corr_klt = _corr_from_cov(cov_klt)
    cov_had = hadamard_combo @ cov @ hadamard_combo.T
    corr_had = _corr_from_cov(cov_had)

    print(f"Processed images: {processed} ({count} samples)")
    print("Before:", _decorrelation_metrics(cov, corr))
    print("KLT:   ", _decorrelation_metrics(cov_klt, corr_klt))
    print("HAD:   ", _decorrelation_metrics(cov_had, corr_had))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mean=mean,
        cov=cov,
        corr=corr,
        klt_forward=klt_fwd,
        klt_inverse=klt_inv,
        eigvals=eigvals,
        order=np.array(order, dtype=np.int32),
        perm=P,
        varnorm=D,
        hadamard=H,
        hadamard_combo=hadamard_combo,
        count=np.array([count], dtype=np.int64),
        processed=np.array([processed], dtype=np.int64),
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
