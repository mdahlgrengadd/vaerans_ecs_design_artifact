"""Hadamard analysis helpers matching the notebook workflow."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.world import World


def _ensure_latent_4(latent: np.ndarray) -> np.ndarray:
    """Normalize latent to shape (4, H, W)."""
    if latent.ndim == 4:
        latent = latent.squeeze(0)
    if latent.ndim != 3 or latent.shape[0] != 4:
        raise ValueError(f"Expected latent shape (4, H, W), got {latent.shape}")
    return latent


def quantize_latent_u8(latent: np.ndarray, bits: int = 8, clip: float = 4.0) -> np.ndarray:
    """Notebook-style latent quantization to uint8/uint16.

    Mirrors SDXL_Codec_Experiments_HighLevelCoidec.ipynb:
      - clip to [-clip, clip]
      - map to [0, 2^bits-1]
      - round to integer
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    latent = _ensure_latent_4(latent).astype(np.float32)
    latent = np.clip(latent, -clip, clip)

    levels = (1 << bits) - 1
    scaled = (latent + clip) / (2 * clip) * levels
    quantized = np.rint(scaled)

    if bits <= 8:
        return quantized.astype(np.uint8)
    return quantized.astype(np.uint16)


def hadamard_4ch_forward(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4-channel normalized Hadamard transform (range-preserving /4)."""
    latent = _ensure_latent_4(latent).astype(np.float32)
    c0, c1, c2, c3 = latent[0], latent[1], latent[2], latent[3]

    y = (c0 + c1 + c2 + c3) / 4.0
    u = (c0 - c1 + c2 - c3) / 4.0
    v = (c0 + c1 - c2 - c3) / 4.0
    w = (c0 - c1 - c2 + c3) / 4.0
    return y, u, v, w


def hadamard4_matrix(normalized: bool = True) -> np.ndarray:
    """Return the 4x4 Hadamard matrix (optionally normalized by 4)."""
    mat = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    if normalized:
        mat = mat / 4.0
    return mat


def _cov_corr(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute covariance and correlation matrices for 4-channel data."""
    data = channels.reshape(4, -1).astype(np.float64)
    data = data - data.mean(axis=1, keepdims=True)
    n = data.shape[1]
    if n < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    cov = (data @ data.T) / (n - 1)
    std = np.sqrt(np.diag(cov))
    denom = std[:, None] * std[None, :]
    denom[denom == 0] = 1.0
    corr = cov / denom
    return cov, corr


def _decorrelation_metrics(cov: np.ndarray, corr: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(cov.shape[0], dtype=bool)
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


def _channel_stats(arr: np.ndarray, name: str) -> dict[str, float | str]:
    arr_f = arr.astype(np.float64)
    return {
        "name": name,
        "mean": float(arr_f.mean()),
        "std": float(arr_f.std()),
        "variance": float(arr_f.var()),
        "min": float(arr_f.min()),
        "max": float(arr_f.max()),
        "dynamic_range": float(arr_f.max() - arr_f.min()),
        "energy": float(np.sum(arr_f * arr_f)),
    }


def hadamard_energy_stats(
    latent: np.ndarray,
    domain: Literal["float", "notebook_u8"] = "notebook_u8",
    bits: int = 8,
    clip: float = 4.0,
) -> dict[str, Any]:
    """Compute per-channel energy stats, matching notebook defaults when requested.

    Args:
        latent: Float latent array (4, H, W) or (1, 4, H, W).
        domain: "float" uses raw latent values; "notebook_u8" quantizes first.
        bits: Quantization bits for "notebook_u8" (default 8).
        clip: Quantization clip range for "notebook_u8" (default 4.0).
    """
    latent_4 = _ensure_latent_4(latent)

    if domain == "notebook_u8":
        latent_4 = quantize_latent_u8(latent_4, bits=bits, clip=clip).astype(np.float32)
    else:
        latent_4 = latent_4.astype(np.float32)

    y, u, v, w = hadamard_4ch_forward(latent_4)

    orig_stats = [_channel_stats(latent_4[i], f"C{i}") for i in range(4)]
    trans_stats = [
        _channel_stats(y, "Y"),
        _channel_stats(u, "U"),
        _channel_stats(v, "V"),
        _channel_stats(w, "W"),
    ]

    total_orig_energy = sum(stat["energy"] for stat in orig_stats)
    total_trans_energy = sum(stat["energy"] for stat in trans_stats)

    for stat in orig_stats:
        stat["energy_pct"] = 0.0 if total_orig_energy == 0 else stat["energy"] / total_orig_energy * 100
    for stat in trans_stats:
        stat["energy_pct"] = 0.0 if total_trans_energy == 0 else stat["energy"] / total_trans_energy * 100

    return {
        "domain": domain,
        "bits": bits,
        "clip": clip,
        "original": {stat["name"]: stat for stat in orig_stats},
        "transformed": {stat["name"]: stat for stat in trans_stats},
        "Y": trans_stats[0],
        "U": trans_stats[1],
        "V": trans_stats[2],
        "W": trans_stats[3],
    }


def hadamard_decorrelation_stats(
    latent: np.ndarray,
    domain: Literal["float", "notebook_u8"] = "float",
    bits: int = 8,
    clip: float = 4.0,
) -> dict[str, Any]:
    """Compute decorrelation metrics (cov/corr) before and after Hadamard.

    The core measure is off-diagonal correlation/energy after transform.
    Lower values indicate stronger decorrelation.
    """
    latent_4 = _ensure_latent_4(latent)
    if domain == "notebook_u8":
        latent_4 = quantize_latent_u8(latent_4, bits=bits, clip=clip).astype(np.float32)
    else:
        latent_4 = latent_4.astype(np.float32)

    cov_before, corr_before = _cov_corr(latent_4)

    y, u, v, w = hadamard_4ch_forward(latent_4)
    transformed = np.stack([y, u, v, w], axis=0)
    cov_after, corr_after = _cov_corr(transformed)

    return {
        "domain": domain,
        "bits": bits,
        "clip": clip,
        "before": _decorrelation_metrics(cov_before, corr_before),
        "after": _decorrelation_metrics(cov_after, corr_after),
        "cov_before": cov_before,
        "cov_after": cov_after,
        "corr_before": corr_before,
        "corr_after": corr_after,
    }


def hadamard_energy_stats_from_world(
    world: World,
    eid: int,
    domain: Literal["float", "notebook_u8"] = "notebook_u8",
    bits: int = 8,
    clip: float = 4.0,
) -> dict[str, Any]:
    """Convenience wrapper to compute stats from a Latent4 component."""
    latent = world.get_component(eid, Latent4)
    data = world.arena.view(latent.z)
    return hadamard_energy_stats(data, domain=domain, bits=bits, clip=clip)
