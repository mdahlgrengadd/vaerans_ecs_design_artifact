"""Utilities for 4-channel linear transforms (KLT, reorder, normalize)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from vaerans_ecs.core.world import World


def _ensure_4ch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        arr = arr.squeeze(0)
    if arr.ndim != 3 or arr.shape[0] != 4:
        raise ValueError(f"Expected shape (4, H, W) or (1, 4, H, W); got {arr.shape}")
    return arr


def covariance_4ch_from_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Compute 4x4 covariance over a list of 4-channel tensors."""
    if not arrays:
        raise ValueError("Need at least one array to compute covariance.")

    sum_vec = np.zeros(4, dtype=np.float64)
    sum_outer = np.zeros((4, 4), dtype=np.float64)
    count = 0

    for arr in arrays:
        data = _ensure_4ch(arr).reshape(4, -1).astype(np.float64)
        sum_vec += data.sum(axis=1)
        sum_outer += data @ data.T
        count += data.shape[1]

    if count < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")

    mean = sum_vec / count
    cov = (sum_outer - count * (mean[:, None] * mean[None, :])) / (count - 1)
    return cov


def covariance_4ch_from_world(
    world: World,
    eids: Iterable[int],
    component: type,
    attr: str,
) -> np.ndarray:
    """Compute 4x4 covariance over entities for a given component/attr."""
    sum_vec = np.zeros(4, dtype=np.float64)
    sum_outer = np.zeros((4, 4), dtype=np.float64)
    count = 0

    for eid in eids:
        comp = world.get_component(eid, component)
        data = _ensure_4ch(world.arena.view(getattr(comp, attr))).reshape(4, -1).astype(
            np.float64
        )
        sum_vec += data.sum(axis=1)
        sum_outer += data @ data.T
        count += data.shape[1]

    if count < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")

    mean = sum_vec / count
    cov = (sum_outer - count * (mean[:, None] * mean[None, :])) / (count - 1)
    return cov


def klt_from_covariance(cov: np.ndarray, sort_desc: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KLT (PCA) transform from covariance.

    Returns (forward, inverse, eigenvalues).
    Forward is V^T, inverse is V, where columns of V are eigenvectors.
    """
    if cov.shape != (4, 4):
        raise ValueError(f"Expected 4x4 covariance, got {cov.shape}")
    eigvals, eigvecs = np.linalg.eigh(cov)
    if sort_desc:
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    forward = eigvecs.T
    inverse = eigvecs
    return forward.astype(np.float32), inverse.astype(np.float32), eigvals.astype(np.float64)


def reorder_indices_by_variance(cov: np.ndarray, descending: bool = True) -> list[int]:
    """Order channel indices by variance (diagonal of covariance)."""
    if cov.shape != (4, 4):
        raise ValueError(f"Expected 4x4 covariance, got {cov.shape}")
    variances = np.diag(cov)
    order = np.argsort(variances)
    if descending:
        order = order[::-1]
    return order.tolist()


def permutation_matrix(order: Sequence[int]) -> np.ndarray:
    """Create a 4x4 permutation matrix for the given order."""
    if len(order) != 4:
        raise ValueError("Permutation order must have length 4.")
    mat = np.zeros((4, 4), dtype=np.float32)
    for i, j in enumerate(order):
        mat[i, int(j)] = 1.0
    return mat


def variance_normalization_matrix(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Diagonal matrix to normalize per-channel variance to 1."""
    if cov.shape != (4, 4):
        raise ValueError(f"Expected 4x4 covariance, got {cov.shape}")
    var = np.diag(cov)
    scale = 1.0 / np.sqrt(np.maximum(var, eps))
    return np.diag(scale.astype(np.float32))


def compose_matrices(*matrices: np.ndarray) -> np.ndarray:
    """Compose transforms left-to-right: M_total = M_n @ ... @ M_2 @ M_1."""
    if not matrices:
        raise ValueError("Need at least one matrix to compose.")
    result = matrices[0].astype(np.float32)
    for mat in matrices[1:]:
        result = mat.astype(np.float32) @ result
    return result
