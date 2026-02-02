"""Hadamard transform system.

The Hadamard 4x4 transform is a simple orthogonal transform that
decorrelates 4-channel data. It's used to transform VAE latent space
into a more compressible representation.

The forward transform uses:
H4 = [[1,1,1,1], [1,1,-1,-1], [1,-1,-1,1], [1,-1,1,-1]] / 2

This matrix is orthogonal (H^T = H^-1), so the inverse is the transpose.
"""

from __future__ import annotations

import numpy as np

from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.core.system import System
from vaerans_ecs.core.world import World


class Hadamard4(System):
    """Hadamard 4-channel orthogonal transform.

    Transforms between Latent4 (z) and YUVW4 (t) representations.

    Modes:
    - 'encode'/'forward': z -> t (Latent4 to YUVW4)
    - 'decode'/'inverse': t -> z (YUVW4 to Latent4)
    """

    # Precomputed Hadamard matrix (orthogonal, no scaling)
    _H4 = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ], dtype=np.float32) / 2.0

    # Inverse is transpose (since matrix is orthogonal)
    _H4_inv = _H4.T

    def required_components(self) -> list[type]:
        """Return required input components."""
        if self.mode in ("encode", "forward"):
            return [Latent4]
        else:  # decode, inverse
            return [YUVW4]

    def produced_components(self) -> list[type]:
        """Return produced output components."""
        if self.mode in ("encode", "forward"):
            return [YUVW4]
        else:  # decode, inverse
            return [Latent4]

    def run(self, world: World, eids: list[int]) -> None:
        """Apply Hadamard transform to entities."""
        if self.mode in ("encode", "forward"):
            self._forward(world, eids)
        else:  # decode, inverse
            self._inverse(world, eids)

    def _forward(self, world: World, eids: list[int]) -> None:
        """Forward transform: Latent4 -> YUVW4."""
        for eid in eids:
            latent = world.get_component(eid, Latent4)
            z = world.arena.view(latent.z)  # (C, H, W)

            # Allocate output tensor
            t_ref = world.arena.alloc_tensor(z.shape, z.dtype)
            t = world.arena.view(t_ref)

            # Apply Hadamard to each spatial location
            # z has shape (4, H, W), we apply H4 to the 4 channels at each pixel
            for h in range(z.shape[1]):
                for w in range(z.shape[2]):
                    # Get 4 channel values at this pixel
                    pixel = z[:, h, w]  # (4,)
                    # Apply transform: t = H4 @ z
                    transformed = self._H4 @ pixel
                    t[:, h, w] = transformed

            world.add_component(eid, YUVW4(t=t_ref))

    def _inverse(self, world: World, eids: list[int]) -> None:
        """Inverse transform: YUVW4 -> Latent4."""
        for eid in eids:
            yuvw = world.get_component(eid, YUVW4)
            t = world.arena.view(yuvw.t)  # (C, H, W)

            # Allocate output tensor
            z_ref = world.arena.alloc_tensor(t.shape, t.dtype)
            z = world.arena.view(z_ref)

            # Apply inverse Hadamard to each spatial location
            for h in range(t.shape[1]):
                for w in range(t.shape[2]):
                    # Get 4 channel values at this pixel
                    pixel = t[:, h, w]  # (4,)
                    # Apply inverse transform: z = H4^T @ t
                    recovered = self._H4_inv @ pixel
                    z[:, h, w] = recovered

            world.add_component(eid, Latent4(z=z_ref))

    @staticmethod
    def _hadamard_batch(data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply Hadamard transform to batch of spatial data (vectorized).

        Args:
            data: (C, H, W) tensor
            matrix: (C, C) transform matrix

        Returns:
            (C, H, W) transformed tensor
        """
        # Reshape to (C, H*W), apply transform, reshape back
        c, h, w = data.shape
        data_flat = data.reshape(c, h * w)  # (C, HW)
        result_flat = matrix @ data_flat  # (C, HW)
        result: np.ndarray = result_flat.reshape(c, h, w)
        return result
