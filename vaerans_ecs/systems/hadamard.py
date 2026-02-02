"""Hadamard transform system.

The Hadamard 4x4 transform is a simple orthogonal transform that
decorrelates 4-channel data. It's used to transform VAE latent space
into a more compressible representation.

The normalized forward transform (divides by 4 to preserve range):
    Y = (C0 + C1 + C2 + C3) / 4  [average, captures structure]
    U = (C0 - C1 + C2 - C3) / 4  [difference, lower energy]
    V = (C0 + C1 - C2 - C3) / 4  [difference, lower energy]
    W = (C0 - C1 - C2 + C3) / 4  [difference, lowest energy]

The inverse transform (no additional scaling):
    C0 = Y + U + V + W
    C1 = Y - U + V - W
    C2 = Y + U - V - W
    C3 = Y - U - V + W

This is perfectly invertible (lossless with float precision).
"""

from __future__ import annotations

import numpy as np

from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.core.system import System
from vaerans_ecs.core.world import World


class Hadamard4(System):
    """Hadamard 4-channel orthogonal transform.

    Transforms between Latent4 (z) and YUVW4 (t) representations.
    Uses normalized Hadamard transform with /4 scaling for range preservation.

    Modes:
    - 'encode'/'forward': z -> t (Latent4 to YUVW4)
    - 'decode'/'inverse': t -> z (YUVW4 to Latent4)

    Energy distribution after transform:
    - Y: ~70-90% of total energy (captures structure from all channels)
    - U, V: ~10-20% each (chroma-like differences)
    - W: ~1-5% (residual, lowest variance)
    """

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
        """Forward transform: Latent4 -> YUVW4.

        Uses normalized Hadamard with /4 scaling to preserve range:
        Y = (C0 + C1 + C2 + C3) / 4  [average, preserves range]
        U = (C0 - C1 + C2 - C3) / 4  [difference, ±range/2]
        V = (C0 + C1 - C2 - C3) / 4  [difference, ±range/2]
        W = (C0 - C1 - C2 + C3) / 4  [difference, ±range/2]
        """
        for eid in eids:
            latent = world.get_component(eid, Latent4)
            z = world.arena.view(latent.z)  # (C, H, W) = (4, H, W)

            # Allocate output tensor
            t_ref = world.arena.alloc_tensor(z.shape, z.dtype)
            t = world.arena.view(t_ref)

            # Extract channels
            C0, C1, C2, C3 = z[0], z[1], z[2], z[3]

            # Normalized Hadamard transform (divide by 4)
            t[0] = (C0 + C1 + C2 + C3) / 4.0  # Y
            t[1] = (C0 - C1 + C2 - C3) / 4.0  # U
            t[2] = (C0 + C1 - C2 - C3) / 4.0  # V
            t[3] = (C0 - C1 - C2 + C3) / 4.0  # W

            world.add_component(eid, YUVW4(t=t_ref))

    def _inverse(self, world: World, eids: list[int]) -> None:
        """Inverse transform: YUVW4 -> Latent4.

        Inverse of normalized Hadamard (no additional scaling needed):
        C0 = Y + U + V + W
        C1 = Y - U + V - W
        C2 = Y + U - V - W
        C3 = Y - U - V + W
        """
        for eid in eids:
            yuvw = world.get_component(eid, YUVW4)
            t = world.arena.view(yuvw.t)  # (C, H, W) = (4, H, W)

            # Allocate output tensor
            z_ref = world.arena.alloc_tensor(t.shape, t.dtype)
            z = world.arena.view(z_ref)

            # Extract YUVW channels
            Y, U, V, W = t[0], t[1], t[2], t[3]

            # Inverse normalized Hadamard transform
            z[0] = Y + U + V + W  # C0
            z[1] = Y - U + V - W  # C1
            z[2] = Y + U - V - W  # C2
            z[3] = Y - U - V + W  # C3

            world.add_component(eid, Latent4(z=z_ref))
