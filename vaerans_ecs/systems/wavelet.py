"""Wavelet decomposition systems using PyWavelets.

Implements multi-level 2D wavelet decomposition for energy concentration
in image compression pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pywt

from vaerans_ecs.components.latent import YUVW4
from vaerans_ecs.components.wavelet import WaveletPyr
from vaerans_ecs.core.arena import TensorRef
from vaerans_ecs.core.system import System

if TYPE_CHECKING:
    from vaerans_ecs.core.world import World


class WaveletCDF53(System):
    """CDF 5/3 biorthogonal wavelet decomposition.

    The CDF 5/3 (Cohen-Daubechies-Feauveau) wavelet is a biorthogonal
    wavelet commonly used in image compression (JPEG 2000).

    Forward mode: YUVW4 → WaveletPyr
    Inverse mode: WaveletPyr → YUVW4
    """

    def __init__(
        self,
        levels: int = 4,
        mode: Literal["forward", "inverse"] = "forward",
    ):
        """Initialize CDF 5/3 wavelet system.

        Args:
            levels: Number of decomposition levels (1-10)
            mode: 'forward' for decomposition, 'inverse' for reconstruction
        """
        super().__init__(mode=mode)
        if not 1 <= levels <= 10:
            raise ValueError(f"levels must be in [1, 10], got {levels}")
        self.levels = levels
        self.wavelet = "bior2.2"  # CDF 5/3 in PyWavelets

    def required_components(self) -> list[type]:
        """Return required component types based on mode."""
        if self.mode == "forward":
            return [YUVW4]
        else:
            return [WaveletPyr]

    def produced_components(self) -> list[type]:
        """Return produced component types based on mode."""
        if self.mode == "forward":
            return [WaveletPyr]
        else:
            return [YUVW4]

    def run(self, world: World, eids: list[int]) -> None:
        """Execute wavelet transform on entities.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        if self.mode == "forward":
            self._run_forward(world, eids)
        else:
            self._run_inverse(world, eids)

    def _run_forward(self, world: World, eids: list[int]) -> None:
        """Forward wavelet decomposition: YUVW4 → WaveletPyr."""
        for eid in eids:
            yuvw = world.get_component(eid, YUVW4)
            t = world.arena.view(yuvw.t)  # Shape: (4, H, W)

            # Apply wavelet decomposition per channel
            coeffs_list = []
            for c in range(t.shape[0]):
                coeffs = pywt.wavedec2(t[c], self.wavelet, level=self.levels)
                coeffs_list.append(coeffs)

            # Pack coefficients into contiguous memory
            packed_ref, index_ref = self._pack_coefficients(world, coeffs_list)

            # Create WaveletPyr component
            pyr = WaveletPyr(
                packed=packed_ref,
                index=index_ref,
                levels=self.levels,
                wavelet=self.wavelet,
            )
            world.add_component(eid, pyr)

    def _run_inverse(self, world: World, eids: list[int]) -> None:
        """Inverse wavelet reconstruction: WaveletPyr → YUVW4."""
        for eid in eids:
            pyr = world.get_component(eid, WaveletPyr)

            # Unpack coefficients
            coeffs_list = self._unpack_coefficients(world, pyr)

            # Reconstruct each channel
            channels = []
            for coeffs in coeffs_list:
                recon = pywt.waverec2(coeffs, pyr.wavelet)
                channels.append(recon)

            # Stack channels
            t_recon = np.stack(channels, axis=0).astype(np.float32)

            # Allocate in arena
            t_ref = world.arena.copy_tensor(t_recon)

            # Create YUVW4 component
            yuvw = YUVW4(t=t_ref)
            world.add_component(eid, yuvw)

    def _pack_coefficients(
        self, world: World, coeffs_list: list[list[np.ndarray]]
    ) -> tuple[TensorRef, TensorRef]:
        """Pack wavelet coefficients into contiguous arena memory.

        Args:
            world: World with arena allocator
            coeffs_list: List of coefficient tuples from pywt.wavedec2

        Returns:
            (packed_ref, index_ref) tuple
        """
        # PyWavelets returns: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        # We flatten all coefficients and store index info for unpacking

        all_coeffs = []
        index_data = []

        for channel_idx, coeffs in enumerate(coeffs_list):
            # coeffs[0] is the approximation (LL) at coarsest level
            # coeffs[1:] are tuples of (LH, HL, HH) details at each level
            cA = coeffs[0]
            all_coeffs.append(cA.ravel())
            # Pad to 5 elements: ch, level, H, W, detail_idx (-1 for approx)
            index_data.append([channel_idx, 0, cA.shape[0], cA.shape[1], -1])

            for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
                for detail_idx, detail in enumerate([cH, cV, cD]):
                    all_coeffs.append(detail.ravel())
                    index_data.append(
                        [channel_idx, level_idx, detail.shape[0],
                            detail.shape[1], detail_idx]
                    )

        # Concatenate all coefficients
        packed_array = np.concatenate(all_coeffs).astype(np.float32)
        packed_ref = world.arena.copy_tensor(packed_array)

        # Store index as structured array (now all rows have 5 elements)
        index_array = np.array(index_data, dtype=np.int32)
        index_ref = world.arena.copy_tensor(index_array)

        return packed_ref, index_ref

    def _unpack_coefficients(self, world: World, pyr: WaveletPyr) -> list[list[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Unpack wavelet coefficients from arena memory.

        Args:
            world: World with arena allocator
            pyr: WaveletPyr component

        Returns:
            List of coefficient tuples for each channel
        """
        packed = world.arena.view(pyr.packed)
        index = world.arena.view(pyr.index)

        # Reconstruct coefficient structure
        # Group by channel
        num_channels = int(index[:, 0].max()) + 1
        coeffs_list: list[list[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            [] for _ in range(num_channels)
        ]

        offset = 0
        for row in index:
            ch, level, h, w, detail_idx = row
            size = h * w
            coeff = packed[offset: offset + size].reshape(h, w)

            if detail_idx == -1:
                # Approximation coefficients (stored first in list)
                coeffs_list[ch].insert(0, coeff)
            else:
                # Detail coefficients
                # Ensure we have the tuple structure at this level
                while len(coeffs_list[ch]) <= level:
                    coeffs_list[ch].append([None, None, None])

                if isinstance(coeffs_list[ch][level], list):
                    coeffs_list[ch][level][detail_idx] = coeff

            offset += size

        # Convert detail lists to tuples
        for ch in range(num_channels):
            for i in range(1, len(coeffs_list[ch])):
                if isinstance(coeffs_list[ch][i], list):
                    coeffs_list[ch][i] = tuple(coeffs_list[ch][i])

        return coeffs_list


class WaveletHaar(WaveletCDF53):
    """Haar wavelet decomposition.

    The Haar wavelet is the simplest wavelet, equivalent to a step function.
    Computationally faster than CDF 5/3 but with less compression efficiency.

    Forward mode: YUVW4 → WaveletPyr
    Inverse mode: WaveletPyr → YUVW4
    """

    def __init__(
        self,
        levels: int = 4,
        mode: Literal["forward", "inverse"] = "forward",
    ):
        """Initialize Haar wavelet system.

        Args:
            levels: Number of decomposition levels (1-10)
            mode: 'forward' for decomposition, 'inverse' for reconstruction
        """
        # Call parent but override wavelet type
        super().__init__(levels=levels, mode=mode)
        self.wavelet = "haar"
