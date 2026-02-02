"""Quality metrics systems for evaluating reconstruction.

Implements PSNR, SSIM, and MS-SSIM metrics using scikit-image.
Metrics store results in World metadata rather than creating components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from vaerans_ecs.components.image import ReconRGB, RGB
from vaerans_ecs.core.system import System

if TYPE_CHECKING:
    from vaerans_ecs.core.world import World


class MetricPSNR(System):
    """Compute Peak Signal-to-Noise Ratio between source and reconstruction.

    PSNR measures reconstruction quality in dB. Higher values indicate
    better quality (typically 20-50 dB for lossy compression).

    Stores result in world.metadata[eid]['psnr'].
    """

    def __init__(
        self,
        src_component: type = RGB,
        recon_component: type = ReconRGB,
        data_range: float | None = None,
    ):
        """Initialize PSNR metric system.

        Args:
            src_component: Source image component type (default: RGB)
            recon_component: Reconstructed image component type (default: ReconRGB)
            data_range: Data range for PSNR calculation (auto-detected if None)
        """
        super().__init__(mode="forward")
        self.src_component = src_component
        self.recon_component = recon_component
        self.data_range = data_range

    def required_components(self) -> list[type]:
        """Return required component types."""
        return [self.src_component, self.recon_component]

    def produced_components(self) -> list[type]:
        """Return produced component types (none - stores in metadata)."""
        return []

    def run(self, world: World, eids: list[int]) -> None:
        """Compute PSNR for entities.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        for eid in eids:
            src = world.get_component(eid, self.src_component)
            recon = world.get_component(eid, self.recon_component)

            src_data = world.arena.view(src.pix)
            recon_data = world.arena.view(recon.pix)

            # Ensure same shape
            if src_data.shape != recon_data.shape:
                raise ValueError(
                    f"Shape mismatch: src {src_data.shape} vs recon {recon_data.shape}"
                )

            # Compute PSNR
            data_range = self.data_range
            if data_range is None:
                # Auto-detect based on dtype
                if src_data.dtype == np.uint8:
                    data_range = 255.0
                else:
                    data_range = 1.0

            psnr_value = peak_signal_noise_ratio(
                src_data, recon_data, data_range=data_range
            )

            # Store in metadata
            if eid not in world.metadata:
                world.metadata[eid] = {}
            world.metadata[eid]["psnr"] = float(psnr_value)


class MetricSSIM(System):
    """Compute Structural Similarity Index between source and reconstruction.

    SSIM measures perceptual quality. Values range from -1 to 1, where
    1 indicates perfect similarity.

    Stores result in world.metadata[eid]['ssim'].
    """

    def __init__(
        self,
        src_component: type = RGB,
        recon_component: type = ReconRGB,
        data_range: float | None = None,
        channel_axis: int = -1,
    ):
        """Initialize SSIM metric system.

        Args:
            src_component: Source image component type (default: RGB)
            recon_component: Reconstructed image component type (default: ReconRGB)
            data_range: Data range for SSIM calculation (auto-detected if None)
            channel_axis: Axis for color channels (default: -1)
        """
        super().__init__(mode="forward")
        self.src_component = src_component
        self.recon_component = recon_component
        self.data_range = data_range
        self.channel_axis = channel_axis

    def required_components(self) -> list[type]:
        """Return required component types."""
        return [self.src_component, self.recon_component]

    def produced_components(self) -> list[type]:
        """Return produced component types (none - stores in metadata)."""
        return []

    def run(self, world: World, eids: list[int]) -> None:
        """Compute SSIM for entities.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        for eid in eids:
            src = world.get_component(eid, self.src_component)
            recon = world.get_component(eid, self.recon_component)

            src_data = world.arena.view(src.pix)
            recon_data = world.arena.view(recon.pix)

            # Ensure same shape
            if src_data.shape != recon_data.shape:
                raise ValueError(
                    f"Shape mismatch: src {src_data.shape} vs recon {recon_data.shape}"
                )

            # Compute SSIM
            data_range = self.data_range
            if data_range is None:
                if src_data.dtype == np.uint8:
                    data_range = 255.0
                else:
                    data_range = 1.0

            ssim_value = structural_similarity(
                src_data,
                recon_data,
                data_range=data_range,
                channel_axis=self.channel_axis,
            )

            # Store in metadata
            if eid not in world.metadata:
                world.metadata[eid] = {}
            world.metadata[eid]["ssim"] = float(ssim_value)


class MetricMSE(System):
    """Compute Mean Squared Error between source and reconstruction.

    MSE is the average squared difference between images.
    Lower values indicate better reconstruction.

    Stores result in world.metadata[eid]['mse'].
    """

    def __init__(
        self,
        src_component: type = RGB,
        recon_component: type = ReconRGB,
    ):
        """Initialize MSE metric system.

        Args:
            src_component: Source image component type (default: RGB)
            recon_component: Reconstructed image component type (default: ReconRGB)
        """
        super().__init__(mode="forward")
        self.src_component = src_component
        self.recon_component = recon_component

    def required_components(self) -> list[type]:
        """Return required component types."""
        return [self.src_component, self.recon_component]

    def produced_components(self) -> list[type]:
        """Return produced component types (none - stores in metadata)."""
        return []

    def run(self, world: World, eids: list[int]) -> None:
        """Compute MSE for entities.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        for eid in eids:
            src: Any = world.get_component(eid, self.src_component)
            recon: Any = world.get_component(eid, self.recon_component)

            src_data = world.arena.view(src.pix)
            recon_data = world.arena.view(recon.pix)

            # Ensure same shape
            if src_data.shape != recon_data.shape:
                raise ValueError(
                    f"Shape mismatch: src {src_data.shape} vs recon {recon_data.shape}"
                )

            # Compute MSE
            mse_value = mean_squared_error(src_data, recon_data)

            # Store in metadata
            if eid not in world.metadata:
                world.metadata[eid] = {}
            world.metadata[eid]["mse"] = float(mse_value)


class MetricMSSSIM(System):
    """Compute Multi-Scale Structural Similarity Index.

    MS-SSIM computes SSIM at multiple scales (resolutions) and averages
    the results. Often better correlated with perceptual quality than SSIM.

    Stores result in world.metadata[eid]['ms_ssim'].
    """

    def __init__(
        self,
        src_component: type = RGB,
        recon_component: type = ReconRGB,
        data_range: float | None = None,
        channel_axis: int = -1,
    ):
        """Initialize MS-SSIM metric system.

        Args:
            src_component: Source image component type (default: RGB)
            recon_component: Reconstructed image component type (default: ReconRGB)
            data_range: Data range for calculation (auto-detected if None)
            channel_axis: Axis for color channels (default: -1)
        """
        super().__init__(mode="forward")
        self.src_component = src_component
        self.recon_component = recon_component
        self.data_range = data_range
        self.channel_axis = channel_axis

    def required_components(self) -> list[type]:
        """Return required component types."""
        return [self.src_component, self.recon_component]

    def produced_components(self) -> list[type]:
        """Return produced component types (none - stores in metadata)."""
        return []

    def run(self, world: World, eids: list[int]) -> None:
        """Compute MS-SSIM for entities.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        from skimage.metrics import structural_similarity

        for eid in eids:
            src: Any = world.get_component(eid, self.src_component)
            recon: Any = world.get_component(eid, self.recon_component)

            src_data = world.arena.view(src.pix)
            recon_data = world.arena.view(recon.pix)

            # Ensure same shape
            if src_data.shape != recon_data.shape:
                raise ValueError(
                    f"Shape mismatch: src {src_data.shape} vs recon {recon_data.shape}"
                )

            # Compute SSIM at multiple scales
            data_range = self.data_range
            if data_range is None:
                if src_data.dtype == np.uint8:
                    data_range = 255.0
                else:
                    data_range = 1.0

            # Compute MS-SSIM by downsampling and averaging
            # Simple implementation: compute SSIM at 3 scales
            scales = []
            current_src = src_data.copy()
            current_recon = recon_data.copy()

            for scale_idx in range(3):
                if min(current_src.shape[:2]) < 16:
                    break  # Image too small for more scales

                ssim_val = structural_similarity(
                    current_src,
                    current_recon,
                    data_range=data_range,
                    channel_axis=self.channel_axis,
                )
                scales.append(ssim_val)

                # Downsample by factor of 2
                if scale_idx < 2:  # Don't downsample after last scale
                    h, w = current_src.shape[:2]
                    new_h, new_w = h // 2, w // 2
                    if new_h < 16 or new_w < 16:
                        break

                    # Simple downsampling (average pooling)
                    if current_src.ndim == 3:
                        current_src = current_src[: new_h * 2, : new_w * 2, :]
                        current_recon = current_recon[: new_h * 2, : new_w * 2, :]
                        current_src = current_src.reshape(
                            new_h, 2, new_w, 2, current_src.shape[2]
                        ).mean(axis=(1, 3))
                        current_recon = current_recon.reshape(
                            new_h, 2, new_w, 2, current_recon.shape[2]
                        ).mean(axis=(1, 3))
                    else:
                        current_src = current_src[: new_h * 2, : new_w * 2]
                        current_recon = current_recon[: new_h * 2, : new_w * 2]
                        current_src = current_src.reshape(new_h, 2, new_w, 2).mean(
                            axis=(1, 3)
                        )
                        current_recon = current_recon.reshape(new_h, 2, new_w, 2).mean(
                            axis=(1, 3)
                        )

            # Average across scales
            ms_ssim = float(np.mean(scales))

            # Store in metadata
            if eid not in world.metadata:
                world.metadata[eid] = {}
            world.metadata[eid]["ms_ssim"] = ms_ssim
