"""Tests for quality metrics systems."""

from __future__ import annotations

import numpy as np
import pytest

from vaerans_ecs.components.image import ReconRGB, RGB
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.metrics import MetricMSE, MetricMSSSIM, MetricPSNR, MetricSSIM


class TestMetricPSNR:
    """Test PSNR metric system."""

    def test_init(self):
        """Test PSNR metric initialization."""
        metric = MetricPSNR()
        assert metric.src_component == RGB
        assert metric.recon_component == ReconRGB
        assert metric.data_range is None

    def test_required_components(self):
        """Test required components."""
        metric = MetricPSNR()
        required = metric.required_components()
        assert RGB in required
        assert ReconRGB in required

    def test_produced_components(self):
        """Test produced components (should be empty)."""
        metric = MetricPSNR()
        produced = metric.produced_components()
        assert produced == []

    def test_perfect_reconstruction(self):
        """Test PSNR with perfect reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Create identical images
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        recon_ref = world.arena.copy_tensor(img)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        # Compute PSNR
        metric = MetricPSNR()
        metric.run(world, [entity])

        # Perfect reconstruction should have infinite PSNR
        psnr = world.metadata[entity]["psnr"]
        assert psnr > 100  # Very high for perfect match

    def test_noisy_reconstruction(self):
        """Test PSNR with noisy reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Original
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        # Add noise
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        recon = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        recon_ref = world.arena.copy_tensor(recon)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        # Compute PSNR
        metric = MetricPSNR()
        metric.run(world, [entity])

        # Noisy reconstruction should have lower PSNR
        psnr = world.metadata[entity]["psnr"]
        assert 20 < psnr < 40  # Typical range for noisy images

    def test_float_images(self):
        """Test PSNR with float32 images."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Float images in [0, 1]
        img = np.random.rand(64, 64, 3).astype(np.float32)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        recon = img + np.random.randn(*img.shape).astype(np.float32) * 0.01
        recon = recon.clip(0, 1)
        recon_ref = world.arena.copy_tensor(recon)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        # Compute PSNR with explicit data range
        metric = MetricPSNR(data_range=1.0)
        metric.run(world, [entity])

        psnr = world.metadata[entity]["psnr"]
        assert psnr > 0

    def test_multiple_entities(self):
        """Test PSNR with multiple entities."""
        world = World(arena_bytes=20 << 20)
        entities = []

        for _ in range(3):
            entity = world.new_entity()
            img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            pix_ref = world.arena.copy_tensor(img)
            world.add_component(entity, RGB(pix=pix_ref))

            recon_ref = world.arena.copy_tensor(img)
            world.add_component(entity, ReconRGB(pix=recon_ref))
            entities.append(entity)

        # Compute for all
        metric = MetricPSNR()
        metric.run(world, entities)

        # All should have PSNR in metadata
        for entity in entities:
            assert "psnr" in world.metadata[entity]


class TestMetricSSIM:
    """Test SSIM metric system."""

    def test_init(self):
        """Test SSIM metric initialization."""
        metric = MetricSSIM()
        assert metric.src_component == RGB
        assert metric.recon_component == ReconRGB

    def test_perfect_reconstruction(self):
        """Test SSIM with perfect reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        recon_ref = world.arena.copy_tensor(img)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        metric = MetricSSIM()
        metric.run(world, [entity])

        # Perfect reconstruction should have SSIM = 1.0
        ssim = world.metadata[entity]["ssim"]
        assert abs(ssim - 1.0) < 1e-6

    def test_noisy_reconstruction(self):
        """Test SSIM with noisy reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        recon = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        recon_ref = world.arena.copy_tensor(recon)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        metric = MetricSSIM()
        metric.run(world, [entity])

        ssim = world.metadata[entity]["ssim"]
        assert 0.7 < ssim < 1.0  # Should still be fairly similar


class TestMetricMSE:
    """Test MSE metric system."""

    def test_init(self):
        """Test MSE metric initialization."""
        metric = MetricMSE()
        assert metric.src_component == RGB
        assert metric.recon_component == ReconRGB

    def test_perfect_reconstruction(self):
        """Test MSE with perfect reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        recon_ref = world.arena.copy_tensor(img)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        metric = MetricMSE()
        metric.run(world, [entity])

        # Perfect reconstruction should have MSE = 0
        mse = world.metadata[entity]["mse"]
        assert mse == 0.0

    def test_noisy_reconstruction(self):
        """Test MSE with noisy reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
        recon = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        recon_ref = world.arena.copy_tensor(recon)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        metric = MetricMSE()
        metric.run(world, [entity])

        mse = world.metadata[entity]["mse"]
        assert mse > 0


class TestMetricMSSSIM:
    """Test MS-SSIM metric system."""

    def test_init(self):
        """Test MS-SSIM metric initialization."""
        metric = MetricMSSSIM()
        assert metric.src_component == RGB
        assert metric.recon_component == ReconRGB

    def test_perfect_reconstruction(self):
        """Test MS-SSIM with perfect reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        # Need larger image for multi-scale
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        recon_ref = world.arena.copy_tensor(img)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        metric = MetricMSSSIM()
        metric.run(world, [entity])

        # Perfect reconstruction should have MS-SSIM ≈ 1.0
        ms_ssim = world.metadata[entity]["ms_ssim"]
        assert abs(ms_ssim - 1.0) < 1e-4

    def test_noisy_reconstruction(self):
        """Test MS-SSIM with noisy reconstruction."""
        world = World(arena_bytes=10 << 20)
        entity = world.new_entity()

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        recon = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        recon_ref = world.arena.copy_tensor(recon)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        metric = MetricMSSSIM()
        metric.run(world, [entity])

        ms_ssim = world.metadata[entity]["ms_ssim"]
        assert 0.5 < ms_ssim < 1.0


class TestMetricsIntegration:
    """Test metrics with compression pipeline."""

    def test_metrics_after_compression(self):
        """Test computing metrics after compression pipeline."""
        from vaerans_ecs.components.latent import Latent4
        from vaerans_ecs.systems.ans import ANSDecode, ANSEncode
        from vaerans_ecs.systems.hadamard import Hadamard4
        from vaerans_ecs.systems.quantize import QuantizeU8
        from vaerans_ecs.systems.wavelet import WaveletCDF53

        world = World(arena_bytes=100 << 20)
        entity = world.new_entity()

        # Create source "image" (just use latent as proxy)
        src_data = np.random.rand(64, 64, 3).astype(np.float32)
        pix_ref = world.arena.copy_tensor(src_data)
        world.add_component(entity, RGB(pix=pix_ref))

        # Create latent (simulated VAE encode)
        latent_data = np.random.randn(4, 8, 8).astype(np.float32)
        latent_ref = world.arena.copy_tensor(latent_data)
        world.add_component(entity, Latent4(z=latent_ref))

        # Compress latent
        Hadamard4(mode="forward").run(world, [entity])
        WaveletCDF53(levels=2, mode="forward").run(world, [entity])
        QuantizeU8(quality=60, mode="forward").run(world, [entity])
        ANSEncode().run(world, [entity])

        # Decompress
        ANSDecode().run(world, [entity])
        QuantizeU8(quality=60, mode="inverse").run(world, [entity])
        WaveletCDF53(levels=2, mode="inverse").run(world, [entity])
        Hadamard4(mode="inverse").run(world, [entity])

        # Simulate reconstruction (just copy for testing)
        latent_recon = world.get_component(entity, Latent4)
        latent_recon_data = world.arena.view(latent_recon.z)

        # Create dummy reconstruction image
        recon_data = np.random.rand(64, 64, 3).astype(np.float32)
        recon_ref = world.arena.copy_tensor(recon_data)
        world.add_component(entity, ReconRGB(pix=recon_ref))

        # Compute all metrics
        MetricPSNR(data_range=1.0).run(world, [entity])
        MetricSSIM(data_range=1.0).run(world, [entity])
        MetricMSE().run(world, [entity])
        MetricMSSSIM(data_range=1.0).run(world, [entity])

        # Verify all metrics computed
        assert "psnr" in world.metadata[entity]
        assert "ssim" in world.metadata[entity]
        assert "mse" in world.metadata[entity]
        assert "ms_ssim" in world.metadata[entity]

        # Verify values are reasonable
        assert world.metadata[entity]["psnr"] > 0
        assert -1 <= world.metadata[entity]["ssim"] <= 1
        assert world.metadata[entity]["mse"] >= 0
        assert 0 <= world.metadata[entity]["ms_ssim"] <= 1
