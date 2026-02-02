"""Integration tests for full compression pipeline.

Tests the complete flow: VAE → Hadamard → Wavelet → Quantize → ANS
"""

from __future__ import annotations

import numpy as np
import pytest

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.image import ReconRGB, RGB
from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.components.quant import SymbolsU8
from vaerans_ecs.components.wavelet import WaveletPyr
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.ans import ANSDecode, ANSEncode
from vaerans_ecs.systems.hadamard import Hadamard4
from vaerans_ecs.systems.quantize import QuantizeU8
from vaerans_ecs.systems.wavelet import WaveletCDF53


class TestFullPipelineWithoutVAE:
    """Test full pipeline without VAE (using dummy latents)."""

    def test_compress_decompress_round_trip(self):
        """Test full compress-decompress round trip with dummy latents."""
        world = World(arena_bytes=100 << 20)
        entity = world.new_entity()

        # Start with dummy latent (simulate VAE output)
        latent_data = np.random.randn(4, 32, 32).astype(np.float32)
        latent_ref = world.arena.copy_tensor(latent_data)
        world.add_component(entity, Latent4(z=latent_ref))

        # COMPRESS PIPELINE: Latent4 → Hadamard → Wavelet → Quantize → ANS
        
        # 1. Hadamard forward
        hadamard_fwd = Hadamard4(mode="forward")
        hadamard_fwd.run(world, [entity])
        assert world.has_component(entity, YUVW4)

        # 2. Wavelet decomposition
        wavelet_fwd = WaveletCDF53(levels=3, mode="forward")
        wavelet_fwd.run(world, [entity])
        assert world.has_component(entity, WaveletPyr)

        # 3. Quantization
        quantize = QuantizeU8(quality=70, mode="forward")
        quantize.run(world, [entity])
        assert world.has_component(entity, SymbolsU8)

        # 4. ANS encoding
        ans_enc = ANSEncode()
        ans_enc.run(world, [entity])
        assert world.has_component(entity, ANSBitstream)

        # Get compressed bitstream
        bitstream = world.get_component(entity, ANSBitstream)
        compressed_data = world.arena.view(bitstream.data)
        print(f"Compressed size: {len(compressed_data)} bytes")

        # DECOMPRESS PIPELINE: ANS → Quantize → Wavelet → Hadamard → Latent4

        # 5. ANS decoding
        ans_dec = ANSDecode()
        ans_dec.run(world, [entity])
        assert world.has_component(entity, SymbolsU8)

        # 6. Dequantization
        dequantize = QuantizeU8(quality=70, mode="inverse")
        dequantize.run(world, [entity])
        assert world.has_component(entity, WaveletPyr)

        # 7. Wavelet reconstruction
        wavelet_inv = WaveletCDF53(levels=3, mode="inverse")
        wavelet_inv.run(world, [entity])
        assert world.has_component(entity, YUVW4)

        # 8. Hadamard inverse
        hadamard_inv = Hadamard4(mode="inverse")
        hadamard_inv.run(world, [entity])
        assert world.has_component(entity, Latent4)

        # Verify reconstruction quality
        latent_recon = world.get_component(entity, Latent4)
        latent_recon_data = world.arena.view(latent_recon.z)

        # Should be close but not exact (lossy compression)
        mse = np.mean((latent_data - latent_recon_data) ** 2)
        print(f"Reconstruction MSE: {mse:.6f}")
        
        # Reasonable error for quality=70
        assert mse < 0.1
        assert latent_recon_data.shape == latent_data.shape

    def test_quality_impact(self):
        """Test that quality parameter affects reconstruction accuracy."""
        world = World(arena_bytes=200 << 20)

        # Original data
        latent_data = np.random.randn(4, 32, 32).astype(np.float32)

        errors = []
        for quality in [20, 50, 80]:
            entity = world.new_entity()
            latent_ref = world.arena.copy_tensor(latent_data)
            world.add_component(entity, Latent4(z=latent_ref))

            # Compress
            Hadamard4(mode="forward").run(world, [entity])
            WaveletCDF53(levels=2, mode="forward").run(world, [entity])
            QuantizeU8(quality=quality, mode="forward").run(world, [entity])
            ANSEncode().run(world, [entity])

            # Decompress
            ANSDecode().run(world, [entity])
            QuantizeU8(quality=quality, mode="inverse").run(world, [entity])
            WaveletCDF53(levels=2, mode="inverse").run(world, [entity])
            Hadamard4(mode="inverse").run(world, [entity])

            # Measure error
            latent_recon = world.get_component(entity, Latent4)
            latent_recon_data = world.arena.view(latent_recon.z)
            mse = np.mean((latent_data - latent_recon_data) ** 2)
            errors.append(mse)
            print(f"Quality {quality}: MSE = {mse:.6f}")

        # Higher quality should have lower error
        assert errors[2] < errors[1]  # quality 80 < 50
        assert errors[1] < errors[0]  # quality 50 < 20

    def test_batch_processing(self):
        """Test pipeline with multiple entities (batch)."""
        world = World(arena_bytes=200 << 20)
        entities = []

        # Create 3 entities
        for _ in range(3):
            entity = world.new_entity()
            latent_data = np.random.randn(4, 32, 32).astype(np.float32)
            latent_ref = world.arena.copy_tensor(latent_data)
            world.add_component(entity, Latent4(z=latent_ref))
            entities.append(entity)

        # Compress all (batch operations)
        Hadamard4(mode="forward").run(world, entities)
        WaveletCDF53(levels=2, mode="forward").run(world, entities)
        QuantizeU8(quality=60, mode="forward").run(world, entities)
        ANSEncode().run(world, entities)

        # All should have bitstreams
        for entity in entities:
            assert world.has_component(entity, ANSBitstream)

        # Decompress all
        ANSDecode().run(world, entities)
        QuantizeU8(quality=60, mode="inverse").run(world, entities)
        WaveletCDF53(levels=2, mode="inverse").run(world, entities)
        Hadamard4(mode="inverse").run(world, entities)

        # All should have reconstructed latents
        for entity in entities:
            assert world.has_component(entity, Latent4)

    def test_pipeline_integration(self):
        """Test all pipeline stages are properly connected."""
        world = World(arena_bytes=100 << 20)
        entity = world.new_entity()

        # Create starting component
        latent_data = np.random.randn(4, 32, 32).astype(np.float32)
        latent_ref = world.arena.copy_tensor(latent_data)
        world.add_component(entity, Latent4(z=latent_ref))

        # Track component transitions
        assert world.has_component(entity, Latent4)
        
        # Stage 1: Hadamard
        Hadamard4(mode="forward").run(world, [entity])
        assert world.has_component(entity, YUVW4)
        
        # Stage 2: Wavelet
        WaveletCDF53(levels=2, mode="forward").run(world, [entity])
        assert world.has_component(entity, WaveletPyr)
        
        # Stage 3: Quantize
        QuantizeU8(quality=50, mode="forward").run(world, [entity])
        assert world.has_component(entity, SymbolsU8)
        
        # Stage 4: ANS
        ANSEncode().run(world, [entity])
        assert world.has_component(entity, ANSBitstream)

        # Verify final bitstream is valid
        bitstream = world.get_component(entity, ANSBitstream)
        data = world.arena.view(bitstream.data)
        probs = world.arena.view(bitstream.probs)
        
        assert len(data) > 0
        assert len(probs) == 256
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_different_wavelet_levels(self):
        """Test pipeline with different wavelet decomposition levels."""
        world = World(arena_bytes=150 << 20)

        for levels in [1, 2, 4]:
            entity = world.new_entity()
            latent_data = np.random.randn(4, 64, 64).astype(np.float32)
            latent_ref = world.arena.copy_tensor(latent_data)
            world.add_component(entity, Latent4(z=latent_ref))

            # Full pipeline with specific level count
            Hadamard4(mode="forward").run(world, [entity])
            WaveletCDF53(levels=levels, mode="forward").run(world, [entity])
            QuantizeU8(quality=50, mode="forward").run(world, [entity])
            ANSEncode().run(world, [entity])

            # Verify bitstream created
            assert world.has_component(entity, ANSBitstream)

            # Decompress
            ANSDecode().run(world, [entity])
            QuantizeU8(quality=50, mode="inverse").run(world, [entity])
            WaveletCDF53(levels=levels, mode="inverse").run(world, [entity])
            Hadamard4(mode="inverse").run(world, [entity])

            # Verify reconstruction
            latent_recon = world.get_component(entity, Latent4)
            assert world.arena.view(latent_recon.z).shape == latent_data.shape


@pytest.mark.skipif(
    True,  # Skip by default since it requires real VAE models
    reason="Requires ONNX VAE models - run manually with real models"
)
class TestFullPipelineWithVAE:
    """Integration tests with real VAE models (requires ONNX files)."""

    def test_full_vae_compression(self):
        """Test complete pipeline including VAE encode/decode."""
        from vaerans_ecs.systems.vae import OnnxVAEDecode, OnnxVAEEncode
        
        world = World(arena_bytes=200 << 20)
        entity = world.new_entity()

        # Start with RGB image
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        pix_ref = world.arena.copy_tensor(img)
        world.add_component(entity, RGB(pix=pix_ref))

        # Full compress pipeline
        OnnxVAEEncode(model="sdxl-vae", mode="encode").run(world, [entity])
        Hadamard4(mode="forward").run(world, [entity])
        WaveletCDF53(levels=3, mode="forward").run(world, [entity])
        QuantizeU8(quality=70, mode="forward").run(world, [entity])
        ANSEncode().run(world, [entity])

        # Full decompress pipeline
        ANSDecode().run(world, [entity])
        QuantizeU8(quality=70, mode="inverse").run(world, [entity])
        WaveletCDF53(levels=3, mode="inverse").run(world, [entity])
        Hadamard4(mode="inverse").run(world, [entity])
        OnnxVAEDecode(model="sdxl-vae", mode="decode").run(world, [entity])

        # Verify reconstruction
        assert world.has_component(entity, ReconRGB)
        recon = world.get_component(entity, ReconRGB)
        recon_data = world.arena.view(recon.pix)
        assert recon_data.shape == img.shape
