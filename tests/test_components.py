"""Tests for component types."""

import numpy as np
import pytest

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.image import RGB, ReconRGB, BlurRGB
from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.components.quant import QuantParams, SymbolsU8
from vaerans_ecs.components.residual import Residual
from vaerans_ecs.components.wavelet import WaveletPyr
from vaerans_ecs.core.arena import Arena


class TestImageComponents:
    """Tests for image components."""

    def test_rgb_creation(self) -> None:
        """Test RGB component creation."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.uint8)

        rgb = RGB(pix=ref, colorspace='sRGB')
        assert rgb.pix == ref
        assert rgb.colorspace == 'sRGB'

    def test_rgb_default_colorspace(self) -> None:
        """Test RGB component with default colorspace."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.uint8)

        rgb = RGB(pix=ref)
        assert rgb.colorspace == 'sRGB'

    def test_recon_rgb_creation(self) -> None:
        """Test ReconRGB component creation."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.float32)

        recon = ReconRGB(pix=ref, colorspace='sRGB')
        assert recon.pix == ref
        assert recon.colorspace == 'sRGB'

    def test_blur_rgb_creation(self) -> None:
        """Test BlurRGB component creation."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.float32)

        blur = BlurRGB(pix=ref, sigma=1.5)
        assert blur.pix == ref
        assert blur.sigma == 1.5

    def test_blur_rgb_invalid_sigma(self) -> None:
        """Test BlurRGB with invalid sigma raises ValidationError."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.float32)

        with pytest.raises(ValueError):  # Pydantic validation
            BlurRGB(pix=ref, sigma=0.0)

        with pytest.raises(ValueError):
            BlurRGB(pix=ref, sigma=-1.0)


class TestLatentComponents:
    """Tests for latent space components."""

    def test_latent4_creation(self) -> None:
        """Test Latent4 component creation."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((4, 32, 32), np.float32)

        latent = Latent4(z=ref)
        assert latent.z == ref

    def test_yuvw4_creation(self) -> None:
        """Test YUVW4 component creation."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((4, 32, 32), np.float32)

        yuvw = YUVW4(t=ref)
        assert yuvw.t == ref


class TestWaveletComponents:
    """Tests for wavelet components."""

    def test_wavelet_pyr_creation(self) -> None:
        """Test WaveletPyr component creation."""
        arena = Arena(size_bytes=100000)
        packed_ref = arena.alloc_tensor((4, 64, 64), np.float32)
        index_ref = arena.alloc_tensor((4, 5), np.int32)

        pyr = WaveletPyr(
            packed=packed_ref,
            index=index_ref,
            levels=4,
            wavelet='bior2.2'
        )
        assert pyr.packed == packed_ref
        assert pyr.index == index_ref
        assert pyr.levels == 4
        assert pyr.wavelet == 'bior2.2'

    def test_wavelet_pyr_default_wavelet(self) -> None:
        """Test WaveletPyr with default wavelet."""
        arena = Arena(size_bytes=100000)
        packed_ref = arena.alloc_tensor((4, 64, 64), np.float32)
        index_ref = arena.alloc_tensor((4, 5), np.int32)

        pyr = WaveletPyr(packed=packed_ref, index=index_ref, levels=4)
        assert pyr.wavelet == 'bior2.2'

    def test_wavelet_pyr_invalid_levels(self) -> None:
        """Test WaveletPyr with invalid levels raises ValidationError."""
        arena = Arena(size_bytes=100000)
        packed_ref = arena.alloc_tensor((4, 64, 64), np.float32)
        index_ref = arena.alloc_tensor((4, 5), np.int32)

        with pytest.raises(ValueError):  # levels > 10
            WaveletPyr(packed=packed_ref, index=index_ref, levels=11)

        with pytest.raises(ValueError):  # levels < 1
            WaveletPyr(packed=packed_ref, index=index_ref, levels=0)


class TestQuantizationComponents:
    """Tests for quantization components."""

    def test_quant_params_creation(self) -> None:
        """Test QuantParams component creation."""
        arena = Arena(size_bytes=100000)
        scales_ref = arena.alloc_tensor((4,), np.float32)
        offsets_ref = arena.alloc_tensor((4,), np.float32)

        params = QuantParams(
            scales=scales_ref,
            offsets=offsets_ref,
            quality=50,
            per_band=True
        )
        assert params.scales == scales_ref
        assert params.offsets == offsets_ref
        assert params.quality == 50
        assert params.per_band is True

    def test_quant_params_invalid_quality(self) -> None:
        """Test QuantParams with invalid quality raises ValidationError."""
        arena = Arena(size_bytes=100000)
        scales_ref = arena.alloc_tensor((4,), np.float32)
        offsets_ref = arena.alloc_tensor((4,), np.float32)

        with pytest.raises(ValueError):  # quality > 100
            QuantParams(
                scales=scales_ref,
                offsets=offsets_ref,
                quality=101
            )

        with pytest.raises(ValueError):  # quality < 1
            QuantParams(
                scales=scales_ref,
                offsets=offsets_ref,
                quality=0
            )

    def test_symbols_u8_creation(self) -> None:
        """Test SymbolsU8 component creation."""
        arena = Arena(size_bytes=100000)
        data_ref = arena.alloc_tensor((1024,), np.uint8)
        scales_ref = arena.alloc_tensor((4,), np.float32)
        offsets_ref = arena.alloc_tensor((4,), np.float32)

        params = QuantParams(
            scales=scales_ref,
            offsets=offsets_ref,
            quality=50
        )

        symbols = SymbolsU8(data=data_ref, params=params)
        assert symbols.data == data_ref
        assert symbols.params == params


class TestEntropyComponents:
    """Tests for entropy coding components."""

    def test_ans_bitstream_creation(self) -> None:
        """Test ANSBitstream component creation."""
        arena = Arena(size_bytes=100000)
        data_ref = arena.alloc_tensor((512,), np.uint8)
        probs_ref = arena.alloc_tensor((256,), np.float32)

        bitstream = ANSBitstream(
            data=data_ref,
            probs=probs_ref,
            initial_state=12345
        )
        assert bitstream.data == data_ref
        assert bitstream.probs == probs_ref
        assert bitstream.initial_state == 12345

    def test_ans_bitstream_zero_state(self) -> None:
        """Test ANSBitstream with zero initial state."""
        arena = Arena(size_bytes=100000)
        data_ref = arena.alloc_tensor((512,), np.uint8)
        probs_ref = arena.alloc_tensor((256,), np.float32)

        bitstream = ANSBitstream(
            data=data_ref,
            probs=probs_ref,
            initial_state=0
        )
        assert bitstream.initial_state == 0

    def test_ans_bitstream_invalid_state(self) -> None:
        """Test ANSBitstream with invalid state raises ValidationError."""
        arena = Arena(size_bytes=100000)
        data_ref = arena.alloc_tensor((512,), np.uint8)
        probs_ref = arena.alloc_tensor((256,), np.float32)

        with pytest.raises(ValueError):  # negative state
            ANSBitstream(
                data=data_ref,
                probs=probs_ref,
                initial_state=-1
            )


class TestResidualComponents:
    """Tests for residual components."""

    def test_residual_creation(self) -> None:
        """Test Residual component creation."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.float32)

        residual = Residual(tensor=ref, scale=0.5)
        assert residual.tensor == ref
        assert residual.scale == 0.5

    def test_residual_default_scale(self) -> None:
        """Test Residual component with default scale."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.float32)

        residual = Residual(tensor=ref)
        assert residual.scale == 1.0


class TestComponentSerialization:
    """Tests for component serialization (Pydantic)."""

    def test_component_model_dump(self) -> None:
        """Test that components can be dumped to dict."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.uint8)

        rgb = RGB(pix=ref, colorspace='sRGB')

        # Pydantic model_dump includes TensorRef as dict
        dumped = rgb.model_dump(mode='python')
        assert 'pix' in dumped
        assert 'colorspace' in dumped
        assert dumped['colorspace'] == 'sRGB'

    def test_component_validation(self) -> None:
        """Test that Pydantic validation works."""
        arena = Arena(size_bytes=100000)
        ref = arena.alloc_tensor((64, 64, 3), np.uint8)

        # Valid creation
        rgb = RGB(pix=ref, colorspace='sRGB')
        assert rgb is not None

        # Test with invalid colorspace type (should work, no constraint)
        rgb2 = RGB(pix=ref, colorspace='unknown')
        assert rgb2.colorspace == 'unknown'

    def test_multiple_components_same_arena(self) -> None:
        """Test creating multiple components from same arena."""
        arena = Arena(size_bytes=100000)

        ref1 = arena.alloc_tensor((64, 64, 3), np.uint8)
        ref2 = arena.alloc_tensor((4, 32, 32), np.float32)
        ref3 = arena.alloc_tensor((32, 32), np.float32)

        rgb = RGB(pix=ref1)
        latent = Latent4(z=ref2)
        residual = Residual(tensor=ref3)

        assert rgb.pix == ref1
        assert latent.z == ref2
        assert residual.tensor == ref3

        # All should be viewable
        assert arena.view(rgb.pix).shape == (64, 64, 3)
        assert arena.view(latent.z).shape == (4, 32, 32)
        assert arena.view(residual.tensor).shape == (32, 32)
