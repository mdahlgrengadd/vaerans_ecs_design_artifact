"""Tests for Arena allocator and TensorRef."""

import numpy as np
import pytest

from vaerans_ecs.core.arena import Arena, TensorRef


class TestTensorRef:
    """Tests for TensorRef dataclass."""

    def test_creation(self) -> None:
        """Test TensorRef creation with valid parameters."""
        ref = TensorRef(
            offset=0,
            shape=(10, 10),
            dtype=np.dtype(np.float32),
            strides=(40, 4),
            generation=0,
        )
        assert ref.offset == 0
        assert ref.shape == (10, 10)
        assert ref.dtype == np.float32
        assert ref.strides == (40, 4)
        assert ref.generation == 0

    def test_negative_offset(self) -> None:
        """Test that negative offset raises ValueError."""
        with pytest.raises(ValueError, match="offset must be non-negative"):
            TensorRef(
                offset=-1,
                shape=(10,),
                dtype=np.dtype(np.float32),
                strides=(4,),
                generation=0,
            )

    def test_mismatched_shape_strides(self) -> None:
        """Test that mismatched shape and strides raises ValueError."""
        with pytest.raises(ValueError, match="shape and strides must have same length"):
            TensorRef(
                offset=0,
                shape=(10, 10),
                dtype=np.dtype(np.float32),
                strides=(4,),  # Wrong length
                generation=0,
            )

    def test_negative_generation(self) -> None:
        """Test that negative generation raises ValueError."""
        with pytest.raises(ValueError, match="generation must be non-negative"):
            TensorRef(
                offset=0,
                shape=(10,),
                dtype=np.dtype(np.float32),
                strides=(4,),
                generation=-1,
            )

    def test_properties(self) -> None:
        """Test TensorRef computed properties."""
        ref = TensorRef(
            offset=0,
            shape=(2, 3, 4),
            dtype=np.dtype(np.float32),
            strides=(48, 16, 4),
            generation=0,
        )
        assert ref.ndim == 3
        assert ref.size == 24
        assert ref.nbytes == 96  # 2*3*4*4 bytes

    def test_subref_simple(self) -> None:
        """Test creating a subref with simple slicing."""
        ref = TensorRef(
            offset=0,
            shape=(4, 3, 256, 256),
            dtype=np.dtype(np.float32),
            strides=(786432, 262144, 1024, 4),
            generation=0,
        )
        # Get second image (index 1)
        subref = ref.subref((1, slice(None), slice(None), slice(None)))
        assert subref.shape == (3, 256, 256)
        assert subref.offset == 786432  # One stride along first dimension
        assert subref.generation == 0

    def test_subref_partial_slice(self) -> None:
        """Test subref with partial slicing."""
        ref = TensorRef(
            offset=0,
            shape=(10, 20),
            dtype=np.dtype(np.float32),
            strides=(80, 4),
            generation=0,
        )
        # Get rows 2:5
        subref = ref.subref((slice(2, 5), slice(None)))
        assert subref.shape == (3, 20)
        assert subref.offset == 160  # 2 * 80
        assert subref.strides == (80, 4)

    def test_subref_negative_index(self) -> None:
        """Test subref with negative indexing."""
        ref = TensorRef(
            offset=0,
            shape=(10, 20),
            dtype=np.dtype(np.float32),
            strides=(80, 4),
            generation=0,
        )
        # Get last row
        subref = ref.subref((-1, slice(None)))
        assert subref.shape == (20,)
        assert subref.offset == 720  # 9 * 80

    def test_subref_out_of_bounds(self) -> None:
        """Test that out-of-bounds subref raises IndexError."""
        ref = TensorRef(
            offset=0,
            shape=(10, 20),
            dtype=np.dtype(np.float32),
            strides=(80, 4),
            generation=0,
        )
        with pytest.raises(IndexError, match="out of bounds"):
            ref.subref((10, slice(None)))  # Index 10 is out of bounds

    def test_subref_strided_not_supported(self) -> None:
        """Test that strided slices raise NotImplementedError."""
        ref = TensorRef(
            offset=0,
            shape=(10, 20),
            dtype=np.dtype(np.float32),
            strides=(80, 4),
            generation=0,
        )
        with pytest.raises(NotImplementedError, match="Strided slices not supported"):
            ref.subref((slice(0, 10, 2), slice(None)))  # step=2


class TestArena:
    """Tests for Arena allocator."""

    def test_creation(self) -> None:
        """Test Arena creation."""
        arena = Arena(size_bytes=1024)
        assert arena.size == 1024
        assert arena.offset == 0
        assert arena.generation == 0
        assert arena.available == 1024

    def test_invalid_size(self) -> None:
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="size_bytes must be positive"):
            Arena(size_bytes=0)
        with pytest.raises(ValueError, match="size_bytes must be positive"):
            Arena(size_bytes=-100)

    def test_alloc_tensor_simple(self) -> None:
        """Test basic tensor allocation."""
        arena = Arena(size_bytes=1024)
        ref = arena.alloc_tensor((10,), np.float32)

        assert ref.shape == (10,)
        assert ref.dtype == np.float32
        assert ref.generation == 0
        assert ref.offset == 0
        assert arena.offset == 40  # 10 * 4 bytes

    def test_alloc_tensor_multiple(self) -> None:
        """Test multiple allocations."""
        arena = Arena(size_bytes=1024)

        ref1 = arena.alloc_tensor((10,), np.float32)
        ref2 = arena.alloc_tensor((5,), np.uint8)

        assert ref1.offset == 0
        assert ref2.offset == 40  # After first allocation
        assert arena.offset == 45  # 40 + 5

    def test_alloc_tensor_alignment(self) -> None:
        """Test dtype alignment."""
        arena = Arena(size_bytes=1024)

        # Allocate uint8 (1-byte aligned)
        ref1 = arena.alloc_tensor((3,), np.uint8)
        assert ref1.offset == 0
        assert arena.offset == 3

        # Allocate float32 (4-byte aligned) - should align to offset 4
        ref2 = arena.alloc_tensor((2,), np.float32)
        assert ref2.offset == 4  # Aligned from 3 to 4
        assert arena.offset == 12  # 4 + 2*4

        # Allocate float64 (8-byte aligned)
        ref3 = arena.alloc_tensor((1,), np.float64)
        assert ref3.offset == 16  # Aligned from 12 to 16
        assert arena.offset == 24  # 16 + 1*8

    def test_view_basic(self) -> None:
        """Test viewing a TensorRef."""
        arena = Arena(size_bytes=1024)
        ref = arena.alloc_tensor((10,), np.float32)

        arr = arena.view(ref)
        assert arr.shape == (10,)
        assert arr.dtype == np.float32
        assert len(arr) == 10

    def test_view_modification(self) -> None:
        """Test that view modifications are reflected in arena."""
        arena = Arena(size_bytes=1024)
        ref = arena.alloc_tensor((5,), np.float32)

        arr = arena.view(ref)
        arr[:] = [1.0, 2.0, 3.0, 4.0, 5.0]

        # View again and check values persist
        arr2 = arena.view(ref)
        assert np.array_equal(arr2, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_view_stale_ref(self) -> None:
        """Test that viewing stale ref after reset raises ValueError."""
        arena = Arena(size_bytes=1024)
        ref = arena.alloc_tensor((10,), np.float32)

        # View works before reset
        arr = arena.view(ref)
        arr[:] = 1.0

        # Reset arena
        arena.reset()

        # Now viewing should fail
        with pytest.raises(ValueError, match="Stale TensorRef"):
            arena.view(ref)

    def test_reset(self) -> None:
        """Test arena reset."""
        arena = Arena(size_bytes=1024)

        # Allocate some tensors
        ref1 = arena.alloc_tensor((10,), np.float32)
        ref2 = arena.alloc_tensor((20,), np.uint8)
        assert arena.offset == 60

        # Reset
        arena.reset()

        # Check state
        assert arena.offset == 0
        assert arena.generation == 1
        assert arena.available == 1024

        # Old refs should be invalid
        with pytest.raises(ValueError, match="Stale TensorRef"):
            arena.view(ref1)
        with pytest.raises(ValueError, match="Stale TensorRef"):
            arena.view(ref2)

        # New allocations should work
        ref3 = arena.alloc_tensor((5,), np.float32)
        assert ref3.generation == 1
        arr = arena.view(ref3)  # Should not raise
        assert arr.shape == (5,)

    def test_out_of_memory(self) -> None:
        """Test allocation failure when arena is full."""
        arena = Arena(size_bytes=100)

        # This should succeed
        ref1 = arena.alloc_tensor((10,), np.float32)  # 40 bytes
        assert ref1 is not None

        # This should fail (need 100 bytes, but only ~60 available)
        with pytest.raises(ValueError, match="Arena out of memory"):
            arena.alloc_tensor((25,), np.float32)  # 100 bytes

    def test_copy_tensor(self) -> None:
        """Test copy_tensor helper."""
        arena = Arena(size_bytes=1024)

        # Create source array
        src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        # Copy to arena
        ref = arena.copy_tensor(src)

        # Verify copy
        arr = arena.view(ref)
        assert np.array_equal(arr, src)
        assert arr.shape == src.shape
        assert arr.dtype == src.dtype

    def test_multidimensional_allocation(self) -> None:
        """Test allocating multidimensional tensors."""
        arena = Arena(size_bytes=10 << 20)  # 10 MB

        # Allocate 3D tensor
        ref = arena.alloc_tensor((4, 3, 256, 256), np.float32)
        arr = arena.view(ref)

        assert arr.shape == (4, 3, 256, 256)
        assert arr.dtype == np.float32

        # Test modification
        arr[0, 0, 0, 0] = 123.0
        arr2 = arena.view(ref)
        assert arr2[0, 0, 0, 0] == 123.0

    def test_different_dtypes(self) -> None:
        """Test allocating tensors with different dtypes."""
        arena = Arena(size_bytes=10000)

        dtypes = [np.uint8, np.int32, np.float32, np.float64]

        for dt in dtypes:
            ref = arena.alloc_tensor((10,), dt)
            arr = arena.view(ref)
            assert arr.dtype == dt
            assert arr.shape == (10,)

    def test_subref_with_arena(self) -> None:
        """Test subref integration with arena viewing."""
        arena = Arena(size_bytes=100000)

        # Allocate batch of images
        batch_ref = arena.alloc_tensor((4, 3, 8, 8), np.float32)
        batch = arena.view(batch_ref)

        # Set each image to different value
        for i in range(4):
            batch[i, :, :, :] = float(i)

        # Create subref for second image
        img1_ref = batch_ref.subref((1, slice(None), slice(None), slice(None)))
        img1 = arena.view(img1_ref)

        assert img1.shape == (3, 8, 8)
        assert np.all(img1 == 1.0)

        # Modify via subref
        img1[:] = 99.0

        # Check that modification is reflected in original batch
        batch2 = arena.view(batch_ref)
        assert np.all(batch2[1] == 99.0)
        assert np.all(batch2[0] == 0.0)  # Other images unchanged

    def test_repr(self) -> None:
        """Test Arena repr."""
        arena = Arena(size_bytes=1024)
        arena.alloc_tensor((10,), np.float32)

        repr_str = repr(arena)
        assert "Arena" in repr_str
        assert "size=1024" in repr_str
        assert "offset=40" in repr_str
        assert "generation=0" in repr_str

    def test_zero_size_tensor(self) -> None:
        """Test allocating empty tensor."""
        arena = Arena(size_bytes=1024)

        ref = arena.alloc_tensor((0,), np.float32)
        arr = arena.view(ref)

        assert arr.shape == (0,)
        assert len(arr) == 0

    def test_large_batch_allocation(self) -> None:
        """Test allocating large batch for realistic use case."""
        arena = Arena(size_bytes=512 << 20)  # 512 MB

        # Allocate batch of 8 images, 256x256 RGB, float32
        batch_ref = arena.alloc_tensor((8, 3, 256, 256), np.float32)
        batch = arena.view(batch_ref)

        expected_bytes = 8 * 3 * 256 * 256 * 4
        assert batch.nbytes == expected_bytes
        assert arena.offset == expected_bytes

        # Verify we can modify all images
        for i in range(8):
            batch[i, 0, 0, 0] = float(i)

        batch2 = arena.view(batch_ref)
        for i in range(8):
            assert batch2[i, 0, 0, 0] == float(i)
